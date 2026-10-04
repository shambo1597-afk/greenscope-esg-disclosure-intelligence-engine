"""Hyperparameter tuning of retrieval, with k-fold cross-validation.

What is tuned (retrieval only, so no API cost):
    chunk_size       how many characters per chunk
    chunk_overlap    how many characters neighbouring chunks share
    embedding model  which model turns text into vectors

Selection metric: MRR@5 (Mean Reciprocal Rank of the first gold-page chunk in
the top 5). It rewards putting the right page FIRST, not just somewhere in the
top 5, so it separates configurations better than Hit@5.

Why cross-validation: if we tried 60 configurations on 40 questions and simply
reported the best score, that score would be optimistic: some configuration
always gets lucky on a small set. So we use 5-fold cross-validation over the
questions (each fold balanced: 4 Wipro + 4 HCLTech questions):
    for each fold: pick the best configuration using the other 32 questions,
                   then score it on the 8 held-out questions it never saw.
The average held-out score is an honest estimate of how well "tune, then use"
works on new questions. The same folds are used to score the original
configuration (800/150, MiniLM) for a fair, paired comparison.

top_k is not tuned this way: Hit@k can only go up as k grows, so a retrieval
metric would always pick the largest k. Instead we report the Hit@k curve for
k = 1..10 and choose k where the curve flattens, weighed against LLM cost.

Grid: the full chunk-size x overlap grid for MiniLM, a 6-point spread of it for
the two larger models (REDUCED_GRID), then every overlap at 400-character chunks
for those models too, where stage 1 found the best region (REFINED_GRID):
36 configurations in total.

Run:  python eval/tune.py         (about 1.5 hours on a 4-core CPU;
                                    embeddings are cached in index/tuning/)
Writes eval/tuning_results.json and eval/tuning_results.md.
"""

import itertools
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from src.data import COMPANIES, RAW_DATA_DIR, chunk_pages, load_pdf  # noqa: E402
from src.model import prepare_query  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
CACHE_DIR = PROJECT_ROOT / "index" / "tuning"

CHUNK_SIZES = [400, 600, 800, 1000, 1200]
OVERLAPS = [0, 75, 150, 250]
# Two-stage search (the larger models are 2-5x slower on CPU):
#   stage 1: the full chunk-size x overlap grid with MiniLM;
#   stage 2: the other models on a spread of the grid (small/medium/large chunks,
#            with and without overlap).
REDUCED_GRID = [(size, overlap) for size in (400, 800, 1200) for overlap in (0, 150)]
# Stage 3 (coarse-to-fine): stage 1 found the best region at 400-character chunks,
# so every model is also run at the remaining 400-character overlaps. This keeps
# the model comparison fair where the best scores are.
REFINED_GRID = [(400, 75), (400, 250)]
MODELS = {
    # name -> (Hugging Face id, prefix added to the QUESTION only)
    "MiniLM-L6": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    # BGE models are trained to expect this instruction in front of search queries.
    "BGE-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "MPNet-base": ("sentence-transformers/all-mpnet-base-v2", ""),
}
BASELINE = ("MiniLM-L6", 800, 150)
N_FOLDS = 5
SEED = 42
TOP_K = 5


def config_name(cfg):
    model, size, overlap = cfg
    return f"{model} / {size} / {overlap}"


def rank_of_first_hit(retrieved_pages, gold_pages):
    gold = set(gold_pages)
    return next((r for r, p in enumerate(retrieved_pages, start=1) if p in gold), None)


def score(ranks, k=TOP_K):
    """MRR@k and Hit@k from a list of first-hit ranks (None = not found)."""
    return {
        "MRR@5": mean(1 / r if r and r <= k else 0 for r in ranks),
        "Hit@1": mean(1.0 if r == 1 else 0.0 for r in ranks),
        "Hit@3": mean(1.0 if r and r <= 3 else 0.0 for r in ranks),
        "Hit@5": mean(1.0 if r and r <= k else 0.0 for r in ranks),
    }


def make_folds(eval_set):
    """Stratified folds: each fold gets the same number of questions per company."""
    rng = random.Random(SEED)
    folds = [[] for _ in range(N_FOLDS)]
    for company in COMPANIES:
        ids = [q["id"] for q in eval_set if q["company"] == company]
        rng.shuffle(ids)
        for i, qid in enumerate(ids):
            folds[i % N_FOLDS].append(qid)
    return folds


def main():
    eval_set = json.loads((EVAL_DIR / "eval_set.json").read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pages = {c: load_pdf(RAW_DATA_DIR / f, c) for c, f in COMPANIES.items()}

    # ranks[config][question_id] = rank of first gold-page chunk (up to rank 10)
    ranks = {}
    chunk_counts = {}
    for model_name, (model_id, query_prefix) in MODELS.items():
        model = SentenceTransformer(model_id)
        queries = [query_prefix + prepare_query(q["question"]) for q in eval_set]
        query_vectors = model.encode(queries, normalize_embeddings=True)
        grid = (list(itertools.product(CHUNK_SIZES, OVERLAPS)) if model_name == BASELINE[0]
                else REDUCED_GRID + REFINED_GRID)
        for size, overlap in grid:
            cfg = (model_name, size, overlap)
            t0 = time.time()
            ranks[cfg] = {}
            for company in COMPANIES:
                chunks = chunk_pages(pages[company], chunk_size=size, chunk_overlap=overlap)
                chunk_pages_list = [c.metadata["page"] for c in chunks]
                cache = CACHE_DIR / f"{model_name}_{size}_{overlap}_{company}.npy"
                if cache.exists():
                    vectors = np.load(cache)
                else:
                    vectors = model.encode([c.page_content for c in chunks], batch_size=64,
                                           normalize_embeddings=True).astype(np.float32)
                    np.save(cache, vectors)
                chunk_counts[(cfg, company)] = len(chunks)
                # Exact inner-product search: the same computation FAISS IndexFlatIP does.
                for qi, q in enumerate(eval_set):
                    if q["company"] != company:
                        continue
                    sims = vectors @ query_vectors[qi]
                    top = np.argsort(-sims)[:10]
                    ranks[cfg][q["id"]] = rank_of_first_hit([chunk_pages_list[i] for i in top], q["gold_pages"])
            s = score(list(ranks[cfg].values()))
            print(f"{config_name(cfg):28} MRR@5 {s['MRR@5']:.3f}  Hit@5 {s['Hit@5']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    configs = list(ranks)
    all_ids = [q["id"] for q in eval_set]
    full = {cfg: score([ranks[cfg][i] for i in all_ids]) for cfg in configs}

    def best_on(ids):
        # Highest MRR@5; ties broken by Hit@5, then the smaller/cheaper model and chunk size.
        model_order = list(MODELS)
        return max(configs, key=lambda c: (
            score([ranks[c][i] for i in ids])["MRR@5"],
            score([ranks[c][i] for i in ids])["Hit@5"],
            -model_order.index(c[0]), -c[1],
        ))

    # --- k-fold cross-validation ------------------------------------------------
    folds = make_folds(eval_set)
    fold_rows = []
    for f, test_ids in enumerate(folds, start=1):
        train_ids = [i for i in all_ids if i not in test_ids]
        chosen = best_on(train_ids)
        fold_rows.append({
            "fold": f,
            "chosen": config_name(chosen),
            "train_MRR@5": score([ranks[chosen][i] for i in train_ids])["MRR@5"],
            "test_tuned": score([ranks[chosen][i] for i in test_ids]),
            "test_baseline": score([ranks[BASELINE][i] for i in test_ids]),
        })

    final = best_on(all_ids)

    # Hit@k curve (k = 1..10) for the baseline and the final configuration.
    def hit_curve(cfg):
        return [mean(1.0 if ranks[cfg][i] and ranks[cfg][i] <= k else 0.0 for i in all_ids) for k in range(1, 11)]

    results = {
        "grid": {"chunk_sizes": CHUNK_SIZES, "overlaps": OVERLAPS, "models": {k: v[0] for k, v in MODELS.items()}},
        "full_set": {config_name(c): v for c, v in full.items()},
        "chunk_counts": {f"{config_name(c)} / {co}": n for (c, co), n in chunk_counts.items()},
        "folds": fold_rows,
        "final_config": config_name(final),
        "baseline_config": config_name(BASELINE),
        "hit_curve": {"baseline": hit_curve(BASELINE), "final": hit_curve(final)},
    }
    (EVAL_DIR / "tuning_results.json").write_text(json.dumps(results, indent=1))
    write_report(results, results["full_set"], final)
    print(f"\nFinal configuration (best on all 40): {config_name(final)}")


def write_report(results, full, final):
    folds = results["folds"]
    tuned_mrr = [r["test_tuned"]["MRR@5"] for r in folds]
    base_mrr = [r["test_baseline"]["MRR@5"] for r in folds]
    tuned_hit = [r["test_tuned"]["Hit@5"] for r in folds]
    base_hit = [r["test_baseline"]["Hit@5"] for r in folds]
    ranked = sorted(full.items(), key=lambda kv: -kv[1]["MRR@5"])

    lines = [
        "# Retrieval tuning with 5-fold cross-validation",
        "",
        "Generated by `python eval/tune.py` (retrieval only, no API cost). "
        f"Grid: chunk size {CHUNK_SIZES} x overlap {OVERLAPS} with MiniLM (stage 1), plus "
        f"{[f'{s}/{o}' for s, o in REDUCED_GRID]} with {list(MODELS)[1:]} (stage 2), plus "
        f"{[f'{s}/{o}' for s, o in REFINED_GRID]} for those models around the best region (stage 3) "
        f"= {len(full)} configurations, on the 40-question eval set. "
        "Selection metric: MRR@5. Company names are removed from the query, as in the app.",
        "",
        "## Cross-validated result (the honest number)",
        "",
        "In each fold the best configuration was chosen on 32 questions and scored on the 8 held-out "
        "questions. The baseline (the original 800/150 MiniLM setup) is scored on the same held-out questions.",
        "",
        "| Fold | Chosen on training questions | Train MRR@5 | Held-out MRR@5 (tuned) | Held-out MRR@5 (baseline) | Held-out Hit@5 (tuned) | Held-out Hit@5 (baseline) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in folds:
        lines.append(
            f"| {r['fold']} | {r['chosen']} | {r['train_MRR@5']:.2f} | {r['test_tuned']['MRR@5']:.2f} | "
            f"{r['test_baseline']['MRR@5']:.2f} | {r['test_tuned']['Hit@5']:.2f} | {r['test_baseline']['Hit@5']:.2f} |"
        )
    lines += [
        f"| **Mean ± std** | | | **{mean(tuned_mrr):.2f} ± {pstdev(tuned_mrr):.2f}** | "
        f"**{mean(base_mrr):.2f} ± {pstdev(base_mrr):.2f}** | {mean(tuned_hit):.2f} ± {pstdev(tuned_hit):.2f} | "
        f"{mean(base_hit):.2f} ± {pstdev(base_hit):.2f} |",
        "",
        f"Train MRR@5 is higher than held-out MRR@5 in most folds: that gap is the optimism a single "
        f"\"best of {len(full)}\" score would have hidden.",
        "",
        f"## Final configuration: {config_name(final)}",
        "",
        "Chosen on all 40 questions (standard practice after cross-validation). Full-set scores "
        "(optimistic, because the same questions were used to choose):",
        "",
        "| Configuration | MRR@5 | Hit@1 | Hit@3 | Hit@5 |",
        "|---|---|---|---|---|",
    ]
    for cfg, s in ranked[:10]:
        mark = " (final)" if cfg == config_name(final) else ""
        lines.append(f"| {cfg}{mark} | {s['MRR@5']:.3f} | {s['Hit@1']:.2f} | {s['Hit@3']:.2f} | {s['Hit@5']:.2f} |")
    base = full[config_name(BASELINE)]
    base_rank = [c for c, _ in ranked].index(config_name(BASELINE)) + 1
    lines += [
        f"| ... | | | | |",
        f"| {config_name(BASELINE)} (baseline, rank {base_rank} of {len(ranked)}) | {base['MRR@5']:.3f} | "
        f"{base['Hit@1']:.2f} | {base['Hit@3']:.2f} | {base['Hit@5']:.2f} |",
        "",
        "## Average MRR@5 by setting (all 40 questions, the 6 grid points every model was run on)",
        "",
    ]
    for label, key_index, values in (("Model", 0, list(MODELS)), ("Chunk size", 1, CHUNK_SIZES), ("Overlap", 2, OVERLAPS)):
        cells = []
        for v in values:
            # Compare like with like: only configurations present for every model.
            matching = [s["MRR@5"] for cfg, s in full.items()
                        if cfg.split(" / ")[key_index] == str(v)
                        and (int(cfg.split(" / ")[1]), int(cfg.split(" / ")[2])) in REDUCED_GRID]
            if matching:
                cells.append(f"{v}: {mean(matching):.3f}")
        lines.append(f"- **{label}:** " + " · ".join(cells))
    curve_b, curve_f = results["hit_curve"]["baseline"], results["hit_curve"]["final"]
    lines += [
        "",
        "## Choosing top_k: Hit@k curve",
        "",
        "| k | " + " | ".join(str(k) for k in range(1, 11)) + " |",
        "|---|" + "---|" * 10,
        "| Baseline | " + " | ".join(f"{x:.2f}" for x in curve_b) + " |",
        "| Final | " + " | ".join(f"{x:.2f}" for x in curve_f) + " |",
        "",
        "## Limitations",
        "",
        "- 40 questions; with 8 questions per held-out fold, one question moves a fold's Hit@5 by 0.125. "
        "Differences of a few hundredths between configurations are within noise.",
        "- Gold pages are not yet human-verified (`verified: false`).",
        "- Only retrieval is tuned; answer quality depends on the LLM as well.",
        "- Chunk sizes above about 1,000 characters can exceed MiniLM's 256-token input limit, so the "
        "end of those chunks is silently ignored by that model (BGE and MPNet accept 512 and 384 tokens).",
        "",
    ]
    (EVAL_DIR / "tuning_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
