"""Experiment: do keyword search (BM25), hybrid search or re-ranking beat dense search?

Methods compared (all on the app's tuned index: BGE-small, 400/250 chunks):
    dense          cosine similarity of BGE vectors (the current app)
    bm25           keyword search: rewards passages containing the question's
                   words, more so for rare words ("CDP", "14001", "nationalities")
    hybrid a=x     x * dense score + (1 - x) * BM25 score, both rescaled to 0..1
    rrf            Reciprocal Rank Fusion: 1/(60 + dense rank) + 1/(60 + BM25 rank)
    ... + rerank   a cross-encoder re-scores the method's top 20 passages; it reads
                   question and passage TOGETHER, which is slower but more precise
                   than comparing two separate vectors

Selection uses the same company-balanced 5-fold cross-validation as eval/tune.py:
choose the method on 32 questions, score it on the 8 held out. Retrieval only, no
API cost. Run:  python eval/retrieval_methods.py  ->  eval/retrieval_methods.md
"""

import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import CrossEncoder  # noqa: E402

from eval.tune import make_folds  # noqa: E402
from src.data import embed_queries  # noqa: E402
from src.model import BM25, ReportIndex, prepare_query  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
POOL = 20      # candidates passed to the re-ranker
TOP_K = 5
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RERANKERS = {
    "MiniLM-CE": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BGE-reranker": "BAAI/bge-reranker-base",
}


def rescale(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def first_hit_rank(order, pages, gold):
    return next((r for r, i in enumerate(order[:TOP_K], start=1) if pages[i] in gold), None)


def scores(ranks):
    return {
        "MRR@5": mean(1 / r if r else 0 for r in ranks),
        "Hit@1": mean(1.0 if r == 1 else 0.0 for r in ranks),
        "Hit@5": mean(1.0 if r else 0.0 for r in ranks),
    }


def main():
    eval_set = json.loads((EVAL_DIR / "eval_set.json").read_text())
    index = ReportIndex()
    bm25 = {c: BM25([r["text"] for r in index.chunks[c]]) for c in index.chunks}
    rerankers = {name: CrossEncoder(model_id) for name, model_id in RERANKERS.items()}

    orders = {}      # method -> question id -> ranked chunk positions
    rerank_time = {name: [] for name in RERANKERS}
    for q in eval_set:
        company, qid = q["company"], q["id"]
        query = prepare_query(q["question"])
        dense = index.vectors[company] @ embed_queries([query])[0]
        keyword = bm25[company].scores(query)
        base = {
            "dense": np.argsort(-dense),
            "bm25": np.argsort(-keyword),
        }
        d, k = rescale(dense), rescale(keyword)
        for a in ALPHAS:
            base[f"hybrid a={a}"] = np.argsort(-(a * d + (1 - a) * k))
        dense_rank = np.empty(len(dense)); dense_rank[base["dense"]] = np.arange(len(dense))
        bm25_rank = np.empty(len(dense)); bm25_rank[base["bm25"]] = np.arange(len(dense))
        # Ties (e.g. ranks 2 and 5 vs 5 and 2) are broken by meaning rank, as in the app.
        base["rrf"] = np.lexsort((dense_rank, -(1 / (60 + dense_rank) + 1 / (60 + bm25_rank))))
        for method, order in base.items():
            orders.setdefault(method, {})[qid] = list(order[:POOL])

        # Re-rank the top 20 of dense, rrf and the hybrid blends (cross-encoder
        # scores are cached per passage, so shared candidates are scored once).
        texts = index.chunks[company]
        for name, model in rerankers.items():
            cache = {}
            for method in ["dense", "rrf"] + [f"hybrid a={a}" for a in ALPHAS]:
                pool = orders[method][qid]
                todo = [i for i in pool if i not in cache]
                if todo:
                    t0 = time.time()
                    for i, s in zip(todo, model.predict([(q["question"], texts[i]["text"]) for i in todo])):
                        cache[i] = float(s)
                    if method == "dense":
                        rerank_time[name].append(time.time() - t0)
                orders.setdefault(f"{method} + {name}", {})[qid] = sorted(pool, key=lambda i: -cache[i])
        print(f"{qid} done", flush=True)

    ranks = {
        m: {q["id"]: first_hit_rank(orders[m][q["id"]], [r["metadata"]["page"] for r in index.chunks[q["company"]]],
                                    set(q["gold_pages"])) for q in eval_set}
        for m in orders
    }
    all_ids = [q["id"] for q in eval_set]
    full = {m: scores([ranks[m][i] for i in all_ids]) for m in orders}

    def best_on(ids):
        # Ties: prefer the simpler method (no re-ranker, then dense over hybrid).
        def simplicity(m):
            return (-("+" in m), -(m != "dense"))
        return max(orders, key=lambda m: (scores([ranks[m][i] for i in ids])["MRR@5"], simplicity(m)))

    folds, fold_rows = make_folds(eval_set), []
    for f, test_ids in enumerate(folds, start=1):
        train_ids = [i for i in all_ids if i not in test_ids]
        chosen = best_on(train_ids)
        fold_rows.append((f, chosen, scores([ranks[chosen][i] for i in test_ids]),
                          scores([ranks["dense"][i] for i in test_ids])))
    final = best_on(all_ids)

    lines = [
        "# Retrieval methods: dense vs keyword (BM25) vs hybrid vs re-ranking",
        "",
        "Generated by `python eval/retrieval_methods.py` (40 questions, BGE-small 400/250 index, "
        "company names removed from the query, retrieval only).",
        "",
        "## Cross-validated comparison",
        "",
        "| Fold | Chosen on 32 questions | Held-out MRR@5 (chosen) | Held-out MRR@5 (dense, current app) | Held-out Hit@5 (chosen) | Held-out Hit@5 (dense) |",
        "|---|---|---|---|---|---|",
    ]
    for f, chosen, s, d in fold_rows:
        lines.append(f"| {f} | {chosen} | {s['MRR@5']:.2f} | {d['MRR@5']:.2f} | {s['Hit@5']:.2f} | {d['Hit@5']:.2f} |")
    c = [r[2]["MRR@5"] for r in fold_rows]; d = [r[3]["MRR@5"] for r in fold_rows]
    ch = [r[2]["Hit@5"] for r in fold_rows]; dh = [r[3]["Hit@5"] for r in fold_rows]
    lines += [
        f"| **Mean ± std** | | **{mean(c):.2f} ± {pstdev(c):.2f}** | **{mean(d):.2f} ± {pstdev(d):.2f}** | "
        f"{mean(ch):.2f} | {mean(dh):.2f} |",
        "",
        f"**Best on all 40 questions:** {final}",
        "",
        "## All methods (all 40 questions; optimistic for the best ones, see the cross-validated table)",
        "",
        "| Method | MRR@5 | Hit@1 | Hit@5 |",
        "|---|---|---|---|",
    ]
    for m, s in sorted(full.items(), key=lambda kv: -kv[1]["MRR@5"]):
        lines.append(f"| {m} | {s['MRR@5']:.3f} | {s['Hit@1']:.2f} | {s['Hit@5']:.2f} |")
    lines += [
        "",
        "## Speed of re-ranking 20 passages (CPU, per question)",
        "",
    ]
    for name, times in rerank_time.items():
        lines.append(f"- {name} (`{RERANKERS[name]}`): {mean(times):.2f} s on average")
    lines.append("")
    (EVAL_DIR / "retrieval_methods.md").write_text("\n".join(lines))
    (EVAL_DIR / "retrieval_methods.json").write_text(json.dumps(
        {"full": full, "folds": [{"fold": f, "chosen": ch_, "test": s, "test_dense": d_} for f, ch_, s, d_ in fold_rows],
         "final": final, "rerank_seconds": {k: mean(v) for k, v in rerank_time.items()}}, indent=1))
    print("\n".join(lines[5:14 + len(fold_rows)]))


if __name__ == "__main__":
    main()
