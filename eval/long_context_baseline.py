"""Baseline: give Claude the WHOLE report instead of 5 retrieved passages.

The obvious alternative to RAG is to paste the full PDF text into a long-context
model. This script answers 10 eval questions that way, with the same model
(Claude Haiku 4.5) and the same rules (SYSTEM_PROMPT), grades the answers with
the same judge as eval/answer_eval.py, and compares accuracy and cost with
GreenScope on the same questions.

Questions: ids 01, 05, 09, 13 and 17 for each company (every 4th question,
fixed in advance, not picked by result).

The report is sent with prompt caching, so repeat questions about the same
report are cheaper; both the cached and the uncached cost per question are
reported.

    python eval/long_context_baseline.py --count   # token count only, free
    python eval/long_context_baseline.py           # about US$0.50 to 1.50
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic  # noqa: E402

from src.data import COMPANIES, RAW_DATA_DIR, load_pdf  # noqa: E402
from src.model import LLM_MODEL, MAX_ANSWER_TOKENS, SYSTEM_PROMPT, get_api_key  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from answer_eval import JUDGE_MODEL, JUDGE_SCHEMA, JUDGE_SYSTEM  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTION_NUMBERS = ["01", "05", "09", "13", "17"]
# Standard (non-batch) prices per million tokens.
HAIKU = {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10}
JUDGE = {"input": 2.00, "output": 10.00}
# GreenScope's measured average cost per single-company answer (eval/answer_quality.json).
GREENSCOPE_COST = 0.003


def report_pages(company):
    pages = load_pdf(RAW_DATA_DIR / COMPANIES[company], company)
    return {p.metadata["page"]: p.page_content for p in pages}


def report_text(company, pages):
    return "\n\n---\n\n".join(f"[{company}, p.{n}]\n{text}" for n, text in sorted(pages.items()))


def haiku_cost(usage, cached):
    """Cost of one call; `cached=False` prices the whole prompt as fresh input."""
    prompt = usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    if not cached:
        return (prompt * HAIKU["input"] + usage.output_tokens * HAIKU["output"]) / 1e6
    return (usage.input_tokens * HAIKU["input"]
            + usage.cache_creation_input_tokens * HAIKU["cache_write"]
            + usage.cache_read_input_tokens * HAIKU["cache_read"]
            + usage.output_tokens * HAIKU["output"]) / 1e6


def main():
    client = anthropic.Anthropic(api_key=get_api_key())
    questions = json.loads((EVAL_DIR / "eval_set.json").read_text(encoding="utf-8"))
    chosen = [q for q in questions if q["id"].split("-")[1] in QUESTION_NUMBERS]
    texts = {c: report_pages(c) for c in COMPANIES}

    if "--count" in sys.argv:
        for company, pages in texts.items():
            n = client.messages.count_tokens(
                model=LLM_MODEL, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": report_text(company, pages)}],
            ).input_tokens
            print(f"{company}: {n:,} tokens -> about US${n * HAIKU['input'] / 1e6:.3f} per uncached question")
        return

    rows = []
    for q in chosen:
        company, pages = q["company"], texts[q["company"]]
        response = client.messages.create(
            model=LLM_MODEL, max_tokens=MAX_ANSWER_TOKENS, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Report excerpts:\n\n" + report_text(company, pages),
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"Question: {q['question']}"},
            ]}],
        )
        answer = "".join(b.text for b in response.content if b.type == "text")
        # The judge sees the gold pages plus every page the answer cites.
        cited = {int(n) for n in re.findall(r"p\.\s*(\d+)", answer)}
        shown = sorted((set(q["gold_pages"]) | cited) & set(pages))
        excerpts = "\n\n---\n\n".join(f"[{company}, p.{n}]\n{pages[n]}" for n in shown)
        judged = client.messages.create(
            model=JUDGE_MODEL, max_tokens=4000, system=JUDGE_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
            messages=[{"role": "user", "content": (
                f"Question: {q['question']}\n\nReference answer: {q['expected_answer']}\n\n"
                f"Report excerpts given to the assistant:\n\n{excerpts}\n\nAssistant's answer:\n{answer}")}],
        )
        verdict = json.loads("".join(b.text for b in judged.content if b.type == "text"))
        rows.append({
            "id": q["id"], "question": q["question"], "answer": answer, **verdict,
            "prompt_tokens": (response.usage.input_tokens + response.usage.cache_creation_input_tokens
                              + response.usage.cache_read_input_tokens),
            "cost_cached": haiku_cost(response.usage, cached=True),
            "cost_uncached": haiku_cost(response.usage, cached=False),
            "judge_cost": (judged.usage.input_tokens * JUDGE["input"]
                           + judged.usage.output_tokens * JUDGE["output"]) / 1e6,
        })
        print(f"{q['id']}: {verdict['verdict']}  (US${rows[-1]['cost_uncached']:.3f} uncached)")

    (EVAL_DIR / "long_context_baseline.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(rows)


def write_report(rows):
    n = len(rows)
    correct = sum(r["verdict"] == "correct" for r in rows)
    cited = sum(r["citations_supported"] for r in rows)
    uncached = sum(r["cost_uncached"] for r in rows) / n
    cached = sum(r["cost_cached"] for r in rows) / n
    spent = sum(r["cost_cached"] + r["judge_cost"] for r in rows)
    graded = json.loads((EVAL_DIR / "answer_quality.json").read_text(encoding="utf-8"))
    lines = [
        "# Baseline: the whole report in the prompt (long context) vs GreenScope",
        "",
        f"Generated by `python eval/long_context_baseline.py`. Same model ({LLM_MODEL}), same answer "
        f"rules and same judge ({JUDGE_MODEL}) as `eval/answer_eval.py`; the only difference is that "
        "the model receives the full report text (every page, labelled with its page number) instead "
        f"of GreenScope's 5 retrieved passages. Questions: ids {', '.join(QUESTION_NUMBERS)} for each "
        f"company, fixed in advance. 1 run per question. Total spent: US${spent:.2f}.",
        "",
        "| Method | Fully correct | Citations supported | Prompt size | Cost per question |",
        "|---|---|---|---|---|",
        f"| Whole report in prompt | {correct} of {n} | {cited} of {n} | "
        f"~{sum(r['prompt_tokens'] for r in rows) // n:,} tokens | US${uncached:.3f} "
        f"(US${cached:.3f} with prompt caching) |",
        "| GreenScope (RAG), same questions | GREENSCOPE_ROW |",
        "",
        "## Per question",
        "",
        "| ID | Question | Long context | GreenScope (3 runs) | Judge's note (long context) |",
        "|---|---|---|---|---|",
    ]
    gs = gs_by_id(graded)
    gs_correct = sum(v.count("C") for i, v in gs.items() if i in {r["id"] for r in rows})
    gs_total = sum(len(v) for i, v in gs.items() if i in {r["id"] for r in rows})
    for r in rows:
        note = r["explanation"].replace("|", "/").replace("\n", " ") if r["verdict"] != "correct" else ""
        lines.append(f"| {r['id']} | {r['question']} | {r['verdict']} | {' '.join(gs.get(r['id'], []))} | {note} |")
    text = "\n".join(lines).replace(
        "GREENSCOPE_ROW",
        f"{gs_correct} of {gs_total} runs | see `eval/answer_quality.md` | ~2,500 tokens | "
        f"about US${GREENSCOPE_COST:.3f}")
    text += ("\n\nLimitations: 10 questions, 1 run each for the long-context method, so small "
             "differences are not meaningful; the cost difference is large and robust.\n")
    (EVAL_DIR / "long_context_baseline.md").write_text(text, encoding="utf-8")
    print(text)


def gs_by_id(graded):
    """Verdict letters (C/P/E/M) per question id from eval/answer_quality.json."""
    letter = {"correct": "C", "partly_correct": "P", "contains_error": "E", "missed": "M"}
    out = {}
    for row in graded if isinstance(graded, list) else graded.get("rows", []):
        qid = row.get("id") or row.get("custom_id", "").split("--")[0]
        out.setdefault(qid, []).append(letter.get(row.get("verdict"), "?"))
    return out


if __name__ == "__main__":
    main()
