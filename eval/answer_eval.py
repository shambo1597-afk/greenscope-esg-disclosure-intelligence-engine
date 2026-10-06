"""Answer-quality evaluation: every eval question, answered 3 times, graded by an AI judge.

Why: retrieval metrics say whether the right page reached the model, not whether
the final answer is right. Answers also vary between runs, so each question is
answered RUNS times to measure consistency.

How:
  1. For each of the 40 questions, retrieve exactly as the app does (hybrid search,
     two-topic splitting, small-to-big context) and build the app's exact request
     (src.model.answer_request).
  2. Send all 120 answer requests as one Message Batch (50% cheaper than normal
     calls; results usually within minutes).
  3. Send each answer to a judge model, Claude Sonnet 5.5, with the question, the
     reference answer from eval_set.json and the passages the answer was based on.
     The judge returns structured JSON: a verdict on the same scale used for hand
     grading, and whether every citation supports its claim. Also one batch.
  4. Write eval/answer_quality.md and eval/answer_quality.json.

The judge is itself a model and can be wrong: a sample of its verdicts was checked
by hand against the PDF text (eval/answer_quality_judge_check.md).

Cost: about US$0.75 for 40 questions x 3 runs (Batch API prices).
Run:  python eval/answer_eval.py        (resumable: batch ids are kept in index/)
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic  # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # noqa: E402
from anthropic.types.messages.batch_create_params import Request  # noqa: E402

from src.model import (  # noqa: E402
    DEFAULT_TOP_K, LLM_MODEL, ReportIndex, answer_request, format_context, get_api_key, has_api_key,
    split_question,
)

EVAL_DIR = PROJECT_ROOT / "eval"
STATE = PROJECT_ROOT / "index" / "answer_eval_state.json"
RUNS = 3
JUDGE_MODEL = "claude-sonnet-5-5"
VERDICTS = ["correct", "partly_correct", "contains_error", "missed"]
# Batch API prices (US$ per million tokens): half of the standard price.
PRICES = {LLM_MODEL: (0.5, 2.5), JUDGE_MODEL: (1.0, 5.0)}

JUDGE_SYSTEM = """You grade answers written by a retrieval-augmented assistant about corporate \
sustainability reports. You receive the question, a reference answer written by a person from the \
report, the report excerpts the assistant was given, and the assistant's answer.

Verdicts:
- correct: gives the key facts needed to answer the question, accurately. Extra claims are fine if \
the excerpts support them. The reference may include extra context that is not required.
- partly_correct: no false statements, but misses part of what the question asks.
- contains_error: at least one figure, rating, year, label or conclusion contradicts the reference \
or the excerpts, or is not supported by them, even if the rest is right. Be strict with numbers and \
ratings ("A" and "A-" are different).
- missed: says the information is not disclosed or not available, although the reference shows it is.

citations_supported: true only if every [Company, p.X] citation points to an excerpt with that page \
that supports the claim it is attached to.

Keep the explanation under 40 words and name the specific problem, if any."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "citations_supported": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["verdict", "citations_supported", "explanation"],
    "additionalProperties": False,
}


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def wait_for(client, batch_id, label):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        counts = batch.request_counts
        print(f"{label}: {counts.processing} processing, {counts.succeeded} done", flush=True)
        time.sleep(30)


def collect(client, batch_id):
    """custom_id -> (text, input tokens, output tokens, stop reason), or an error string."""
    out = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            out[result.custom_id] = (text, msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason)
        else:
            out[result.custom_id] = f"batch result: {result.result.type}"
    return out


def main():
    if not has_api_key():
        sys.exit("Needs an Anthropic API key.")
    client = anthropic.Anthropic(api_key=get_api_key())
    eval_set = json.loads((EVAL_DIR / "eval_set.json").read_text())
    by_id = {q["id"]: q for q in eval_set}
    state = load_state()

    # 1-2. Retrieve like the app, then batch the answer requests.
    if "answer_batch" not in state:
        index = ReportIndex()
        contexts, requests, split_cost = {}, [], 0.0
        for q in eval_set:
            split = split_question(q["question"])
            split_cost += split.cost_usd
            chunks = index.retrieve_multi(q["question"], q["company"], DEFAULT_TOP_K, split.queries)
            contexts[q["id"]] = format_context(chunks)
            for run in range(1, RUNS + 1):
                requests.append(Request(custom_id=f"{q['id']}--{run}",
                                        params=MessageCreateParamsNonStreaming(**answer_request(q["question"], chunks))))
        batch = client.messages.batches.create(requests=requests)
        state.update(answer_batch=batch.id, contexts=contexts, split_cost=split_cost)
        save_state(state)
        print(f"Answer batch {batch.id}: {len(requests)} requests")
    wait_for(client, state["answer_batch"], "answers")
    answers = collect(client, state["answer_batch"])

    # 3. Judge every answer.
    if "judge_batch" not in state:
        requests = []
        for cid, result in answers.items():
            if isinstance(result, str):
                continue
            q = by_id[cid.split("--")[0]]
            content = (f"Question: {q['question']}\n\nReference answer: {q['expected_answer']}\n\n"
                       f"Report excerpts given to the assistant:\n\n{state['contexts'][q['id']]}\n\n"
                       f"Assistant's answer:\n{result[0]}")
            requests.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                model=JUDGE_MODEL, max_tokens=4000, system=JUDGE_SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                messages=[{"role": "user", "content": content}],
            )))
        batch = client.messages.batches.create(requests=requests)
        state["judge_batch"] = batch.id
        save_state(state)
        print(f"Judge batch {batch.id}: {len(requests)} requests")
    wait_for(client, state["judge_batch"], "judge")
    judged = collect(client, state["judge_batch"])

    # 4. Report.
    rows, cost = [], state.get("split_cost", 0.0)
    for cid in sorted(answers, key=lambda c: (by_id[c.split("--")[0]]["company"] != "Wipro", c)):
        qid, run = cid.split("--")
        answer, verdict = answers[cid], judged.get(cid)
        if not isinstance(answer, str):
            cost += (answer[1] * PRICES[LLM_MODEL][0] + answer[2] * PRICES[LLM_MODEL][1]) / 1e6
        if isinstance(verdict, tuple):
            cost += (verdict[1] * PRICES[JUDGE_MODEL][0] + verdict[2] * PRICES[JUDGE_MODEL][1]) / 1e6
        if isinstance(verdict, tuple) and verdict[3] != "refusal":
            parsed = json.loads(verdict[0])
        else:
            parsed = {"verdict": "judge_failed", "citations_supported": None,
                      "explanation": verdict if isinstance(verdict, str) else "judge refused or failed"}
        rows.append({"id": qid, "run": int(run), "company": by_id[qid]["company"], "question": by_id[qid]["question"],
                     "answer": answer if isinstance(answer, str) else answer[0], **parsed})
    (EVAL_DIR / "answer_quality.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    write_report(rows, cost)


def write_report(rows, cost):
    def share(subset, verdict):
        return sum(r["verdict"] == verdict for r in subset) / len(subset)

    groups = {"Wipro": [r for r in rows if r["company"] == "Wipro"],
              "HCLTech": [r for r in rows if r["company"] == "HCLTech"], "Overall": rows}
    ids = sorted({r["id"] for r in rows}, key=lambda i: (not i.startswith("wipro"), i))
    per_q = {i: [r for r in rows if r["id"] == i] for i in ids}
    always_correct = sum(all(r["verdict"] == "correct" for r in v) for v in per_q.values())
    consistent = sum(len({r["verdict"] for r in v}) == 1 for v in per_q.values())
    judged = [r for r in rows if r["citations_supported"] is not None]

    lines = [
        "# Answer quality: 40 questions x 3 runs, graded by an AI judge",
        "",
        f"Generated by `python eval/answer_eval.py`. Answers from the app's exact pipeline (Claude Haiku 4.5), "
        f"graded by {JUDGE_MODEL} against the reference answers in `eval_set.json` and the passages each "
        f"answer was based on. Both steps used the Message Batches API. Total cost: US${cost:.2f}.",
        "",
        "## Verdicts (share of answers)",
        "",
        "| Group | Answers | Correct | Partly correct | Contains an error | Missed |",
        "|---|---|---|---|---|---|",
    ]
    for name, subset in groups.items():
        lines.append(f"| {name} | {len(subset)} | {share(subset, 'correct'):.0%} | {share(subset, 'partly_correct'):.0%} | "
                     f"{share(subset, 'contains_error'):.0%} | {share(subset, 'missed'):.0%} |")
    failed = sum(r["verdict"] == "judge_failed" for r in rows)
    lines += [
        "",
        f"- **Citations supported** (every citation backs its claim): "
        f"{sum(bool(r['citations_supported']) for r in judged) / max(len(judged), 1):.0%} of graded answers.",
        f"- **Consistency:** {consistent} of {len(per_q)} questions got the same verdict in all {RUNS} runs; "
        f"{always_correct} of {len(per_q)} were correct in every run.",
    ]
    if failed:
        lines.append(f"- {failed} answers could not be graded (judge refused or failed).")
    lines += ["", "## Per question", "", "| ID | Question | Run 1 | Run 2 | Run 3 | Judge's note on the first non-correct run |",
              "|---|---|---|---|---|---|"]
    short = {"correct": "C", "partly_correct": "P", "contains_error": "**E**", "missed": "**M**", "judge_failed": "?"}
    for i, v in per_q.items():
        v = sorted(v, key=lambda r: r["run"])
        note = next((r["explanation"] for r in v if r["verdict"] != "correct"), "")
        lines.append(f"| {i} | {v[0]['question']} | " + " | ".join(short[r["verdict"]] for r in v) + f" | {note} |")
    lines += [
        "",
        "C correct · P partly correct · **E** contains an error · **M** missed.",
        "",
        "## Limitations",
        "",
        "- The judge is a language model and can be wrong. Hand checks of 20 of its verdicts against "
        "the PDF text, across two runs, agreed with all 20 (`answer_quality_judge_check.md`); it is "
        "strict about exact figures.",
        "- Reference answers are short summaries written from the PDF text and not yet human-verified.",
        "- 40 questions; one question is 2.5% of the total.",
        "",
    ]
    (EVAL_DIR / "answer_quality.md").write_text("\n".join(lines))
    print("\n".join(lines[6:12 + len(groups)]))


if __name__ == "__main__":
    main()
