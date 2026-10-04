# GreenScope: ESG Disclosure Intelligence Engine

A Streamlit app that answers questions about the sustainability (ESG) reports of two
Indian IT companies, **Wipro (FY2024-25)** and **HCLTech (FY2024)**, using
Retrieval-Augmented Generation (RAG). Every claim in an answer is cited to a page,
e.g. *[Wipro, p.67]*, and a **Compare both** mode answers side by side.

New to RAG? Read [`docs/EXPLAINER.md`](docs/EXPLAINER.md) for a plain-language walkthrough.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Create a file called `.env` in the project root (it's gitignored, so it's never committed):

```
ANTHROPIC_API_KEY=your-key-here
```

In a Claude Code cloud environment, `ANTHROPIC_API_KEY` is not passed to sessions, so
set `GREENSCOPE_ANTHROPIC_KEY` there instead; the app accepts either name.

Without a key the app still works: it shows the most relevant report passages, but
doesn't write an AI answer. Each answer costs about $0.002 to $0.0035 with Claude Haiku 4.5
(shown under every answer; Compare mode makes two calls).

## Run

```bash
streamlit run app.py
```

The first run reads and embeds both PDFs (about a minute) and saves the result in
`index/`. Later runs load it in seconds. To force a rebuild: `python -m src.data --rebuild`.

## Architecture

```
                         ONE-OFF INGESTION  (src/data.py, cached in index/)
 ┌──────────────────────┐   ┌───────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
 │ data/raw/*.pdf       │──►│ PyPDFLoader       │──►│ Recursive splitter  │──►│ all-MiniLM-L6-v2 │
 │ Wipro 163 pp.        │   │ 1 doc per page    │   │ 800 chars, 150 over-│   │ 384-dim vectors, │
 │ HCLTech 113 pp.      │   │ page = PDF page   │   │ lap, drop < 50 chars│   │ normalized       │
 └──────────────────────┘   └───────────────────┘   └─────────────────────┘   └────────┬─────────┘
                                                                                       │
                                     index/  Wipro_vectors.npy + Wipro_chunks.json  ◄──┘
                                             HCLTech_vectors.npy + HCLTech_chunks.json

                         PER QUESTION  (src/model.py, app.py)
 ┌──────────────┐   ┌────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
 │ Question     │──►│ prepare_query:     │──►│ FAISS IndexFlatIP        │──►│ Claude Haiku 4.5     │
 │ (Streamlit)  │   │ company name ->    │   │ one index per company,   │   │ answer ONLY from the │
 │ mode, top_k  │   │ "the company";     │   │ cosine similarity,       │   │ passages, cite       │
 └──────────────┘   │ embed with MiniLM  │   │ top_k passages each      │   │ [Company, p.X]       │
                    └────────────────────┘   └──────────────────────────┘   └──────────┬───────────┘
                                                                                       ▼
                                       Answer + expandable passages (company, page, similarity)
                                       Compare mode: one column per company
```

| File | Role |
|---|---|
| `app.py` | Streamlit interface: mode, top_k slider, example questions, answers, passages |
| `src/data.py` | Load PDFs, chunk, embed, save/load the `index/` cache |
| `src/model.py` | FAISS retrieval (single + comparison) and grounded generation with Claude |
| `eval/eval_set.json` | 16 test questions (8 per company) with the PDF pages holding the answer |
| `eval/run_eval.py` | Retrieval evaluation, writes `eval/results.md` |
| `docs/EXPLAINER.md` | Plain-language explanation and likely examiner questions |

## Evaluation

```bash
python eval/run_eval.py
```

Runs retrieval only (no API calls, no cost) and writes [`eval/results.md`](eval/results.md).
Current results (top 5 passages per question):

| Group | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Precision@5 |
|---|---|---|---|---|---|
| Wipro (8) | 0.50 | 0.75 | 1.00 | 0.68 | 0.50 |
| HCLTech (8) | 0.62 | 0.88 | 0.88 | 0.75 | 0.40 |
| Overall (16) | 0.56 | 0.81 | 0.94 | 0.72 | 0.45 |

Searching with the company name left in the question gives overall Hit@5 = 0.56; see
the results file for why the app removes it. The eval set is small, matches by page
only, and its gold pages are not yet human-verified (`"verified": false`).

## Notes

- The reports cover **different fiscal years** (Wipro FY2024-25, HCLTech FY2024), so
  side-by-side figures aren't always like-for-like.
- Page numbers are PDF file pages (page 1 = first page of the file), which may differ
  from the numbers printed on the pages.
- Charts and images aren't read; only extractable text is searched.
- Source PDFs:
  [Wipro](https://www.wipro.com/content/dam/nexus/en/sustainability/sustainability_reports/wipro-sustainability-report-fy-2024-2025.pdf),
  [HCLTech](https://www.hcltech.com/sites/default/files/documents/resources/pdf-landing-page/files/2024/08/08/hcltech-sustainability-report-august.pdf).
