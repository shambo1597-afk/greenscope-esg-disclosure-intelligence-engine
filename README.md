# GreenScope: ESG Disclosure Intelligence Engine

A Streamlit app that answers questions about the sustainability (ESG) reports of two
Indian IT companies, **Wipro (FY2024-25)** and **HCLTech (FY2024)**, using
Retrieval-Augmented Generation (RAG). Every claim in an answer is cited to a page,
e.g. *[Wipro, p.67]*, a **Compare both** mode answers side by side, and every
retrieved passage comes with a word-level explanation of why it matched.

| Read this | For |
|---|---|
| [`docs/EXPLAINER.md`](docs/EXPLAINER.md) | How it works in plain language, every parameter choice with its evidence, rubric mapping, likely examiner questions |
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Persona, ROI model, market strategy, risks |
| [`docs/VIBE_CODING_LOG.md`](docs/VIBE_CODING_LOG.md) | How the project was built with AI agents |

## Setup

Requires Python 3.10+ (tested on 3.11). Versions in `requirements.txt` are the ones tested.

```bash
pip install -r requirements.txt
```

Create a file called `.env` in the project root (it's gitignored, so it's never committed):

```
ANTHROPIC_API_KEY=your-key-here
```

In a Claude Code cloud environment, `ANTHROPIC_API_KEY` is not passed to sessions, so
set `GREENSCOPE_ANTHROPIC_KEY` there instead; the app accepts either name.

Without a key the app still works: it shows the most relevant report passages and their
explanations, but doesn't write an AI answer. Each answer costs about US$0.002 to US$0.0035
with Claude Haiku 4.5 and takes about 3 seconds (Compare mode makes two calls).

## Run

```bash
streamlit run app.py
```

The first run downloads the embedding model, reads and embeds both PDFs (about 2 minutes)
and saves the result in `index/`. Later runs load it in seconds. To force a rebuild:
`python -m src.data --rebuild`.

## Architecture

```
                         ONE-OFF INGESTION  (src/data.py, cached in index/)
 ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ data/raw/*.pdf   │──►│ PyPDFLoader      │──►│ Recursive splitter   │──►│ BAAI/bge-small-en-   │
 │ Wipro 163 pp.    │   │ 1 doc per page   │   │ 400 chars, 250 over- │   │ v1.5: 384-dim unit   │
 │ HCLTech 113 pp.  │   │ page = PDF page  │   │ lap, drop < 50 chars │   │ vectors (~4,000)     │
 └──────────────────┘   └──────────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                                                     │
                                   index/  <Company>_vectors.npy + <Company>_chunks.json
                                                                                     │
                         PER QUESTION  (src/model.py, app.py)                        ▼
 ┌──────────────┐   ┌───────────────────┐   ┌───────────────────────┐   ┌──────────────────────┐
 │ Question     │──►│ prepare_query:    │──►│ FAISS IndexFlatIP     │──►│ Small-to-big: widen  │
 │ mode, top_k  │   │ name -> "the      │   │ one index per company │   │ each match to ~2,000 │
 │ (Streamlit)  │   │ company"; embed   │   │ cosine, top 5 each    │   │ chars of its page    │
 └──────────────┘   │ with BGE + prefix │   └───────────┬───────────┘   └──────────┬───────────┘
                    └───────────────────┘               │                          ▼
                                                        ▼               ┌──────────────────────┐
                                        ┌──────────────────────────┐    │ Claude Haiku 4.5:    │
                                        │ explain(): exact Shapley │    │ answer ONLY from the │
                                        │ value per question word  │    │ passages, cite       │
                                        └─────────────┬────────────┘    │ [Company, p.X]       │
                                                      │                 └──────────┬───────────┘
                                                      ▼                            ▼
                     Answer + passages (company, page, similarity, coloured word contributions)
                     Compare mode: one column per company · sidebar: time saved and AI cost
```

| File | Role |
|---|---|
| `app.py` | Streamlit interface: mode, top_k slider, example questions, answers, passages, explanations, session metrics |
| `src/data.py` | Load PDFs, chunk, embed, save/load the `index/` cache; the tuned parameters and why |
| `src/model.py` | Query clean-up, FAISS retrieval, small-to-big context, Shapley explanations, grounded generation with Claude |
| `eval/eval_set.json` | 40 test questions (20 per company) with the PDF pages holding the answer |
| `eval/run_eval.py` | Retrieval evaluation (Hit@k, MRR, Precision@5), writes `eval/results.md` |
| `eval/tune.py` | Grid search with 5-fold cross-validation, writes `eval/tuning_results.md` |
| `eval/make_charts.py` | Charts of the tuning results, written to `docs/figures/` |
| `eval/distance_metrics.py` | Cosine vs dot product vs Euclidean vs Manhattan |
| `eval/compare_pdf_parsers.py` | Experiment: PyPDF vs Docling |

## Evaluation

All of these are retrieval-only (no API calls, no cost) unless noted.

```bash
python eval/run_eval.py          # main retrieval metrics       -> eval/results.md
python eval/tune.py              # tuning + cross-validation    -> eval/tuning_results.md (~1.5 h, cached)
python eval/make_charts.py       # charts                       -> docs/figures/
python eval/distance_metrics.py  # distance metrics             -> eval/distance_metrics.md
```

Current retrieval results (40 questions, top 5 passages):

| Group | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Precision@5 |
|---|---|---|---|---|---|
| Wipro (20) | 0.70 | 0.85 | 0.95 | 0.77 | 0.57 |
| HCLTech (20) | 0.85 | 0.90 | 0.90 | 0.87 | 0.60 |
| Overall (40) | 0.78 | 0.88 | 0.93 | 0.82 | 0.58 |

**Tuning with cross-validation.** 36 settings (embedding model × chunk size × overlap);
in each of 5 folds the best was chosen on 32 questions and scored on the 8 held out. The
same setting won every fold: held-out MRR@5 **0.82 ± 0.13** vs **0.67 ± 0.15** for the
original setting (MiniLM, 800/150). Details:
[`eval/tuning_results.md`](eval/tuning_results.md).

**Answer quality** (graded by hand, costs a few cents):
[`eval/answer_check_tuned.md`](eval/answer_check_tuned.md): 13 of 16 fully correct, with
each error traced to its cause.

**Other experiments:** removing company names from the query (Hit@5 0.70 to 0.93;
`eval/results.md`), distance metrics ([`eval/distance_metrics.md`](eval/distance_metrics.md)),
and PyPDF vs the layout-aware parser Docling, on retrieval and graded answers
([`eval/pdf_parser_comparison.md`](eval/pdf_parser_comparison.md); reproduce with
`python eval/compare_pdf_parsers.py --answers`, needs `pip install docling`).

Limitations: 40 questions is small; matching is by page, not passage; gold pages are not
yet human-verified (`"verified": false`); answers were graded once per question.

## Notes

- The reports cover **different fiscal years** (Wipro FY2024-25, HCLTech FY2024), so
  side-by-side figures aren't always like-for-like.
- Page numbers are PDF file pages (page 1 = first page of the file), which may differ
  from the numbers printed on the pages.
- Charts and images aren't read; only extractable text is searched.
- Source PDFs:
  [Wipro](https://www.wipro.com/content/dam/nexus/en/sustainability/sustainability_reports/wipro-sustainability-report-fy-2024-2025.pdf),
  [HCLTech](https://www.hcltech.com/sites/default/files/documents/resources/pdf-landing-page/files/2024/08/08/hcltech-sustainability-report-august.pdf).
