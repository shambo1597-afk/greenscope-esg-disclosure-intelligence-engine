# GreenScope: ESG Disclosure Intelligence Engine

A Streamlit app that answers questions about the sustainability (ESG) reports of two
Indian IT companies, **Wipro** and **HCLTech**, both for **FY2024-25** (April 2024 to March 2025), using
Retrieval-Augmented Generation (RAG). Every claim in an answer is cited to a page,
e.g. *[Wipro, p.67]*, a **Compare both** mode answers side by side, and every
retrieved passage comes with a word-level explanation of why it matched.

| Read this | For |
|---|---|
| [`docs/EXPLAINER.md`](docs/EXPLAINER.md) | How it works in plain language, every parameter choice with its evidence, rubric mapping, likely examiner questions |
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Persona, ROI model, market strategy, risks |
| [`docs/VIBE_CODING_LOG.md`](docs/VIBE_CODING_LOG.md) | How the project was built with AI agents |

## Team (Group 2)

| Name | Reg No |
|---|---|
| Shambadeb Ghosh | DBM/1069/03 |
| Madhav Narayan Yadav | DBM/1045/03 |
| Iringakaran Rhishi Sasidharan | DBM/1029/03 |
| Arjun Madanan | DBM/1009/03 |
| Lakshya Maheshwari | DBM/1044/03 |

## Setup

Requires **Python 3.11 or newer** (tested on 3.11; the pinned numpy needs 3.11+). Versions in
`requirements.txt` are the ones tested.

```bash
python -m pip install -r requirements.txt
```

Use `python -m pip` (not plain `pip`) and start the app with `python -m streamlit run app.py`,
so the packages and the app use the same Python. If `pip` and `streamlit` belong to different
Python installations, the app fails with `ModuleNotFoundError` (e.g. `No module named 'faiss'`).

**Windows:** in the project folder, a virtual environment avoids mixing installations:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.model --check
python -m streamlit run app.py
```

(Optional, smaller download: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
before the requirements, for the CPU-only build of PyTorch.)

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
python -m streamlit run app.py
```

The pre-built index is committed in `index/`, so the app starts in seconds; the first
run only downloads the embedding model (~130 MB). If the PDFs or settings change, the
index is rebuilt automatically (about 2 minutes); to force it: `python -m src.data --rebuild`.

## Before a live demo

1. On the presenting laptop, with internet: `python -m pip install -r requirements.txt`, then
   `python -m src.model --check --live`. Every line should say PASS. This downloads the
   embedding model, so search keeps working later even if the Wi-Fi drops (only the AI
   answer needs internet).
2. Start the app (`python -m streamlit run app.py`) 5 minutes early and ask one example question,
   so everything is loaded.
3. Have a fallback: if the API fails, the retrieved passages and their explanations are
   still shown, and each one is a cited answer in itself.

## Architecture

```
                         ONE-OFF INGESTION  (src/data.py, cached in index/)
 ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ data/raw/*.pdf   │──►│ PyPDFLoader      │──►│ Recursive splitter   │──►│ BAAI/bge-small-en-   │
 │ Wipro 163 pp.    │   │ 1 doc per page   │   │ 400 chars, 250 over- │   │ v1.5: 384-dim unit   │
 │ HCLTech 122 pp.  │   │ page = PDF page  │   │ lap, drop < 50 chars │   │ vectors (~3,800)     │
 └──────────────────┘   └──────────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                                                     │
                                   index/  <Company>_vectors.npy + <Company>_chunks.json
                                                                                     │
                         PER QUESTION  (src/model.py, app.py)                        ▼
 ┌──────────────┐   ┌───────────────────┐   ┌───────────────────────┐   ┌──────────────────────┐
 │ Question     │──►│ split_question:   │──►│ Hybrid search per     │──►│ Small-to-big: widen  │
 │ mode, top_k  │   │ one query per     │   │ company: FAISS cosine │   │ each match to ~2,000 │
 │ (Streamlit)  │   │ topic; name ->    │   │ + BM25 keywords,      │   │ chars of its page    │
 └──────────────┘   │ "the company"     │   │ fused by RRF, top 5   │   └──────────┬───────────┘
                    └───────────────────┘   └───────────┬───────────┘              ▼
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
| `src/model.py` | Query clean-up and splitting, hybrid retrieval (FAISS + BM25, RRF), small-to-big context, Shapley explanations, grounded generation with Claude, pre-demo check |
| `eval/eval_set.json` | 40 test questions (20 per company) with the PDF pages holding the answer |
| `eval/run_eval.py` | Retrieval evaluation (Hit@k, MRR, Precision@5), writes `eval/results.md` |
| `eval/tune.py` | Grid search with 5-fold cross-validation, writes `eval/tuning_results.md` |
| `eval/make_charts.py` | Charts of the tuning results, written to `docs/figures/` |
| `eval/distance_metrics.py` | Cosine vs dot product vs Euclidean vs Manhattan |
| `eval/retrieval_methods.py` | Dense vs BM25 vs hybrid vs re-ranking, cross-validated |
| `eval/multi_topic_eval.py` | Two-topic questions: single vs split search |
| `eval/answer_eval.py` | 40 questions x 3 runs, graded by an AI judge via the Batch API |
| `tests/test_core.py` | 15 automated tests, run on every push by GitHub Actions |
| `eval/compare_pdf_parsers.py` | Experiment: PyPDF vs Docling |

## Evaluation

All of these are retrieval-only (no API calls, no cost) unless noted.

```bash
python eval/run_eval.py          # main retrieval metrics       -> eval/results.md
python eval/tune.py              # tuning + cross-validation    -> eval/tuning_results.md (~1.5 h, cached)
python eval/make_charts.py       # charts                       -> docs/figures/
python eval/distance_metrics.py  # distance metrics             -> eval/distance_metrics.md
python eval/retrieval_methods.py # hybrid search / re-ranking   -> eval/retrieval_methods.md
python eval/multi_topic_eval.py  # two-topic questions (API, <1 cent) -> eval/multi_topic_results.md
python eval/answer_eval.py       # answer quality (API, ~US$0.50)    -> eval/answer_quality.md
pytest -q                        # automated tests
```

Current retrieval results (40 questions, top 5 passages):

| Group | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Precision@5 |
|---|---|---|---|---|---|
| Wipro (20) | 0.80 | 0.95 | 0.95 | 0.85 | 0.67 |
| HCLTech (20) | 0.95 | 1.00 | 1.00 | 0.97 | 0.73 |
| Overall (40) | 0.88 | 0.97 | 0.97 | 0.91 | 0.70 |

**Tuning with cross-validation.** 36 settings (embedding model × chunk size × overlap);
in each of 5 folds the best was chosen on 32 questions and scored on the 8 held out. The
same setting won every fold: held-out MRR@5 **0.84 ± 0.09** vs **0.69 ± 0.12** for the
original setting (MiniLM, 800/150). Details:
[`eval/tuning_results.md`](eval/tuning_results.md).

**Hybrid search.** Keyword search (BM25) fused with vector search by Reciprocal Rank
Fusion beat vector search alone (MRR@5 0.912 vs 0.837 on all 40; held-out 0.89 vs 0.84);
re-rankers were tested and made results worse ([`eval/retrieval_methods.md`](eval/retrieval_methods.md)).

**Answer quality.** 120 answers (40 questions × 3 runs) graded by Claude Sonnet 5.5 against
reference answers and the source passages: **86% correct**, 5% partly correct, 7% with
an error, 2% missed; citations supported in 90%
([`eval/answer_quality.md`](eval/answer_quality.md)). Hand checks of 20 judge verdicts
agreed with all 20 ([`eval/answer_quality_judge_check.md`](eval/answer_quality_judge_check.md)).

**Other experiments:** splitting two-topic questions
([`eval/multi_topic_results.md`](eval/multi_topic_results.md)), removing company names
from the query (Hit@5 0.82 to 0.97;
`eval/results.md`), distance metrics ([`eval/distance_metrics.md`](eval/distance_metrics.md)),
and PyPDF vs the layout-aware parser Docling, on retrieval and graded answers
([`eval/pdf_parser_comparison.md`](eval/pdf_parser_comparison.md); reproduce with
`python eval/compare_pdf_parsers.py --answers`, needs `pip install docling`).

Limitations: 40 questions is small; matching is by page, not passage; gold pages are not
yet human-verified (`"verified": false`); the answer judge was checked on 20 verdicts.

## Notes

- Both reports cover **FY2024-25** (April 2024 to March 2025). An earlier version used
  HCLTech's FY2024 report; it was replaced so comparisons are like-for-like. Companies can
  still use different baselines and definitions (e.g. Wipro's 2017/2020 vs HCLTech's FY20).
- Page numbers are PDF file pages (page 1 = first page of the file), which may differ
  from the numbers printed on the pages.
- Charts and images aren't read; only extractable text is searched.
- Source PDFs:
  [Wipro](https://www.wipro.com/content/dam/nexus/en/sustainability/sustainability_reports/wipro-sustainability-report-fy-2024-2025.pdf),
  [HCLTech](https://www.hcltech.com/sites/default/files/documents/resources/pdf-landing-page/files/2025/10/15/sustainability-report-fy25.pdf).
