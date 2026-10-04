# GreenScope business case

> **Status of the numbers.** Product figures (report length, cost per answer, speed,
> accuracy) are **measured** in this repository. Market and time-saving figures are
> **assumptions**, stated openly so they can be challenged and replaced with
> measurements (see "Validating the key assumption"). Regulatory facts should be
> checked against the latest SEBI / EU sources before presenting.

## 1. The problem

Sustainability (ESG) reports are long, unstructured PDFs. The two reports in GreenScope
are **163 pages (Wipro FY2024-25)** and **113 pages (HCLTech FY2024)**. Answering one
question such as *"What is each company's Scope 3 target?"* means searching both PDFs,
reading the surrounding text to check the scope, baseline year and fiscal year, and
recording the page so a reviewer can check it. Doing this for dozens of metrics across
several peers is slow, repetitive and error-prone.

Demand for this work is growing. In India, SEBI requires the top 1,000 listed companies
to publish a Business Responsibility and Sustainability Report (BRSR), and buyers of IT
services, especially in Europe and the US, increasingly ask suppliers for ESG data for
their own value-chain (Scope 3) reporting.

General-purpose chatbots are fast but answer from memory, can invent figures and give
no page reference. ESG data vendors provide standardized scores, but they are expensive
and don't show the source text behind each number.

## 2. Target persona

**Primary: Priya, ESG research analyst** at an Indian asset manager or ESG advisory firm.
- Benchmarks peer companies on climate, diversity and governance disclosures for
  investment notes and client reports.
- Needs exact figures **with the page reference**, because every number in her report is reviewed.
- Pain: hours per week spent in PDF search; risk of mixing up scopes, baselines and fiscal years.
- Not a programmer: uses the web app, never the code.

**Secondary personas**
- **Procurement / vendor-risk teams** at multinational clients assessing IT suppliers'
  ESG commitments (e.g. renewable-energy and Scope 3 targets) during supplier due diligence.
- **Corporate sustainability teams** benchmarking their own disclosures against peers.

## 3. The product

A web app where the analyst asks a question in plain English and gets, for one company
or **side by side for both**, an answer in which every claim is cited to a page, with the
exact supporting passages one click away. Measured in this repository:

| Measure | Value | Source |
|---|---|---|
| Cost per answer (Claude Haiku 4.5) | about US$0.002 to US$0.0035 | measured token usage, shown in the app |
| Right page among the 5 passages given to the AI | 93% of 40 test questions (Hit@5; also 93% on held-out questions in cross-validation) | `eval/results.md`, `eval/tuning_results.md` |
| Fully correct answers on the 16-question graded test | 13 of 16 (1 partly correct, 2 with an error) | `eval/answer_check_tuned.md` |
| Time per answer | about 3.5 s (search 0.03 s, word explanation 0.2 s, AI answer 3.3 s) | measured on a 4-core CPU |
| Setup | one-off indexing of both reports in about 2 minutes on a laptop | `src/data.py` |

Accuracy is high but not perfect, so the workflow keeps a human in the loop: the analyst
reads the cited passage before using a figure. The time-saving assumption below includes
that check.

## 4. ROI model

### Assumptions

| Assumption | Value | Basis |
|---|---|---|
| Manual time to find, check and record one metric for one company | 10 min | assumption, to be measured (section 6) |
| Time with GreenScope (read answer, check cited passage) | 2 min | assumption, to be measured |
| **Time saved per company lookup** | **8 min** | difference of the two |
| Lookups per year for one analyst team | 2,400 | e.g. 10 benchmarking projects × 8 companies × 30 metrics |
| Loaded analyst cost: India scenario | ₹1,000/hour (≈ US$12) | assumption |
| Loaded analyst cost: global consultancy scenario | US$60/hour | assumption |
| AI cost per lookup | US$0.003 | measured |
| Hosting (small cloud VM) | US$30/month = US$360/year | assumption |
| Maintenance (adding reports, re-running the evaluation) | 1 day/month = 96 hours/year at the analyst rate | assumption |

### Result (per team, per year)

| | India scenario | Global scenario |
|---|---|---|
| Hours saved (2,400 × 8 min) | 320 h | 320 h |
| **Value of time saved** | **US$3,840** (₹3.2 lakh) | **US$19,200** |
| AI cost (2,400 × US$0.003) | US$7 | US$7 |
| Hosting | US$360 | US$360 |
| Maintenance (96 h) | US$1,152 | US$5,760 |
| **Total cost** | **US$1,519** | **US$6,127** |
| **Net benefit** | US$2,321 | US$13,073 |
| **ROI** = net benefit ÷ cost | **153%** | **213%** |
| Break-even | about 950 lookups/year | about 770 lookups/year |

The AI cost is negligible (under 1% of total cost). The economics are driven almost
entirely by **analyst time saved** versus **maintenance time**.

### Sensitivity: hours saved per year

| Lookups per year → / minutes saved per lookup ↓ | 1,000 | 2,400 | 5,000 |
|---|---|---|---|
| 4 min | 67 h | 160 h | 333 h |
| 8 min | 133 h | 320 h | 667 h |
| 12 min | 200 h | 480 h | 1,000 h |

Multiply by the hourly cost for the value. Even in the most pessimistic cell
(4 minutes, 1,000 lookups, India rate) the time saved (67 h ≈ US$800) does not cover the
fixed costs, which is why the go-to-market below starts with teams that do recurring
benchmarking at volume.

### Non-financial benefits
- **Auditability:** every figure carries a page reference, which speeds up review.
- **Consistency:** the same question gets the same retrieval for every company.
- **Error reduction:** side-by-side answers with the fiscal-year warning make
  apples-to-oranges comparisons visible.

## 5. Market strategy

**Positioning:** "Source-grounded ESG answers with page citations, at a fraction of a
cent per question." It sits between generic chatbots (fast, unreliable, no sources) and
ESG data vendors (curated scores, expensive, little source text).

**Go-to-market in three steps**
1. **Pilot (months 0-3):** free pilot with one ESG advisory firm or university research
   group. Run the time trial (section 6) to replace the 8-minute assumption with a
   measurement, and grow the evaluation set with their real questions.
2. **Sector pack (months 3-9):** cover the main Indian IT services peers (e.g. the Nifty IT
   constituents), so a full peer benchmark is possible in one tool. Adding a report is a
   one-line change plus re-indexing (see README).
3. **Scale (months 9+):** a self-service upload feature and other sectors; an enterprise
   option that indexes a client's own document library. The embedding model and search
   run locally, and only the few retrieved passages are sent to the AI model.

**Pricing (illustrative)**

| Tier | Price | For |
|---|---|---|
| Free | 20 questions/day, public sector packs | students, journalists, trial users |
| Analyst | US$99 (≈ ₹8,000) per seat per month | ESG analysts and procurement teams |
| Enterprise | custom | private document libraries, single sign-on |

Unit economics: a heavy user asking 500 questions a month costs about US$1.50 in AI
usage, so gross margin on an Analyst seat is above 95% before hosting and support.

## 6. Validating the key assumption: a time trial

The ROI depends mostly on *minutes saved per lookup*. The team can measure it directly:
1. Take 10 questions from `eval/eval_set.json`.
2. Half the team answers them manually from the PDFs (find, check, record value and page),
   timing each one; the other half uses GreenScope and checks the cited passage.
3. Swap question sets and repeat, so each person does both methods.
4. Compare the median minutes per lookup and the error rate of each method.

Replacing the assumed 10 and 2 minutes with measured values turns the ROI table from an
estimate into evidence.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wrong or invented figures | Answers restricted to retrieved text, page citations, evidence panel, word-level explanation, documented evaluation and known failure cases |
| Charts and tables extracted badly | Neighbouring-chunk context and prompt rules; layout-aware parsing evaluated (`eval/pdf_parser_comparison.md`) and listed as future work |
| Comparing different years or definitions | Fiscal-year note on every page; citations show the exact wording |
| Dependence on one AI provider | Retrieval runs locally; the generation step is one function and can be switched to another model |
| Confidential client documents | Embeddings computed locally; only the top passages leave the machine |
| Liability | Positioned as decision support for analysts, not investment advice |

## 8. Success metrics (KPIs)

- Median minutes per lookup (target: under 3, including verification)
- Share of answers accepted without correction (target: over 90%)
- Retrieval Hit@5 on the growing evaluation set (target: over 90%)
- Cost per answer (target: under US$0.005)
- Weekly active analysts and lookups per analyst
