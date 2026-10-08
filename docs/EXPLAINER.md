# GreenScope, explained in plain language

This document walks through how GreenScope works, why each setting was chosen and how
we tested it, where it can still go wrong, and how to answer the questions an examiner
is likely to ask. No coding knowledge is needed.

---

## 1. The problem

Wipro's sustainability report is 163 pages long and HCLTech's is 122; both cover the
same fiscal year, FY2024-25 (April 2024 to March 2025). Finding one
figure, such as a Scope 3 emissions target, means a lot of scrolling, and comparing
the two companies means doing it twice. A general chatbot could answer from memory,
but it might be out of date or invent a number, and it can't tell you which page
the figure came from.

GreenScope answers questions **only from the two reports**, and every claim comes
with a page reference you can check. (Who needs this, and what it is worth, is in
`docs/BUSINESS_CASE.md`.)

## 2. The big idea: Retrieval-Augmented Generation (RAG)

RAG is two steps:

1. **Retrieval**: find the few passages in the reports most likely to contain the answer.
2. **Generation**: give only those passages to an AI language model (Claude) and ask it
   to write an answer from them, with citations.

It's an open-book exam: instead of answering from memory, the model is handed the
relevant pages and told to answer from those alone.

```
 Your question
      │
      ▼
 ┌────────────────────┐   "What is the Scope 3 target?"  (company name -> "the company")
 │ 1. Embed question  │──► list of 384 numbers that capture its meaning (BGE-small)
 └────────────────────┘
      │
      ▼
 ┌────────────────────┐   rank ~1,700-2,400 passages per report twice: by meaning
 │ 2. Hybrid search   │   (FAISS, vectors) and by keywords (BM25); fuse the two
 └────────────────────┘──► top 5 400-character passages (+ page numbers)
      │
      ▼
 ┌────────────────────┐   widen each match to ~2,000 characters of the same page
 │ 3. Small-to-big    │──► enough context around each figure
 └────────────────────┘
      │
      ▼
 ┌────────────────────┐   "Answer ONLY from these passages, cite [Company, p.X]"
 │ 4. Claude Haiku    │──► answer with page citations
 └────────────────────┘
      │
      ▼
 ┌────────────────────┐   which words of the question made each passage match?
 │ 5. Explanation     │──► Shapley value per word, shown in colour
 └────────────────────┘
```

## 3. Each step in THIS project

### 3.1 Loading the PDFs (`src/data.py`)

`PyPDFLoader` reads each PDF page by page and extracts the text. Each page keeps
its **page number in the PDF file**: page 1 is the first page of the file. This can
differ from the number printed on the page, since covers usually aren't numbered.
We use the file page because that's what you type into a PDF viewer to jump there.

Limitation: only text is extracted. Numbers inside images or charts are invisible,
and complex tables come out as jumbled lines of text (section 6 has the evidence and
the experiments we ran on this).

### 3.2 Chunking: cutting pages into passages

A page can cover several topics, which is too much to compare with one short
question. So we cut the text into **chunks** of up to **400 characters** (two to four
sentences), where each chunk repeats the last **250 characters** of the previous one
(the **overlap**).

- **Why small chunks (400)? To keep matches focused.** Each chunk is turned into a
  single "meaning fingerprint" (next section). If a chunk mixes emissions, water use
  and board diversity, its fingerprint is a blur of all three and matches none of
  them well. ESG reports put a different metric in almost every sentence, so short
  chunks match questions precisely. We **measured** this: average retrieval score
  (MRR@5) fell from 0.74 at 400 characters to 0.67 at 800 and 0.61 at 1,200 (section 5).
- **Why a large overlap (250)? To keep context across cuts.** The cut is mechanical and
  can land mid-sentence: "...reduce emissions by" | "42% by FY30". With overlap, a fact
  that straddles one cut appears whole in another chunk, and every sentence sits near
  the middle of at least one chunk. Measured at 400 characters: overlap 0 / 75 / 150 /
  250 gave MRR@5 0.80 / 0.80 / 0.74 / 0.84.
- **The trade-off and how we handle it.** A small chunk can separate a number from the
  words explaining it ("42%" without "reduction in Scope 3 by FY30"). So search uses the
  small chunks, but the model receives each match **widened to about 2,000 characters
  of the same page** ("small-to-big", section 3.4). Search is precise; the answer
  still has context.
- The splitter (`RecursiveCharacterTextSplitter`) cuts at the most natural place
  available: between paragraphs, then lines, then words.
- Chunks shorter than **50 characters** are dropped: cover titles, page footers and
  section dividers with no facts in them.

Result: **2,357 chunks for Wipro and 1,487 for HCLTech**.

> History: the first version used 800-character chunks with 150 overlap, chosen by
> reasoning alone. Testing showed smaller chunks work better, so we changed it.

### 3.3 Embeddings: turning text into meaning fingerprints

An **embedding model** turns a piece of text into a list of numbers (a *vector*,
here 384 numbers). Texts with similar meaning get similar vectors, even when they
use different words: "renewable electricity share" lands close to "84% of our
power comes from green sources".

We use **`BAAI/bge-small-en-v1.5`** (a Hugging Face model, run with
sentence-transformers / PyTorch):
- it's **trained specifically for search** (matching a question to the passage that
  answers it), which is exactly our task;
- it's small (~130 MB), free, and runs on a laptop CPU, with no API cost and no data
  leaving your machine for this step;
- it reads up to 512 word-pieces, so a 400-character chunk is never cut off;
- BGE expects a short instruction in front of search *questions* ("Represent this
  sentence for searching relevant passages: "), which we add (`QUERY_PREFIX`).

We compared it against two alternatives, **all-MiniLM-L6-v2** (smaller, general
purpose; our first choice) and **all-mpnet-base-v2** (twice the size, slower). BGE won
(section 5).

The vectors have length 1 (**normalized**). That makes the similarity calculation in
the next step a simple multiplication.

Embedding happens **once**: results are saved in the `index/` folder and reloaded on
later runs (building it takes about 2 minutes). If the PDFs or settings change, the
index is rebuilt automatically.

### 3.4 Cosine similarity and FAISS: finding the closest passages

To compare the question with a chunk, we measure the **angle** between their two
vectors. This is called **cosine similarity**:
- 1.0 = pointing the same way (same meaning),
- around 0 = unrelated.

Because the vectors have length 1, cosine similarity equals the *dot product*
(multiply matching numbers, add them up), and ranking by Euclidean (straight-line)
distance gives exactly the same order, since for unit vectors
distance² = 2 − 2 × cosine. We checked this on all 40 test questions
(`eval/distance_metrics.md`): cosine, dot product and Euclidean gave identical top-5
results; Manhattan distance differed and scored lower.

**FAISS** (Facebook AI Similarity Search) does this search very quickly. We use
`IndexFlatIP`:
- *Flat* = compare against every chunk (exact, no shortcuts). With about 2,000 chunks
  per report this takes about 0.03 seconds, so approximate methods aren't needed.
- *IP* = inner product, i.e. the dot product, i.e. cosine similarity for our vectors.

There is **one index per company**. This guarantees we always know which report a
passage came from. In **Compare** mode it also guarantees both companies get their
own top 5. A single combined search could return five Wipro passages and none from
HCLTech, which would make the comparison unfair.

**Hybrid search: meaning plus keywords.** Meaning search is good at paraphrases ("green
power" finds "renewable electricity") but blurs exact terms: "CDP", "ISO 14001",
"nationalities". So every passage is also ranked by **BM25**, the classic keyword
score: a passage scores higher the more of the question's words it contains, rare words
counting more ("nationalities" beats "company"), with a small penalty for long passages.
The two rankings are combined with **Reciprocal Rank Fusion (RRF)**:

> score = 1 / (60 + meaning rank) + 1 / (60 + keyword rank)

A passage near the top of either list rises; near the top of both, it wins. RRF has no
weight to tune (60 is the standard constant), which matters on a small test set
(section 5). Results: MRR@5 0.84 -> 0.91, Hit@1 0.78 -> 0.88. Example: "How many
nationalities are in Wipro's workforce?" was a miss with meaning search alone; with
keywords, page 7 ("146 Nationalities") comes first. The app shows each passage's
meaning rank, keyword rank and matched keywords.

We also tested **re-ranking**: a cross-encoder model reads the question and each of the
top 20 passages *together* and re-scores them. On our questions the BGE re-ranker made
results worse (MRR@5 0.75 vs 0.91, Hit@5 0.95 vs 0.97), added about 4.5 s per company on
our CPU and needs a 1.1 GB model, so we left it out (`eval/retrieval_methods.md`). A
smaller re-ranker trained on web searches was also worse.

**Two-topic questions.** "What are the water **and** waste goals?" is really two
searches. Questions that contain "and", "&", "as well as" or ";" (a free check) are sent
to Claude Haiku (about US$0.0002), which returns one search query per topic, or the
question unchanged ("Scope 1 **and** 2 emissions" is one topic). Each topic gets its own
top 5. On 8 two-topic test questions, both topics reached the model in 8 of 8 vs 7 of 8
with a single search; a single search with 10 passages also got 8 of 8, so the gain is
"more passages, only where needed" rather than anything cleverer (`eval/multi_topic_results.md`).
One HCLTech test question (water and waste goals) was replaced when we moved to the FY2025
report, because that report states no water goal.

**Query clean-up.** Before searching, GreenScope replaces the company name in your
question with "the company" (e.g. "HCLTech's water target" becomes "the company's
water target"). Since we're already searching inside one company's report, the name
adds nothing. Worse, it made the question look similar to useless page footers like
"About HCLTech / HCLTech Sustainability Report 2024 / 92" (a footer in the FY2024
HCLTech report we started with). With the final settings,
this raises the share of questions with a correct page in the top 5 from **0.82 to
0.97** (`eval/results.md`; with meaning search alone the effect was larger: 0.56 to 0.94
on our first test set). Claude still sees your original question.

**Small-to-big context.** Each match is widened with neighbouring chunks, alternately
after and before it, until it holds about **2,000 characters**, but never beyond its
own page, so every citation stays correct. Text already sent with a better match is
not repeated. The *Retrieved passages* panel shows exactly this widened text. Section 6
shows the real error that led to this.

### 3.5 Prompt grounding: making the model stick to the evidence

The retrieved passages are sent to **Claude Haiku 4.5**, each labelled with its
source, e.g. `[Wipro, p.67]`. The **system prompt** (standing instructions) sets the rules:

1. answer **only** from the provided passages, never from outside knowledge;
2. cite every claim as **[Company, p.X]**;
3. keep numbers, units, years and ratings (e.g. "A-") **exactly as written**; don't
   convert, round or add up figures;
4. if the passages don't contain the answer, say **"This is not disclosed in the
   retrieved text"**;
5. text from charts and tables can be scrambled, so only link a figure to a
   category (Scope 1 and 2 vs Scope 3, target vs achieved, a year) when the text
   explicitly connects them; otherwise say it's unclear;
6. don't draw conclusions the text doesn't state (e.g. "the target has been met");
7. be concise.

This is called **grounding**: the answer is tied to evidence you can inspect. The
app shows the passages under every answer, with their similarity score and page, so
you can check each claim yourself.

### 3.6 Explaining why a passage matched (Shapley values)

A similarity score of 0.79 says *how* close a passage is, not *why*. Under every
retrieved passage the app colours each word of your question: **green** if it pushed
the passage up, **red** if it pulled it down, with the size of the effect.

How it's computed (`ReportIndex.explain` in `src/model.py`): the content words of the
question are treated as "players" in a game, and the "payout" is the cosine similarity
between the question and the passage. Filler words ("what", "the", "of") stay in every
version. For a 5-word question we embed all 2⁵ = 32 versions of the question with some
words removed, and compute each word's **Shapley value**: its average extra similarity
over every order in which the words could have been added. This is the same idea as
**SHAP**, applied to retrieval instead of a classifier.

Two properties make it trustworthy and easy to explain:
- **It adds up.** For each passage, the word values sum exactly to (similarity of the
  full question) − (similarity with no content words). We check this in testing.
- **It's exact, not sampled.** With up to 8 content words (256 versions) it computes
  every combination in about 0.2 seconds. Longer questions fall back to leave-one-out
  (remove one word, measure the drop).

What it revealed: for *"What is Wipro's target for reducing Scope 3 emissions?"*, the
word **"3" contributes least** (+0.002 to +0.03, against up to +0.12 for "target"). Both
embedding models we checked barely distinguish "Scope 3" from "Scope 1 and 2". That
explains why Scope 1/2 passages compete with Scope 3 passages, and why the prompt rule
against unconnected figures matters.

## 4. How we measure quality

**Retrieval** (`eval/run_eval.py`, free): 40 questions, 20 per company. For each, we
listed the **gold pages**, the PDF pages where the answer actually appears, found by
searching the extracted text. We then check whether the search returns those pages.
Final system:

| Metric | Meaning | Score |
|---|---|---|
| Hit@1 | The very first passage is from a gold page | 0.88 |
| Hit@3 | A gold page is in the top 3 | 0.97 |
| Hit@5 | A gold page is in the top 5 (what the model sees) | 0.97 |
| MRR@5 | Average of 1/rank of the first gold page (1 = always first) | 0.91 |
| Precision@5 | Share of the 5 passages that are from gold pages | 0.70 |

**Answers** (`eval/answer_eval.py`, US$0.52): every one of the 40 questions is answered
**3 times** through the app's exact pipeline (answers vary between runs), and each of
the 120 answers is graded by a stronger model, **Claude Sonnet 5.5**, acting as a judge.
It sees the question, the reference answer and the passages the answer was based on,
and returns a verdict on the scale we use by hand (correct / partly correct / contains
an error / missed) plus whether every citation supports its claim. Both steps ran
through the **Message Batches API**, which costs half as much and returns within minutes.

| Correct | Partly correct | Contains an error | Missed | Citations all supported | Correct in all 3 runs |
|---|---|---|---|---|---|
| **86%** | 5% | 7% | 2% | 90% | 31 of 40 questions |

An AI judge can be wrong too, so we checked 20 of its verdicts by hand against the PDF
(over two runs): it agreed in all 20, and it is strict about exact figures (`eval/answer_quality_judge_check.md`).
Most errors are figures paired with the wrong label from scrambled chart or table text;
the misses are facts that only appear on infographic pages (section 6).

In counts: of the 120 answers, **103 were correct**, 6 partly correct, 8 contained an
error and 3 missed the answer. The 8 errors come from 5 questions (Wipro's Scope 1 and 2
target in all 3 runs, plus 4 single runs); the 3 misses are one question, Wipro's 13.5%
supplier-diversity spend, which only appears on an infographic highlights page. A typical
error: asked for Wipro's 2030 Scope 1 and 2 target, the answer gave the correct 59% but
then said the reduction from 195,453 to 31,462 tCO2e "represents" that target. On page 67
those figures are 2025 performance, not the target. The 120 answers are 3 runs of the same
40 questions, so they are not 120 independent tests.

### Which numbers are held-out, and which are not (read this before quoting a number)

Only some of our numbers come from questions that played no part in choosing the
settings:

| Number | Held-out? |
|---|---|
| Tuned chunking and embedding model: MRR@5 **0.84 ± 0.09**, Hit@5 **0.97 ± 0.05** | Yes, 5-fold cross-validation |
| Hybrid search (RRF): MRR@5 **0.89** vs 0.84 for meaning search alone | Yes, cross-validated |
| Final system on all 40: Hit@1 0.88, Hit@5 0.97 (39 of 40), MRR@5 0.91 | No: same 40 questions used to choose settings |
| Answers: 86% correct (103 of 120) | No: same 40 questions |

Five later design choices were made by looking at results on the same 40 questions,
without cross-validation: replacing the company name with "the company", splitting
two-topic questions, the 2,000-character page context, RRF over meaning search, and
rejecting the re-rankers. So the full-set numbers are somewhat **optimistic**. The honest
summary is: **tuned and tested on the same 40 questions, with cross-validation for the
main settings**. The fix is a fresh, frozen test set (20 to 30 questions, including table
questions, unanswerable questions and comparisons) written by someone who never tuned
against it. We list it as the first next step. Note also that with 40 questions one
question moves Hit@5 by 2.5 points.

### A fresh, frozen test set (the most honest number)

To get a number nobody tuned against, we wrote **20 new questions** (`eval/fresh_test_set.json`):
8 per company, deliberately including table, infographic and text questions, plus 4 questions
the reports don't answer. They were committed to GitHub **before** the first run, the system was
not changed afterwards, and each was answered 3 times and graded by the same judge
(`eval/fresh_test_results.md`, US$0.47):

| | Result |
|---|---|
| Answers fully correct | **57 of 60 (95%)** |
| Questions correct in all 3 runs | 19 of 20 |
| Table questions (6) / infographic (3) / unanswerable (4) | all runs correct |
| Right page in the top 5 (16 answerable questions) | 16 of 16; MRR@5 0.86 |
| Citations supported | 56 of 60 answers |

The one failure is the known weakness: asked how HCLTech's training hours changed, the answer
gave the correct 27% but paired the chart's numbers with the wrong categories. Two honest
caveats: the questions were written by the same assistant that built the system (not an
outsider), and they are single-fact questions, which are easier than some of the multi-part
questions in the original 40. So 95% is not "better than 86%"; it shows the system was not
merely fitted to the original 40 questions.

### MRR@5, one progression

MRR@5 means: how high the first passage from a correct page ranks, averaged over questions
(1.0 = always first, 0.5 = typically second).

| Step | MRR@5 | Measured on |
|---|---|---|
| Original setup (MiniLM, 800/150, meaning search) | 0.69 ± 0.12 | held-out folds of the 40 |
| + tuned chunks and model (BGE-small, 400/250) | 0.84 ± 0.09 | held-out folds of the 40 |
| + hybrid search (RRF) | 0.89 ± 0.08 | held-out folds of the 40 |
| Final system, all 40 questions | 0.91 | the 40 used for tuning (optimistic) |
| Final system, fresh frozen questions | 0.86 | 16 new answerable questions |

The hybrid step (+0.05) is small next to the fold spread (±0.09), so it is suggestive rather
than proven; we kept it because it was picked in 4 of 5 folds and costs nothing extra.

### Two extra checks

**Whole report in the prompt instead of RAG** (`eval/long_context_baseline.md`, US$0.37).
Both reports fit in Claude Haiku 4.5's context window (about 97,000 tokens for Wipro and
66,000 for HCLTech), so the obvious alternative is to skip retrieval. On 10 questions
fixed in advance (ids 01, 05, 09, 13, 17 per company), same model, rules and judge:

| Method | Fully correct | Cost per question |
|---|---|---|
| Whole report in prompt | 8 of 10 | US$0.083 (US$0.028 with prompt caching) |
| GreenScope, same 10 questions | 23 of 30 runs | about US$0.003 |

Accuracy is about the same on this small sample (the two methods failed on
partly different questions), but GreenScope costs about **28 times less** per question (9 times less even with
caching), sends about 2,500 tokens instead of 82,000, and shows the exact passages it used.
Long context would also stop fitting as reports are added or get longer.

**Questions the reports don't answer** (`eval/out_of_scope.md`, under one cent). We asked
four questions with no answer in the reports (Wipro's water use in 2010, its cryptocurrency
holdings, HCLTech's EV charging points in Brazil, HCLTech's share-price target). GreenScope
said "This is not disclosed in the retrieved text" for **all 4**, and did not use outside
knowledge.

## 5. Tuning the settings, with cross-validation

`eval/tune.py` tried **36 combinations** in three stages: MiniLM with 5 chunk sizes
(400 to 1,200 characters) × 4 overlaps (0 to 250) = 20; then BGE-small and MPNet on 6
promising size/overlap pairs = 12; then 2 more overlaps for those two models = 4. Each was
scored by MRR@5 on
the 40 questions. Retrieval only, so no API cost.

**Why cross-validation?** Pick the best of 36 settings on 40 questions and report its
score, and the score is too optimistic: with that many tries, something always gets
lucky. So we used **5-fold cross-validation**: the 40 questions are split into 5 groups
of 8 (4 per company). Five times over, we choose the best setting using 32 questions
and score it on the 8 it never saw. The average of those held-out scores is the honest
estimate.

| | Held-out MRR@5 | Held-out Hit@5 |
|---|---|---|
| Original setting (MiniLM, 800/150) | 0.69 ± 0.12 | 0.85 |
| Tuned setting, chosen without seeing the test questions | **0.84 ± 0.09** | **0.97** |

The same setting (**BGE-small, 400, 250**) won in **all 5 folds**, and it beat the
original in 4 of 5. Charts: `docs/figures/tuning_chunk_size.png`,
`docs/figures/cross_validation.png`.

**What the search showed**
- Smaller chunks scored higher for every model (400 characters was best for all three).
- BGE-small beat MiniLM and the larger MPNet: a model trained for search matters more
  than model size.
- The search was **coarse-to-fine**: after the first round showed the best region (400
  characters), every model was also run at every overlap there. That mattered: without
  it, we would have picked MiniLM; with it, BGE won clearly.

**Choosing top_k (5).** top_k can't be tuned the same way: showing more passages can
only raise Hit@k, so the metric would always pick the biggest number. Instead we looked
at the curve (`docs/figures/hit_at_k.png`): Hit@k reaches 0.97 at k = 5 and stays 0.97
at k = 10, while each extra passage adds cost and noise to the prompt. So 5 it is (the
sidebar slider allows 3 to 8).

**After tuning we re-checked the answers**, not just retrieval: the graded 16 questions
came out about the same (13 correct in both), with several answers more complete.
Retrieval improved clearly, so we kept the tuned setting.

**Retrieval method, also cross-validated** (`eval/retrieval_methods.py`): with the same 5
folds we compared meaning search, keyword search (BM25), weighted blends of the two (7
weights), RRF, and each of those followed by two re-rankers. Every blend beat meaning
search alone on the full set; RRF was best (MRR@5 0.912 vs 0.837) and was picked in 4
of 5 folds (held-out 0.89 vs 0.84). The "best" blend weight changed from fold to fold,
a sign that tuning it would mostly fit noise, which is why we chose weight-free RRF.

## 6. Where hallucination (made-up content) can still happen

Grounding reduces the risk a lot, but doesn't remove it:

- **Retrieval misses.** If the right passage isn't in the top 5 (7% of test questions),
  the model should say "not disclosed" (2% of graded answers), but it may stretch a
  related passage into an answer. The misses are facts printed only on infographic
  pages, such as Wipro's 13.5% supplier diversity spend on its highlights page.
- **Garbled extraction.** Tables and infographics come out as jumbled text. The model
  may pair a number with the wrong label (examples below). This is the main source of
  the 7% of answers with an error in the graded evaluation.
- **Run-to-run variation.** The same question with the same passages can be answered
  slightly differently. Once in four runs, the model wrote Wipro's CDP rating as "A"
  although the passage says "A-".
- **Different baselines and definitions** remain even though the fiscal years now match
  (see "Fiscal years" below).
- **Different definitions.** "Renewable share" may mean purchased electricity in one
  report and total energy in another; baselines differ (2017 vs FY20).
- **Model knowledge leaking in.** Claude may know facts about these companies from
  training. The prompt forbids using them, but that can't be guaranteed 100%.
- **Questions about several topics.** Now split into one search per topic, but only
  when the question contains a joining word like "and"; a two-topic question phrased
  without one is still searched once.

### A real example we found and fixed

Asked for Wipro's Scope 3 target, the first version answered **"2030 Target: 59%"**.
The report says **55% by 2030** (59% is the Scope 1 and 2 target). Page 67 has a
"Targets Vs Performance" chart that the PDF reader turned into loose fragments:

```
Targets Vs Performance
84%2025 Performance
59%2030 Target
2025 Performance 55%**
```

The retrieved chunk held these fragments and the words "Scope 3", but not the heading
that says which chart is which. The fixes, all in `src/model.py`:
1. **context around each match** is now sent too (today: small-to-big), so the clear
   sentence "55% reduction in Scope 3 from 2020 baseline" reaches the model;
2. **prompt rules 5 and 6**: don't pair figures with labels the text doesn't connect,
   and don't draw conclusions such as "target exceeded". The second rule fixed a
   separate answer (on the FY2024 HCLTech report we used at first) that wrongly said its
   42% Scope 3 target was "already exceeded" (29% was achieved; the report says it beat
   an *interim* pathway).

The answer now gives 55% by 2030 correctly.

We also tried fixing the extraction at the source with two other PDF readers:
- **PyMuPDF:** its default mode had the same problem; its position-sorted mode mixed
  the page's left column into the chart line by line, and retrieval got worse. Rejected.
- **Docling**, a layout-analysis library that uses machine-learning models to find
  headings, columns and tables. Its text for page 67 is genuinely clean. But when we
  answered 16 questions with both readers and graded every answer, PyPDF scored 13
  correct / 2 partly / 1 error / 0 missed, Docling 12 / 1 / 1 / 2. Docling also needs a
  large install and about 8 minutes of conversion. We kept PyPDF
  (`eval/pdf_parser_comparison.md`).

A typical example of the remaining errors (from the FY2024 HCLTech report we used at
first): asked how much HCLTech's Scope 1 and 2 emissions fell, the answer correctly said
25% but added "173,743 mtCO2 in FY24". The chart's values were
extracted as an unordered list (`167,426 162,407 158,810 224,094 173,743`) and the model
picked the wrong one; FY24 is 167,426. Prompt rule 5 reduces this kind of error but
cannot prevent it, because the labels needed to pair the numbers are not in the
extracted text.

The safeguard is the **Retrieved passages** panel: every figure can be checked against
the exact text and page in seconds.

### Fiscal years and data integrity: what is and isn't guaranteed

**Fiscal years.** Both reports cover FY2024-25 (April 2024 to March 2025). Our first
version paired Wipro FY2024-25 with HCLTech **FY2024**; we replaced HCLTech's report
because figures genuinely move between years (its Scope 3 reduction since FY20 is 29% in
the FY24 report and 22% in FY25), so mixed years would have made side-by-side answers
misleading.

**What we control**
- **Same period, official sources.** Both PDFs are downloaded unmodified from the
  companies' websites (links in the README) and committed, so anyone can check them.
- **Traceability.** Every claim is cited to a PDF page, and the exact passage is shown.
- **Answers only from the reports.** The model is instructed to use nothing else.
- **Measured accuracy.** Retrieval and answer quality are evaluated and reported,
  including the errors.

**What we cannot guarantee**
- **The companies' own numbers.** GreenScope reports what each company discloses. It
  does not audit it. Parts of these reports are externally assured (both include
  assurance statements), but not every figure.
- **Different baselines and definitions.** Wipro measures against 2017 (Scope 1 and 2)
  and 2020 (Scope 3); HCLTech against FY20. Same year, different yardsticks.
- **Reports that contradict themselves.** HCLTech's FY25 report says on page 8 that 98%
  of its owned buildings are Platinum-rated, and on page 32 that all of them are. We left
  that fact out of the test set because it has no single right answer.
- **Extraction.** Charts and tables can come out scrambled, which causes most of the
  answer errors (section 4).
- **Our test set.** Gold pages and reference answers were found by searching the text
  and have not yet been checked by a person; the questions were written while reading the
  reports, so their wording may be closer to the reports' than a real user's.

## 7. Why each setting was chosen (summary)

| Setting | Value | Why |
|---|---|---|
| Chunk size | 400 characters | Best in tuning for all three models; precise matches (section 5) |
| Overlap | 250 characters | Best at 400 characters (MRR@5 0.84 vs 0.80 with none); facts straddling a cut survive whole |
| Context per match | ~2,000 characters, same page | Small-to-big: the model sees the sentence that explains each figure |
| Minimum chunk | 50 characters | Removes covers, footers and headings that have no facts |
| Embedding model | BAAI/bge-small-en-v1.5 | Trained for search; won the cross-validated comparison against MiniLM and MPNet |
| Similarity | Cosine (unit vectors + inner product) | Matches how the model was trained; equals dot product and Euclidean ranking here |
| Index | FAISS IndexFlatIP, one per company | Exact search in 0.03 s; per-company keeps sources and comparisons fair |
| Ranking | Hybrid: meaning (FAISS) + keywords (BM25), fused with RRF (k = 60) | MRR@5 0.84 -> 0.91; no weight to tune; a re-ranker was tested and left out |
| Two-topic questions | Split into one search per topic (free gate + Haiku) | Both topics found in 8/8 vs 7/8 two-topic questions |
| top_k | 5 (slider 3 to 8) | Hit@k reaches 0.97 at 5, unchanged up to 10 |
| Query clean-up | company name -> "the company" | Hit@5 0.82 -> 0.97 |
| LLM | Claude Haiku 4.5, max 600 tokens | Cheapest current Claude model; the job is reading supplied text and citing it. Measured: about 3 seconds and US$0.002 to US$0.0035 per answer |

## 8. How this project maps to the course rubric

The rubric is written for classical machine learning. Here is how each term applies
to a RAG system:

| Rubric term | In GreenScope | Where |
|---|---|---|
| Preprocessing | PDF text extraction, chunking, removing near-empty chunks, query clean-up | `src/data.py`, `prepare_query` in `src/model.py` |
| Scaling | Normalizing every vector to length 1, so cosine = inner product | `embed_texts` in `src/data.py`, `eval/distance_metrics.md` |
| Feature selection | Which text represents a passage (chunk size and overlap) and which embedding model turns it into features | `eval/tune.py` |
| Hyperparameter tuning | Grid search over 36 combinations of model x chunk size x overlap; top_k from the Hit@k curve | `eval/tuning_results.md` |
| Model optimization | Hybrid search (BM25 + vectors, RRF) adopted; re-rankers tested and rejected; two-topic splitting | `eval/retrieval_methods.md`, `eval/multi_topic_results.md` |
| Cross-validation | 5-fold, company-balanced, choose on 32 questions, score on 8 (for settings and for retrieval methods) | `eval/tuning_results.md`, `eval/retrieval_methods.md` |
| Model explainability (SHAP/LIME) | Exact Shapley values per question word for every retrieved passage, plus page citations | `ReportIndex.explain`, the app's passage panel |
| Error analysis | Inspected every retrieval miss; 120 answers graded by an AI judge checked by hand; errors traced to scrambled charts; two PDF readers tested and rejected | sections 4 to 6, `eval/` |
| Production-grade pipeline | Automated tests (15) run on every push via GitHub Actions; pinned versions; committed index; pre-demo check | `tests/`, `.github/workflows/tests.yml`, `python -m src.model --check` |
| Real-time inference | Each question is embedded and searched live (0.03 s), answered in about 3 s | the app |
| Business metrics / ROI | Persona, ROI model, live "time saved" and cost in the app sidebar | `docs/BUSINESS_CASE.md` |

## 9. Likely examiner questions, with answers

**1. Why not just ask ChatGPT or Claude directly?**
Without retrieval the model answers from memory: it may be out of date, may mix up
fiscal years or companies, may invent a figure, and gives no page reference. RAG
restricts the answer to the actual reports and makes every claim checkable.

**2. Why chunk size 400 and overlap 250?**
They won a grid search of 36 settings, and the same choice won in all 5 folds of
cross-validation (held-out MRR@5 0.84 vs 0.69 for our original 800/150). The intuition:
small chunks give focused fingerprints, because ESG reports pack a metric into almost
every sentence; large overlap keeps facts that straddle a cut whole. The cost, less
context per chunk, is handled by sending ~2,000 characters around each match to the model.

**3. What is an embedding, and why BGE-small?**
A list of numbers that represents meaning, so similar texts have similar vectors.
BGE-small is trained specifically to match questions to passages, which is our task. In
our tests it beat both a general model of the same size (MiniLM) and a model twice its
size (MPNet). It is free and runs on a laptop.

**4. What is cross-validation here, and why did you need it?**
We tried 36 settings. Reporting the best score on the same 40 questions would be
optimistic, since some setting always gets lucky. So we split the questions into 5
groups, chose the setting on 4 groups and tested it on the fifth, five times. The tuned
setting scored 0.84 on questions it had never seen, against 0.69 for the original.

**5. How do you explain the model's decisions (SHAP/LIME)?**
For every retrieved passage we compute exact Shapley values, the method behind SHAP:
each word of the question gets a share of the similarity score, averaged over all orders
in which words could be added. The shares add up exactly to the total. The app colours
the words green or red. It showed us, for instance, that the model barely uses the "3"
in "Scope 3".

**6. Which distance metric do you use, and why?**
Cosine similarity, computed as an inner product on unit-length vectors (FAISS
IndexFlatIP). On unit vectors, cosine, dot product and Euclidean distance give exactly
the same ranking (distance² = 2 − 2 × cosine); we verified this on all 40 questions.
Manhattan distance ranked differently and scored lower (MRR@5 0.812 vs 0.837).

**7. What does FAISS do, and is it overkill for ~3,800 chunks?**
It finds the vectors most similar to the question. At this size plain numpy would also
be fast. FAISS keeps the code standard and would scale to thousands of reports. We use
its exact (flat) index, so no accuracy is traded for speed.

**8. Why combine vector search with keyword search?**
Vectors capture meaning but blur exact terms such as "CDP", "ISO 14001" or
"nationalities"; keyword search (BM25) catches exactly those. We merge the two rankings
with Reciprocal Rank Fusion, which needs no weight to tune. Cross-validated, it raised
MRR@5 from 0.84 to 0.89 on held-out questions (0.91 on the full set). We also tried
re-ranking models; on our questions they made results worse and added seconds per answer.

**9. How do you know the answers are correct?**
Three layers. Retrieval: a correct page is in the top 5 for 97% of 40 test questions.
Answers: we generated 120 answers (40 questions × 3 runs) and had a stronger model grade
each against a reference answer and the source passages: 86% fully correct, 7% with an
error, 2% missed. We checked the judge itself by hand on 20 verdicts (all 20 agreed). And
every answer in the app shows its citations, the exact passages and the word
explanation, so a reader can verify each claim.

**10. How do you know the code works?**
15 automated tests check the core logic: query clean-up, chunking, the BM25 formula,
hybrid ranking, that context never crosses a page (which would break citations), that
Shapley values add up exactly, and the behaviour without an API key. One test checks that
retrieval quality on the real index hasn't dropped. GitHub runs them all on every push.

**11. What happens when the answer isn't in the reports?**
The model is told to say "This is not disclosed in the retrieved text". This can also
happen when the information *is* in the report but retrieval missed it, which is why
we show the passages. Raising top_k with the slider can help.

**12. Why Claude Haiku rather than a bigger model?**
The task is reading five supplied passages and quoting them with citations, which
doesn't need deep reasoning. Haiku 4.5 is the cheapest current Claude model (about
US$0.002 to US$0.0035 and 3 seconds per answer). Our tests show answer quality depends
mostly on whether retrieval found the right passages and whether the PDF text is clean,
which a bigger model doesn't fix.

**13. Is the comparison between Wipro and HCLTech fair?**
Mostly. Both reports cover the same fiscal year (FY2024-25), and retrieval is fair: each
company gets its own top 5. Our first version paired Wipro FY2024-25 with HCLTech
FY2024; we replaced HCLTech's report because the figures really move between years (its
Scope 3 reduction since FY20 is 29% in the FY24 report and 22% in FY25). What still
differs: baselines (Wipro uses 2017 and 2020, HCLTech FY20) and some definitions. The
app says so, and the citations let users check the exact wording.

**14. What was the biggest problem you found, and how did you fix it?**
Two. First, the company name in questions matched page footers ("About HCLTech ...
92" in the FY2024 report we started with); replacing it with "the company" raised Hit@5 from 0.56 to 0.94 on our first test
set. Second, a chart on Wipro's page 67 made the model report 59% (the Scope 1 and 2
target) as the Scope 3 target; we traced it to scrambled chart text and fixed it with
context around each match and two prompt rules. Section 6 has the details, including
two PDF readers we tested and rejected.

**15. What is the business value?**
For an ESG analyst, a lookup that takes about 10 minutes by hand takes about 2 with
GreenScope, including checking the cited passage. For a team doing 2,400 lookups a year
that is 320 hours. The ROI is 104% at Indian analyst rates and 213% at global
consultancy rates, after hosting and maintenance. The 10 and 2 minutes are assumptions;
`docs/BUSINESS_CASE.md` describes a time trial to measure them.

**16. What are the main limitations, and what would you improve next?**
Charts and images aren't read, and tables are extracted poorly: that causes most of
the 7% of answers with an error and the 2% misses. The evaluation set (40 questions) is
small, its gold pages are not yet human-verified, and the answer judge was checked on
only 20 verdicts. Page numbers are PDF file pages, not printed pages. Next steps:
layout-aware text for chart and table pages only (or a vision model reading those
pages), a larger human-verified
evaluation set, and the time trial to measure the business case.

---

## 10. Small examples to work on the board

Each example uses tiny made-up numbers so it can be done by hand in a minute or two. The
method is exactly what GreenScope does; only the sizes are smaller. Where the real
numbers differ, the last line of each example says what they are.

### 10.1 Chunking with overlap

Text: the 26 letters `ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Chunk size 10, overlap 4, so each new
chunk starts 10 − 4 = 6 letters after the previous one:

| Chunk | Positions | Letters |
|---|---|---|
| 1 | 1–10 | ABCDEFGHIJ |
| 2 | 7–16 | GHIJKLMNOP |
| 3 | 13–22 | MNOPQRSTUV |
| 4 | 19–26 | STUVWXYZ |

A fact sitting on a cut, say "IJK", is split between chunks 1 and 2 without overlap, but
with overlap chunk 2 holds it whole. **Real:** 400 characters with 250 overlap, and the
splitter prefers to cut at paragraph, line or word breaks rather than mid-word.

### 10.2 Cosine similarity: which passage is closest?

Pretend embeddings have only 2 numbers instead of 384. Divide each vector by its length
so it has length 1 (this is what `normalize_embeddings=True` does):

| | Vector | Length | Unit vector |
|---|---|---|---|
| Question | (3, 4) | 5 | (0.6, 0.8) |
| Passage A | (4, 3) | 5 | (0.8, 0.6) |
| Passage B | (0, 10) | 10 | (0, 1) |
| Passage C | (5, 0) | 5 | (1, 0) |

Cosine similarity = multiply matching numbers and add (the dot product of unit vectors):

- A: 0.6 × 0.8 + 0.8 × 0.6 = **0.96**
- B: 0.6 × 0 + 0.8 × 1 = **0.80**
- C: 0.6 × 1 + 0.8 × 0 = **0.60**

Ranking: A, B, C. Note B's raw vector is long (10) but that doesn't help it: only the
direction counts. **Why Euclidean distance gives the same order:** for unit vectors,
distance² = 2 − 2 × cosine, so A = 0.08, B = 0.40, C = 0.80 (smallest distance = most
similar, same order). **Real:** 384 numbers per vector, and FAISS does these
multiplications for about 2,000 passages per report in about 0.03 seconds.

### 10.3 BM25 keyword score: rare words count more

Imagine a report of N = 4 passages. Each query word gets a rarity weight (IDF):

> IDF = ln( 1 + (N − n + 0.5) / (n + 0.5) ), where n = passages containing the word

- "CDP" appears in 1 passage: ln(1 + 3.5 / 1.5) = ln(3.33) ≈ **1.20**
- "company" appears in all 4: ln(1 + 0.5 / 4.5) = ln(1.11) ≈ **0.11**

So a match on "CDP" is worth about 11 times a match on "company". Repetition helps, but
with diminishing returns: for a passage of average length, the count factor is
count × 2.5 / (count + 1.5) (k1 = 1.5):

| Times the word appears | 1 | 2 | 10 |
|---|---|---|---|
| Count factor | 1.00 | 1.43 | 2.17 |

Ten mentions are worth only about twice one mention, so a passage can't win by
repeating a word. Passage score = sum over query words of IDF × count factor.

### 10.4 Reciprocal Rank Fusion: combining the two rankings

Every passage gets a rank from meaning search and a rank from keyword search.
Score = 1/(60 + meaning rank) + 1/(60 + keyword rank):

| Passage | Meaning rank | Keyword rank | Score |
|---|---|---|---|
| X | 1 | 5 | 1/61 + 1/65 = 0.01639 + 0.01538 = **0.03178** |
| Y | 3 | 1 | 1/63 + 1/61 = 0.01587 + 0.01639 = **0.03227** |
| Z | 2 | 40 | 1/62 + 1/100 = 0.01613 + 0.01000 = **0.02613** |

Final order: **Y, X, Z**. Y is not first by meaning, but being near the top of *both*
lists beats being first in one. Z ranks second by meaning but the keyword search barely
finds it, so it drops. Only ranks are used, never raw scores, so there's no weight to
tune between a cosine score (0 to 1) and a BM25 score (0 to 20 or more).

### 10.5 Shapley values: sharing the credit between two words

Question with two content words, "water" and "target". Suppose these are the similarities
to one passage (filler words like "what" and "the" are always kept):

| Words kept | Similarity |
|---|---|
| neither | 0.50 |
| water only | 0.70 |
| target only | 0.60 |
| both | 0.75 |

A word's Shapley value = its average extra similarity over both orders of adding words:

- **water**: added first: 0.70 − 0.50 = 0.20; added second: 0.75 − 0.60 = 0.15. Average = **0.175**
- **target**: added first: 0.60 − 0.50 = 0.10; added second: 0.75 − 0.70 = 0.05. Average = **0.075**

Check: 0.175 + 0.075 = 0.25 = 0.75 − 0.50, so the values add up to the full effect. In the
app "water" would be shown in a darker green than "target". **Real:** up to 8 content
words, so up to 2⁸ = 256 versions of the question are embedded per answer.

### 10.6 Hit@k and MRR: grading the search

Four test questions. The rank of the first passage from a correct page:

| Question | First correct rank | 1 / rank |
|---|---|---|
| Q1 | 1 | 1 |
| Q2 | 3 | 0.333 |
| Q3 | not in top 5 | 0 |
| Q4 | 2 | 0.5 |

- Hit@1 = 1 of 4 = **0.25**; Hit@3 = 3 of 4 = **0.75**; Hit@5 = **0.75**
- MRR@5 = (1 + 0.333 + 0 + 0.5) / 4 = **0.46**
- Precision@5 for Q1, if 3 of its 5 passages come from correct pages: 3 / 5 = **0.6**

**Real (40 questions):** Hit@1 0.88, Hit@5 0.97, MRR@5 0.91, Precision@5 0.70.

### 10.7 Five-fold cross-validation

Split the 40 questions into 5 folds of 8 (4 Wipro + 4 HCLTech each). Draw five rows:

| Round | Choose the best setting using | Score it on |
|---|---|---|
| 1 | folds 2, 3, 4, 5 (32 questions) | fold 1 (8 questions) |
| 2 | folds 1, 3, 4, 5 | fold 2 |
| 3 | folds 1, 2, 4, 5 | fold 3 |
| 4 | folds 1, 2, 3, 5 | fold 4 |
| 5 | folds 1, 2, 3, 4 | fold 5 |

Every question is used for scoring exactly once, and never by the round that chose the
setting. The result is the average of the five held-out scores ± their spread.
**Real:** held-out MRR@5 0.84 ± 0.09 (tuned) vs 0.69 ± 0.12 (original); the same setting
won all 5 rounds.

### 10.8 The ROI, India scenario

- Time saved: 2,400 lookups × 8 minutes = 19,200 minutes = **320 hours**
- Value: 320 h × ₹500/h = ₹1.6 lakh ≈ **US$1,920** (₹500 ≈ US$6)
- Cost: AI 2,400 × US$0.003 ≈ US$7, hosting US$360, maintenance 96 h × US$6 = US$576;
  total **US$943**
- Net benefit: 1,920 − 943 = US$977; ROI = 977 / 943 ≈ **104%**
- Break-even: each lookup earns 8/60 h × US$6 − US$0.003 ≈ US$0.80 against fixed costs
  of US$936, so 936 / 0.80 ≈ **1,200 lookups a year**

---

## 11. Q&A cheat sheet: the hard questions

**17. Why not just give the whole PDF to a long-context model, or use NotebookLM?**
We tested it (section 4, "Two extra checks"). Same model, whole report in the prompt: 8 of
10 correct, against 23 of 30 runs for GreenScope on the same questions, so about the same
accuracy on a small sample. But it costs US$0.083 per question (US$0.028 with caching)
against about US$0.003 for GreenScope: 9 to 28 times more. GreenScope also shows the exact
passages and why they matched, and keeps working when there are more reports than fit in
one prompt. Tools like NotebookLM do cite sources; our difference is page-level citations
restricted to the reports, a side-by-side mode, and measured accuracy.

**18. How do you know you didn't overfit to your 40 questions?**
Partly we do, and we say so. The main settings (chunk size, overlap, embedding model) were
chosen with 5-fold cross-validation: held-out MRR@5 0.84 ± 0.09 vs 0.69 for the original.
Hybrid search was also cross-validated (held-out 0.89 vs 0.84). But five later choices
were made on the same 40 questions, so the final full-set numbers (Hit@5 0.97, 86% correct)
are somewhat optimistic. So we wrote 20 fresh questions, froze them on GitHub before the first
run, and changed nothing afterwards: 57 of 60 answers were correct and the right page was in the
top 5 for 16 of 16. Caveat: we wrote them ourselves, and they are single-fact questions; a set
written by an outside analyst is the next step.

**19. What happens with tables, charts, or a question the reports don't answer?**
Unanswerable: we asked 4 such questions and GreenScope declined all 4 ("This is not
disclosed in the retrieved text"). Tables and charts: this is our main weakness. PDF text
extraction turns them into unlabelled lists of numbers, and that causes most of the 8 wrong
answers. We tried a layout-aware parser (Docling): no better on graded answers and much
slower. The next step is to send only the table and chart pages to a vision model, which
reads the page as an image; as a rough estimate this is a one-off cost of well under US$1
for both reports at indexing time.

**20. Does it generalise beyond two Indian IT companies?**
Not tested. Nothing in the code is specific to these companies: adding a report is one line
in `src/data.py` plus re-indexing (about 2 minutes). But our accuracy numbers only hold for
these two reports, and other sectors (e.g. manufacturing) have more tables. A new report
needs new test questions before we can claim the same accuracy.

**21. Why Claude Haiku 4.5? Did you compare models?**
It is the cheapest current Claude model, and answer cost drives the business case. We did
not run a systematic model comparison. The evidence that the model is not the bottleneck:
most errors come from scrambled table text in the passages, which a bigger model would
receive in the same form. The grader is a stronger model (Claude Sonnet 5.5). It is from
the same family, so self-preference is possible; that is why we hand-checked 20 of its
verdicts against the PDF (all 20 agreed).

**22. A 7% error rate is risky for analysts. How do you handle that?**
GreenScope is decision support, not an oracle. Every claim carries a page citation, the
exact passages are one click away, and the word explanation shows why each passage was
picked. The ROI assumes the analyst checks the cited passage (2 minutes per lookup). The
errors are mostly a figure paired with the wrong label from a table, which the cited
passage makes visible. Citations were fully supported in 90% of answers, so the analyst
should check the citation, not just the number.

**23. Who would pay US$99 a month, and what evidence do you have?**
No direct evidence yet: that is what the pilot and the time trial are for. The arithmetic:
an analyst doing about 200 lookups a month saves about 27 hours at our assumed 8 minutes
each. At ₹500/hour that is worth about ₹13,300 (about US$160), so US$99 is a thin margin in
India and regional pricing would likely be needed; at a global consultancy rate (US$60/hour)
the same hours are worth about US$1,600. The ">95% gross margin" counts only AI usage, not
hosting or support.

**24. What exactly is the Shapley value function?**
The "players" are the content words of the question; filler words ("what", "the", "of")
are always kept. For a set S of content words, v(S) = cosine similarity between the
passage and the question containing only the words in S (plus the filler words), each
embedded with BGE-small. A word's Shapley value is its average extra v when added, over
all orders of adding words. The values add up to v(all words) − v(no content words). That
second term is not zero: the filler-only question ("what are the company's ...") already
has some similarity to most passages. That is why the word values on the slide add up to
about 0.30 while the full similarity is 0.786. "Scope" and "3" are scored as separate words
because the method works word by word; it showed that "3" adds very little, i.e. the
embedding barely tells Scope 3 from Scope 1 and 2.

**25. Your ROI costs: where do they come from?**
Per team per year: AI 2,400 lookups × US$0.003 ≈ US$7 (single-company answers; if every
lookup were a side-by-side comparison, about US$14). Hosting US$360. Maintenance 96 hours,
priced at the analyst's own rate: US$576 in India (US$6/hour), US$5,760 for a global
consultancy (US$60/hour). Totals: US$943 and US$6,127. The 8 minutes saved per lookup is
an assumption; the 10-question time trial in the business case would replace it with a
measurement.
