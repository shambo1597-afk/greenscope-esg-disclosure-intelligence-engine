# GreenScope, explained in plain language

This document walks through how GreenScope works, why each setting was chosen,
where it can still go wrong, and how to answer the questions an examiner is
likely to ask. No coding knowledge is needed.

---

## 1. The problem

Wipro's sustainability report is 163 pages long and HCLTech's is 113. Finding one
figure, such as a Scope 3 emissions target, means a lot of scrolling, and comparing
the two companies means doing it twice. A general chatbot could answer from memory,
but it might be out of date or invent a number, and it can't tell you which page
the figure came from.

GreenScope answers questions **only from the two reports**, and every claim comes
with a page reference you can check.

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
 ┌───────────────────┐   "What is the Scope 3 target?"
 │ 1. Embed question │──► list of 384 numbers that capture its meaning
 └───────────────────┘
      │
      ▼
 ┌───────────────────┐   compare with ~600 stored passage vectors per report
 │ 2. FAISS search   │──► top 5 most similar passages (+ page numbers)
 └───────────────────┘
      │
      ▼
 ┌───────────────────┐   "Answer ONLY from these passages, cite [Company, p.X]"
 │ 3. Claude Haiku   │──► answer with page citations
 └───────────────────┘
```

## 3. Each step in THIS project

### 3.1 Loading the PDFs (`src/data.py`)

`PyPDFLoader` reads each PDF page by page and extracts the text. Each page keeps
its **page number in the PDF file**: page 1 is the first page of the file. This can
differ from the number printed on the page, since covers usually aren't numbered.
We use the file page because that's what you type into a PDF viewer to jump there.

Limitation: only text is extracted. Numbers inside images or charts are invisible,
and complex tables come out as jumbled lines of text.

### 3.2 Chunking: cutting pages into passages

A page can cover several topics, which is too much to compare with one short
question. So we cut the text into **chunks** of up to **800 characters**
(about 130 to 160 words, one or two paragraphs), with **150 characters of overlap**
between neighbouring chunks.

- **Why 800 characters? To keep chunks focused.** Each chunk is turned into a single
  "meaning fingerprint" (next section). If a chunk mixes emissions, water use and
  board diversity, its fingerprint is a blur of all three and matches none of them
  well. 800 characters is usually one topic. Much smaller chunks (200 to 300
  characters) would split a number from the words explaining it: "42%" means
  nothing without "reduction in Scope 3 by FY30". 800 is also safely under the
  embedding model's input limit (256 word-pieces), so no text is silently cut off.
- **Why 150 overlap? To keep context across cuts.** The cut is mechanical and can
  land mid-sentence: "...reduce emissions by" | "42% by FY30". Repeating the last
  ~150 characters (about one sentence) at the start of the next chunk means a fact
  that crosses the cut still appears whole in at least one chunk. Under 20% overlap
  keeps duplicated text low.
- The splitter (`RecursiveCharacterTextSplitter`) cuts at the most natural place
  available: between paragraphs, then lines, then words.
- Chunks shorter than **50 characters** are dropped. They are cover titles, page
  footers and section dividers with no facts in them.

Result: **636 chunks for Wipro and 460 for HCLTech**.

### 3.3 Embeddings: turning text into meaning fingerprints

An **embedding model** turns a piece of text into a list of numbers (a *vector*,
here 384 numbers). Texts with similar meaning get similar vectors, even when they
use different words: "renewable electricity share" lands close to "84% of our
power comes from green sources".

We use **`sentence-transformers/all-MiniLM-L6-v2`**:
- it's free and runs on a normal laptop CPU, with no API cost and no data leaving
  your machine for this step;
- it's small (~90 MB) and fast: both reports embed in about a minute;
- it's a well-known, widely used baseline, which makes the results easy to compare and defend.

The vectors are **normalized** (scaled to length 1). That makes the similarity
calculation in the next step a simple multiplication.

Embedding happens **once**: results are saved in the `index/` folder and reloaded
on later runs. If the PDFs or the chunk settings change, the index is rebuilt automatically.

### 3.4 Cosine similarity and FAISS: finding the closest passages

To compare the question with a chunk, we measure the **angle** between their two
vectors. This is called **cosine similarity**:
- 1.0 = pointing the same way (same meaning),
- around 0 = unrelated.

Because the vectors are normalized, cosine similarity equals the *dot product*
(multiply matching numbers, add them up). **FAISS** (Facebook AI Similarity Search)
is a library that does this very quickly. We use `IndexFlatIP`:
- *Flat* = compare against every chunk (exact, no shortcuts). With ~600 chunks per
  report this takes well under a millisecond, so faster approximate methods aren't needed.
- *IP* = inner product, i.e. the dot product, i.e. cosine similarity for our vectors.

There is **one index per company**. This guarantees we always know which report a
passage came from. In **Compare** mode it also guarantees both companies get their
own top 5. A single combined search could return five Wipro passages and none from
HCLTech, which would make the comparison unfair.

**Query clean-up.** Before searching, GreenScope replaces the company name in your
question with "the company" (e.g. "HCLTech's water target" becomes "the company's
water target"). Since we're already searching inside one company's report, the name
adds nothing. Worse, it made the question look similar to useless page footers like
"About HCLTech / HCLTech Sustainability Report 2024 / 92". This one change raised the
share of questions with a correct page in the top 5 from **0.56 to 0.94** (see
`eval/results.md`). Claude still sees your original question.

**Neighbouring text.** The search matches 800-character chunks, but the sentence
that explains a number is often in the chunk right next to it. So each match is
sent to Claude together with the chunk before and after it **on the same page**
(overlaps merged, repeats removed). Staying on the same page keeps every
citation correct. The *Retrieved passages* panel shows exactly this widened text.
See section 5 for the real error that led to this.

### 3.5 Prompt grounding: making the model stick to the evidence

The retrieved passages are sent to **Claude Haiku 4.5**, each labelled with its
source, e.g. `[Wipro, p.67]`. The **system prompt** (standing instructions) sets the rules:

1. answer **only** from the provided passages, never from outside knowledge;
2. cite every claim as **[Company, p.X]**;
3. keep numbers, units and years **exactly as written**;
4. if the passages don't contain the answer, say **"This is not disclosed in the
   retrieved text"**;
5. text from charts and tables can be scrambled, so only link a figure to a
   category (Scope 1 and 2 vs Scope 3, target vs achieved, a year) when the text
   explicitly connects them; otherwise say it's unclear;
6. don't draw conclusions the text doesn't state (e.g. "the target has been met");
7. be concise.

This is called **grounding**: the answer is tied to evidence you can inspect. The
app shows the passages under every answer (*Retrieved passages*), with their
similarity score and page, so you can check each claim yourself.

## 4. Why each setting was chosen

| Setting | Value | Why |
|---|---|---|
| Chunk size | 800 characters | Focused on one topic, keeps a figure with its label/unit/year, fits the embedding model's limit |
| Overlap | 150 characters | About one sentence: facts split by a cut survive whole in one chunk; under 20% keeps duplication low |
| Minimum chunk | 50 characters | Removes covers, footers and headings that have no facts |
| Embedding model | all-MiniLM-L6-v2 | Free, local, fast, well-known baseline; 384-number vectors |
| Similarity | Cosine (normalized + inner product) | Compares meaning regardless of passage length |
| Index | FAISS IndexFlatIP, one per company | Exact search is instant at this size; per-company keeps sources and comparisons fair |
| top_k | 5 (slider 3 to 8) | Eval: the right page is in the top 5 for 15 of 16 questions but only in the top 1 for 9. Five passages (~1,000 words) give enough evidence without drowning the answer in noise or cost |
| LLM | Claude Haiku 4.5, max 600 tokens | Cheapest current Claude model; the job is reading supplied text and citing it, not deep reasoning. Measured cost: about $0.002 to $0.0035 per answer (≈1,500 to 2,500 input + 100 to 300 output tokens at $1 / $5 per million tokens). Compare mode makes two calls. $5 of credit covers well over 1,000 answers. The app shows the cost under every answer |

## 5. Where hallucination (made-up content) can still happen

Grounding reduces the risk a lot, but doesn't remove it:

- **Retrieval misses.** If the right passage isn't in the top 5, the model should
  say "not disclosed", but it may stretch a related passage into an answer. Example:
  answering "what is the target" with a progress figure.
- **Mis-citation.** The model can attach the right fact to the wrong page tag when
  several passages are similar.
- **Garbled extraction.** Tables and infographics come out as jumbled text (e.g.
  numbers listed after all their labels). The model may pair a number with the wrong label.
- **Different fiscal years.** Wipro's report is FY2024-25 and HCLTech's is FY2024.
  A side-by-side answer can look like a like-for-like comparison when it isn't.
- **Different definitions.** "Renewable share" may mean purchased electricity in one
  report and total energy in another; baselines differ (2017 vs FY20).
- **Paraphrase drift.** Despite the instruction, the model may round or reword a figure.
- **Model knowledge leaking in.** Claude may know facts about these companies from
  training. The prompt forbids using them, but that can't be guaranteed 100%.

- **Questions about two topics at once.** "What are the water **and** waste goals?"
  is turned into one search, which returned only water passages, so the answer said
  waste goals weren't disclosed (both reports do state them). Ask about one topic at a time.

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

The retrieved chunk held these fragments and the words "Scope 3", but not the
heading that says which chart is which. Two fixes, both in `src/model.py`:
1. **neighbouring chunks** are now sent too, so the clear sentence "55% reduction in
   Scope 3 from 2020 baseline" from the adjacent chunk reaches the model;
2. **two prompt rules** (5 and 6 above): don't pair figures with labels the text
   doesn't connect, and don't draw conclusions such as "target exceeded". The second
   rule fixed a separate HCLTech answer that wrongly said its 42% Scope 3 target was
   "already exceeded" (29% was achieved; the report says it beat an *interim* pathway).

After the fixes the answer gives 55% by 2030 correctly. One small slip remained:
it describes the 233,303 tCO2e reduction (already achieved) as "the 2030 target".
This is why the evidence panel matters.

We also tried fixing the extraction at the source with two other PDF readers:
- **PyMuPDF:** its default mode had the same problem; its position-sorted mode mixed
  the page's left column into the chart line by line, and retrieval Hit@5 fell from
  0.94 to 0.88. Rejected.
- **Docling**, a layout-analysis library that uses machine-learning models to find
  headings, columns and tables. Its text for page 67 is genuinely clean (Scope 3 figures
  in their own table under a "Scope 3 emissions" heading). But when we answered all 16
  eval questions with both readers and graded every answer, PyPDF scored 13 correct,
  2 partly correct, 1 with an error; Docling 12 correct, 1 partly correct, 1 with an
  error and 2 misses. Retrieval Hit@5 was the same (0.94). Docling also needs a large
  install and about 8 minutes of conversion. We kept PyPDF. Full results:
  `eval/pdf_parser_comparison.md`.

That answer-quality test also found an error that is still there: asked how much
HCLTech's Scope 1 and 2 emissions fell, the answer correctly says 25% but adds
"173,743 mtCO2 in FY24". The chart's values were extracted as an unordered list
(`167,426 162,407 158,810 224,094 173,743`) and the model picked the wrong one; FY24 is
167,426. Prompt rule 5 reduces this kind of error but cannot prevent it, because the
labels needed to pair the numbers are simply not in the extracted text.

The safeguard is the **Retrieved passages** panel: every figure can be checked
against the exact text and page in seconds.

## 6. How we measured retrieval quality

`eval/run_eval.py` runs 16 questions (8 per company). For each, we listed the
**gold pages**, the PDF pages where the answer actually appears, found by searching
the extracted text. We then check whether the search returns those pages:

| Metric | Meaning | Overall |
|---|---|---|
| Hit@1 | The very first passage is from a gold page | 0.56 |
| Hit@3 | A gold page is in the top 3 | 0.81 |
| Hit@5 | A gold page is in the top 5 (what the LLM sees by default) | 0.94 |
| MRR@5 | Average of 1/rank of the first gold page | 0.72 |
| Precision@5 | Share of the 5 passages that are from gold pages | 0.45 |

This costs nothing (no AI calls). Limitations: only 16 questions; matching is by
page, not by exact passage; the gold pages haven't yet been double-checked by a
person; and it measures retrieval, not answer correctness. Full table and details:
`eval/results.md`.

## 7. Likely examiner questions, with answers

**1. Why not just ask ChatGPT or Claude directly?**
Without retrieval the model answers from memory: it may be out of date, may mix up
fiscal years or companies, may invent a figure, and gives no page reference. RAG
restricts the answer to the actual reports and makes every claim checkable.

**2. Why chunk size 800 and overlap 150?**
800 characters is about one topic, so each chunk's embedding is focused, and it
keeps a number with its label, unit and year. It also fits under the embedding
model's 256-token limit. The 150 overlap (about one sentence) means a fact split by a
cut still appears whole in one chunk, while keeping duplication under 20%. We did not
run a full sweep of alternatives; that's a sensible next experiment using the eval script.

**3. What is an embedding, and why MiniLM?**
A list of numbers that represents meaning, so similar texts have similar vectors.
MiniLM is free, runs locally, is fast, and is a standard baseline. A larger embedding
model might retrieve better; the eval script makes that easy to test.

**4. What does FAISS do, and is it overkill for ~1,100 chunks?**
It finds the vectors most similar to the question. At this size a plain numpy
calculation would also be instant. FAISS keeps the code standard and would scale to
thousands of reports. We use its exact (flat) index, so no accuracy is traded for speed.

**5. How do you know the answers are correct?**
Two layers. First, the retrieval eval: a correct page is in the top 5 for 94% of test
questions. Second, every answer shows its citations and the exact passages, so a
reader can verify each claim. We have not yet run a systematic evaluation of the
generated answers themselves; that would need human grading or an LLM judge.

**6. What happens when the answer isn't in the reports?**
The system prompt tells the model to say "This is not disclosed in the retrieved
text". This can also happen when the information *is* in the report but retrieval
missed it, which is why we show the passages. Raising top_k with the slider can help.

**7. Why Claude Haiku rather than a bigger model?**
The task is reading five supplied passages and quoting them with citations, which
doesn't need deep reasoning. Haiku 4.5 is the cheapest current Claude model (~$0.005
per answer), which matters on a student budget. Answer quality depends mostly on
whether retrieval found the right passages, which the model size doesn't change.

**8. Is the comparison between Wipro and HCLTech fair?**
Only partly. Retrieval is fair: each company gets its own top 5. But the reports
cover different fiscal years (FY2024-25 vs FY2024), use different baselines
(2017/2020 vs FY20) and sometimes different definitions. The app states the
fiscal-year difference on every page, and the citations let users check definitions.

**9. What was the biggest problem you found, and how did you fix it?**
The first eval found a correct page in the top 5 for only 56% of questions, and just
38% for HCLTech. Looking at the misses showed the company name in the question was
matching page footers like "About HCLTech ... 92". Since each search is already
restricted to one company, we replace the name with "the company" before searching.
Hit@5 rose to 94%. The eval report keeps both numbers so the effect can be reproduced.

**10. Did you find any wrong answers? What did you do?**
Yes. The Wipro Scope 3 target came out as 59% instead of 55% because a chart on
page 67 was extracted as loose numbers and labels. We traced it to the exact
passage, then sent neighbouring chunks along with each match and added prompt rules
against pairing unconnected figures and against unstated conclusions. The answer is
now correct; section 5 has the details, including a fix we tried and rejected
(two other PDF readers) and the evidence for rejecting them.

**11. What are the main limitations, and what would you improve next?**
Charts and images aren't read, and tables are extracted poorly. The eval set is small
and not yet human-verified. Page numbers are PDF file pages, not printed pages. We tested a
layout-aware parser (Docling): cleaner tables, but no better answers on our test, so
we kept the simpler one (see `eval/pdf_parser_comparison.md`). Next steps: using
layout-aware text only for chart and table pages, hybrid search (meaning plus keyword matching, which helps
with exact terms like "CDP" or "Scope 3"), a re-ranking step, a larger human-verified
eval set, and an evaluation of answer faithfulness.
