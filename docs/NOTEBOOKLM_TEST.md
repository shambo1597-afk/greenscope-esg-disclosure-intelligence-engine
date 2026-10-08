# Competitor check: NotebookLM vs GreenScope (about 15 minutes)

NotebookLM is the closest free alternative: upload PDFs, ask questions, get cited answers.
This test asks it the same 10 questions GreenScope was graded on, so we can say in Q&A how we compare.

The 10 questions were chosen before seeing any NotebookLM answer. **They are deliberately hard for us:**
5 of them are questions GreenScope gets wrong or only partly right, so this is not a cherry-picked win.

## Steps

1. Go to notebooklm.google.com, then **New notebook**.
2. Upload the two PDFs from `data/raw/` in the GitHub repo: `wipro_sustainability_2024_25.pdf` and `hcltech_sustainability_fy2025.pdf` (on GitHub, open each file and click the download button). Nothing else.
3. Paste each question below **exactly as written**, one at a time, in a fresh chat each time if possible.
4. For each answer, compare it with the "Correct answer" column and write one letter in the NotebookLM column:
   - **C** = correct and complete
   - **P** = partly correct (true, but misses the main number)
   - **E** = contains a wrong number or fact
   - **M** = missed: says it can't find something that is in the report
   - For the two "not in the report" questions: **C** if it says it isn't disclosed, **E** if it gives a number.
5. Also tick **Cite OK** if its citation points to the page in the "Page" column (hover over the citation number).
6. Send me the filled table (a photo is fine) and I'll add the result to the explainer.

## Questions

| # | Question (paste exactly) | Correct answer | Page | GreenScope (3 runs) | NotebookLM | Cite OK |
|---|---|---|---|---|---|---|
| 1 | What is Wipro's 2030 target for Scope 1 and 2 emissions? | 59% reduction by 2030 from a 2017 baseline | 46, 63, 67, 88 | E E E | C | not checked |
| 2 | What share of Wipro's spend goes to diverse suppliers? | 13.5% supplier diversity spend | 7 | M M M | C | not checked |
| 3 | What was Wipro's Lost Time Injury Frequency Rate (LTIFR) in FY25? | 0.27 per million person-hours (0.18 in FY24) | 111 | C C C | C | not checked |
| 4 | What was Wipro's total energy consumption in FY25? | About 754 trillion Joules | 65 | C C C | C | not checked |
| 5 | What internal carbon price per tonne of CO2 does Wipro use? | Not in the report | – | C C C | P | not checked |
| 6 | What percentage of HCLTech's employees are women? | 28.8% in FY25; target 40% by 2030 | 11, 49 | E C E | C | not checked |
| 7 | How has HCLTech reduced its water consumption? | 32.2% less Pan-India water vs 2019-20; 31x replenished in India | 6, 13, 35 | P P P | C | not checked |
| 8 | How much did HCLTech's average training hours per employee change from FY24 to FY25? | Up 27% for both male and female employees | 46 | E E E | E | not checked |
| 9 | How many members are on HCLTech's Board, and what share are independent directors? | Ten members, 70% independent | 85 | C C C | C | not checked |
| 10 | By what year does HCLTech aim to become water positive? | Not in the report (no target year) | – | C C C | C | not checked |

GreenScope on these 10: 16 of 30 runs fully correct (53%), versus 86% on all 40 original questions and 95% on the fresh set.
That gap is the point: these are its hardest questions.

## Result (run on 8 October 2026, graded by Claude against the report text)

**NotebookLM: 8 of 10 correct, 1 partly correct, 1 with an error. GreenScope on the same 10: 16 of 30 runs (53%), 5 questions correct in a majority of runs.**
NotebookLM is clearly better on these hard questions. It got all four that GreenScope misses or gets partly right
because the answer sits in a chart, a highlights infographic or a long table (questions 1, 2, 6 and 7).

| # | NotebookLM | Note |
|---|---|---|
| 1 | C | 59% by 2030 from 2017; also correctly adds that 84% is already achieved (p.67-68). |
| 2 | C | 13.5% (p.7 infographic, which GreenScope misses every time). |
| 3 | C | 0.27 (0.18 in FY24). |
| 4 | C | 754 TJ with the full breakdown. |
| 5 | P | Wipro has no internal carbon price. NotebookLM gave the US$124.66 social cost of carbon Wipro uses for natural capital valuation (p.51), correctly labelled, but under an "internal carbon price" heading instead of saying there is none. Its HCLTech figures (US$10 benchmark, ₹1,609 in FY25, p.26-27) are right. |
| 6 | C | 28.8% (and 28.66% of headcount, p.49); does not mention the 40% target. |
| 7 | C | 32.2% and 31x, plus a long list of measures. |
| 8 | E | Wrong chart pairing (male 22.12 → 25.84; overall "22.50", a number not in the report). Same failure as GreenScope. |
| 9 | C | Ten members, 70% independent. |
| 10 | C | Says there is no target year. Its claim that HCLTech "achieved net water positivity" is its own inference from the 31x figure; the report does not use that phrase. |

Citations were not checked (the pasted answers had no citation markers).

**Finding: question 8 has no clean answer.** The report text (HCLTech p.46) says training hours rose "27% among both
male and female employees", but the chart on the same page shows male 25.80 → 22.12 (down 14%), female 25.84 → 29.63
(up 15%) and overall 23.18 → 26.89 (up 16%). The report contradicts itself. Our reference answer used the text; both
tools mangled the chart. This is a real example of why GreenScope shows the passage instead of only a number.

## How to use the result in Q&A

- **NotebookLM about as accurate:** "A general tool does about as well on single lookups. Our edge is side-by-side
  comparison, word-level explanations of why a passage was picked, a measured accuracy number, and cost control
  (under half a US cent per answer, and it runs as our own app rather than a consumer product)."
- **NotebookLM clearly better:** "It likely reads charts and tables better. That matches our own error analysis,
  and it's why a vision model for table and chart pages is our first next step."
- **NotebookLM worse:** "It's one small test on 10 questions with one run each, so it's a signal, not proof."

Limits: 10 questions, 1 run for NotebookLM, graded by Claude reading the report rather than by the judge model used
for GreenScope, and the questions were chosen to be GreenScope's hardest. It shows where NotebookLM is stronger
(charts and infographics); it does not say which tool is better overall.
