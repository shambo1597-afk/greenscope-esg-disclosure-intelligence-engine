# Experiment: PyPDF vs Docling for PDF text extraction

**Question.** PyPDF (used by the app) scrambles charts and tables: on Wipro p.67 it
separated "59% 2030 Target" from its heading, and the model reported 59% as the Scope 3
target (it is the Scope 1 and 2 target). Would a layout-aware parser,
[Docling](https://github.com/docling-project/docling), give better answers?

**Method.** Both reports were converted with Docling (one markdown string per page,
OCR off, table detection on), then run through exactly the same chunking (800/150),
embedding (MiniLM), retrieval (FAISS, company names removed from the query,
neighbouring chunks added) and generation (Claude Haiku 4.5, same prompt) as the app.
Reproduce with `python eval/compare_pdf_parsers.py --answers` (needs `pip install docling`;
conversion takes about 8 minutes, the 32 answers cost about US$0.08).

## What Docling does better

On Wipro p.67 the extraction is much cleaner. PyPDF:

```
Targets Vs Performance
84%2025 Performance
59%2030 Target
2025 Performance 55%**
```

Docling:

```
Scope 1 and 2 emissions
2030 Target
59%
2025 Performance
84%
...
## Scope 3 emissions
| 2030 Target      | 55%     |
| 2025 Performance | 55% **  |
| 2025 Performance | 57% *** |
```

## Retrieval (16 questions, top 5 passages)

| Parser | Group | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|
| PyPDF | Wipro | 0.50 | 0.75 | 1.00 | 0.68 |
| PyPDF | HCLTech | 0.62 | 0.88 | 0.88 | 0.75 |
| PyPDF | **Overall** | **0.56** | **0.81** | **0.94** | **0.72** |
| Docling | Wipro | 0.62 | 1.00 | 1.00 | 0.79 |
| Docling | HCLTech | 0.38 | 0.50 | 0.88 | 0.53 |
| Docling | **Overall** | 0.50 | 0.75 | **0.94** | 0.66 |

Same Hit@5 (what the model sees); Docling ranks better for Wipro and worse for HCLTech.
With 8 questions per company, one question moves a score by 0.125, so these
differences are within noise.

## Answer quality (16 questions, graded against `expected_answer`)

Grades: **C** correct · **P** partly correct (incomplete) · **E** contains a factual error ·
**M** missed (said "not disclosed" although the report states it). Answers are in
`parser_answers.json`. Grading was done by the AI assistant that built the project,
checking each figure against the extracted PDF text; it has not been reviewed by a person.

| ID | PyPDF | Docling | Notes |
|---|---|---|---|
| wipro-01 Scope 3 target | C | C | Both: 55% by 2030 from a 2020 baseline |
| wipro-02 Net-zero year | C | C | 2040 |
| wipro-03 Renewable share | C | C | 84% |
| wipro-04 Women in workforce | C | **M** | Docling: p.114 was retrieved but its gender table did not yield 37.1% |
| wipro-05 Water targets | C | C | Both list all four targets |
| wipro-06 Waste recycled | C | C | 97.61% recycled, 0.73% landfill |
| wipro-07 CDP rating | C | C | A- (climate, water), A (supply chain) |
| wipro-08 Voluntary attrition | C | C | Region, age and gender breakdown |
| hcltech-01 Scope 3 target | C | C | 42% by FY30 |
| hcltech-02 Net-zero year | C | C | 2040 |
| hcltech-03 Renewable target | C | **M** | Docling: the 80%-by-2030 passage was not retrieved |
| hcltech-04 Women employees | C | C | 29.1% |
| hcltech-05 Water reduction | P | P | Both describe measures; neither gives the 40% reduction in India |
| hcltech-06 Landfill goal | C | C | Zero waste to landfill by FY25 |
| hcltech-07 CDP rating | P | **C** | PyPDF found only the "A" supplier-engagement rating, not the A- climate rating |
| hcltech-08 Scope 1+2 fall | **E** | **E** | Both give 25% correctly. PyPDF adds "173,743 mtCO2 in FY24" (FY24 is 167,426; the chart values were extracted out of order). Docling says the "2030 target" was exceeded; the report's 2030 target is 50%, and what was exceeded is the 20% interim pathway |
| **Total** | **13 C, 2 P, 1 E, 0 M** | **12 C, 1 P, 1 E, 2 M** | |

## Decision

**Keep PyPDF.** Docling produces visibly better text for charts and tables, but on this
evaluation it gives no better answers (one fewer correct, two misses), while adding a
large dependency (layout models of several hundred MB) and about 8 minutes of
conversion. The neighbouring-chunk context and prompt rules already fixed the original
p.67 error with PyPDF.

## Limitations

- 16 questions; a one-question difference is not statistically meaningful.
- One generation per question; Claude's answers vary a little between runs.
- Grades are by the AI assistant and not yet reviewed by a person.
- Docling's markdown tables change chunk boundaries (797 vs 636 Wipro chunks), so
  differences mix two effects: extraction quality and chunking.
- A natural next experiment is a hybrid: Docling text only for pages where PyPDF output
  looks like a chart or table (many short lines and bare numbers).
