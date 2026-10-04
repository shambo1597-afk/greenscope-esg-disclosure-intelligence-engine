# Answer check after tuning

After `eval/tune.py` selected **BGE-small / 400 / 250** (with "small-to-big" context of
about 2,000 characters per match), the 16 original graded questions were answered again
and graded the same way as in `pdf_parser_comparison.md`: **C** correct, **P** partly
correct, **E** contains an error, **M** missed. Answers: `tuned_answers.json`. Cost: US$0.04.

| ID | Original (MiniLM 800/150) | Tuned (BGE 400/250) | Notes on the tuned answer |
|---|---|---|---|
| wipro-01 Scope 3 target | C | C | 55% by 2030 from 2020 baseline |
| wipro-02 Net-zero year | C | C | 2040, Scope 1, 2 and 3 |
| wipro-03 Renewable share | C | C | 84%, target 100% by 2030 |
| wipro-04 Women in workforce | C | C | 37.1% |
| wipro-05 Water targets | C | C | All four revised targets plus FY25 performance (more complete) |
| wipro-06 Waste recycled | C | C | 97.61% recycled, 0.73% landfill; also added up a total itself (rule slip) |
| wipro-07 CDP rating | C | **E** | Said "A" for Climate Change and Water; the passage it received says "A-". Re-asked 3 times: correct all 3 times, so an occasional slip (1 in 4 runs) |
| wipro-08 Voluntary attrition | C | C | Region, age, gender |
| hcltech-01 Scope 3 target | C | C | 42% by FY30 |
| hcltech-02 Net-zero year | C | C | 2040 |
| hcltech-03 Renewable target | C | C | 80% by 2030 |
| hcltech-04 Women employees | C | C | 29.1% (29.13% permanent) |
| hcltech-05 Water reduction | P | P | Describes measures, not the 40% India reduction |
| hcltech-06 Landfill goal | C | C | Zero waste to landfill by FY25 |
| hcltech-07 CDP rating | P | **C** | Now gives both the A- climate rating and the A supplier-engagement rating |
| hcltech-08 Scope 1+2 fall | E | E | 25% correct; FY24 tonnage 173,743 wrong (167,426), from chart values extracted out of order |
| **Total** | **13 C, 2 P, 1 E** | **13 C, 1 P, 2 E** | |

**Conclusion.** Answer quality is about the same on these 16 questions, while retrieval is
clearly better (held-out MRR@5 0.82 vs 0.67). The tuned configuration is kept.

**What this check taught us**
- **Answers vary between runs.** The same question with the same passages gave a wrong
  rating once and the right one three times. A single run per question measures quality
  with noise; repeated runs (or a larger set) would give firmer numbers.
- In response, prompt rule 3 now names ratings ("A-", "BBB+") and forbids adding up
  figures. That change was not separately measured.
