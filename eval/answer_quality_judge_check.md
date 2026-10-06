# Checking the AI judge by hand

## Run 2: matched-year reports (current)

After HCLTech's FY2024 report was replaced by its FY2025 report, the evaluation was run
again. 6 new verdicts were checked against the PDF text (2 "contains an error", plus 4
picked at random from HCLTech's "correct" verdicts, random seed 11):

| Answer | Judge | Hand check | Evidence |
|---|---|---|---|
| hcltech-04 run 1 | contains error | agree | The answer says 28.75%; p.49 gives 28.66% female (p.11: "28.8% gender diversity"). 28.75% appears nowhere |
| hcltech-08 run 3 | contains error | agree | Claims Scope 1 alone fell 46%; p.16 gives 22,121 to 12,901 tCO2e (about 42%); 46% is Scope 1 and 2 combined |
| hcltech-13 run 2 | correct | agree | p.11: 34% of electricity from renewables in FY25; p.31: up from 19% in FY24; 80% target by 2030 |
| hcltech-13 run 3 | correct | agree | Same figures, same pages |
| hcltech-15 run 3 | correct | agree | p.8: Gold rating from EcoVadis; p.88: top 5% highest rated |
| hcltech-20 run 2 | correct | agree | p.90: recognised as one of Ethisphere's 2025 World's Most Ethical Companies |

**Agreement: 6 of 6** (20 of 20 across both runs).

## Run 1: earlier version (Wipro FY2024-25 with HCLTech FY2024)

The answer-quality evaluation was graded by Claude Sonnet 5.5. A judge
model can be wrong, so 14 of its 120 verdicts were checked by hand against the extracted
PDF text: 8 picked at random from its "correct" verdicts (random seed 7) and 6 of its
"contains an error" / "missed" verdicts.

| Answer | Judge | Hand check | Evidence |
|---|---|---|---|
| wipro-17 run 1 | correct | agree | p.15: "Two of the six independent directors are women" |
| wipro-07 run 3 | correct | agree | p.11: "A- in CDP Climate Change and Water Disclosure, and A in CDP Supply Chain" |
| wipro-20 run 2 | correct | agree | p.7 and p.119: NPS "increased by 640 bps" |
| hcltech-15 run 3 | correct | agree | p.97: "Gold rating from EcoVadis" |
| wipro-03 run 1 | correct | agree | p.65: 84% of purchased electricity renewable, 100% target by 2030 |
| wipro-04 run 1 | correct | agree | p.7: 37.1% women employees; p.4: one in five senior leaders |
| hcltech-08 run 1 | correct | agree | p.56: 25% fall; "emission intensity have reduced by 52%" |
| wipro-05 run 1 | correct | agree | p.74: the 3% freshwater, 45% treated water and ZLD targets |
| wipro-09 (all runs) | contains error | agree | p.67: the 84% / 195,453 to 31,462 tCO2e figure is FY25 performance, not the 59% target |
| hcltech-09 (all runs) | contains error | agree | 2030 target is 50% (p.5, p.54); the 20% on p.56 is the interim pathway |
| hcltech-16 run 2 | contains error | agree (strict) | p.31 says "over 15740 hours"; the answer cited it for "15,700+" |
| wipro-06 run 2 | contains error | agree | p.79: 97.61% is waste diverted from disposal, not "directed to disposal" |
| wipro-12 (all runs) | missed | agree | 13.5% supplier diversity spend is on p.7 (infographic page), not retrieved |
| hcltech-10 (all runs) | missed | agree | 30% senior-leadership target is on p.5, not retrieved |

**Agreement: 14 of 14.** The judge is strict about exact wording and figures (hcltech-16 is
a borderline call), which makes its error rate conservative rather than flattering.
Limitations: 14 is a small sample, and the checker is the same assistant that built the
project; a team member should repeat a few checks independently.
