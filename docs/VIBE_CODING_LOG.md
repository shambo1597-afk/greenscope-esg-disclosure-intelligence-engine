# Vibe Coding Log: GreenScope

*Team: [add names] · Course: AI and ML for Digital Business Managers · October 2026*

## Platforms used

| Tool | Used for |
|---|---|
| **Claude Code** (cloud sessions, via the Claude app and claude.ai/code) | Writing and running all code, evaluation scripts and docs; git commits and pushes |
| **GitHub** | Repository, version history, one pull request, PDF uploads |
| **Anthropic API** (Claude Haiku 4.5) | Answer generation inside the app; also used to test answer quality |
| **Streamlit** | Web interface (tested headlessly with Streamlit's `AppTest`) |

## Prompt engineering strategies that worked

1. **Spec-first prompts with hard constraints.** The first prompt asked for *only*
   `src/data.py`, fixed the parameters (chunk_size=800, overlap=150, MiniLM), required
   specific metadata, and ended with "stop and let me review". Small, reviewable steps
   kept us in control.
2. **Asking for the "why", not just the code.** We required comments explaining *why*
   800/150 were chosen, so we could defend them. That led to *testing* them: a
   cross-validated grid search replaced them with 400/250 and a better embedding model.
3. **A verification checklist inside the prompt.** The full-build prompt listed checks
   (imports, eval run, a retrieval sanity check, a real API call "if the key is available;
   if not, say so; don't fake it", no secrets staged). The agent reported each one,
   including what it could *not* verify.
4. **Explicit budget and scope limits.** "I have US$5 of credits": the agent then
   tracked the cost of every test call. Total API spend for development: about US$0.20.
5. **Challenging the agent's recommendations.** When the agent dismissed a better PDF
   parser as "a bigger change", we asked *"why not?"*. It tested the idea instead of
   assuming, and that turned into two documented experiments (PyMuPDF, Docling).

## Agentic workflow

```
 prompt with spec + checks ─► agent writes code ─► agent runs it on the real PDFs
          ▲                                               │
          │                                   evaluation / live test finds a problem
          │                                               │
 we review results, decide ◄── agent reports evidence, proposes fixes with costs
```

- **Evaluation-driven development.** Every change was measured on a gold-page evaluation
  set (16, later 40 questions) before being kept. Two "obvious" improvements were
  *rejected* on evidence (PyMuPDF and Docling parsers), and one non-obvious fix was found
  from the data: removing company names from the search query (Hit@5 0.56 to 0.94).
  Tuning was cross-validated so the reported gain (held-out MRR@5 0.67 to 0.82) is not
  an artefact of choosing the best of many settings on the same questions.
- **Human-in-the-loop on decisions, agent on execution.** We chose the domain, the
  documents, the model budget and which experiments to run; the agent wrote, ran and
  reported.
- **Small logical commits** (data, model, app, eval, docs), each with a message explaining
  the reason, so the history doubles as a design record.

## Where the AI went wrong, and how we caught it

| Problem | How it was caught |
|---|---|
| Generated answer said Wipro's Scope 3 target was 59% (it is 55%) | Live test of the app; traced to scrambled chart text on page 67; fixed with neighbouring-chunk context and two prompt rules |
| Answer claimed HCLTech had "already exceeded" its 42% target | Reading answers against the PDF; fixed with a "no unstated conclusions" rule |
| A code parameter (`temperature`) rejected by the current SDK | Running a deliberate error-path test before relying on the code |
| Eval gold list missed a page holding the answer | Inspecting every retrieval "miss" instead of trusting the score |
| One remaining error (HCLTech FY24 tonnage) | Graded answer test; documented as a known limitation, not hidden |

## What we own (the "golden rule")

The agent produced the scaffolding, but every design decision is documented with its
evidence and can be explained by the team: chunking and overlap (`docs/EXPLAINER.md`
and `eval/tuning_results.md`), embeddings and cosine similarity
(`eval/distance_metrics.md`), retrieval quality (`eval/results.md`), explainability
(Shapley values in `src/model.py`), parser choice (`eval/pdf_parser_comparison.md`) and
business value (`docs/BUSINESS_CASE.md`).

## Lessons learned

- Agents are fast at code and slow to doubt themselves: **measure, don't trust.**
- A small, honest evaluation set was the most valuable artefact we built.
- Asking "why not?" and "can you prove it?" improved the product more than any single feature prompt.
