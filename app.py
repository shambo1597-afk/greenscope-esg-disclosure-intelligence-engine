"""GreenScope: Streamlit user interface.

Run with:  streamlit run app.py

This file only handles the screen. The real work lives in:
    src/data.py   - reading the PDFs, chunking, embedding (cached in index/)
    src/model.py  - searching the chunks and asking Claude for a cited answer
"""

import html
import re
from typing import List

import streamlit as st

from src.model import (
    DEFAULT_TOP_K,
    Answer,
    ReportIndex,
    RetrievedChunk,
    generate_answer,
    has_api_key,
    split_question,
)

MODES = {
    "Wipro only": ["Wipro"],
    "HCLTech only": ["HCLTech"],
    "Compare both": ["Wipro", "HCLTech"],
}

EXAMPLE_QUESTIONS = [
    "What are the company's Scope 3 emissions reduction targets?",
    "What share of the company's electricity comes from renewable energy?",
    "What percentage of the workforce are women?",
    "What are the company's water and waste management goals?",
]

# Business-case assumption (see docs/BUSINESS_CASE.md): finding, checking and
# recording one metric for one company by hand takes ~10 minutes; with GreenScope,
# reading the cited answer and checking the passage takes ~2. Saving: 8 minutes.
MINUTES_SAVED_PER_LOOKUP = 8

st.set_page_config(page_title="GreenScope", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading the report index (first run builds it, about 2 minutes)...")
def load_index() -> ReportIndex:
    """Load the FAISS indexes once per server process, not on every click."""
    return ReportIndex()


def use_example(question: str) -> None:
    """Button callback: put an example into the question box and run it."""
    st.session_state.question = question
    st.session_state.run_now = True


def word_contributions_html(chunk: RetrievedChunk) -> str:
    """Colour each question word by how much it raised (green) or lowered (red) the match."""
    scored = [v for _, v in chunk.word_contributions if v is not None]
    largest = max((abs(v) for v in scored), default=0) or 1.0
    spans = []
    for word, value in chunk.word_contributions:
        word = html.escape(word)
        if value is None:
            spans.append(f'<span style="opacity:0.55">{word}</span>')
            continue
        strength = 0.15 + 0.65 * abs(value) / largest
        colour = f"rgba(34,139,34,{strength:.2f})" if value >= 0 else f"rgba(200,40,40,{strength:.2f})"
        spans.append(
            f'<span title="{value:+.3f}" style="background:{colour};padding:1px 4px;'
            f'border-radius:4px">{word} <small>{value:+.2f}</small></span>'
        )
    return " ".join(spans)


def show_sources(chunks: List[RetrievedChunk], expanded: bool = False) -> None:
    """Expandable list of the passages the answer was based on."""
    with st.expander(f"Retrieved passages ({len(chunks)})", expanded=expanded):
        st.caption(
            "Passages are ranked by combining two searches: by **meaning** (similarity of the text's "
            "meaning to your question) and by **keywords** (exact words in common). Each is shown with its "
            "neighbouring text on the same page, exactly as sent to the AI. **Why it matched:** each word of "
            "your question is coloured by how much it raised (green) or lowered (red) the meaning similarity "
            "(Shapley values); keyword matches are listed below it."
        )
        for rank, c in enumerate(chunks, start=1):
            ranks = f" · meaning rank {c.meaning_rank} · keyword rank {c.keyword_rank}" if c.keyword_rank else ""
            if c.sub_query:
                ranks += f" · found for *{c.sub_query}*"
            st.markdown(f"**#{rank} · {c.company}, page {c.page}** · similarity {c.score:.3f}{ranks}")
            if c.word_contributions:
                st.markdown(word_contributions_html(c), unsafe_allow_html=True)
            if c.keyword_matches is not None:
                matches = ", ".join(f"{w} (+{v:.1f})" for w, v in c.keyword_matches) or "none"
                st.caption(f"Keyword matches: {matches}")
            if c.context == "":
                st.caption("Already included in a higher-ranked passage above.")
            else:
                st.text(c.text if c.context is None else c.context)
            if rank < len(chunks):
                st.divider()


def record_usage(answers: List[Answer], extra_cost: float = 0.0) -> None:
    """Add this question's answers to the session's running business metrics."""
    usage = st.session_state.setdefault("usage", {"questions": 0, "lookups": 0, "cost": 0.0})
    usage["questions"] += 1
    usage["lookups"] += sum(1 for a in answers if not a.error)
    usage["cost"] += sum(a.cost_usd for a in answers) + extra_cost


def show_usage(panel) -> None:
    usage = st.session_state.get("usage", {"questions": 0, "lookups": 0, "cost": 0.0})
    with panel.container():
        st.subheader("This session")
        st.metric("Questions asked", usage["questions"])
        st.metric("Analyst time saved (est.)", f"{usage['lookups'] * MINUTES_SAVED_PER_LOOKUP} min",
                  help=f"{MINUTES_SAVED_PER_LOOKUP} minutes per company answer, compared with finding and "
                       "checking the figure in the PDF by hand. Assumption explained in docs/BUSINESS_CASE.md.")
        st.metric("AI cost", f"US$ {usage['cost']:.3f}")


def show_answer(answer: Answer) -> None:
    if answer.error:
        st.info(answer.error)
    else:
        # The model sometimes writes "•" bullets; Markdown needs "- " to show a list.
        st.markdown(re.sub(r"^\s*•\s*", "- ", answer.text, flags=re.MULTILINE))
        st.caption(f"{answer.input_tokens:,} input + {answer.output_tokens:,} output tokens · about US\$ {answer.cost_usd:.4f}")
    # With no generated answer the passages are the result, so open them.
    show_sources(answer.chunks, expanded=bool(answer.error))


# --- Header ----------------------------------------------------------------------
st.title("🌱 GreenScope")
st.markdown(
    "Ask questions about the sustainability (ESG) reports of **Wipro** and **HCLTech**. "
    "GreenScope finds the most relevant passages in each report and has an AI write an answer "
    "**using only those passages**, citing the page for every claim, e.g. *[Wipro, p.70]*. "
    "Open *Retrieved passages* under any answer to check the evidence yourself."
)

# --- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Which report(s)?", list(MODES), index=2)
    top_k = st.slider(
        "Passages to retrieve per company", min_value=3, max_value=8, value=DEFAULT_TOP_K,
        help="More passages = more context for the answer, but also more noise and cost.",
    )
    if not has_api_key():
        st.warning("No ANTHROPIC_API_KEY found: search works, but AI answers are disabled.")
    st.divider()
    usage_panel = st.empty()  # filled at the end of the script, after this run's answers

# --- Question input --------------------------------------------------------------
st.session_state.setdefault("question", "")
st.markdown("**Try an example:**")
example_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, q in zip(example_cols, EXAMPLE_QUESTIONS):
    col.button(q, on_click=use_example, args=(q,), use_container_width=True)

question = st.text_input("Your question", key="question", placeholder="e.g. What is the net-zero target year?")
ask = st.button("Ask", type="primary")
run_now = st.session_state.pop("run_now", False)

# --- Answer ----------------------------------------------------------------------
if (ask or run_now) and question.strip():
    index = load_index()
    companies = MODES[mode]
    # Questions about two topics ("water and waste goals") are searched once per
    # topic, so both are represented in the passages (see split_question).
    split = split_question(question)
    if len(split.queries) > 1:
        st.caption("Searched separately for: " + " · ".join(f"*{q}*" for q in split.queries))

    if len(companies) == 1:
        with st.spinner(f"Searching the {companies[0]} report and writing an answer..."):
            chunks = index.retrieve_multi(question, companies[0], top_k, split.queries)
            index.explain(question, chunks, companies[0])
            answer = generate_answer(question, chunks)
        record_usage([answer], split.cost_usd)
        st.subheader(companies[0])
        show_answer(answer)
    else:
        with st.spinner("Searching both reports and writing two answers..."):
            per_company = index.retrieve_comparison(question, top_k, split.queries)
            for company, chunks in per_company.items():
                index.explain(question, chunks, company)
            answers = {c: generate_answer(question, chunks) for c, chunks in per_company.items()}
        record_usage(list(answers.values()), split.cost_usd)
        columns = st.columns(len(answers))
        for col, (company, answer) in zip(columns, answers.items()):
            with col:
                st.subheader(company)
                show_answer(answer)
elif ask:
    st.warning("Please type a question first.")

# --- Footer ----------------------------------------------------------------------
st.divider()
st.caption(
    "Note: the two reports cover different fiscal years: **Wipro FY2024-25** and **HCLTech FY2024** "
    "(April 2023 to March 2024). Keep this in mind when comparing figures. Page numbers refer to the "
    "page of the PDF file, which may differ from the number printed on the page. "
    "AI answers can still contain mistakes, so verify important figures in the cited passages."
)

show_usage(usage_panel)
