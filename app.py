"""GreenScope: Streamlit user interface.

Run with:  streamlit run app.py

This file only handles the screen. The real work lives in:
    src/data.py   - reading the PDFs, chunking, embedding (cached in index/)
    src/model.py  - searching the chunks and asking Claude for a cited answer
"""

from typing import List

import streamlit as st

from src.model import (
    DEFAULT_TOP_K,
    Answer,
    ReportIndex,
    RetrievedChunk,
    generate_answer,
    has_api_key,
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

st.set_page_config(page_title="GreenScope", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading the report index (first run builds it, about a minute)...")
def load_index() -> ReportIndex:
    """Load the FAISS indexes once per server process, not on every click."""
    return ReportIndex()


def use_example(question: str) -> None:
    """Button callback: put an example into the question box and run it."""
    st.session_state.question = question
    st.session_state.run_now = True


def show_sources(chunks: List[RetrievedChunk], expanded: bool = False) -> None:
    """Expandable list of the passages the answer was based on."""
    with st.expander(f"Retrieved passages ({len(chunks)})", expanded=expanded):
        st.caption("Each match is shown with its neighbouring text on the same page, exactly as sent to the AI.")
        for rank, c in enumerate(chunks, start=1):
            st.markdown(f"**#{rank} · {c.company}, page {c.page}** · similarity {c.score:.3f}")
            if c.context == "":
                st.caption("Already included in a higher-ranked passage above.")
            else:
                st.text(c.text if c.context is None else c.context)
            if rank < len(chunks):
                st.divider()


def show_answer(answer: Answer) -> None:
    if answer.error:
        st.info(answer.error)
    else:
        st.markdown(answer.text)
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

    if len(companies) == 1:
        with st.spinner(f"Searching the {companies[0]} report and writing an answer..."):
            chunks = index.retrieve(question, companies[0], top_k)
            answer = generate_answer(question, chunks)
        st.subheader(companies[0])
        show_answer(answer)
    else:
        with st.spinner("Searching both reports and writing two answers..."):
            per_company = index.retrieve_comparison(question, top_k)
            answers = {c: generate_answer(question, chunks) for c, chunks in per_company.items()}
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
