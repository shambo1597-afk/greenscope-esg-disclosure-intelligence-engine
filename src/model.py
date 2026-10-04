"""Retrieval + generation: find the relevant report passages, then answer from them.

Two steps, the classic RAG (Retrieval-Augmented Generation) pattern:

1. RETRIEVE  - turn the question into a vector and ask FAISS for the chunks whose
               vectors are most similar (cosine similarity). One index per
               company, so we always know which report a passage came from.
2. GENERATE  - send ONLY those chunks to Claude with strict rules: answer from
               the given text, cite every claim as [Company, p.X], and say so
               when the answer is not there.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv

from src.data import COMPANIES, build_index, embed_texts

# Read ANTHROPIC_API_KEY from a local .env file if present (it is gitignored).
# The key is only ever passed to the Anthropic client; it is never printed or logged.
load_dotenv()

LLM_MODEL = "claude-haiku-4-5-20251001"  # cheapest current Claude model
MAX_ANSWER_TOKENS = 600                   # enough for a cited paragraph or short list
DEFAULT_TOP_K = 5


@dataclass
class RetrievedChunk:
    """One search result: the passage, where it came from, and how similar it is."""
    text: str
    company: str
    page: int
    chunk_index: int
    source: str
    score: float  # cosine similarity: 1.0 = same meaning, ~0 = unrelated
    # The text actually sent to Claude: this chunk plus its neighbours on the same
    # page (see add_neighbours). "" = already sent as part of a better result;
    # None = not widened, so the chunk text itself is sent.
    context: Optional[str] = None


@dataclass
class Answer:
    """What generate_answer returns. `error` is set instead of crashing."""
    text: str
    error: Optional[str] = None
    chunks: List[RetrievedChunk] = field(default_factory=list)


# --- Retrieval -----------------------------------------------------------------
# Company names as people might type them ("Wipro's", "HCL Tech", "HCLTech Ltd").
_COMPANY_NAME = r"\b(?:Wipro|HCL ?Tech(?:nologies)?|HCL)(?: Limited| Ltd\.?)?"


def prepare_query(question: str) -> str:
    """Replace company names in the question with "the company" before searching.

    Each search already runs inside ONE company's report, so the name tells the
    search nothing new. Worse, it makes the question look similar to chunks that
    are mostly the company name: page headers and footers such as
    "About HCLTech / HCLTech Sustainability Report 2024 / 92". Those boilerplate
    chunks then crowd out the passages that hold the actual answer. In our eval
    this one change raised Hit@5 from 0.56 to 0.94 (see eval/results.md).
    Only the search query is changed; Claude still sees the original question.
    """
    query = re.sub(_COMPANY_NAME + r"(?:'s|’s)", "the company's", question, flags=re.IGNORECASE)
    return re.sub(_COMPANY_NAME + r"\b", "the company", query, flags=re.IGNORECASE)


def _join_overlapping(first: str, second: str) -> str:
    """Join two consecutive chunks, dropping the text they share.

    Consecutive chunks overlap by up to CHUNK_OVERLAP (150) characters: the end
    of one is repeated at the start of the next. Keep that shared part once.
    """
    for size in range(min(len(first), len(second), 300), 0, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n" + second


class ReportIndex:
    """Holds one FAISS index + chunk list per company.

    IndexFlatIP = exact ("flat", no approximation) search by inner product (IP).
    Because every vector was normalized to length 1 in data.py, the inner
    product of two vectors IS their cosine similarity. With ~600 chunks per
    report an exact search takes well under a millisecond, so no approximate
    index is needed.
    """

    def __init__(self, force_rebuild: bool = False):
        data = build_index(force=force_rebuild)
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}
        self.chunks: Dict[str, List[dict]] = {}
        for company, entry in data.items():
            vectors = np.ascontiguousarray(entry["vectors"], dtype=np.float32)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self.indexes[company] = index
            self.chunks[company] = entry["chunks"]

    def retrieve(
        self,
        question: str,
        company: str,
        top_k: int = DEFAULT_TOP_K,
        neutralize_names: bool = True,
        neighbours: bool = True,
    ) -> List[RetrievedChunk]:
        """Return the top_k chunks from one company's report, most similar first.

        neutralize_names=False searches with the question exactly as typed
        (only used by the evaluation to measure the effect of prepare_query).
        neighbours=False sends only the matching chunks themselves to Claude.
        """
        if company not in self.indexes:
            raise ValueError(f"Unknown company '{company}'. Choose from {list(COMPANIES)}.")

        query = prepare_query(question) if neutralize_names else question
        query_vector = embed_texts([query])
        scores, positions = self.indexes[company].search(query_vector, top_k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:  # FAISS pads with -1 when there are fewer than top_k chunks
                continue
            record = self.chunks[company][pos]
            meta = record["metadata"]
            results.append(RetrievedChunk(
                text=record["text"],
                company=meta["company"],
                page=meta["page"],
                chunk_index=meta["chunk_index"],
                source=meta["source"],
                score=float(score),
            ))
        if neighbours:
            self.add_neighbours(results, company)
        return results

    def add_neighbours(self, results: List[RetrievedChunk], company: str) -> None:
        """Widen each result with the chunk before and after it on the same page.

        Why: the search matches small chunks, but the sentence that explains a
        number is often in the next chunk. Example (Wipro p.67): one chunk holds
        the chart text "84% 2025 Performance / 59% 2030 Target", while the
        sentence "55% reduction in Scope 3 from 2020 baseline" sits in the chunk
        before it. Seeing only the chart fragment, the model attached 59% (the
        Scope 1 and 2 target) to Scope 3. Sending the neighbours too gives it the
        surrounding explanation. Neighbours are taken from the same page only,
        so every citation [Company, p.X] stays correct.

        Results are processed best-first; chunks already sent with a better
        result are not repeated (that result's context becomes "").
        """
        chunks = self.chunks[company]
        already_sent = set()
        for result in results:
            window = [
                i for i in (result.chunk_index - 1, result.chunk_index, result.chunk_index + 1)
                if 0 <= i < len(chunks)
                and chunks[i]["metadata"]["page"] == result.page
                and i not in already_sent
            ]
            already_sent.update(window)
            context = ""
            for i in window:
                context = _join_overlapping(context, chunks[i]["text"]) if context else chunks[i]["text"]
            result.context = context

    def retrieve_comparison(self, question: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, List[RetrievedChunk]]:
        """Search each report separately so both companies are always represented.

        A single combined search could return 5 Wipro chunks and 0 HCLTech
        chunks if Wipro happens to phrase things closer to the question, which
        would make a fair comparison impossible.
        """
        return {company: self.retrieve(question, company, top_k) for company in self.indexes}


# --- Generation ----------------------------------------------------------------
SYSTEM_PROMPT = """You are GreenScope, an assistant that answers questions about corporate \
sustainability (ESG) reports.

Rules:
1. Answer ONLY from the report excerpts provided in the user message. Never use outside \
knowledge, even if you believe you know the answer.
2. Cite every factual claim with its source in the form [Company, p.X], using the company \
and page shown in the excerpt header.
3. Keep numbers, units, percentages and years exactly as written in the excerpts. Do not \
convert, round or calculate new figures.
4. If the excerpts do not contain the answer, say: "This is not disclosed in the retrieved \
text." You may mention closely related information that IS in the excerpts, with citations.
5. Be concise: a short paragraph or a few bullet points."""


def get_api_key() -> str:
    """The configured key. Cloud environments don't pass ANTHROPIC_API_KEY through
    (Claude Code reserves it), so GREENSCOPE_ANTHROPIC_KEY is accepted as a fallback."""
    for name in ("ANTHROPIC_API_KEY", "GREENSCOPE_ANTHROPIC_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def has_api_key() -> bool:
    """True if an Anthropic API key is configured (env var or .env file)."""
    return bool(get_api_key())


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Label each excerpt with its citation so the model can copy it exactly.

    Uses each result's widened `context`; results whose text was already sent
    as part of a better result's context (context == "") are skipped.
    """
    blocks = [
        f"[{c.company}, p.{c.page}]\n{c.text if c.context is None else c.context}"
        for c in chunks
        if c.context != ""
    ]
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> Answer:
    """Ask Claude to answer `question` using only `chunks`.

    Never raises: problems (no key, network error, bad key) come back as
    Answer.error with a friendly message, so the app keeps working.
    """
    if not chunks:
        return Answer(text="", error="No relevant passages were found in the report.", chunks=chunks)
    if not has_api_key():
        return Answer(
            text="",
            error=(
                "No Anthropic API key found, so an AI-written answer can't be generated. "
                "The most relevant report passages are still shown below. "
                "To enable answers, add ANTHROPIC_API_KEY=... to a .env file and restart the app."
            ),
            chunks=chunks,
        )

    import anthropic  # imported here so retrieval works even without the package configured

    user_message = (
        f"Report excerpts:\n\n{format_context(chunks)}\n\n"
        f"Question: {question}"
    )
    try:
        client = anthropic.Anthropic(api_key=get_api_key())
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_ANSWER_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        return Answer(text="", error="The Anthropic API key was rejected. Please check it in your .env file.", chunks=chunks)
    except anthropic.RateLimitError:
        return Answer(text="", error="The Anthropic API rate limit was reached. Please wait a moment and try again.", chunks=chunks)
    except anthropic.APIConnectionError:
        return Answer(text="", error="Could not reach the Anthropic API. Check your internet connection.", chunks=chunks)
    except anthropic.APIStatusError as e:
        return Answer(text="", error=f"The Anthropic API returned an error (status {e.status_code}).", chunks=chunks)

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n_(Answer cut short at the length limit.)_"
    return Answer(text=text, chunks=chunks)


if __name__ == "__main__":
    # Quick smoke test: retrieval for one question per company (no API cost),
    # plus a generated answer if a key is configured.
    idx = ReportIndex()
    q = "What are the company's Scope 3 emissions reduction targets?"
    for company in COMPANIES:
        top = idx.retrieve(q, company, top_k=3)
        print(f"\n== {company} ==")
        for c in top:
            print(f"  {c.score:.3f}  p.{c.page:<4} {c.text[:110]!r}")
        if has_api_key():
            ans = generate_answer(q, top)
            print("\n" + (ans.error or ans.text))
    if not has_api_key():
        print("\n(No ANTHROPIC_API_KEY set: generation skipped.)")
