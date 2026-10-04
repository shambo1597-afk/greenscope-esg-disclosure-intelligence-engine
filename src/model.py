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


@dataclass
class Answer:
    """What generate_answer returns. `error` is set instead of crashing."""
    text: str
    error: Optional[str] = None
    chunks: List[RetrievedChunk] = field(default_factory=list)


# --- Retrieval -----------------------------------------------------------------
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

    def retrieve(self, question: str, company: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
        """Return the top_k chunks from one company's report, most similar first."""
        if company not in self.indexes:
            raise ValueError(f"Unknown company '{company}'. Choose from {list(COMPANIES)}.")

        query_vector = embed_texts([question])
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
        return results

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


def has_api_key() -> bool:
    """True if an Anthropic API key is configured (env var or .env file)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Label each excerpt with its citation so the model can copy it exactly."""
    blocks = [f"[{c.company}, p.{c.page}]\n{c.text}" for c in chunks]
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
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=0,  # same question -> same answer; no creative paraphrasing
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
