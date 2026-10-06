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
from collections import Counter, defaultdict
from math import log
from dataclasses import dataclass, field
from math import factorial
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv

from src.data import CHUNK_OVERLAP, COMPANIES, build_index, embed_queries

# Read ANTHROPIC_API_KEY from a local .env file if present (it is gitignored).
# The key is only ever passed to the Anthropic client; it is never printed or logged.
load_dotenv()

LLM_MODEL = "claude-haiku-4-5-20251001"  # cheapest current Claude model
MAX_ANSWER_TOKENS = 600                   # enough for a cited paragraph or short list
# Claude Haiku 4.5 list prices in US dollars per million tokens, used only to show
# an approximate cost per answer in the app.
PRICE_PER_MTOK_INPUT = 1.00
PRICE_PER_MTOK_OUTPUT = 5.00
DEFAULT_TOP_K = 5
# Reciprocal Rank Fusion constant (the standard value from the original RRF paper).
RRF_K = 60
# Each match is sent to the model with surrounding text from the same page, up to
# about this many characters (see ReportIndex.add_neighbours).
CONTEXT_CHARS = 2000


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
    # Filled by ReportIndex.explain(): (word, contribution) for each word of the
    # question; contribution is None for words that were not scored ("the", "of").
    word_contributions: Optional[List[Tuple[str, Optional[float]]]] = None
    explanation_method: str = ""
    # Hybrid search: this passage's rank by meaning (vectors) and by keywords
    # (BM25), and the question words it matched with their BM25 contribution.
    meaning_rank: int = 0
    keyword_rank: int = 0
    # For two-topic questions: the sub-question this passage was found for.
    sub_query: str = ""
    keyword_matches: Optional[List[Tuple[str, float]]] = None


@dataclass
class Answer:
    """What generate_answer returns. `error` is set instead of crashing."""
    text: str
    error: Optional[str] = None
    chunks: List[RetrievedChunk] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """Approximate cost of this answer at Haiku 4.5 list prices."""
        return (self.input_tokens * PRICE_PER_MTOK_INPUT
                + self.output_tokens * PRICE_PER_MTOK_OUTPUT) / 1_000_000


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
    this one change raised Hit@5 from 0.56 to 0.94 with the original settings,
    and from 0.70 to 0.93 with the tuned ones (see eval/results.md).
    Only the search query is changed; Claude still sees the original question.
    """
    query = re.sub(_COMPANY_NAME + r"(?:'s|’s)", "the company's", question, flags=re.IGNORECASE)
    return re.sub(_COMPANY_NAME + r"\b", "the company", query, flags=re.IGNORECASE)


def _join_overlapping(first: str, second: str) -> str:
    """Join two consecutive chunks, dropping the text they share.

    Consecutive chunks overlap by up to CHUNK_OVERLAP characters: the end
    of one is repeated at the start of the next. Keep that shared part once.
    """
    for size in range(min(len(first), len(second), CHUNK_OVERLAP + 50), 0, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n" + second


# --- Explanation ---------------------------------------------------------------
MAX_SHAPLEY_WORDS = 8

# Words that carry little meaning on their own. They stay in every version of the
# question when computing word contributions, so only content words are scored.
# "company"/"company's" are here because prepare_query() inserted them.
_FILLER_WORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "by", "with", "at", "from",
    "is", "are", "was", "were", "be", "been", "does", "do", "did", "has", "have", "had",
    "what", "which", "who", "how", "when", "where", "why", "much", "many",
    "its", "it", "this", "that", "these", "those", "their", "there",
    "company", "company's", "companys",
}


def _normalise_word(word: str) -> str:
    return word.lower().strip("?.,!:;\"'()").replace("’", "'")


# --- Keyword search (BM25) --------------------------------------------------------
def _tokens(text: str) -> List[str]:
    """Lower-case words and numbers ("14001", "1,23,456", "37.1")."""
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


class BM25:
    """Okapi BM25, the classic keyword-search score.

    For each word of the question that appears in a passage:
        idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average_length))
    - idf: rare words count more ("nationalities" beats "company");
    - tf: how often the word appears, with diminishing returns (k1 = 1.5);
    - b = 0.75: long passages are penalised a little, so they don't win just by
      containing more words.
    """

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [Counter(_tokens(t)) for t in texts]
        self.n_docs = len(docs)
        self.lengths = np.array([sum(d.values()) for d in docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.n_docs else 0.0
        self.postings = defaultdict(list)  # word -> [(passage position, count), ...]
        for i, doc in enumerate(docs):
            for word, count in doc.items():
                self.postings[word].append((i, count))
        self.idf = {
            w: log((self.n_docs - len(p) + 0.5) / (len(p) + 0.5) + 1) for w, p in self.postings.items()
        }

    @staticmethod
    def query_words(query: str) -> List[str]:
        """The words of a query that are scored (filler words ignored), in order."""
        seen, words = set(), []
        for w in _tokens(query):
            if w not in _FILLER_WORDS and w != "s" and w not in seen:
                seen.add(w)
                words.append(w)
        return words

    def _term(self, word: str, i: int, count: int) -> float:
        norm = 1 - self.b + self.b * self.lengths[i] / self.avg_length
        return self.idf[word] * count * (self.k1 + 1) / (count + self.k1 * norm)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every passage for the query."""
        result = np.zeros(self.n_docs, dtype=np.float32)
        for word in self.query_words(query):
            for i, count in self.postings.get(word, []):
                result[i] += self._term(word, i, count)
        return result

    def word_scores(self, query: str, i: int) -> List[Tuple[str, float]]:
        """Each query word's contribution to passage i's score (they add up to it)."""
        out = []
        for word in self.query_words(query):
            count = next((c for j, c in self.postings.get(word, []) if j == i), 0)
            if count:
                out.append((word, float(self._term(word, i, count))))
        return out


class ReportIndex:
    """Holds one FAISS index + chunk list per company.

    IndexFlatIP = exact ("flat", no approximation) search by inner product (IP).
    Because every vector was normalized to length 1 in data.py, the inner
    product of two vectors IS their cosine similarity. With about 2,000 chunks
    per report an exact search takes about 0.03 seconds, so no approximate
    index is needed.
    """

    def __init__(self, force_rebuild: bool = False):
        data = build_index(force=force_rebuild)
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}
        self.chunks: Dict[str, List[dict]] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.bm25: Dict[str, BM25] = {}
        for company, entry in data.items():
            vectors = np.ascontiguousarray(entry["vectors"], dtype=np.float32)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self.indexes[company] = index
            self.chunks[company] = entry["chunks"]
            self.vectors[company] = vectors
            self.bm25[company] = BM25([record["text"] for record in entry["chunks"]])

    def retrieve(
        self,
        question: str,
        company: str,
        top_k: int = DEFAULT_TOP_K,
        neutralize_names: bool = True,
        neighbours: bool = True,
        hybrid: bool = True,
    ) -> List[RetrievedChunk]:
        """Return the top_k chunks from one company's report, best first.

        Hybrid search (default): passages are ranked twice, by meaning (cosine
        similarity of BGE vectors, via FAISS) and by keywords (BM25), and the two
        rankings are combined with Reciprocal Rank Fusion:
            score = 1 / (RRF_K + meaning rank) + 1 / (RRF_K + keyword rank)
        A passage near the top of either list rises; one near the top of both wins.
        Meaning search finds paraphrases ("green power" ~ "renewable electricity");
        keyword search catches exact terms the vectors blur ("CDP", "14001",
        "nationalities"). RRF needs no weight to tune. On the 40-question set it
        raised MRR@5 from 0.82 to 0.86 (eval/retrieval_methods.md).

        neutralize_names=False searches with the question exactly as typed, and
        hybrid=False uses meaning search only (both used by the evaluations).
        neighbours=False sends only the matching chunks themselves to Claude.
        """
        if company not in self.indexes:
            raise ValueError(f"Unknown company '{company}'. Choose from {list(COMPANIES)}.")

        query = prepare_query(question) if neutralize_names else question
        query_vector = embed_queries([query])
        n_chunks = self.indexes[company].ntotal
        cosine, order = self.indexes[company].search(query_vector, n_chunks if hybrid else top_k)
        cosine, order = cosine[0], order[0]
        similarity = {int(pos): float(s) for s, pos in zip(cosine, order) if pos >= 0}

        meaning_rank = {int(pos): rank for rank, pos in enumerate(order) if pos >= 0}
        keyword_rank = {}
        if hybrid:
            keyword_order = np.argsort(-self.bm25[company].scores(query))
            keyword_rank = {int(pos): rank for rank, pos in enumerate(keyword_order)}
            fused = {pos: 1 / (RRF_K + meaning_rank[pos]) + 1 / (RRF_K + keyword_rank[pos]) for pos in meaning_rank}
            top = sorted(fused, key=lambda pos: -fused[pos])[:top_k]
        else:
            top = [int(pos) for pos in order[:top_k] if pos >= 0]

        results = []
        for pos in top:
            record = self.chunks[company][pos]
            meta = record["metadata"]
            results.append(RetrievedChunk(
                text=record["text"],
                company=meta["company"],
                page=meta["page"],
                chunk_index=meta["chunk_index"],
                source=meta["source"],
                score=similarity[pos],
                meaning_rank=meaning_rank[pos] + 1,
                keyword_rank=keyword_rank.get(pos, -1) + 1,
            ))
        if neighbours:
            self.add_neighbours(results, company)
        return results

    def add_neighbours(self, results: List[RetrievedChunk], company: str) -> None:
        """Widen each result with the text around it on the same page ("small-to-big").

        Search works best with SMALL chunks (precise matches, see
        eval/tuning_results.md), but the model answers best with MORE context:
        the sentence that explains a number is often next to it. Example (Wipro
        p.67): one chunk held the chart text "84% 2025 Performance / 59% 2030
        Target", while "55% reduction in Scope 3 from 2020 baseline" sat in the
        chunk before it; seeing only the fragment, the model reported 59% (the
        Scope 1 and 2 target) as the Scope 3 target.

        So each match is grown with neighbouring chunks, alternately after and
        before it, until it holds about CONTEXT_CHARS characters. Growth stops at
        the page boundary, so every citation [Company, p.X] stays correct, and at
        text already sent with a better-ranked match, so nothing is repeated
        (a match that is fully covered already gets context "").
        """
        chunks = self.chunks[company]

        def usable(i: int, page: int) -> bool:
            return 0 <= i < len(chunks) and chunks[i]["metadata"]["page"] == page and i not in already_sent

        already_sent = set()
        for result in results:
            if result.chunk_index in already_sent:
                result.context = ""
                continue
            first = last = result.chunk_index
            context = chunks[first]["text"]
            while len(context) < CONTEXT_CHARS:
                grew = False
                if usable(last + 1, result.page):
                    last += 1
                    context = _join_overlapping(context, chunks[last]["text"])
                    grew = True
                if len(context) < CONTEXT_CHARS and usable(first - 1, result.page):
                    first -= 1
                    context = _join_overlapping(chunks[first]["text"], context)
                    grew = True
                if not grew:
                    break
            already_sent.update(range(first, last + 1))
            result.context = context

    def explain(self, question: str, results: List[RetrievedChunk], company: str) -> None:
        """Explain each match: how much did each word of the question contribute?

        Method: exact Shapley values (the idea behind SHAP), applied to retrieval.
        The "players" are the content words of the question; filler words such as
        "what", "the", "of" stay in every version of the question. For every subset
        of content words we embed the shortened question and measure its cosine
        similarity with the passage. A word's Shapley value is its average extra
        similarity over all the orders in which words could be added.

        Useful property: for each passage the values add up exactly to
        (similarity of the full question) - (similarity with no content words).

        With n content words this needs 2**n small embeddings (n <= 8: at most
        256, well under a second). Longer questions use leave-one-out instead:
        each word's value = full similarity - similarity without that word.
        Results are stored on each RetrievedChunk (word_contributions).
        """
        if not results:
            return
        query = prepare_query(question)
        for result in results:
            result.keyword_matches = self.bm25[company].word_scores(query, result.chunk_index)
        words = query.split()
        players = [i for i, w in enumerate(words) if _normalise_word(w) not in _FILLER_WORDS]
        n = len(players)
        if n == 0:
            return

        def question_with(kept_players):
            return " ".join(w for i, w in enumerate(words) if i not in players or i in kept_players)

        exact = n <= MAX_SHAPLEY_WORDS
        if exact:
            masks = list(range(2 ** n))  # bit j set = player j kept
        else:
            full_mask = 2 ** n - 1
            masks = [full_mask] + [full_mask & ~(1 << j) for j in range(n)]
        texts = [question_with({players[j] for j in range(n) if mask >> j & 1}) for mask in masks]
        chunk_vectors = self.vectors[company][[r.chunk_index for r in results]]
        sims = embed_queries(texts) @ chunk_vectors.T  # rows: question versions, cols: passages
        row = {mask: i for i, mask in enumerate(masks)}

        values = np.zeros((n, len(results)))
        if exact:
            weight = [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)]
            for j in range(n):
                for mask in masks:
                    if not mask >> j & 1:
                        gain = sims[row[mask | 1 << j]] - sims[row[mask]]
                        values[j] += weight[bin(mask).count("1")] * gain
        else:
            for j in range(n):
                values[j] = sims[0] - sims[1 + j]

        for col, result in enumerate(results):
            by_word = {players[j]: float(values[j, col]) for j in range(n)}
            result.word_contributions = [(w, by_word.get(i)) for i, w in enumerate(words)]
            result.explanation_method = "Shapley values" if exact else "leave-one-out"

    def retrieve_multi(
        self, question: str, company: str, top_k: int = DEFAULT_TOP_K, sub_queries: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """Retrieve for a question that may cover several topics.

        With one (sub-)query this is plain retrieve(). With several (from
        split_question), each sub-query gets its own top_k, and the lists are
        interleaved (1st of each, then 2nd of each, ...) without duplicates, so
        every topic is represented. Example: "water and waste goals" searched as
        one question returned only water passages; searched as two, both appear.
        """
        if not sub_queries or len(sub_queries) < 2:
            return self.retrieve(question, company, top_k)
        lists = [self.retrieve(q, company, top_k, neighbours=False) for q in sub_queries]
        for q, found in zip(sub_queries, lists):
            for result in found:
                result.sub_query = q
        merged, seen = [], set()
        for position in range(top_k):
            for found in lists:
                if position < len(found) and found[position].chunk_index not in seen:
                    seen.add(found[position].chunk_index)
                    merged.append(found[position])
        self.add_neighbours(merged, company)
        return merged

    def retrieve_comparison(
        self, question: str, top_k: int = DEFAULT_TOP_K, sub_queries: Optional[List[str]] = None
    ) -> Dict[str, List[RetrievedChunk]]:
        """Search each report separately so both companies are always represented.

        A single combined search could return 5 Wipro chunks and 0 HCLTech
        chunks if Wipro happens to phrase things closer to the question, which
        would make a fair comparison impossible.
        """
        return {company: self.retrieve_multi(question, company, top_k, sub_queries) for company in self.indexes}


# --- Generation ----------------------------------------------------------------
SYSTEM_PROMPT = """You are GreenScope, an assistant that answers questions about corporate \
sustainability (ESG) reports.

Rules:
1. Answer ONLY from the report excerpts provided in the user message. Never use outside \
knowledge, even if you believe you know the answer.
2. Cite every factual claim with its source in the form [Company, p.X], using the company \
and page shown in the excerpt header.
3. Keep numbers, units, percentages, years and ratings (e.g. "A-", "BBB+") exactly as written \
in the excerpts. Do not convert, round, add up or calculate new figures.
4. If the excerpts do not contain the answer, say: "This is not disclosed in the retrieved \
text." You may mention closely related information that IS in the excerpts, with citations.
5. Excerpts may contain text extracted from charts and tables, where numbers can appear \
separated from their labels or out of order. Only state that a figure belongs to a category \
(e.g. Scope 1 and 2 vs Scope 3, target vs achieved, a specific year) when the text explicitly \
connects them, ideally in a full sentence. If the connection is unclear, say so instead of guessing.
6. Do not draw conclusions the excerpts do not state, such as whether a target has been met, \
exceeded or missed, or how two figures compare. Report the figures and let the reader compare.
7. Be concise: a short paragraph or a few bullet points."""


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


# --- Splitting two-topic questions -----------------------------------------------
# Only questions that could cover more than one topic are sent to the model to be
# split; everything else is searched as typed, at no extra cost.
_MAYBE_SEVERAL_TOPICS = re.compile(r"\band\b|&|\bas well as\b|;|\?.+\?", re.IGNORECASE)

SPLIT_PROMPT = """You turn questions about company sustainability reports into search queries.
If the question asks about one topic, return it unchanged.
If it asks about several distinct topics, return one short, complete question per topic (at most 3).

Examples:
"What are the water and waste goals?" ->
What are the water goals?
What are the waste goals?

"What is the Scope 1 and 2 target?" ->
What is the Scope 1 and 2 target?

"How many women work there and what is the attrition rate?" ->
How many women work there?
What is the attrition rate?

Output only the questions, one per line, with no numbering or other text."""


@dataclass
class SplitQuestion:
    queries: List[str]
    cost_usd: float = 0.0


def split_question(question: str) -> SplitQuestion:
    """Split a multi-topic question into one search query per topic.

    A free regular-expression check runs first; only questions containing
    "and", "&", "as well as", ";" or two question marks are sent to Claude
    Haiku (about US$0.0002). Any problem (no key, API error, odd output) falls
    back to searching the question as typed.
    """
    if not _MAYBE_SEVERAL_TOPICS.search(question) or not has_api_key():
        return SplitQuestion([question])
    import anthropic

    try:
        response = anthropic.Anthropic(api_key=get_api_key()).messages.create(
            model=LLM_MODEL,
            max_tokens=150,
            system=SPLIT_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.APIError:
        return SplitQuestion([question])
    cost = (response.usage.input_tokens * PRICE_PER_MTOK_INPUT
            + response.usage.output_tokens * PRICE_PER_MTOK_OUTPUT) / 1_000_000
    text = "".join(b.text for b in response.content if b.type == "text")
    queries = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    queries = [q for q in queries if len(q) > 3][:3]
    return SplitQuestion(queries or [question], cost)


def answer_request(question: str, chunks: List[RetrievedChunk]) -> dict:
    """The exact Claude request used to answer a question (also used by the
    evaluation, so it tests precisely what the app sends)."""
    return {
        "model": LLM_MODEL,
        "max_tokens": MAX_ANSWER_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Report excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"}],
    }


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

    try:
        client = anthropic.Anthropic(api_key=get_api_key())
        response = client.messages.create(**answer_request(question, chunks))
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
    return Answer(
        text=text,
        chunks=chunks,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def preflight(live: bool = False) -> bool:
    """Pre-demo check: is everything the app needs in place? Prints PASS/FAIL lines.

    Run on the presenting machine before a demo:  python -m src.model --check [--live]
    --live also makes one tiny Claude call (about US$0.0001) to prove the key works.
    """
    import time

    from src.data import RAW_DATA_DIR, _cache_is_valid

    ok = True

    def report(passed: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}{': ' + detail if detail else ''}")

    for company, filename in COMPANIES.items():
        report((RAW_DATA_DIR / filename).exists(), f"{company} PDF present", filename)
    report(_cache_is_valid(), "Pre-built index matches the PDFs and settings",
           "" if _cache_is_valid() else "it will be rebuilt on first use (about 2 minutes)")
    t0 = time.time()
    index = ReportIndex()
    report(True, "Index and embedding model loaded", f"{time.time() - t0:.0f} s")
    for company, question, page in (("Wipro", "What is Wipro's Scope 3 target?", {46, 63, 67, 88}),
                                    ("HCLTech", "What is HCLTech's Scope 3 target?", {29})):
        pages = [r.page for r in index.retrieve(question, company, top_k=5)]
        report(bool(page & set(pages)), f"{company} sample search finds the right page", f"pages {pages}")
    report(has_api_key(), "Anthropic API key configured",
           "" if has_api_key() else "AI answers disabled; search still works")
    if live and has_api_key():
        import anthropic

        try:
            anthropic.Anthropic(api_key=get_api_key()).messages.create(
                model=LLM_MODEL, max_tokens=5, messages=[{"role": "user", "content": "Reply OK"}])
            report(True, "Live API call")
        except anthropic.APIError as e:
            report(False, "Live API call", type(e).__name__)
    print("\nReady for the demo." if ok else "\nFix the FAIL lines before the demo.")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if preflight(live="--live" in sys.argv) else 1)
