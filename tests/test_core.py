"""Tests for GreenScope's core logic.

Most tests use a tiny fake embedding model (words hashed into a vector), so they
run in seconds without downloading anything. The last test loads the real index
and checks that retrieval quality has not regressed.

Run:  pytest -q
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

import src.model as model
from src.data import CHUNK_SIZE, MIN_CHUNK_CHARS, chunk_pages

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Query clean-up ------------------------------------------------------------
@pytest.mark.parametrize("question, expected", [
    ("What is Wipro's Scope 3 target?", "What is the company's Scope 3 target?"),
    ("How has HCL Tech reduced water use?", "How has the company reduced water use?"),
    ("Compare HCLTech Ltd and Wipro Limited", "Compare the company and the company"),
    ("What is the net-zero year?", "What is the net-zero year?"),
])
def test_prepare_query_replaces_company_names(question, expected):
    assert model.prepare_query(question) == expected


def test_prepare_query_leaves_other_words_alone():
    # "HCLFoundation" is a different name and must not be rewritten.
    assert "HCLFoundation" in model.prepare_query("How many lives has the HCLFoundation impacted?")


# --- Chunking ----------------------------------------------------------------------
def test_chunk_pages_respects_size_min_length_and_metadata():
    text = " ".join(f"Sentence number {i} about emissions and water." for i in range(60))
    pages = [
        Document(page_content=text, metadata={"company": "Wipro", "source": "w.pdf", "page": 7}),
        Document(page_content="Contents", metadata={"company": "Wipro", "source": "w.pdf", "page": 8}),
    ]
    chunks = chunk_pages(pages)
    assert chunks, "expected at least one chunk"
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert all(len(c.page_content.strip()) >= MIN_CHUNK_CHARS for c in chunks)  # "Contents" dropped
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c.metadata["page"] for c in chunks} == {7}


def test_join_overlapping_keeps_shared_text_once():
    assert model._join_overlapping("abc reduced emissions by", "reduced emissions by 42%") == "abc reduced emissions by 42%"
    assert model._join_overlapping("first part", "unrelated") == "first part\nunrelated"


# --- Keyword search (BM25) -------------------------------------------------------
def test_bm25_prefers_rare_matching_words_and_word_scores_add_up():
    texts = ["the company has 146 nationalities in its workforce",
             "the company reports scope 3 emissions",
             "the company workforce is large"]
    bm25 = model.BM25(texts)
    scores = bm25.scores("How many nationalities are in the company's workforce?")
    assert scores.argmax() == 0
    assert sum(v for _, v in bm25.word_scores("nationalities workforce", 0)) == pytest.approx(
        bm25.scores("nationalities workforce")[0], rel=1e-5)
    assert bm25.scores("the company")[0] == 0  # filler words are ignored


# --- Retrieval, context and explanations with a fake embedding model ---------------
def fake_embed(texts):
    """Deterministic stand-in for BGE: hash each word into 64 dimensions."""
    vectors = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in model._tokens(text):
            vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture
def toy_index(monkeypatch):
    records = [
        ("Scope 3 emissions target is 55% by 2030.", 10),
        ("The baseline year for Scope 3 is 2020.", 10),
        ("Renewable electricity share is 84%.", 11),
        ("Women are 37.1% of the workforce.", 12),
        ("Water reuse reached 31% of total use.", 12),
    ]
    chunks = [{"text": t, "metadata": {"company": "Wipro", "source": "w.pdf", "page": p, "chunk_index": i}}
              for i, (t, p) in enumerate(records)]
    monkeypatch.setattr(model, "embed_queries", fake_embed)
    monkeypatch.setattr(model, "build_index", lambda force=False: {
        "Wipro": {"vectors": fake_embed([t for t, _ in records]), "chunks": chunks}})
    return model.ReportIndex()


def test_retrieve_ranks_relevant_passage_first_and_records_ranks(toy_index):
    results = toy_index.retrieve("What is Wipro's renewable electricity share?", "Wipro", top_k=3)
    assert results[0].page == 11
    assert results[0].meaning_rank >= 1 and results[0].keyword_rank >= 1


def test_context_grows_within_the_page_only(toy_index):
    results = toy_index.retrieve("Scope 3 emissions target", "Wipro", top_k=1)
    context = results[0].context
    assert "55% by 2030" in context and "baseline year" in context  # neighbour on the same page added
    assert "84%" not in context                                       # next page never added


def test_shapley_contributions_add_up(toy_index):
    question = "What is the Scope 3 emissions target?"
    results = toy_index.retrieve(question, "Wipro", top_k=2)
    toy_index.explain(question, results, "Wipro")
    words = model.prepare_query(question).split()
    empty = " ".join(w for w in words if model._normalise_word(w) in model._FILLER_WORDS)
    for r in results:
        total = sum(v for _, v in r.word_contributions if v is not None)
        empty_similarity = float(fake_embed([empty])[0] @ toy_index.vectors["Wipro"][r.chunk_index])
        assert total == pytest.approx(r.score - empty_similarity, abs=1e-5)
        assert r.explanation_method == "Shapley values"


def test_retrieve_multi_represents_every_sub_question(toy_index):
    results = toy_index.retrieve_multi("water and women", "Wipro", top_k=1,
                                       sub_queries=["water reuse", "women workforce"])
    assert {r.sub_query for r in results} == {"water reuse", "women workforce"}


# --- Generation without an API key ----------------------------------------------------
def test_no_api_key_gives_friendly_message_not_a_crash(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "GREENSCOPE_ANTHROPIC_KEY"):
        monkeypatch.delenv(name, raising=False)
    chunk = model.RetrievedChunk("text", "Wipro", 3, 0, "w.pdf", 0.5)
    answer = model.generate_answer("Question?", [chunk])
    assert answer.error and "API key" in answer.error
    assert model.split_question("water and waste goals?").queries == ["water and waste goals?"]


def test_format_context_cites_pages_and_skips_repeated_passages():
    a = model.RetrievedChunk("first", "Wipro", 3, 0, "w.pdf", 0.9, context="first, widened")
    b = model.RetrievedChunk("second", "Wipro", 3, 1, "w.pdf", 0.8, context="")  # already in a's context
    assert model.format_context([a, b]) == "[Wipro, p.3]\nfirst, widened"


# --- Regression check on the real index (needs the embedding model) --------------------
@pytest.mark.slow
def test_real_index_retrieval_quality_has_not_regressed():
    eval_set = json.loads((PROJECT_ROOT / "eval" / "eval_set.json").read_text())
    index = model.ReportIndex()
    hits = 0
    for q in eval_set:
        pages = {r.page for r in index.retrieve(q["question"], q["company"], top_k=5, neighbours=False)}
        hits += bool(pages & set(q["gold_pages"]))
    assert hits / len(eval_set) >= 0.90  # currently 0.93 (eval/results.md)
