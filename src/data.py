"""Ingestion: turn the two PDF reports into searchable vectors.

Pipeline (runs once, then is cached on disk):

    PDF  ->  pages  ->  chunks  ->  embeddings  ->  index/ folder

Every chunk carries metadata so that any answer can be traced back to the
exact report and page it came from:
    company      "Wipro" or "HCLTech"
    source       PDF filename, e.g. "wipro_sustainability_2024_25.pdf"
    page         1-indexed PDF page (page 1 = first page of the file, which is
                 what a PDF viewer shows; it may differ from the number printed
                 on the page itself)
    chunk_index  position of the chunk within its company's report (0, 1, 2, ...)

Run `python -m src.data` to build the index and print a quick summary.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# pypdf logs harmless font warnings for some PDFs (e.g. HCLTech's custom fonts).
# The text still extracts correctly, so keep the console readable.
logging.getLogger("pypdf").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "index"

# The two reports, keyed by the company name shown in the app.
COMPANIES: Dict[str, str] = {
    "Wipro": "wipro_sustainability_2024_25.pdf",
    "HCLTech": "hcltech_sustainability_fy2024.pdf",
}

# --- Tuned parameters ----------------------------------------------------------
# All three values below were chosen by a grid search over 36 combinations of
# embedding model x chunk size x overlap, with 5-fold cross-validation on a
# 40-question evaluation set (eval/tune.py, results in eval/tuning_results.md).
# The same combination won in all 5 folds. On questions NOT used to choose it,
# it ranked the right page higher than the original setting (MiniLM, 800/150):
# held-out MRR@5 0.82 vs 0.67, Hit@5 0.93 vs 0.85.

# EMBEDDING MODEL: BAAI/bge-small-en-v1.5 (384-number vectors, reads up to 512
# word-pieces). Same size and speed class as all-MiniLM-L6-v2, but trained
# specifically for search (question -> passage), which is our task. It beat
# MiniLM (fast, general purpose) and all-mpnet-base-v2 (2x larger, slower).
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE models are trained to expect this instruction in front of search QUESTIONS
# (not in front of the passages). Leaving it out lowers search quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# CHUNK_SIZE = 400 characters (~60-80 words, two to four sentences).
# Why small chunks keep matches FOCUSED:
#   * Each chunk becomes ONE vector that summarises its meaning. The more topics a
#     chunk mixes (emissions + water + board diversity), the more its vector is an
#     average that matches no single question well. ESG reports pack a different
#     metric into almost every sentence, so short chunks match questions precisely.
#   * Measured: average MRR@5 fell steadily with size for every model
#     (400: 0.72, 800: 0.66, 1200: 0.57).
# The risk of small chunks is losing context: a figure such as "42%" cut off from
# the words saying WHAT it measures. That is handled separately: the model is
# given each match WITH its surrounding text from the same page ("small-to-big",
# CONTEXT_CHARS in model.py), so search is precise and the answer still has context.
#
# CHUNK_OVERLAP = 250 characters: each chunk repeats the last 250 characters of
# the previous one, so a new chunk starts every ~150 characters.
# Why overlap preserves CONTEXT across boundaries:
#   * Splitting is mechanical, so a cut can land mid-thought:
#     "...reduced emissions by" | "42% against the FY20 baseline".
#     With overlap, a fact that straddles one cut appears whole in another chunk.
#   * With high overlap, every sentence appears in two or three chunks, at least
#     one of which has it near the middle with its neighbours, which helps short
#     chunks most. Measured at 400 characters: overlap 0 / 75 / 150 / 250 gave
#     MRR@5 0.74 / 0.79 / 0.74 / 0.82 with BGE.
#   * The cost is more chunks to store (about 4,000 instead of 1,100), which is
#     still tiny for FAISS. Repeated text is merged before it reaches the model.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 250

# Chunks shorter than this are mostly cover pages, section dividers or stray
# headings ("Contents", "2024 Sustainability Report"). They carry no facts and
# would only clutter search results.
MIN_CHUNK_CHARS = 50


# --- 1. Loading ----------------------------------------------------------------
def load_pdf(file_path: Path, company: str) -> List[Document]:
    """Load a PDF and return one Document per page with clean metadata."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}. Put the reports in {RAW_DATA_DIR}."
        )

    pages = PyPDFLoader(str(file_path)).load()

    # PyPDFLoader adds lots of PDF properties (producer, creation date, ...) and
    # a 0-indexed "page". Keep only what we need, with 1-indexed page numbers.
    for page in pages:
        page.metadata = {
            "company": company,
            "source": file_path.name,
            "page": page.metadata.get("page", 0) + 1,
        }
    return pages


# --- 2. Chunking ---------------------------------------------------------------
def chunk_pages(
    pages: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> List[Document]:
    """Split pages into overlapping chunks and number them.

    RecursiveCharacterTextSplitter tries the most natural break first
    (blank line -> line break -> space -> character), so chunks end at
    paragraph or sentence boundaries whenever possible.

    Splitting is done page by page, so every chunk inherits its page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chars]

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


# --- 3. Embedding --------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading takes a few seconds).

    The same model embeds both the report chunks and the user's question, so
    the two live in the same "meaning space" and can be compared.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """Turn texts into 384-number vectors of length 1 (normalized).

    Normalizing means the dot product of two vectors equals their cosine
    similarity, which is exactly what the FAISS index in model.py computes.
    """
    vectors = get_embedding_model().encode(
        texts,
        batch_size=64,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)  # FAISS requires float32


def embed_queries(questions: List[str]) -> np.ndarray:
    """Embed search questions (adds QUERY_PREFIX if the model needs one)."""
    return embed_texts([QUERY_PREFIX + q for q in questions])


def embed_chunks(chunks: List[Document]) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Embed chunks; return (embeddings, texts, metadatas) in matching order."""
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    return embed_texts(texts, show_progress=True), texts, metadatas


# --- 4. Build / load the on-disk cache ------------------------------------------
# index/
#   manifest.json           settings used for the build (to detect changes)
#   <Company>_vectors.npy   one row per chunk
#   <Company>_chunks.json   text + metadata per chunk, same order as the rows


def _manifest() -> dict:
    """Describe everything that, if changed, should trigger a rebuild."""
    pdfs = {}
    for company, filename in COMPANIES.items():
        path = RAW_DATA_DIR / filename
        pdfs[company] = {"file": filename, "bytes": path.stat().st_size if path.exists() else None}
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "query_prefix": QUERY_PREFIX,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "pdfs": pdfs,
    }


def _cache_is_valid() -> bool:
    manifest_path = INDEX_DIR / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        saved = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    files_present = all(
        (INDEX_DIR / f"{c}_vectors.npy").exists() and (INDEX_DIR / f"{c}_chunks.json").exists()
        for c in COMPANIES
    )
    return files_present and saved == _manifest()


def build_index(force: bool = False) -> Dict[str, dict]:
    """Return {company: {"vectors": np.ndarray, "chunks": [{"text", "metadata"}]}}.

    Loads from index/ if a valid cache exists; otherwise loads, chunks and
    embeds both PDFs (about a minute on a laptop CPU) and saves the result.
    """
    if not force and _cache_is_valid():
        data = {}
        for company in COMPANIES:
            data[company] = {
                "vectors": np.load(INDEX_DIR / f"{company}_vectors.npy"),
                "chunks": json.loads((INDEX_DIR / f"{company}_chunks.json").read_text()),
            }
        return data

    INDEX_DIR.mkdir(exist_ok=True)
    data = {}
    for company, filename in COMPANIES.items():
        pages = load_pdf(RAW_DATA_DIR / filename, company)
        chunks = chunk_pages(pages)
        vectors, texts, metadatas = embed_chunks(chunks)
        records = [{"text": t, "metadata": m} for t, m in zip(texts, metadatas)]

        np.save(INDEX_DIR / f"{company}_vectors.npy", vectors)
        (INDEX_DIR / f"{company}_chunks.json").write_text(json.dumps(records, ensure_ascii=False))
        data[company] = {"vectors": vectors, "chunks": records}

    # Written last, so an interrupted build is never mistaken for a valid cache.
    (INDEX_DIR / "manifest.json").write_text(json.dumps(_manifest(), indent=2))
    return data


if __name__ == "__main__":
    import sys

    index = build_index(force="--rebuild" in sys.argv)
    for company, entry in index.items():
        chunks = entry["chunks"]
        print("=" * 70)
        print(f"{company}: {len(chunks)} chunks, vectors {entry['vectors'].shape}")
        for record in chunks[:2]:
            print("-" * 70)
            print(f"metadata: {record['metadata']}")
            print(record["text"])
