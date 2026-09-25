"""Ingestion: PDF loading, chunking, embedding.

Pipeline:  PDF file -> pages (one Document per page) -> chunks -> embeddings

Every chunk carries metadata so that any answer generated later can be traced
back to the exact report and page it came from:
    - source_file  : PDF filename, e.g. "wipro_sustainability_2024_25.pdf"
    - page_number  : 1-indexed page number (matches what a human sees in a PDF viewer)
    - chunk_index  : position of the chunk within its document (0, 1, 2, ...)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

PDF_FILES = [
    "wipro_sustainability_2024_25.pdf",
    "hcltech_sustainability_fy2024.pdf",
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking parameters -----------------------------------------------------
# CHUNK_SIZE = 800 characters (~130-160 words, ~180-200 tokens)
#
# Why 800 — keeping chunks FOCUSED:
#   * An embedding is a single vector summarising the whole chunk. The bigger
#     the chunk, the more topics get averaged together (e.g. Scope 1 emissions
#     + water usage + board diversity), and the vector stops matching any one
#     question well. ~800 chars is roughly one or two paragraphs, or one table
#     section, in an ESG report — usually a single topic/metric.
#   * all-MiniLM-L6-v2 truncates input at 256 word-piece tokens. 800 chars sits
#     comfortably under that limit, so no text is silently dropped when
#     embedding. (Much larger chunks, e.g. 2000 chars, would be cut off and the
#     tail of each chunk would be invisible to retrieval.)
#   * Small, focused chunks also mean we can pass several of them to the LLM
#     later without blowing up the prompt with irrelevant text.
#
# Why not smaller (e.g. 200-300):
#   * Too small and a chunk loses its context: a number like "1,23,456 tCO2e"
#     separated from the sentence saying WHAT it measures and for WHICH YEAR
#     is useless. 800 chars is enough to keep a metric together with its label,
#     unit, and reporting period.
#
# CHUNK_OVERLAP = 150 characters (~19% of chunk size) — CONTEXT CONTINUITY:
#   * Splitting is mechanical, so a boundary can fall mid-thought, e.g.
#     "...reduced absolute emissions by" | "42% against the FY20 baseline".
#     Repeating the last ~150 chars (roughly 1-2 sentences) at the start of the
#     next chunk means a fact that straddles a boundary still appears whole in
#     at least one chunk.
#   * ~150 is about one sentence — enough to bridge a split, but small enough
#     (under 20%) that we don't store and retrieve lots of near-duplicate text.
#     Much higher overlap would inflate the index and cause the same passage to
#     come back multiple times in retrieval results.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF and return one LangChain Document per page."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = PyPDFLoader(str(file_path)).load()

    # PyPDFLoader stores the full path in "source" and a 0-indexed "page".
    # Add clean, human-friendly fields we can cite in answers later.
    for page in pages:
        page.metadata["source_file"] = file_path.name
        page.metadata["page_number"] = page.metadata.get("page", 0) + 1
    return pages


def chunk_pages(
    pages: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split page Documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries separators in order
    (paragraph "\\n\\n" -> line "\\n" -> word " " -> character), so it splits on
    the most natural boundary that still fits within chunk_size.

    Splitting is done per page, so each chunk inherits its page's metadata
    (source_file, page_number). chunk_index is then assigned sequentially across
    the whole document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # char offset of the chunk within its page
    )
    chunks = splitter.split_documents(pages)

    # Drop chunks that are empty/whitespace (e.g. from image-only pages).
    chunks = [c for c in chunks if c.page_content.strip()]

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading is slow)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_chunks(
    chunks: List[Document], batch_size: int = 64
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Embed chunks with all-MiniLM-L6-v2.

    Returns:
        embeddings : np.ndarray of shape (n_chunks, 384)
        texts      : original chunk text, same order as embeddings
        metadatas  : chunk metadata, same order as embeddings
    """
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    model = get_embedding_model()
    # normalize_embeddings=True -> unit-length vectors, so cosine similarity
    # equals a plain dot product (convenient for the vector store later).
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings, texts, metadatas


if __name__ == "__main__":
    for filename in PDF_FILES:
        pdf_path = RAW_DATA_DIR / filename
        print("=" * 80)
        print(f"Document: {filename}")
        print("=" * 80)

        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        embeddings, texts, metadatas = embed_chunks(chunks)

        print(f"Pages loaded      : {len(pages)}")
        print(f"Total chunks      : {len(chunks)}")
        print(f"Embeddings shape  : {embeddings.shape}")

        for i in range(min(2, len(chunks))):
            print("-" * 80)
            print(f"Chunk {i}")
            print(f"Metadata: {metadatas[i]}")
            print(f"Length  : {len(texts[i])} chars")
            print("Text:")
            print(texts[i])
        print()
