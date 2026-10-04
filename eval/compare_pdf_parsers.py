"""Experiment: does a layout-aware PDF parser (Docling) beat PyPDF for GreenScope?

PyPDF (used by the app) extracts text in roughly the order it was drawn, which
scrambles charts and tables. Docling uses layout-analysis models to recover
headings, reading order and tables. This script compares the two on the eval set.

Steps:
  1. Convert both PDFs with Docling, one markdown string per page
     (about 8 minutes on a laptop CPU; cached in index/docling_pages/).
  2. Build a second index from that text (in index/docling/) with the same
     chunking, embedding and retrieval as the app.
  3. Retrieval eval for both (free).
  4. With --answers: generate an answer to every eval question from both
     indexes (32 Claude Haiku calls, about US$0.08) and save them to
     eval/parser_answers.json for manual grading.

Docling is NOT needed to run the app. Install it only for this experiment:
    pip install docling
    python eval/compare_pdf_parsers.py [--answers]

Results and grades: eval/pdf_parser_comparison.md
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

import src.data as data  # noqa: E402
import src.model as model  # noqa: E402
from eval.run_eval import evaluate, group_rows, summarise  # noqa: E402

DOCLING_PAGES_DIR = data.INDEX_DIR / "docling_pages"


def docling_pages(pdf_path: Path) -> dict:
    """Return {page_number: markdown} for one PDF, converting once and caching."""
    cache = DOCLING_PAGES_DIR / f"{pdf_path.stem}.json"
    if cache.exists():
        return {int(n): text for n, text in json.loads(cache.read_text()).items()}

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)  # text PDFs, keep tables
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    print(f"Converting {pdf_path.name} with Docling (several minutes)...")
    doc = converter.convert(str(pdf_path)).document
    pages = {n: doc.export_to_markdown(page_no=n) for n in sorted(doc.pages)}
    DOCLING_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False))
    return pages


def load_with_docling(file_path, company):
    """Drop-in replacement for data.load_pdf that uses Docling's page text."""
    file_path = Path(file_path)
    return [
        Document(page_content=text, metadata={"company": company, "source": file_path.name, "page": page})
        for page, text in docling_pages(file_path).items()
    ]


def main():
    eval_set = json.loads((PROJECT_ROOT / "eval" / "eval_set.json").read_text())

    indexes = {"PyPDF": model.ReportIndex()}  # the app's own index
    # Same pipeline, Docling text, separate cache folder (the app's index is untouched).
    data.load_pdf = load_with_docling
    data.INDEX_DIR = PROJECT_ROOT / "index" / "docling"
    indexes["Docling"] = model.ReportIndex()

    print(f"{'':16}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR@5':>8}")
    for name, index in indexes.items():
        for group, rows in group_rows(evaluate(index, eval_set, True)).items():
            s = summarise(rows)
            print(f"{name + ' ' + group:16}" + "".join(f"{s[m]:>8.2f}" for m in ("Hit@1", "Hit@3", "Hit@5", "MRR@5")))

    if "--answers" in sys.argv:
        answers, total = {}, 0.0
        for name, index in indexes.items():
            for item in eval_set:
                chunks = index.retrieve(item["question"], item["company"], top_k=5)
                answer = model.generate_answer(item["question"], chunks)
                total += answer.cost_usd
                answers[f"{name}|{item['id']}"] = answer.error or answer.text
        out = PROJECT_ROOT / "eval" / "parser_answers.json"
        out.write_text(json.dumps(answers, indent=1, ensure_ascii=False) + "\n")
        print(f"\n{len(answers)} answers written to {out} (cost about US${total:.3f})")


if __name__ == "__main__":
    main()
