from __future__ import annotations

from pathlib import Path
from typing import Any


def looks_like_pdf(path: Path) -> bool:
    """True only for a real PDF (``%PDF`` magic header). Guards against feeding a notebook
    or other JSON/text file to pypdf, which otherwise crashes with 'invalid pdf header'."""
    try:
        with path.open("rb") as handle:
            return handle.read(5).startswith(b"%PDF")
    except OSError:
        return False


def _not_a_pdf_report(path: Path) -> dict[str, Any]:
    return {
        "page_count": 0,
        "encrypted": False,
        "text": "",
        "pages_read": [],
        "characters_returned": 0,
        "truncated": False,
        "error": "not_a_pdf",
        "warnings": [
            f"{path.name} is not a PDF (no %PDF header). Use file_read_text or "
            "file_profile_json instead — e.g. for a .ipynb notebook read its code/markdown."
        ],
    }


def extract_pdf_text(path: Path, *, page_start: int, max_pages: int, max_chars: int) -> dict[str, Any]:
    from pypdf import PdfReader

    if not looks_like_pdf(path):
        return _not_a_pdf_report(path)

    reader = PdfReader(str(path))
    warnings: list[str] = []

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return {
                "page_count": len(reader.pages),
                "encrypted": True,
                "text": "",
                "pages_read": [],
                "warnings": ["PDF is encrypted and could not be opened with an empty password."],
            }

    page_count = len(reader.pages)
    end = min(page_count, page_start + max_pages)
    chunks: list[str] = []
    pages_read: list[int] = []
    truncated = False

    for page_index in range(page_start, end):
        try:
            page_text = reader.pages[page_index].extract_text() or ""
        except Exception as exc:
            warnings.append(f"Could not extract text from page {page_index}: {exc}")
            continue
        if not page_text:
            warnings.append(f"Page {page_index} had no extractable text.")
            continue
        pages_read.append(page_index)
        chunks.append(page_text)
        if sum(len(chunk) for chunk in chunks) >= max_chars:
            truncated = True
            break

    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return {
        "page_count": page_count,
        "encrypted": bool(reader.is_encrypted),
        "page_start": page_start,
        "max_pages": max_pages,
        "pages_read": pages_read,
        "text": text,
        "characters_returned": len(text),
        "truncated": truncated,
        "warnings": warnings,
    }
