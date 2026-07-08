from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_pdf_text(path: Path, *, page_start: int, max_pages: int, max_chars: int) -> dict[str, Any]:
    from pypdf import PdfReader

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
