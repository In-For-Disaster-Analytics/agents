"""Map-reduce PDF summarization.

Reads a (possibly long) PDF in order, summarizes it section-by-section ("map"), then combines
those section summaries into one document summary ("reduce") — so metadata is written from the
report's actual content rather than just its first pages. Bounded by page/char/window caps.

The LLM call is injected as ``chat(system_prompt, user_text) -> str`` so this is unit-testable
without a live model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

ChatFn = Callable[[str, str], str]

MAX_PAGES = 60
MAX_CHARS = 120_000
WINDOW_CHARS = 8_000
MAX_WINDOWS = 12

_MAP_SYS = (
    "You are summarizing one section of a longer document so its dataset metadata can be written. "
    "In 2-3 sentences capture the subject, study area/region, methods, data, and any findings present "
    "in THIS section. Omit boilerplate, tables of contents, and headers/footers."
)
_REDUCE_SYS = (
    "Combine these ordered section summaries of a single document into one 3-5 sentence summary. "
    "State the document's overall subject, geographic/temporal scope, methods/data, and key findings. "
    "Be specific and factual; do not invent details not present in the summaries."
)


def _extract_full_text(path: Path, max_pages: int, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    parts: list[str] = []
    total = 0
    for page in reader.pages[:max_pages]:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def map_reduce_text_summary(
    text: str,
    *,
    chat: ChatFn,
    window_chars: int = WINDOW_CHARS,
    max_windows: int = MAX_WINDOWS,
) -> dict[str, Any]:
    """Summarize text in windows (map), then combine (reduce)."""
    if not text.strip():
        return {"summary": "", "windows": 0, "truncated": False}
    chunks = [text[i : i + window_chars] for i in range(0, len(text), window_chars)]
    truncated = len(chunks) > max_windows
    chunks = chunks[:max_windows]

    if len(chunks) == 1:
        return {"summary": chat(_MAP_SYS, chunks[0]).strip(), "windows": 1, "truncated": truncated}

    section_summaries = [chat(_MAP_SYS, chunk).strip() for chunk in chunks]
    combined = "\n".join(f"Section {i + 1}: {s}" for i, s in enumerate(section_summaries))
    summary = chat(_REDUCE_SYS, combined).strip()
    return {
        "summary": summary,
        "windows": len(chunks),
        "truncated": truncated,
        "section_summaries": section_summaries,
    }


def map_reduce_pdf_summary(
    path: Path,
    *,
    chat: ChatFn,
    max_pages: int = MAX_PAGES,
    max_chars: int = MAX_CHARS,
    window_chars: int = WINDOW_CHARS,
    max_windows: int = MAX_WINDOWS,
) -> dict[str, Any]:
    from app.files.extractors.pdf import looks_like_pdf

    if not looks_like_pdf(path):
        return {
            "summary": "",
            "windows": 0,
            "truncated": False,
            "error": "not_a_pdf",
            "note": (
                f"{path.name} is not a PDF (no %PDF header). For a notebook/code/text file use "
                "file_read_text or file_profile_json instead of a PDF tool."
            ),
        }
    text = _extract_full_text(path, max_pages, max_chars)
    if not text.strip():
        return {"summary": "", "windows": 0, "truncated": False, "note": "No extractable PDF text (or pypdf missing)."}
    result = map_reduce_text_summary(text, chat=chat, window_chars=window_chars, max_windows=max_windows)
    result["chars_read"] = len(text)
    return result
