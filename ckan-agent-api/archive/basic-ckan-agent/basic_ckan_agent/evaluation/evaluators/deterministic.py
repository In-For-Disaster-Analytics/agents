"""Deterministic (code-only) evaluators for obvious metadata failures.

These are cheap, reproducible gates that run before any LLM judging. They follow
the LangSmith functional-evaluator form: take ``outputs`` (and optionally
``reference_outputs``) and return a ``{"key", "score", "comment"}`` dict.
"""

from __future__ import annotations

from typing import Any

BAD_TITLES = {
    "",
    "dataset",
    "untitled",
    "untitled dataset",
    "metadata",
    "results",
    "file upload",
    "data",
    "new dataset",
}

RAW_EXTENSIONS = (".csv", ".zip", ".xlsx", ".json", ".tsv", ".txt", ".pdf", ".nc")

TITLE_MIN_LEN = 10
TITLE_MAX_LEN = 120
DESCRIPTION_MIN_LEN = 80
DESCRIPTION_MAX_LEN = 2000

FILLER_TERMS = ("lorem ipsum", "todo", "tbd", "placeholder", "xxx", "fixme")


def title_basic_checks(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Reject empty/placeholder/filename-only/too-short/too-long titles."""
    title = str(outputs.get("title", "")).strip()
    lowered = title.lower()

    reasons: list[str] = []
    if not (TITLE_MIN_LEN <= len(title) <= TITLE_MAX_LEN):
        reasons.append(f"length {len(title)} outside [{TITLE_MIN_LEN},{TITLE_MAX_LEN}]")
    if lowered in BAD_TITLES:
        reasons.append("generic placeholder title")
    if lowered.endswith(RAW_EXTENSIONS) and not _filename_is_descriptive(title):
        reasons.append("looks like a raw filename")

    passed = not reasons
    return {
        "key": "title_basic_quality",
        "score": passed,
        "comment": (f"title={title!r}; " + ("OK" if passed else "; ".join(reasons))),
    }


def description_basic_checks(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Reject empty/too-short/too-long/filler/duplicate-of-title descriptions."""
    title = str(outputs.get("title", "")).strip()
    description = str(outputs.get("description", "")).strip()
    lowered = description.lower()

    reasons: list[str] = []
    if not (DESCRIPTION_MIN_LEN <= len(description) <= DESCRIPTION_MAX_LEN):
        reasons.append(f"length {len(description)} outside [{DESCRIPTION_MIN_LEN},{DESCRIPTION_MAX_LEN}]")
    found_filler = [term for term in FILLER_TERMS if term in lowered]
    if found_filler:
        reasons.append(f"filler text: {', '.join(found_filler)}")
    if description and lowered == title.lower():
        reasons.append("description duplicates the title")

    passed = not reasons
    return {
        "key": "description_basic_quality",
        "score": passed,
        "comment": (f"description length={len(description)}; " + ("OK" if passed else "; ".join(reasons))),
    }


def must_mention_terms(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Check that required terms (from the example) appear in title+description.

    Reference key ``must_mention`` is a list of terms; matching is
    case-insensitive substring. Examples without the key score ``None`` (skipped).
    """
    reference_outputs = reference_outputs or {}
    required = _as_terms(reference_outputs.get("must_mention"))
    if not required:
        return {"key": "must_mention_terms", "score": None, "comment": "no must_mention terms"}

    haystack = (str(outputs.get("title", "")) + " " + str(outputs.get("description", ""))).lower()
    missing = [term for term in required if term.lower() not in haystack]
    passed = not missing
    return {
        "key": "must_mention_terms",
        "score": passed,
        "comment": "all present" if passed else f"missing: {', '.join(missing)}",
    }


# Generic tokens that don't make a filename descriptive on their own.
_GENERIC_FILENAME_TOKENS = {
    "data", "final", "draft", "copy", "new", "old", "output", "outputs",
    "results", "result", "file", "files", "dataset", "v1", "v2", "v3",
    "rev", "version", "temp", "tmp", "export", "raw", "clean", "processed",
}


def _filename_is_descriptive(title: str) -> bool:
    stem = title.rsplit(".", 1)[0]
    words = [w for w in stem.replace("_", " ").replace("-", " ").split() if w]
    meaningful = [w for w in words if w.lower() not in _GENERIC_FILENAME_TOKENS and not w.isdigit()]
    return len(meaningful) >= 3


def _as_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []
