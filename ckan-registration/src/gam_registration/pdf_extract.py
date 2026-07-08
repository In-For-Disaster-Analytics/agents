"""PDF report pipeline for TWDB GAM metadata enrichment.

Map-reduce strategy:
  - MAP: extract_metadata_from_chunk — one LLM call per ~8500-char chunk
  - REDUCE: consolidate_chunk_metadata — one LLM call merging all per-chunk results

PDF input is treated as UNTRUSTED: size and content-type are checked before
writing to a temp file.

Lazy-imports pymupdf (fitz) so the module is importable without pymupdf installed.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

# Lazy import: pymupdf is optional at import time; functions that need it raise
# ImportError with a helpful message when the library is absent.
fitz = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Configurable inter-call delay (seconds) read once at module import time.
_LLM_CALL_DELAY_SECONDS: float = float(os.environ.get("LLM_CALL_DELAY_SECONDS", "4"))

# Safety limits for untrusted PDF downloads.
_MAX_PDF_BYTES: int = 200 * 1024 * 1024  # 200 MB hard cap
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "binary/octet-stream",
    "application/octet-stream",
}


# ---------------------------------------------------------------------------
# Utils re-use — import the LLM call helper and JSON parser from utils.py.
# We do NOT re-implement the HTTP call; we delegate to the existing helper.
# Assumption: utils._chat_completion_content(*, model, api_key, system_prompt,
#   user_payload, base_url=None, temperature=0.1, timeout=120) -> str
# Assumption: utils._parse_llm_json(content: str) -> dict[str, Any]
# ---------------------------------------------------------------------------
from .utils import _chat_completion_content, _parse_llm_json  # noqa: E402


def _lazy_fitz():
    """Return the fitz (PyMuPDF) module, importing lazily and raising if absent."""
    global fitz
    if fitz is None:
        try:
            import fitz as _fitz  # type: ignore[import]
            fitz = _fitz
        except ImportError as exc:
            raise ImportError(
                "pymupdf (fitz) is required for PDF extraction. "
                "Install it with: pip install pymupdf"
            ) from exc
    return fitz


# ---------------------------------------------------------------------------
# B1 helper: fetch a PDF to a temp file
# ---------------------------------------------------------------------------

def fetch_pdf_to_temp(
    url: str,
    *,
    timeout: int = 120,
) -> Path:
    """Stream-download a PDF from *url* to a NamedTemporaryFile.

    Performs content-type and size sanity checks before writing.
    The caller is responsible for deleting the temp file when done.

    Parameters
    ----------
    url:
        HTTPS/HTTP URL pointing to a PDF file. Treated as untrusted input.
    timeout:
        Request timeout in seconds (applied to each chunk read, not total time).

    Returns
    -------
    Path
        Path to the temp file containing the PDF bytes.

    Raises
    ------
    ValueError
        If the content-type header does not indicate a PDF or octet-stream,
        or if the download exceeds the 200 MB size cap.
    requests.HTTPError
        If the server returns a non-2xx status code.
    """
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower().split(";")[0].strip()
    # Allow octet-stream for servers that don't set application/pdf.
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Unexpected content-type '{content_type}' when fetching PDF from {url!r}. "
            f"Expected one of: {sorted(_ALLOWED_CONTENT_TYPES)}"
        )

    # Write to a named temp file so callers can pass the Path to PyMuPDF.
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                total_bytes += len(chunk)
                if total_bytes > _MAX_PDF_BYTES:
                    raise ValueError(
                        f"PDF download from {url!r} exceeded the {_MAX_PDF_BYTES // (1024*1024)} MB size limit."
                    )
                tmp.write(chunk)
        tmp.flush()
        return Path(tmp.name)
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    finally:
        tmp.close()


# ---------------------------------------------------------------------------
# B2 helper: extract text chunks from a local PDF file
# ---------------------------------------------------------------------------

def extract_pdf_text_chunks(
    pdf_path: Path,
    *,
    chunk_chars: int = 8500,
    max_chunks: int | None = None,
) -> list[str]:
    """Extract text from *pdf_path* using PyMuPDF and split into chunks.

    Parameters
    ----------
    pdf_path:
        Path to a local PDF file (may be a temp file from fetch_pdf_to_temp).
    chunk_chars:
        Target character count per chunk. Splits are made on whitespace
        boundaries where possible to avoid cutting mid-word.
    max_chunks:
        If set and the number of extracted chunks exceeds this value, only
        the first *max_chunks* chunks are returned.  The cap is ALWAYS logged
        explicitly — never silently truncated.

    Returns
    -------
    list[str]
        List of text chunks, each at most *chunk_chars* characters.
        Returns an empty list if extraction fails (best-effort).
    """
    _fitz = _lazy_fitz()

    try:
        doc = _fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("pdf_extract: failed to open PDF %s: %s", pdf_path, exc)
        return []

    try:
        pages_text: list[str] = []
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text()
                if text:
                    pages_text.append(text)
            except Exception as exc:
                logger.warning(
                    "pdf_extract: failed to extract text from page %d of %s: %s",
                    page_num,
                    pdf_path,
                    exc,
                )
    finally:
        doc.close()

    if not pages_text:
        logger.warning("pdf_extract: no text extracted from %s", pdf_path)
        return []

    full_text = " ".join(pages_text)

    # Split into chunks of approximately chunk_chars characters.
    chunks: list[str] = []
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = start + chunk_chars
        if end >= text_len:
            # Last chunk.
            chunk = full_text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to split at a whitespace boundary to avoid cutting mid-word.
        split_pos = full_text.rfind(" ", start, end)
        if split_pos <= start:
            # No whitespace found in the window; hard split.
            split_pos = end

        chunk = full_text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        start = split_pos + 1

    total_chunks = len(chunks)

    if max_chunks is not None and total_chunks > max_chunks:
        logger.warning(
            "pdf_extract: max_chunks cap of %d hit for %s "
            "(total chunks extracted: %d). Only the first %d chunks will be processed. "
            "Some PDF content will NOT be included in the metadata extraction.",
            max_chunks,
            pdf_path,
            total_chunks,
            max_chunks,
        )
        chunks = chunks[:max_chunks]

    return chunks


# ---------------------------------------------------------------------------
# Local report PDF finder
# ---------------------------------------------------------------------------

def find_local_report_pdf(package_dir: Path) -> "Path | None":
    """Return the best candidate report PDF found under *package_dir*, or None.

    Search is purely filesystem — no network calls.  If *package_dir* does not
    exist, returns None gracefully.

    Ranking (highest wins):
      (a) PDFs inside a directory whose name (case-insensitive) is
          ``report`` or ``reports``.
      (b) PDFs whose filename (stem or full name, case-insensitive) contains
          ``report``.
      (c) Any remaining PDF, largest by file size.

    Within each tier, files are sorted for determinism (largest size first for
    tier c; lexicographic for tiers a and b).

    Parameters
    ----------
    package_dir:
        Root directory of a GAM model package (e.g. ``Yegua-Jackson_Aquifer_GAM/``).

    Returns
    -------
    Path or None
        Absolute path to the best PDF candidate, or None if none found.
    """
    if not package_dir or not Path(package_dir).exists():
        return None

    package_dir = Path(package_dir)

    tier_a: list[Path] = []  # inside a "report"/"reports" dir
    tier_b: list[Path] = []  # filename contains "report"
    tier_c: list[Path] = []  # any other PDF

    for pdf in package_dir.rglob("*"):
        if not pdf.is_file():
            continue
        if pdf.suffix.lower() != ".pdf":
            continue

        # Determine which tier this PDF belongs to.
        parent_name = pdf.parent.name.lower()
        if parent_name in ("report", "reports"):
            tier_a.append(pdf)
        elif "report" in pdf.name.lower():
            tier_b.append(pdf)
        else:
            tier_c.append(pdf)

    if tier_a:
        # Within tier a, prefer largest file; break ties lexicographically.
        return sorted(tier_a, key=lambda p: (-p.stat().st_size, str(p)))[0]
    if tier_b:
        return sorted(tier_b, key=lambda p: (-p.stat().st_size, str(p)))[0]
    if tier_c:
        return sorted(tier_c, key=lambda p: (-p.stat().st_size, str(p)))[0]

    return None


# ---------------------------------------------------------------------------
# MAP step: extract candidate metadata from one chunk
# ---------------------------------------------------------------------------

_MAP_SYSTEM_PROMPT = """\
You are extracting structured metadata from a section of a technical groundwater report.
Your job is to identify values for the requested fields from THIS chunk of text only.
If a field cannot be determined from this chunk, return null for that field.
DO NOT invent, guess, or extrapolate facts not stated in the text.
Return STRICT JSON only — no markdown, no comments, no trailing commas.
Include a "chunk_summary" key with a single sentence summarizing the main topic of this chunk.
"""


def extract_metadata_from_chunk(
    chunk: str,
    schema_hint: list[str],
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
) -> dict[str, Any]:
    """MAP step — extract candidate metadata fields from one PDF text chunk.

    Asks the LLM to extract candidate values for each field in *schema_hint*
    plus a short "chunk_summary" sentence.  The LLM is instructed NOT to invent
    facts — unknown fields must be returned as null.

    Parameters
    ----------
    chunk:
        A ~8500-char excerpt of PDF text.
    schema_hint:
        List of metadata field names the LLM should attempt to extract
        (e.g. ["title", "temporal_coverage_start", "author", ...]).
    llm_model / llm_api_key / llm_base_url:
        LLM endpoint configuration (forwarded to _chat_completion_content).

    Returns
    -------
    dict
        Keys are the *schema_hint* field names plus "chunk_summary".
        Values are extracted strings or null.
    """
    field_list = ", ".join(f'"{f}"' for f in schema_hint)
    user_payload: dict[str, Any] = {
        "fields_to_extract": schema_hint,
        "instructions": (
            f"Extract values for these fields: [{field_list}]. "
            "Return null for any field not clearly stated in the text. "
            'Also return "chunk_summary": one sentence describing this chunk\'s main topic. '
            "Output must be strict JSON."
        ),
        "chunk_text": chunk,
    }

    content = _chat_completion_content(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        system_prompt=_MAP_SYSTEM_PROMPT,
        user_payload=user_payload,
        temperature=0.1,
    )
    parsed = _parse_llm_json(content)

    # Ensure all requested fields are present in the result (set absent ones to None).
    result: dict[str, Any] = {"chunk_summary": parsed.get("chunk_summary") or ""}
    for field in schema_hint:
        result[field] = parsed.get(field)
    return result


# ---------------------------------------------------------------------------
# REDUCE step: consolidate per-chunk results into final metadata
# ---------------------------------------------------------------------------

_REDUCE_SYSTEM_PROMPT = """\
You are consolidating candidate metadata extracted from multiple chunks of a technical
groundwater report, plus context from the dataset landing page.
Your job is to merge the per-chunk candidates into a single best-estimate metadata record.
Rules:
- Prefer the most specific, most frequently occurring value across chunks.
- If chunks disagree on a field, choose the value with the strongest textual support.
- If no chunk provides a value for a field, return null.
- DO NOT invent, guess, or extrapolate facts not present in the source material.
- Mark any field that could not be determined with null and add a "_gap" annotation
  string explaining why (e.g., "_gap_title": "not found in any chunk").
Return STRICT JSON only — no markdown, no comments, no trailing commas.
"""


def consolidate_chunk_metadata(
    chunk_results: list[dict[str, Any]],
    landing_page_excerpt: str = "",
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
) -> dict[str, Any]:
    """REDUCE step — merge per-chunk candidate metadata into a final metadata dict.

    Parameters
    ----------
    chunk_results:
        List of dicts returned by extract_metadata_from_chunk — one per chunk.
    landing_page_excerpt:
        Optional 6000-char excerpt from the TWDB landing page, used as additional
        context for the consolidation LLM call.
    llm_model / llm_api_key / llm_base_url:
        LLM endpoint configuration.

    Returns
    -------
    dict
        Consolidated metadata dict with the same structure as the
        propose_ckan_dataset_metadata_with_llm output (dataset_name, dataset_title,
        dataset_notes, temporal_coverage_start, temporal_coverage_end, dataset_tags,
        plus any additional fields extracted from the PDF).
    """
    chunk_summaries = [r.get("chunk_summary", "") for r in chunk_results if r.get("chunk_summary")]

    user_payload: dict[str, Any] = {
        "instructions": (
            "Merge the per-chunk candidates below into a single metadata record. "
            "See system prompt for merge rules. "
            "Output strict JSON matching CKAN dataset metadata field names."
        ),
        "chunk_summaries": chunk_summaries,
        "per_chunk_candidates": chunk_results,
        "landing_page_excerpt": landing_page_excerpt or "",
    }

    content = _chat_completion_content(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        system_prompt=_REDUCE_SYSTEM_PROMPT,
        user_payload=user_payload,
        temperature=0.1,
    )
    return _parse_llm_json(content)


# ---------------------------------------------------------------------------
# Orchestrator: run the full map-reduce pipeline over a PDF
# ---------------------------------------------------------------------------

def run_pdf_map_reduce(
    pdf_path: Path,
    landing_page_excerpt: str = "",
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
    schema_hint: list[str] | None = None,
    chunk_chars: int = 8500,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Run the full MAP → REDUCE pipeline over a PDF file.

    Extracts text chunks, runs one LLM MAP call per chunk (with a configurable
    inter-call sleep), then runs one REDUCE call to consolidate all chunk results.

    Parameters
    ----------
    pdf_path:
        Path to a local PDF file.
    landing_page_excerpt:
        Optional landing-page context passed to the REDUCE step.
    llm_model / llm_api_key / llm_base_url:
        LLM endpoint configuration.
    schema_hint:
        Field names to extract.  Defaults to the standard CKAN/SUBSIDE fields.
    chunk_chars:
        Target chunk size in characters.
    max_chunks:
        Cap on number of chunks processed.  Hitting the cap is always logged.

    Returns
    -------
    dict
        Consolidated metadata dict from the REDUCE step.
    """
    if schema_hint is None:
        schema_hint = [
            "dataset_title",
            "dataset_notes",
            "dataset_author",
            "dataset_version",
            "temporal_coverage_start",
            "temporal_coverage_end",
            "dataset_tags",
            "spatial",
            "collection_method",
            "quality_control_level",
            "program_area",
            "categories",
        ]

    chunks = extract_pdf_text_chunks(pdf_path, chunk_chars=chunk_chars, max_chunks=max_chunks)
    if not chunks:
        logger.warning("pdf_extract: no text chunks extracted from %s; skipping map-reduce.", pdf_path)
        return {}

    chunk_results: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(_LLM_CALL_DELAY_SECONDS)
        logger.debug("pdf_extract: MAP call %d/%d for %s", i + 1, len(chunks), pdf_path)
        result = extract_metadata_from_chunk(
            chunk,
            schema_hint,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
        chunk_results.append(result)

    # Sleep before the REDUCE call.
    time.sleep(_LLM_CALL_DELAY_SECONDS)
    logger.debug("pdf_extract: REDUCE call for %s (%d chunks)", pdf_path, len(chunks))
    return consolidate_chunk_metadata(
        chunk_results,
        landing_page_excerpt=landing_page_excerpt,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )
