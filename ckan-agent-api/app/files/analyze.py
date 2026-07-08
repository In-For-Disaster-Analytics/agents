"""Extension-routed file analysis for the persona author's evidence.

Migrated from ``basic-ckan-agent`` (spec Increment 4). Each supplied file is passed
through ``safety.validate_readable_file`` first (size limit + sensitive-path refusal,
spec S-4) and then routed to the appropriate extractor. Heavy/optional extractor
dependencies (pypdf, rasterio, fiona) are imported lazily inside the extractors and
degrade gracefully when absent.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from app.files.extractors.image import inspect_image
from app.files.extractors.json_data import profile_geojson, profile_json
from app.files.extractors.pdf import extract_pdf_text
from app.files.extractors.spatial import profile_raster, profile_shapefile_zip
from app.files.extractors.tabular import profile_csv
from app.files.extractors.text import read_text_sample
from app.files.safety import FileSafetyError, validate_readable_file

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".xml", ".html", ".htm",
    ".py", ".yaml", ".yml", ".ini", ".cfg", ".sql",
}
MAX_DIRECTORY_FILES = 50


def analyze_path(path: Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Analyze a single file into a report dict. Never raises."""
    report: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "extension": path.suffix.lower(),
        "format": (path.suffix.lower().lstrip(".") or "").upper() or "UNKNOWN",
    }
    try:
        safe = validate_readable_file(str(path), base_dir=base_dir)
    except FileSafetyError as exc:
        report["error"] = {"code": exc.code, "message": exc.message}
        return report

    report["size_bytes"] = safe.size_bytes
    report["mime_type"] = safe.mime_type
    ext = safe.extension
    p = safe.path
    try:
        if ext in {".csv", ".tsv"}:
            report["tabular"] = profile_csv(p, max_rows=20)
        elif ext == ".geojson":
            report["geojson"] = profile_geojson(p, max_sample_chars=4000)
        elif ext in {".json", ".ipynb"}:
            report["json"] = profile_json(p, max_sample_chars=4000)
        elif ext == ".pdf":
            # Read enough to get past cover/TOC into the abstract/intro of longer reports.
            report["pdf"] = extract_pdf_text(p, page_start=0, max_pages=12, max_chars=12000)
        elif ext in {".tif", ".tiff"}:
            report["raster"] = profile_raster(p)
        elif ext in {".jpg", ".jpeg", ".png", ".gif"}:
            report["image"] = inspect_image(p)
        elif ext == ".zip":
            report["archive"] = profile_shapefile_zip(p)
        elif _looks_like_json(p):
            # JSON/ipynb saved with a .txt (or other) extension — parse by content, not name.
            report["json"] = profile_json(p, max_sample_chars=4000)
        elif ext in TEXT_SUFFIXES or safe.mime_type.startswith("text/"):
            report["text"] = read_text_sample(p, max_chars=4000)
    except Exception as exc:  # noqa: BLE001 - analysis is best-effort evidence
        report["parse_warning"] = str(exc)
    return report


def _looks_like_json(path: Path) -> bool:
    """True if the file's first non-whitespace byte is '{' or '[' (likely JSON/ipynb)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(2048)
    except OSError:
        return False
    stripped = head.lstrip()
    return stripped[:1] in (b"{", b"[")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _ref_path(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("path", "local_path", "file_path", "filepath", "upload_path"):
            if item.get(key):
                return str(item[key])
    return ""


def _resolve_file_references(request: dict[str, Any], base_dir: Path | None) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            paths.append(p)

    for item in _as_list(request.get("files")) + _as_list(request.get("uploaded_files")):
        raw = _ref_path(item)
        if raw:
            add(_candidate(raw, base_dir))

    for raw_dir in _as_list(request.get("upload_dir")) + _as_list(request.get("upload_dirs")):
        directory = _candidate(str(raw_dir), base_dir)
        if not directory.is_dir():
            add(directory)
            continue
        children = sorted(c for c in directory.rglob("*") if c.is_file())
        if len(children) > MAX_DIRECTORY_FILES:
            warnings.append(
                f"Directory {directory} has {len(children)} files; analyzing the first {MAX_DIRECTORY_FILES}."
            )
            children = children[:MAX_DIRECTORY_FILES]
        for child in children:
            add(child)
    return paths, warnings


def _candidate(raw: str, base_dir: Path | None) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute() and base_dir is not None:
        return base_dir / p
    return p


def _analyze_inline_files(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Analyze pasted/attached content (chat ``inline_files`` / ``file_contents``).

    Each item carries the file's content directly (not a path), so we materialize it to a
    temp file with the right suffix and run the normal extractor dispatch — this is what
    lets an attached notebook/CSV/PDF actually reach the author as structured evidence.
    """
    reports: list[dict[str, Any]] = []
    items = _as_list(request.get("inline_files")) + _as_list(request.get("file_contents"))
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("text") or item.get("data")
        if content is None:
            continue
        name = str(item.get("name") or item.get("filename") or "pasted.txt")
        suffix = Path(name).suffix or ".txt"
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as handle:
                handle.write(str(content))
                tmp = Path(handle.name)
            report = analyze_path(tmp)
            report["name"] = name  # show the real name, not the temp path
            report["inline"] = True
            reports.append(report)
        except Exception:  # noqa: BLE001 - best-effort evidence
            continue
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return reports


def analyze_request_files(
    request: dict[str, Any], *, base_dir: Path | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve file/upload-dir references AND inline content from a request and analyze each."""
    paths, warnings = _resolve_file_references(request, base_dir)
    reports = [analyze_path(p, base_dir=base_dir) for p in paths]
    reports.extend(_analyze_inline_files(request))
    return reports, warnings


def gather_file_evidence(
    request: dict[str, Any], *, base_dir: Path | None = None, deep_threshold: int = 3
) -> dict[str, Any]:
    """Evidence for the (tool-calling) author.

    Always returns a cheap ``file_heads`` inventory of every supplied/extracted file (with its
    path, so the author can deep-review via tools). When the number of on-disk files is small
    (<= ``deep_threshold``) they are also fully analyzed up front into ``file_reports``; above
    the threshold the author deep-reviews the most informative ones itself via the file/GDAL
    tools. Inline (pasted) content is always fully analyzed (it has no persistent path to tool-call).
    """
    paths, warnings = _resolve_file_references(request, base_dir)
    heads = build_head_inventory(paths)
    inline_reports = _analyze_inline_files(request)
    if len(paths) <= deep_threshold:
        reports = [analyze_path(p, base_dir=base_dir) for p in paths] + inline_reports
    else:
        reports = inline_reports
    return {"file_heads": heads, "file_reports": reports, "file_warnings": warnings}


_KIND_BY_EXT = {
    ".ipynb": "notebook", ".csv": "tabular", ".tsv": "tabular",
    ".json": "json", ".geojson": "geojson", ".pdf": "pdf",
    ".tif": "raster", ".tiff": "raster", ".shp": "vector",
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image", ".zip": "archive",
}


def _kind(ext: str, mime: str) -> str:
    if ext in _KIND_BY_EXT:
        return _KIND_BY_EXT[ext]
    if ext in TEXT_SUFFIXES or mime.startswith("text/"):
        return "text"
    return "binary"


def _head_preview(path: Path) -> str:
    """A cheap text head (first ~600 chars) for triage; empty for binary."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(2048)
    except OSError:
        return ""
    if b"\x00" in raw:
        return ""
    return raw.decode("utf-8", errors="replace")[:600]


def build_head_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    """Cheap per-file heads for agent triage — name/size/kind + a short preview, NOT a deep parse.

    The tool-calling author uses this to decide which files to deep-review with the file/GDAL tools.
    """
    import mimetypes

    inventory: list[dict[str, Any]] = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        ext = path.suffix.lower()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        entry: dict[str, Any] = {
            "name": path.name,
            "path": str(path),
            "extension": ext,
            "size_bytes": size,
            "kind": _kind(ext, mime),
        }
        head = _head_preview(path)
        if head:
            entry["head"] = head
        inventory.append(entry)
    return inventory


def build_file_inventory(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact inventory (counts + filenames) for the author's temporal/format inference."""
    names = [r.get("name", "") for r in reports]
    ext_counts = Counter((r.get("extension") or "[none]") for r in reports)
    inventory: dict[str, Any] = {
        "file_count": len(reports),
        "extension_counts": dict(ext_counts),
        "filenames": names[:200],
    }
    if len(names) > 200:
        inventory["filenames_truncated"] = True
    return inventory
