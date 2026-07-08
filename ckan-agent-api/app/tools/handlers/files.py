"""File-extractor tool handlers.

Each wraps a migrated extractor and runs ``validate_readable_file`` first (size limit +
sensitive-path refusal, spec S-4), so every file tool is safety-checked before reading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.files.extractors import archive, image, json_data, pdf, spatial, tabular, text
from app.files.safety import validate_readable_file


def _safe_path(args: dict[str, Any]) -> Path:
    return validate_readable_file(str(args["path"])).path


def read_text(args: dict[str, Any]) -> Any:
    path = _safe_path(args)
    max_chars = int(args.get("max_chars", 4000))
    # For notebooks, return the readable cell sources (markdown + code) instead of the raw
    # escaped notebook JSON, so the author can scan the actual code for bbox/date/CRS values.
    if path.suffix.lower() == ".ipynb" or json_data.looks_like_notebook(path):
        return json_data.read_notebook_source(path, max_chars=max(max_chars, 6000))
    return text.read_text_sample(path, max_chars=max_chars)


def profile_csv(args: dict[str, Any]) -> Any:
    return tabular.profile_csv(_safe_path(args), max_rows=int(args.get("max_rows", 20)))


def profile_json(args: dict[str, Any]) -> Any:
    return json_data.profile_json(_safe_path(args), max_sample_chars=int(args.get("max_sample_chars", 4000)))


def profile_geojson(args: dict[str, Any]) -> Any:
    return json_data.profile_geojson(_safe_path(args), max_sample_chars=int(args.get("max_sample_chars", 4000)))


def extract_pdf_text(args: dict[str, Any]) -> Any:
    return pdf.extract_pdf_text(
        _safe_path(args),
        page_start=int(args.get("page_start", 0)),
        max_pages=int(args.get("max_pages", 5)),
        max_chars=int(args.get("max_chars", 6000)),
    )


def inspect_image(args: dict[str, Any]) -> Any:
    return image.inspect_image(_safe_path(args))


def inspect_zip(args: dict[str, Any]) -> Any:
    return archive.inspect_zip(_safe_path(args), max_members=int(args.get("max_members", 100)))


def profile_raster(args: dict[str, Any]) -> Any:
    return spatial.profile_raster(_safe_path(args))


def profile_shapefile_zip(args: dict[str, Any]) -> Any:
    return spatial.profile_shapefile_zip(_safe_path(args))
