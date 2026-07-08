from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from basic_ckan_agent.files.catalog import description_for_tool
from basic_ckan_agent.files.extractors.archive import inspect_zip
from basic_ckan_agent.files.extractors.image import inspect_image
from basic_ckan_agent.files.extractors.json_data import profile_geojson, profile_json
from basic_ckan_agent.files.extractors.pdf import extract_pdf_text
from basic_ckan_agent.files.extractors.spatial import profile_raster, profile_shapefile_zip
from basic_ckan_agent.files.extractors.tabular import profile_csv
from basic_ckan_agent.files.extractors.text import read_text_sample
from basic_ckan_agent.files.results import tool_error, tool_success
from basic_ckan_agent.files.safety import (
    FileSafetyError,
    normalize_allowed_paths,
    resolve_user_path,
    validate_readable_file,
)
from basic_ckan_agent.files.schemas import (
    CsvProfileInput,
    FilePathInput,
    JsonProfileInput,
    PdfTextInput,
    TextFileInput,
    ZipInspectInput,
)
from basic_ckan_agent.logging_config import debug_print
from basic_ckan_agent.utils import safe_json_dumps


def build_file_tools(allowed_paths: list[str] | None = None, *, base_dir: Path | None = None) -> list[StructuredTool]:
    allowed = normalize_allowed_paths(allowed_paths, base_dir=base_dir)

    def file_stat(path: str) -> str:
        return _execute("file_stat", lambda: _stat_path(path, allowed_paths=allowed, base_dir=base_dir))

    def file_read_text(path: str, max_chars: int = 8000) -> str:
        return _execute(
            "file_read_text",
            lambda: read_text_sample(
                validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path,
                max_chars=max_chars,
            ),
        )

    def file_extract_pdf_text(
        path: str,
        page_start: int = 0,
        max_pages: int = 5,
        max_chars: int = 12000,
    ) -> str:
        return _execute(
            "file_extract_pdf_text",
            lambda: extract_pdf_text(
                validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path,
                page_start=page_start,
                max_pages=max_pages,
                max_chars=max_chars,
            ),
        )

    def file_profile_csv(path: str, max_rows: int = 25) -> str:
        return _execute(
            "file_profile_csv",
            lambda: profile_csv(
                validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path,
                max_rows=max_rows,
            ),
        )

    def file_profile_json(path: str, max_sample_chars: int = 12000) -> str:
        return _execute(
            "file_profile_json",
            lambda: profile_json(
                validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path,
                max_sample_chars=max_sample_chars,
            ),
        )

    def file_profile_geojson(path: str, max_sample_chars: int = 12000) -> str:
        return _execute(
            "file_profile_geojson",
            lambda: profile_geojson(
                validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path,
                max_sample_chars=max_sample_chars,
            ),
        )

    def file_inspect_image(path: str) -> str:
        return _execute(
            "file_inspect_image",
            lambda: inspect_image(validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path),
        )

    def file_inspect_zip(path: str, max_members: int = 100) -> str:
        return _execute(
            "file_inspect_zip",
            lambda: inspect_zip(
                validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path,
                max_members=max_members,
            ),
        )

    def file_profile_raster(path: str) -> str:
        return _execute(
            "file_profile_raster",
            lambda: profile_raster(validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path),
        )

    def file_profile_shapefile_zip(path: str) -> str:
        return _execute(
            "file_profile_shapefile_zip",
            lambda: profile_shapefile_zip(validate_readable_file(path, allowed_paths=allowed, base_dir=base_dir).path),
        )

    return [
        _structured_tool(file_stat, "file_stat", FilePathInput),
        _structured_tool(file_read_text, "file_read_text", TextFileInput),
        _structured_tool(file_extract_pdf_text, "file_extract_pdf_text", PdfTextInput),
        _structured_tool(file_profile_csv, "file_profile_csv", CsvProfileInput),
        _structured_tool(file_profile_json, "file_profile_json", JsonProfileInput),
        _structured_tool(file_profile_geojson, "file_profile_geojson", JsonProfileInput),
        _structured_tool(file_inspect_image, "file_inspect_image", FilePathInput),
        _structured_tool(file_inspect_zip, "file_inspect_zip", ZipInspectInput),
        _structured_tool(file_profile_raster, "file_profile_raster", FilePathInput),
        _structured_tool(file_profile_shapefile_zip, "file_profile_shapefile_zip", FilePathInput),
    ]


def _structured_tool(func: Callable[..., str], name: str, args_schema: type) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description_for_tool(name, f"Local file metadata tool: {name}"),
        args_schema=args_schema,
    )


def _execute(tool_name: str, handler: Callable[[], dict[str, Any]]) -> str:
    debug_print(f"File tool called: {tool_name}", {})
    try:
        result = handler()
        warnings = result.pop("warnings", None)
        payload = tool_success(tool_name, result, warnings=warnings if isinstance(warnings, list) else None)
    except FileSafetyError as exc:
        payload = tool_error(tool_name, exc.code, exc.message)
    except json.JSONDecodeError as exc:
        payload = tool_error(tool_name, "invalid_json", str(exc))
    except UnicodeDecodeError as exc:
        payload = tool_error(tool_name, "decode_error", str(exc))
    except Exception as exc:
        payload = tool_error(tool_name, "tool_exception", str(exc))
    debug_print(f"File tool result: {tool_name}", payload)
    return safe_json_dumps(payload)


def _stat_path(path: str, *, allowed_paths: set[str] | None, base_dir: Path | None) -> dict[str, Any]:
    resolved = resolve_user_path(path, allowed_paths=allowed_paths, base_dir=base_dir)
    exists = resolved.exists()
    if not exists:
        return {
            "path": path,
            "resolved_path": resolved.as_posix(),
            "exists": False,
        }

    if not resolved.is_file():
        return {
            "path": path,
            "resolved_path": resolved.as_posix(),
            "exists": True,
            "is_file": False,
            "message": "Path exists but is not a regular file.",
        }

    safe_file = validate_readable_file(path, allowed_paths=allowed_paths, base_dir=base_dir)
    stat = safe_file.path.stat()
    return {
        "path": path,
        "resolved_path": safe_file.path.as_posix(),
        "exists": True,
        "is_file": True,
        "size_bytes": safe_file.size_bytes,
        "extension": safe_file.extension,
        "mime_type": safe_file.mime_type,
        "modified_time": stat.st_mtime,
    }
