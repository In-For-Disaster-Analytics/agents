#!/usr/bin/env python3
"""CKAN registration worker CLI.

This module wraps the standalone CKAN helpers in utils.py with a small JSON
command interface (analyze / dry-run / revise / apply / show commands).
It is callable from a notebook, a shell script, or any orchestration layer.

For the GAM-specific SUBSIDE registration pipeline (Capabilities A/B/D),
see orchestrate.py which wires together discovery, TWDB enrichment, PDF
map-reduce, persona loop, and SUBSIDE field mapping into a single
per-model function (run_registration) and a manifest-level loop
(run_manifest_registration).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from . import utils as u  # noqa: E402


STATE_SCHEMA_VERSION = 1
DEFAULT_STATE_DIR = Path(os.getenv("CKAN_AGENT_STATE_DIR", "/tmp/ckan-registration-agent"))

TEXT_EXTENSIONS = {
    ".csv",
    ".geojson",
    ".html",
    ".htm",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sql",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

ARTIFACT_REFERENCE_RE = re.compile(
    r"\b("
    r"attach(?:ed|ment)?|base data|csv|data file|file|ipynb|jupyter|local file|"
    r"notebook|python file|script|uploaded?"
    r")\b",
    re.IGNORECASE,
)
NO_EXISTING_DATASET_RE = re.compile(
    r"\b("
    r"do(?:n't| not) have (?:an? )?(?:existing )?(?:ckan )?dataset|"
    r"no (?:existing )?(?:ckan )?dataset|"
    r"does(?:n't| not) exist(?: in ckan)?|"
    r"new (?:ckan )?dataset"
    r")\b",
    re.IGNORECASE,
)
DATASET_DETAIL_RE = re.compile(
    r"\b(title|name|notes|description|author|maintainer|license|tag|spatial|temporal|resource|owner[_ -]?org)\s*[:=]",
    re.IGNORECASE,
)

DATASET_FIELD_ALIASES = {
    "dataset_name": "name",
    "dataset_title": "title",
    "dataset_notes": "notes",
    "dataset_url": "url",
    "dataset_author": "author",
    "dataset_author_email": "author_email",
    "dataset_maintainer": "maintainer",
    "dataset_maintainer_email": "maintainer_email",
    "dataset_license_id": "license_id",
    "dataset_version": "version",
    "dataset_type": "type",
    "dataset_isopen": "isopen",
    "dataset_spatial": "spatial",
}

ALLOWED_DATASET_FIELDS = {
    "name",
    "title",
    "notes",
    "url",
    "owner_org",
    "private",
    "author",
    "author_email",
    "maintainer",
    "maintainer_email",
    "license_id",
    "version",
    "type",
    "isopen",
    "spatial",
    "temporal_coverage_start",
    "temporal_coverage_end",
    "tags",
}

SENSITIVE_TRACE_KEYS = {
    "authorization",
    "ckan-api-token",
    "ckan_api_token",
    "ckan-password",
    "ckan_password",
    "cookie",
    "openai-api-key",
    "openai_api_key",
    "password",
    "request-headers",
    "request_headers",
    "secret",
    "token",
}
TRACE_VALUE_MAX_CHARS = 500
TRACE_LIST_MAX_ITEMS = 20


class AgentError(RuntimeError):
    """Expected user/configuration error returned as JSON."""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def is_sensitive_trace_key(key: object) -> bool:
    text = str(key).strip().lower().replace("_", "-")
    return text in SENSITIVE_TRACE_KEYS or any(part in text for part in ("password", "secret", "token", "api-key"))


def trace_value(value: Any, *, max_chars: int = TRACE_VALUE_MAX_CHARS) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = "<redacted>" if is_sensitive_trace_key(key) else trace_value(item, max_chars=max_chars)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [trace_value(item, max_chars=max_chars) for item in items[:TRACE_LIST_MAX_ITEMS]]
        if len(items) > TRACE_LIST_MAX_ITEMS:
            out.append({"truncated_items": len(items) - TRACE_LIST_MAX_ITEMS})
        return out
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return u.clean_text(value, max_chars)


def trace_event(trace: list[dict[str, Any]] | None, step: str, **details: Any) -> None:
    if trace is None:
        return
    event: dict[str, Any] = {"at": now_iso(), "step": step}
    for key, value in details.items():
        event[key] = trace_value(value)
    trace.append(event)


def dataset_override_keys(request: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    dataset = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
    keys.update(str(key) for key in dataset)
    for key in set(DATASET_FIELD_ALIASES) | ALLOWED_DATASET_FIELDS:
        if key in request:
            keys.add(str(key))
    return sorted(keys)


def request_source_summary(request: dict[str, Any]) -> dict[str, Any]:
    upload_dirs = as_list(request.get("upload_dir")) + as_list(request.get("upload_dirs"))
    file_items = as_list(request.get("files")) + as_list(request.get("uploaded_files"))
    file_refs: list[str] = []
    for item in file_items:
        if isinstance(item, dict):
            value = item.get("path") or item.get("local_path") or item.get("file_path") or item.get("name")
        else:
            value = item
        if value is not None:
            file_refs.append(str(value))

    metadata_keys = []
    for key in request:
        if is_sensitive_trace_key(key):
            continue
        metadata_keys.append(str(key))

    message = u.clean_text(request.get("message"), 240)
    source_urls = as_list(request.get("source_url")) + as_list(request.get("source_urls"))
    return {
        "message_present": bool(message),
        "message_excerpt": message,
        "metadata_keys": sorted(metadata_keys),
        "source_url_count": len([value for value in source_urls if u.clean_text(value)]),
        "source_urls": [u.clean_text(value, 240) for value in source_urls if u.clean_text(value)],
        "upload_dir_count": len(upload_dirs),
        "upload_dirs": [str(value) for value in upload_dirs],
        "file_ref_count": len(file_refs),
        "file_refs": file_refs,
        "dataset_override_fields": dataset_override_keys(request),
        "client_context": request.get("agent_context") or {},
    }


def response_should_include_trace(request: dict[str, Any]) -> bool:
    return parse_bool(
        request.get("debug_trace"),
        parse_bool(os.getenv("CKAN_AGENT_TRACE_RESPONSE"), False),
    )


def has_file_inputs(request: dict[str, Any]) -> bool:
    return bool(as_list(request.get("upload_dir")) or as_list(request.get("upload_dirs")) or as_list(request.get("files")) or as_list(request.get("uploaded_files")))


def has_source_inputs(request: dict[str, Any]) -> bool:
    return any(u.clean_text(value) for value in as_list(request.get("source_url")) + as_list(request.get("source_urls")))


def has_existing_dataset_input(request: dict[str, Any]) -> bool:
    return bool(u.clean_text(request.get("existing_ckan_entry") or request.get("existing_dataset")))


def has_dataset_override_inputs(request: dict[str, Any]) -> bool:
    return bool(dataset_override_keys(request))


def message_has_dataset_details(message: str) -> bool:
    text = u.clean_text(message, 4000)
    if DATASET_DETAIL_RE.search(text):
        return True
    if len(text) >= 160 and re.search(r"\b(dataset|data|measurements|observations|model|campaign|survey)\b", text, re.I):
        return True
    return False


def analyze_preflight_issue(request: dict[str, Any]) -> dict[str, Any] | None:
    message = u.clean_text(request.get("message"), 4000)
    has_files = has_file_inputs(request)
    has_source = has_source_inputs(request)
    has_existing = has_existing_dataset_input(request)
    has_overrides = has_dataset_override_inputs(request)
    allow_metadata_only = parse_bool(request.get("allow_metadata_only"), False)

    if has_files or has_source or has_existing or has_overrides or allow_metadata_only:
        return None

    if message and ARTIFACT_REFERENCE_RE.search(message):
        return {
            "code": "missing_readable_files",
            "message": (
                "I can see that the request mentions a notebook, script, attachment, or data file, "
                "but the CKAN worker did not receive a readable file path or upload directory."
            ),
            "next_steps": [
                "Send metadata.files with local file paths the API service can read.",
                "Or send metadata.upload_dir pointing to the staged notebook/data directory.",
                "Or send metadata.source_url if the dataset can be analyzed from a source URL.",
            ],
        }

    if not message or NO_EXISTING_DATASET_RE.search(message) or len(message) < 80:
        return {
            "code": "insufficient_dataset_input",
            "message": (
                "I do not have enough dataset input to analyze. A missing existing CKAN dataset is fine; "
                "that means this should be prepared as a new CKAN dataset, but I still need source files, "
                "a source URL, an upload directory, or explicit dataset metadata."
            ),
            "next_steps": [
                "Provide files, uploaded_files, upload_dir, or upload_dirs.",
                "Or provide source_url/source_urls.",
                "Or provide dataset overrides such as dataset.title and dataset.notes.",
                "For an intentional metadata-only record, set allow_metadata_only=true.",
            ],
        }

    if not message_has_dataset_details(message):
        return {
            "code": "missing_dataset_details",
            "message": (
                "The request has no files, source URL, existing CKAN entry, or explicit dataset fields, "
                "and the chat text does not contain enough dataset details for a safe metadata-only proposal."
            ),
            "next_steps": [
                "Provide a readable local path/upload_dir, source URL, or explicit dataset metadata.",
                "For an intentional metadata-only record, set allow_metadata_only=true.",
            ],
        }

    return None


def build_needs_input_markdown(issue: dict[str, Any]) -> str:
    lines = ["## More Dataset Input Needed", "", str(issue.get("message") or "More input is required.")]
    next_steps = [str(item) for item in as_list(issue.get("next_steps")) if u.clean_text(item)]
    if next_steps:
        lines.extend(["", "### Next Steps"])
        lines.extend([f"- {step}" for step in next_steps])
    return "\n".join(lines)


def needs_input_response(session_id: str, issue: dict[str, Any], trace: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    result = {
        "ok": False,
        "command": "analyze",
        "status": "needs_input",
        "session_id": session_id,
        "error": issue.get("message"),
        "issue": issue,
        "review_markdown": build_needs_input_markdown(issue),
    }
    if response_should_include_trace(request):
        result["trace"] = trace
    return result


def load_env_file(path: Path | None, *, override: bool = False) -> None:
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def normalized_headers(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    return {str(key).strip().lower().replace("_", "-"): str(value).strip() for key, value in headers.items() if value is not None}


def header_lookup(headers: dict[str, str], *names: str) -> str:
    for name in names:
        normalized = name.strip().lower().replace("_", "-")
        value = headers.get(normalized)
        if value:
            return value
    return ""


def apply_secret_headers(request: dict[str, Any]) -> None:
    """Map n8n/webhook headers to environment variables without saving them."""

    headers = normalized_headers(request.get("headers") or request.get("request_headers"))
    if not headers:
        return

    ckan_api_token = header_lookup(headers, "CKAN_API_TOKEN", "ckan-api-token", "x-ckan-api-token")
    openai_api_key = header_lookup(headers, "OPENAI_API_KEY", "openai-api-key", "x-openai-api-key")
    ckan_auth_mode = header_lookup(headers, "CKAN_AUTH_MODE", "ckan-auth-mode", "x-ckan-auth-mode")
    openai_base_url = header_lookup(headers, "OPENAI_BASE_URL", "openai-base-url", "x-openai-base-url")
    ckan_llm_model = header_lookup(headers, "CKAN_LLM_MODEL", "ckan-llm-model", "x-ckan-llm-model")

    if ckan_api_token:
        os.environ["CKAN_API_TOKEN"] = ckan_api_token
        os.environ["CKAN_AUTH_MODE"] = ckan_auth_mode or "api_token"
    elif ckan_auth_mode:
        os.environ["CKAN_AUTH_MODE"] = ckan_auth_mode

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if openai_base_url:
        os.environ["OPENAI_BASE_URL"] = openai_base_url
    if ckan_llm_model:
        os.environ["CKAN_LLM_MODEL"] = ckan_llm_model


def load_json_input(path: str | None) -> dict[str, Any]:
    if not path or path == "-":
        text = sys.stdin.read()
        return json.loads(text) if text.strip() else {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AgentError("Input JSON must be an object.")
    return payload


def load_json_input_b64(value: str) -> dict[str, Any]:
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True).decode("utf-8")
        payload = json.loads(decoded) if decoded.strip() else {}
    except Exception as exc:
        raise AgentError(f"Could not decode --input-b64 JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentError("--input-b64 JSON must decode to an object.")
    return payload


def write_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def sanitize_session_id(value: str | None) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return text or uuid.uuid4().hex


def state_path(state_dir: Path, session_id: str) -> Path:
    return state_dir / f"{sanitize_session_id(session_id)}.json"


def save_state(state: dict[str, Any], state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    path = state_path(state_dir, state["session_id"])
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return path


def load_state(input_payload: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    explicit_path = input_payload.get("state_path")
    if explicit_path:
        path = Path(str(explicit_path)).expanduser()
    else:
        session_id = sanitize_session_id(input_payload.get("session_id"))
        path = state_path(state_dir, session_id)
    if not path.exists():
        raise AgentError(f"No saved CKAN agent state found at {path}.")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise AgentError(f"State file is not a JSON object: {path}")
    return state


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def resolve_input_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise AgentError(f"Input file or directory does not exist: {path}")
    return path.resolve()


def collect_file_records(request: dict[str, Any], trace: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    max_files = int(request.get("max_files") or os.getenv("MAX_FILES", "5000"))
    upload_inputs = as_list(request.get("upload_dir")) + as_list(request.get("upload_dirs"))
    file_inputs = as_list(request.get("files")) + as_list(request.get("uploaded_files"))

    trace_event(
        trace,
        "files.collect.start",
        upload_dir_count=len(upload_inputs),
        file_ref_count=len(file_inputs),
        max_files=max_files,
    )

    def add_path(path: Path, metadata: dict[str, Any] | None = None, *, explicit: bool = False) -> None:
        metadata = metadata or {}
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if len(records) >= max_files:
                    return
                if not child.is_file():
                    continue
                if is_hidden_path(child.relative_to(path)):
                    continue
                add_path(child, {"root": str(path)}, explicit=False)
            return

        if not path.is_file():
            return
        if not explicit and path.name.startswith("."):
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        records.append(
            {
                "path": resolved,
                "provided_name": metadata.get("name") or metadata.get("filename"),
                "description": metadata.get("description") or metadata.get("notes"),
                "root": metadata.get("root"),
            }
        )

    for directory in upload_inputs:
        add_path(resolve_input_path(directory), explicit=True)

    for item in file_inputs:
        if isinstance(item, dict):
            raw_path = item.get("path") or item.get("local_path") or item.get("file_path")
            if not raw_path:
                continue
            add_path(resolve_input_path(raw_path), item, explicit=True)
        else:
            add_path(resolve_input_path(item), explicit=True)

    trace_event(
        trace,
        "files.collect.done",
        collected_count=len(records),
        sample_paths=[str(record["path"]) for record in records[:10]],
    )
    return records[:max_files]


def common_root(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    try:
        return Path(os.path.commonpath([str(path.parent if path.is_file() else path) for path in paths]))
    except ValueError:
        return None


def relative_name(path: Path, root: Path | None) -> str:
    if root:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_ipynb_preview(path: Path, max_chars: int) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells") if isinstance(payload, dict) else []
    parts: list[str] = []
    for cell in cells[:40]:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(line) for line in source)
        if source:
            parts.append(str(source))
        for output in cell.get("outputs", []) or []:
            if not isinstance(output, dict):
                continue
            text = output.get("text")
            if isinstance(text, list):
                text = "".join(str(line) for line in text)
            if text:
                parts.append(str(text))
    return u.clean_text(" ".join(parts), max_chars=max_chars)


def extract_docx_preview(path: Path, max_chars: int) -> str:
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as handle:
            xml = handle.read()
    text = decode_text_bytes(xml)
    text = re.sub(r"<[^>]+>", " ", text)
    return u.clean_text(text, max_chars=max_chars)


def extract_pdf_preview(path: Path, max_chars: int) -> str:
    reader_cls = None
    try:
        from pypdf import PdfReader  # type: ignore

        reader_cls = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader_cls = PdfReader
        except Exception:
            return ""

    reader = reader_cls(str(path))
    parts = []
    for page in reader.pages[:6]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return u.clean_text(" ".join(parts), max_chars=max_chars)


def extract_text_preview(path: Path, max_chars: int = 1800) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_EXTENSIONS:
            with path.open("rb") as handle:
                return u.clean_text(decode_text_bytes(handle.read(max_chars * 4)), max_chars=max_chars)
        if suffix == ".ipynb":
            return extract_ipynb_preview(path, max_chars)
        if suffix == ".docx":
            return extract_docx_preview(path, max_chars)
        if suffix == ".pdf":
            return extract_pdf_preview(path, max_chars)
    except Exception:
        return ""
    return ""


def infer_tags(path: Path, preview: str) -> list[str]:
    tags = []
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        tags.append(suffix)

    text = f"{path.name} {preview}".lower()
    keyword_tags = {
        "groundwater": "groundwater",
        "aquifer": "aquifer",
        "modflow": "modflow",
        "subsidence": "subsidence",
        "notebook": "notebook",
        "jupyter": "notebook",
        "geojson": "geojson",
        "csv": "csv",
        "raster": "raster",
        "timeseries": "time-series",
        "time series": "time-series",
        "model": "model",
    }
    for keyword, tag in keyword_tags.items():
        if keyword in text:
            tags.append(tag)
    return [tag["name"] for tag in u.dedupe_tags(tags)]


def build_resource_plan(
    request: dict[str, Any],
    trace: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    records = collect_file_records(request, trace)
    paths = [record["path"] for record in records]
    root = common_root(paths)
    source_urls = extract_source_urls(request)
    source_url = source_urls[0] if source_urls else ""
    trace_event(
        trace,
        "resources.plan.start",
        collected_file_count=len(records),
        common_root=root,
        source_url=source_url,
    )

    plan: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for record in records:
        path: Path = record["path"]
        root_hint = Path(record["root"]) if record.get("root") else root
        rel = relative_name(path, root_hint)
        resource_name = rel.replace(os.sep, "__")
        if resource_name in used_names:
            resource_name = f"{path.stem}-{len(used_names)}{path.suffix}"
        used_names.add(resource_name)

        preview = extract_text_preview(path)
        provided_description = u.clean_text(record.get("description"), 1200)
        if provided_description:
            description = provided_description
        else:
            description = f"Uploaded file from {rel}."
        if preview:
            description = u.clean_text(f"{description} Text preview: {preview}", 3000)

        mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        suffix = path.suffix.lower().lstrip(".")
        plan.append(
            {
                "resource_name": resource_name,
                "resource_title": u.clean_text(record.get("provided_name") or u.resource_title_from_path(path), 180),
                "resource_description": description,
                "resource_tags": infer_tags(path, preview),
                "source_url": source_url,
                "local_path": str(path),
                "relative_path": rel,
                "format": suffix.upper() if suffix else "BIN",
                "mimetype": mimetype,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "text_preview": preview,
            }
        )

    if not plan:
        warnings.append("No local files were supplied; the worker will produce metadata-only CKAN package state.")
    trace_event(
        trace,
        "resources.plan.done",
        resource_count=len(plan),
        resource_names=[item.get("resource_name") for item in plan[:10]],
        warnings=warnings,
    )
    return plan, warnings


def extract_source_urls(request: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for value in as_list(request.get("source_url")) + as_list(request.get("source_urls")):
        text = u.clean_text(value)
        if text:
            urls.append(text)

    message = u.clean_text(request.get("message"))
    urls.extend(re.findall(r"https?://[^\s)>\]]+", message))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        clean_url = url.rstrip(".,;")
        if clean_url and clean_url not in seen:
            deduped.append(clean_url)
            seen.add(clean_url)
    return deduped


def get_dataset_request(request: dict[str, Any]) -> dict[str, Any]:
    dataset = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
    merged = dict(dataset)
    for key in set(DATASET_FIELD_ALIASES) | ALLOWED_DATASET_FIELDS:
        if key in request and key not in merged:
            merged[key] = request[key]
    return merged


def owner_org_from_request(request: dict[str, Any]) -> str:
    ckan = request.get("ckan") if isinstance(request.get("ckan"), dict) else {}
    return u.clean_text(
        ckan.get("owner_org")
        or ckan.get("owner_org_id")
        or request.get("owner_org")
        or request.get("owner_org_id")
        or os.getenv("CKAN_OWNER_ORG_ID")
        or os.getenv("CKAN_OWNER_ORG")
    )


def ckan_url_from_request(request: dict[str, Any]) -> str:
    ckan = request.get("ckan") if isinstance(request.get("ckan"), dict) else {}
    return u.clean_text(ckan.get("url") or request.get("ckan_url") or os.getenv("CKAN_URL") or "https://ckan.tacc.utexas.edu")


def merged_ckan_request(state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    request_ckan = request.get("ckan") if isinstance(request.get("ckan"), dict) else {}
    return {**request, "ckan": {**(state.get("ckan") or {}), **request_ckan}}


def dataset_preferences(request: dict[str, Any]) -> dict[str, Any]:
    dataset = get_dataset_request(request)
    return {
        "name": dataset.get("name") or dataset.get("dataset_name") or os.getenv("CKAN_DATASET_NAME") or "",
        "title": dataset.get("title") or dataset.get("dataset_title") or os.getenv("CKAN_DATASET_TITLE") or "",
        "url": dataset.get("url") or dataset.get("dataset_url") or "",
        "author": dataset.get("author") or dataset.get("dataset_author") or os.getenv("CKAN_DATASET_AUTHOR") or "",
        "author_email": dataset.get("author_email") or dataset.get("dataset_author_email") or os.getenv("CKAN_DATASET_AUTHOR_EMAIL") or "",
        "maintainer": dataset.get("maintainer") or dataset.get("dataset_maintainer") or os.getenv("CKAN_DATASET_MAINTAINER") or "",
        "maintainer_email": dataset.get("maintainer_email")
        or dataset.get("dataset_maintainer_email")
        or os.getenv("CKAN_DATASET_MAINTAINER_EMAIL")
        or "",
        "license_id": dataset.get("license_id") or dataset.get("dataset_license_id") or os.getenv("CKAN_DATASET_LICENSE_ID") or "cc-by",
        "version": dataset.get("version") or dataset.get("dataset_version") or os.getenv("CKAN_DATASET_VERSION") or "",
        "type": dataset.get("type") or dataset.get("dataset_type") or os.getenv("CKAN_DATASET_TYPE") or "dataset",
        "isopen": parse_bool(dataset.get("isopen") if "isopen" in dataset else dataset.get("dataset_isopen"), parse_bool(os.getenv("CKAN_DATASET_ISOPEN"), True)),
        "spatial": dataset.get("spatial") or dataset.get("dataset_spatial") or os.getenv("CKAN_DATASET_SPATIAL") or "",
        "temporal_coverage_start": dataset.get("temporal_coverage_start") or os.getenv("CKAN_TEMPORAL_COVERAGE_START") or "",
        "temporal_coverage_end": dataset.get("temporal_coverage_end") or os.getenv("CKAN_TEMPORAL_COVERAGE_END") or "",
        "private": parse_bool(dataset.get("private"), parse_bool(os.getenv("CKAN_DATASET_PRIVATE"), False)),
        "tags": dataset.get("tags") or dataset.get("dataset_tags") or [],
    }


def fallback_title(request: dict[str, Any], resource_plan: list[dict[str, Any]]) -> str:
    message = u.clean_text(request.get("message"), 220)
    if message:
        first_sentence = re.split(r"[.!?]\s+", message)[0]
        if len(first_sentence) >= 8:
            return u.clean_text(first_sentence, 140)
    if resource_plan:
        if len(resource_plan) == 1:
            return u.clean_text(resource_plan[0].get("resource_title") or "Uploaded Dataset", 140)
        stems = [Path(item["relative_path"]).stem for item in resource_plan[:8]]
        common = os.path.commonprefix(stems).strip("_- ")
        if len(common) >= 5:
            return u.clean_text(common.replace("_", " ").replace("-", " "), 140)
    return "CKAN Chat Registration Dataset"


def fallback_title_source(request: dict[str, Any], resource_plan: list[dict[str, Any]], prefs: dict[str, Any]) -> str:
    if u.clean_text(prefs.get("title")):
        return "dataset.title override"
    message = u.clean_text(request.get("message"), 220)
    if message:
        first_sentence = re.split(r"[.!?]\s+", message)[0]
        if len(first_sentence) >= 8:
            return "chat message"
    if resource_plan:
        if len(resource_plan) == 1:
            return "single resource title"
        stems = [Path(item["relative_path"]).stem for item in resource_plan[:8]]
        common = os.path.commonprefix(stems).strip("_- ")
        if len(common) >= 5:
            return "common resource filename prefix"
    return "default fallback"


def fallback_name_source(request: dict[str, Any], prefs: dict[str, Any]) -> str:
    dataset = get_dataset_request(request)
    if u.clean_text(dataset.get("name") or dataset.get("dataset_name")):
        return "dataset.name override"
    if u.clean_text(os.getenv("CKAN_DATASET_NAME")):
        return "CKAN_DATASET_NAME env"
    if u.clean_text(prefs.get("name")):
        return "preferred dataset name"
    return "slugified fallback title"


def fallback_metadata(request: dict[str, Any], resource_plan: list[dict[str, Any]], prefs: dict[str, Any], source_urls: list[str]) -> dict[str, Any]:
    title = u.clean_text(prefs.get("title") or fallback_title(request, resource_plan), 140)
    name = u.slugify(str(prefs.get("name") or title)) or "ckan-chat-registration-dataset"
    resource_count = len(resource_plan)

    notes_parts = []
    message = u.clean_text(request.get("message"), 1200)
    if message:
        notes_parts.append(message)
    notes_parts.append(f"Registered through the CKAN chat registration workflow with {resource_count} resource file(s).")
    if source_urls:
        notes_parts.append(f"Source URL: {source_urls[0]}")

    tag_texts: list[str] = []
    for item in resource_plan:
        tag_texts.extend(item.get("resource_tags") or [])
    tag_texts.extend(as_list(prefs.get("tags")))
    tags = u.dedupe_tags([str(tag) for tag in tag_texts])

    return {
        "dataset_name": name,
        "dataset_title": title,
        "dataset_notes": u.clean_text(" ".join(notes_parts), 3000),
        "dataset_url": u.clean_text(prefs.get("url") or (source_urls[0] if source_urls else "")),
        "dataset_author": u.clean_text(prefs.get("author")),
        "dataset_author_email": u.clean_text(prefs.get("author_email")),
        "dataset_maintainer": u.clean_text(prefs.get("maintainer")),
        "dataset_maintainer_email": u.clean_text(prefs.get("maintainer_email")),
        "dataset_license_id": u.clean_text(prefs.get("license_id")),
        "dataset_version": u.clean_text(prefs.get("version")),
        "dataset_type": u.clean_text(prefs.get("type") or "dataset"),
        "dataset_isopen": parse_bool(prefs.get("isopen"), True),
        "dataset_spatial": u.clean_text(prefs.get("spatial")),
        "temporal_coverage_start": u.clean_text(prefs.get("temporal_coverage_start")),
        "temporal_coverage_end": u.clean_text(prefs.get("temporal_coverage_end")),
        "dataset_tags": tags,
    }


def resource_plan_for_utils(resource_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for item in resource_plan:
        next_item = dict(item)
        next_item["local_path"] = Path(str(item["local_path"]))
        converted.append(next_item)
    return converted


def propose_metadata(
    request: dict[str, Any],
    resource_plan: list[dict[str, Any]],
    *,
    use_llm: bool,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    trace: list[dict[str, Any]] = []
    prefs = dataset_preferences(request)
    source_urls = extract_source_urls(request)
    fallback = fallback_metadata(request, resource_plan, prefs, source_urls)
    trace_event(
        trace,
        "metadata.fallback.prepared",
        dataset_name=fallback.get("dataset_name"),
        dataset_title=fallback.get("dataset_title"),
        name_source=fallback_name_source(request, prefs),
        title_source=fallback_title_source(request, resource_plan, prefs),
        notes_uses_message=bool(u.clean_text(request.get("message"))),
        resource_count=len(resource_plan),
        source_url_count=len(source_urls),
    )

    api_key = u.clean_text(os.getenv("OPENAI_API_KEY"))
    model = u.clean_text(os.getenv("CKAN_LLM_MODEL") or os.getenv("TOPIC_LABELER_MODEL") or "Meta-Llama-3.3-70B-Instruct")
    base_url = u.clean_text(os.getenv("OPENAI_BASE_URL"))

    if not use_llm:
        warnings.append("LLM metadata proposal skipped by request; fallback metadata was used.")
        trace_event(trace, "metadata.llm.skipped", reason="request disabled LLM")
        return fallback, warnings, trace
    if not api_key:
        warnings.append("OPENAI_API_KEY is not set; fallback metadata was used.")
        trace_event(trace, "metadata.llm.skipped", reason="OPENAI_API_KEY is not set")
        return fallback, warnings, trace
    if not resource_plan:
        warnings.append("No resource files were supplied; fallback metadata was used.")
        trace_event(trace, "metadata.llm.skipped", reason="no resource files supplied")
        return fallback, warnings, trace

    try:
        trace_event(
            trace,
            "metadata.llm.requested",
            model=model,
            base_url=base_url,
            resource_count=len(resource_plan),
            source_metadata_url=source_urls[0] if source_urls else "",
        )
        proposal = u.propose_ckan_dataset_metadata_with_llm(
            resource_plan_for_utils(resource_plan),
            model=model,
            api_key=api_key,
            base_url=base_url or None,
            source_metadata_url=source_urls[0] if source_urls else None,
            preferred_dataset_name=prefs.get("name"),
            preferred_dataset_title=prefs.get("title"),
            preferred_dataset_url=prefs.get("url"),
            preferred_dataset_author=prefs.get("author"),
            preferred_dataset_author_email=prefs.get("author_email"),
            preferred_dataset_maintainer=prefs.get("maintainer"),
            preferred_dataset_maintainer_email=prefs.get("maintainer_email"),
            preferred_dataset_license_id=prefs.get("license_id"),
            preferred_dataset_version=prefs.get("version"),
            preferred_dataset_type=prefs.get("type"),
            preferred_dataset_isopen=prefs.get("isopen"),
            preferred_dataset_spatial=prefs.get("spatial"),
            preferred_temporal_coverage_start=prefs.get("temporal_coverage_start"),
            preferred_temporal_coverage_end=prefs.get("temporal_coverage_end"),
            preferred_dataset_tags=u.dedupe_tags([str(tag) for tag in as_list(prefs.get("tags"))]),
            preserve_preferred_values=False,
        )
        merged = dict(fallback)
        proposal_values = {key: value for key, value in proposal.items() if value not in (None, "", [])}
        merged.update(proposal_values)
        trace_event(
            trace,
            "metadata.llm.merged",
            proposed_fields=sorted(proposal_values),
            fallback_fields_used=sorted(key for key in fallback if key not in proposal_values),
        )
        return merged, warnings, trace
    except Exception as exc:
        warnings.append(f"LLM metadata proposal failed; fallback metadata was used: {exc}")
        trace_event(trace, "metadata.llm.failed", error=str(exc))
        return fallback, warnings, trace


def desired_payload_from_metadata(
    request: dict[str, Any],
    llm_dataset: dict[str, Any],
    resource_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    prefs = dataset_preferences(request)
    owner_org = owner_org_from_request(request)

    tag_texts: list[str] = []
    for tag in as_list(llm_dataset.get("dataset_tags")):
        if isinstance(tag, dict):
            tag_texts.append(str(tag.get("name", "")))
        else:
            tag_texts.append(str(tag))
    for item in resource_plan:
        tag_texts.extend([str(tag) for tag in item.get("resource_tags", [])])
    tag_texts.extend([str(tag) for tag in as_list(prefs.get("tags"))])

    desired = {
        "name": u.slugify(str(llm_dataset.get("dataset_name") or prefs.get("name") or "ckan-chat-registration-dataset")),
        "title": u.clean_text(llm_dataset.get("dataset_title") or prefs.get("title") or "CKAN Chat Registration Dataset", 140),
        "notes": u.clean_text(llm_dataset.get("dataset_notes") or "", 3000),
        "url": u.clean_text(llm_dataset.get("dataset_url") or prefs.get("url")),
        "owner_org": owner_org,
        "private": parse_bool(prefs.get("private"), False),
        "tags": u.dedupe_tags(tag_texts),
        "author": u.clean_text(llm_dataset.get("dataset_author") or prefs.get("author")),
        "author_email": u.clean_text(llm_dataset.get("dataset_author_email") or prefs.get("author_email")),
        "maintainer": u.clean_text(llm_dataset.get("dataset_maintainer") or prefs.get("maintainer")),
        "maintainer_email": u.clean_text(llm_dataset.get("dataset_maintainer_email") or prefs.get("maintainer_email")),
        "license_id": u.clean_text(llm_dataset.get("dataset_license_id") or prefs.get("license_id")),
        "version": u.clean_text(llm_dataset.get("dataset_version") or prefs.get("version")),
        "type": u.clean_text(llm_dataset.get("dataset_type") or prefs.get("type") or "dataset"),
        "isopen": parse_bool(llm_dataset.get("dataset_isopen"), parse_bool(prefs.get("isopen"), True)),
        "spatial": u.clean_text(llm_dataset.get("dataset_spatial") or prefs.get("spatial")),
        "temporal_coverage_start": u.clean_text(llm_dataset.get("temporal_coverage_start") or prefs.get("temporal_coverage_start")),
        "temporal_coverage_end": u.clean_text(llm_dataset.get("temporal_coverage_end") or prefs.get("temporal_coverage_end")),
    }
    apply_dataset_overrides(desired, request)
    return desired


def apply_dataset_overrides(desired: dict[str, Any], request: dict[str, Any]) -> None:
    dataset = get_dataset_request(request)
    for raw_key, value in dataset.items():
        key = DATASET_FIELD_ALIASES.get(raw_key, raw_key)
        if key not in ALLOWED_DATASET_FIELDS:
            continue
        if key == "name":
            desired[key] = u.slugify(str(value))
        elif key == "tags":
            desired[key] = u.dedupe_tags([str(tag.get("name", "")) if isinstance(tag, dict) else str(tag) for tag in as_list(value)])
        elif key in {"private", "isopen"}:
            desired[key] = parse_bool(value, bool(desired.get(key)))
        else:
            desired[key] = u.clean_text(value) if value is not None else ""


def normalize_existing_ckan_entry(value: Any) -> str:
    text = u.clean_text(value)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if "dataset" in parts:
            index = parts.index("dataset")
            if len(parts) > index + 1:
                return parts[index + 1]
        if parts:
            return parts[-1]
    return text


def auth_header_for_read() -> str | None:
    mode = u.clean_text(os.getenv("CKAN_AUTH_MODE") or "api_token").lower()
    token = u.clean_text(os.getenv("CKAN_API_TOKEN") or os.getenv("CKAN_API_KEY"))
    username = u.clean_text(os.getenv("CKAN_USERNAME"))
    password = os.getenv("CKAN_PASSWORD") or ""
    if mode == "api_token" and token:
        return token
    if mode == "tapis_password" and username and password:
        try:
            return u.build_ckan_auth_header(
                auth_mode=mode,
                username=username,
                password=password,
                tapis_url=os.getenv("CKAN_TAPIS_URL") or u.DEFAULT_TAPIS_URL,
            )
        except Exception:
            return None
    return None


def auth_header_required() -> str:
    try:
        return u.build_ckan_auth_header(
            auth_mode=os.getenv("CKAN_AUTH_MODE") or "api_token",
            api_token=os.getenv("CKAN_API_TOKEN") or os.getenv("CKAN_API_KEY") or "",
            username=os.getenv("CKAN_USERNAME") or "",
            password=os.getenv("CKAN_PASSWORD") or "",
            tapis_url=os.getenv("CKAN_TAPIS_URL") or u.DEFAULT_TAPIS_URL,
        )
    except Exception as exc:
        raise AgentError(f"CKAN authentication is required for apply: {exc}") from exc


def build_review_markdown(state: dict[str, Any]) -> str:
    desired = state["desired_dataset_payload"]
    resources = state.get("resource_plan", [])
    lines = [
        "## Proposed CKAN Registration",
        f"- Dataset name: `{desired.get('name')}`",
        f"- Title: {desired.get('title')}",
        f"- Owner org: `{desired.get('owner_org') or '<not set>'}`",
        f"- Private: `{desired.get('private')}`",
        f"- Resource count: `{len(resources)}`",
        "",
        "Say `REGISTER` only after reviewing the dry-run output.",
    ]
    if state.get("warnings"):
        lines.extend(["", "### Warnings"])
        lines.extend([f"- {warning}" for warning in state["warnings"]])
    return "\n".join(lines)


def analyze(request: dict[str, Any], state_dir: Path, *, use_llm: bool) -> dict[str, Any]:
    session_id = sanitize_session_id(request.get("session_id"))
    trace: list[dict[str, Any]] = []
    trace_event(
        trace,
        "analyze.start",
        session_id=session_id,
        input_sources=request_source_summary(request),
        use_llm=use_llm,
    )
    preflight_issue = analyze_preflight_issue(request)
    if preflight_issue:
        trace_event(
            trace,
            "analyze.preflight.needs_input",
            code=preflight_issue.get("code"),
            reason=preflight_issue.get("message"),
            next_steps=preflight_issue.get("next_steps"),
        )
        return needs_input_response(session_id, preflight_issue, trace, request)

    resource_plan, warnings = build_resource_plan(request, trace)
    llm_dataset, metadata_warnings, metadata_trace = propose_metadata(request, resource_plan, use_llm=use_llm)
    trace.extend(metadata_trace)
    warnings.extend(metadata_warnings)
    desired_payload = desired_payload_from_metadata(request, llm_dataset, resource_plan)
    trace_event(
        trace,
        "dataset.desired_payload",
        dataset_name=desired_payload.get("name"),
        dataset_title=desired_payload.get("title"),
        resource_count=len(resource_plan),
        dataset_override_fields=dataset_override_keys(request),
        tags=desired_payload.get("tags"),
    )

    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "analyzed",
        "message": u.clean_text(request.get("message"), 3000),
        "source_urls": extract_source_urls(request),
        "existing_ckan_entry": normalize_existing_ckan_entry(request.get("existing_ckan_entry") or request.get("existing_dataset")),
        "ckan": {
            "url": ckan_url_from_request(request),
            "owner_org": owner_org_from_request(request),
            "upload_resources": parse_bool(request.get("upload_resources"), True),
            "remove_stale_resources": parse_bool(request.get("remove_stale_resources"), False),
            "resource_extra_fields": [str(value) for value in as_list(request.get("resource_extra_fields"))],
        },
        "llm_dataset": llm_dataset,
        "desired_dataset_payload": desired_payload,
        "resource_plan": resource_plan,
        "warnings": warnings,
        "trace": trace,
    }
    path = save_state(state, state_dir)
    result = {
        "ok": True,
        "command": "analyze",
        "state_path": str(path),
        "session_id": state["session_id"],
        "status": state["status"],
        "desired_dataset_payload": state["desired_dataset_payload"],
        "resource_count": len(resource_plan),
        "warnings": warnings,
        "review_markdown": build_review_markdown(state),
    }
    if response_should_include_trace(request):
        result["trace"] = trace
    return result


def resource_delta(existing: dict[str, Any] | None, resource_plan: list[dict[str, Any]]) -> dict[str, Any]:
    planned_names = {str(item.get("resource_name", "")) for item in resource_plan if item.get("resource_name")}
    existing_resources = existing.get("resources", []) if existing else []
    existing_names = {str(item.get("name", "")) for item in existing_resources if item.get("name")}
    return {
        "create": sorted(planned_names - existing_names),
        "update": sorted(planned_names & existing_names),
        "delete_candidates": sorted(existing_names - planned_names),
        "planned_count": len(planned_names),
        "existing_count": len(existing_names),
    }


def dry_run(request: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    state = load_state(request, state_dir)
    trace = state.setdefault("trace", [])
    trace_event(
        trace,
        "dry_run.start",
        session_id=state.get("session_id"),
        request=request_source_summary(request),
    )
    apply_dataset_overrides(state["desired_dataset_payload"], request)
    ckan_url = ckan_url_from_request(merged_ckan_request(state, request))
    lookup = normalize_existing_ckan_entry(
        request.get("existing_ckan_entry")
        or state.get("existing_ckan_entry")
        or state["desired_dataset_payload"].get("name")
    )
    existing = u.fetch_existing_dataset_or_none(ckan_url, lookup, auth_header_for_read()) if lookup else None
    changes = u.compare_dataset_metadata(existing, state["desired_dataset_payload"])
    resource_changes = resource_delta(existing, state.get("resource_plan", []))
    trace_event(
        trace,
        "dry_run.compare",
        ckan_url=ckan_url,
        lookup=lookup,
        existing_dataset_found=bool(existing),
        metadata_change_count=len(changes),
        resource_changes=resource_changes,
    )

    state["status"] = "dry_run"
    state["dry_run"] = {
        "ckan_url": ckan_url,
        "lookup": lookup,
        "existing_dataset_found": bool(existing),
        "changes": changes,
        "resource_changes": resource_changes,
    }
    path = save_state(state, state_dir)
    result = {
        "ok": True,
        "command": "dry-run",
        "state_path": str(path),
        "session_id": state["session_id"],
        "existing_dataset_found": bool(existing),
        "changes": changes,
        "resource_changes": resource_changes,
        "review_markdown": u.render_changes_table_markdown(changes),
    }
    if response_should_include_trace(request):
        result["trace"] = trace
    return result


def revise(request: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    state = load_state(request, state_dir)
    trace = state.setdefault("trace", [])
    trace_event(
        trace,
        "revise.start",
        session_id=state.get("session_id"),
        request=request_source_summary(request),
    )
    apply_dataset_overrides(state["desired_dataset_payload"], request)

    excludes = {u.clean_text(value) for value in as_list(request.get("exclude_resources")) if u.clean_text(value)}
    if excludes:
        kept = []
        for item in state.get("resource_plan", []):
            candidates = {
                u.clean_text(item.get("resource_name")),
                u.clean_text(item.get("relative_path")),
                u.clean_text(item.get("local_path")),
            }
            if candidates.isdisjoint(excludes):
                kept.append(item)
        state["resource_plan"] = kept
        notes = u.clean_text(state["desired_dataset_payload"].get("notes"), 3000)
        if notes:
            state["desired_dataset_payload"]["notes"] = re.sub(
                r"with \d+ resource file\(s\)",
                f"with {len(kept)} resource file(s)",
                notes,
            )
        trace_event(
            trace,
            "revise.resources_excluded",
            requested_excludes=sorted(excludes),
            remaining_count=len(kept),
        )

    if "upload_resources" in request:
        state.setdefault("ckan", {})["upload_resources"] = parse_bool(request.get("upload_resources"), True)
    if "remove_stale_resources" in request:
        state.setdefault("ckan", {})["remove_stale_resources"] = parse_bool(request.get("remove_stale_resources"), False)

    state["status"] = "revised"
    trace_event(
        trace,
        "revise.done",
        dataset_name=state["desired_dataset_payload"].get("name"),
        dataset_title=state["desired_dataset_payload"].get("title"),
        resource_count=len(state.get("resource_plan", [])),
    )
    path = save_state(state, state_dir)
    result = {
        "ok": True,
        "command": "revise",
        "state_path": str(path),
        "session_id": state["session_id"],
        "desired_dataset_payload": state["desired_dataset_payload"],
        "resource_count": len(state.get("resource_plan", [])),
        "review_markdown": build_review_markdown(state),
    }
    if response_should_include_trace(request):
        result["trace"] = trace
    return result


def require_register_approval(request: dict[str, Any]) -> None:
    approval = u.clean_text(request.get("approval") or request.get("confirmation"))
    if approval != "REGISTER":
        raise AgentError("Apply refused. Send approval exactly as REGISTER after reviewing the dry-run output.")


def apply_registration(request: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    require_register_approval(request)
    state = load_state(request, state_dir)
    trace = state.setdefault("trace", [])
    trace_event(
        trace,
        "apply.start",
        session_id=state.get("session_id"),
        request=request_source_summary(request),
    )
    apply_dataset_overrides(state["desired_dataset_payload"], request)

    desired = state["desired_dataset_payload"]
    ckan_url = ckan_url_from_request(merged_ckan_request(state, request))
    auth_header = auth_header_required()
    dataset_after = u.create_or_update_ckan_dataset(
        ckan_url,
        dataset_name=desired["name"],
        dataset_title=desired["title"],
        dataset_notes=desired["notes"],
        dataset_tags=desired.get("tags") or [],
        auth_header=auth_header,
        owner_org=desired.get("owner_org"),
        private=parse_bool(desired.get("private"), False),
        dataset_author=desired.get("author"),
        dataset_author_email=desired.get("author_email"),
        dataset_maintainer=desired.get("maintainer"),
        dataset_maintainer_email=desired.get("maintainer_email"),
        dataset_license_id=desired.get("license_id"),
        dataset_url=desired.get("url"),
        dataset_version=desired.get("version"),
        dataset_type=desired.get("type") or "dataset",
        dataset_isopen=parse_bool(desired.get("isopen"), True),
        dataset_spatial=desired.get("spatial"),
        temporal_coverage_start=desired.get("temporal_coverage_start"),
        temporal_coverage_end=desired.get("temporal_coverage_end"),
    )

    upload_resources = parse_bool(request.get("upload_resources"), parse_bool(state.get("ckan", {}).get("upload_resources"), True))
    uploaded: list[dict[str, Any]] = []
    created_count = 0
    updated_count = 0
    if upload_resources and state.get("resource_plan"):
        uploaded, created_count, updated_count = u.upsert_resources(
            ckan_url,
            dataset_after,
            resource_plan_for_utils(state["resource_plan"]),
            auth_header,
            extra_resource_fields=state.get("ckan", {}).get("resource_extra_fields") or [],
        )

    removed_count = 0
    remove_stale = parse_bool(request.get("remove_stale_resources"), parse_bool(state.get("ckan", {}).get("remove_stale_resources"), False))
    if remove_stale:
        delete_approval = u.clean_text(request.get("delete_approval"))
        if delete_approval != "DELETE_STALE_RESOURCES":
            raise AgentError("Stale resource removal requires delete_approval exactly as DELETE_STALE_RESOURCES.")
        latest_dataset = u.fetch_ckan_dataset(ckan_url, desired["name"], auth_header=auth_header)
        removed_count = u.remove_stale_resources(
            ckan_url,
            latest_dataset,
            {item["resource_name"] for item in state.get("resource_plan", [])},
            auth_header,
        )

    dataset_url = f"{ckan_url.rstrip('/')}/dataset/{dataset_after.get('name') or desired['name']}"
    state["status"] = "applied"
    state["last_apply_result"] = {
        "dataset_name": dataset_after.get("name") or desired["name"],
        "dataset_url": dataset_url,
        "resource_count": len(uploaded),
        "resource_created": created_count,
        "resource_updated": updated_count,
        "resource_removed": removed_count,
    }
    trace_event(
        trace,
        "apply.done",
        dataset_name=state["last_apply_result"]["dataset_name"],
        dataset_url=dataset_url,
        upload_resources=upload_resources,
        resource_created=created_count,
        resource_updated=updated_count,
        resource_removed=removed_count,
    )
    path = save_state(state, state_dir)
    result = {
        "ok": True,
        "command": "apply",
        "state_path": str(path),
        "session_id": state["session_id"],
        **state["last_apply_result"],
    }
    if response_should_include_trace(request):
        result["trace"] = trace
    return result


def show_state(request: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    state = load_state(request, state_dir)
    return {
        "ok": True,
        "command": "show",
        "state": state,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CKAN chat registration worker for n8n.")
    parser.add_argument("command", choices=["analyze", "dry-run", "revise", "apply", "show"])
    parser.add_argument("--input", "-i", help="JSON request file. Use '-' or omit to read stdin.")
    parser.add_argument("--input-b64", help="Base64-encoded JSON request object. Useful for restricted n8n Code nodes.")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR), help="Directory for saved session state.")
    parser.add_argument("--env-file", default=str(SCRIPT_DIR / ".env"), help="Optional .env file to load before running.")
    parser.add_argument("--secret-env-file", help="Optional env file for per-request secrets. Values override .env.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM metadata proposal during analyze.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    try:
        env_file = Path(args.env_file).expanduser() if args.env_file else None
        load_env_file(env_file)
        secret_env_file = Path(args.secret_env_file).expanduser() if args.secret_env_file else None
        load_env_file(secret_env_file, override=True)
        if secret_env_file and secret_env_file.exists():
            try:
                secret_env_file.unlink()
            except OSError:
                pass
        request = load_json_input_b64(args.input_b64) if args.input_b64 else load_json_input(args.input)
        apply_secret_headers(request)
        request.pop("headers", None)
        request.pop("request_headers", None)
        state_dir = Path(args.state_dir).expanduser()

        if args.command == "analyze":
            result = analyze(request, state_dir, use_llm=not args.no_llm and parse_bool(request.get("use_llm"), True))
        elif args.command == "dry-run":
            result = dry_run(request, state_dir)
        elif args.command == "revise":
            result = revise(request, state_dir)
        elif args.command == "apply":
            result = apply_registration(request, state_dir)
        elif args.command == "show":
            result = show_state(request, state_dir)
        else:  # pragma: no cover
            raise AgentError(f"Unsupported command: {args.command}")
        write_json(result)
        return 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
