"""Standalone CKAN + LLM utilities for  registration notebook."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
import random
import re
import time
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_TAPIS_URL = "https://portals.tapis.io/v3/oauth2/tokens"

# Retryable HTTP status codes for LLM calls.
_LLM_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# SDK exception class names that trigger a retry (inspected by name to avoid
# hard-importing openai exception classes beyond the top-level OpenAI import).
_LLM_RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "APIStatusError",
}


def _llm_max_retries() -> int:
    """Maximum number of *retries* after the initial attempt (default 5).

    Read from ``LLM_MAX_RETRIES`` env var.  The total number of attempts is
    ``LLM_MAX_RETRIES + 1``.
    """
    raw = os.getenv("LLM_MAX_RETRIES", "5").strip()
    try:
        value = int(raw)
    except ValueError:
        return 5
    return max(0, value)


def _llm_backoff_base_seconds() -> float:
    """Base delay in seconds for exponential back-off (default 2.0).

    Read from ``LLM_BACKOFF_BASE_SECONDS`` env var.
    """
    raw = os.getenv("LLM_BACKOFF_BASE_SECONDS", "2.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 2.0
    return max(0.0, value)


def _llm_backoff_max_seconds() -> float:
    """Maximum delay cap in seconds for exponential back-off (default 60.0).

    Read from ``LLM_BACKOFF_MAX_SECONDS`` env var.
    """
    raw = os.getenv("LLM_BACKOFF_MAX_SECONDS", "60.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 60.0
    return max(0.0, value)


def _llm_should_retry_exception(exc: Exception) -> tuple[bool, int | None]:
    """Return ``(should_retry, status_code_or_None)`` for an SDK exception.

    Detects retryable conditions defensively via ``getattr`` so that no
    specific openai exception class needs to be imported.
    """
    exc_name = type(exc).__name__
    if exc_name in _LLM_RETRYABLE_EXCEPTION_NAMES:
        # Try to extract a numeric status code from the exception.
        status = getattr(exc, "status_code", None)
        if status is None:
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                status = getattr(response_obj, "status_code", None)
        try:
            status_int = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_int = None
        return True, status_int

    # Also retry if the exception exposes a retryable status_code directly.
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            if int(status) in _LLM_RETRYABLE_STATUSES:
                return True, int(status)
        except (TypeError, ValueError):
            pass

    return False, None


def _llm_retry_after_from_exception(exc: Exception) -> float | None:
    """Extract a ``Retry-After`` value (seconds) from an SDK exception, if present."""
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        response_obj = getattr(exc, "response", None)
        if response_obj is not None:
            headers = getattr(response_obj, "headers", {}) or {}
            raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _chat_completion_content(
    *,
    model: str,
    api_key: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    base_url: str | None = None,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """Call an LLM and return the assistant message content string.

    Both the OpenAI SDK branch and the HTTP-fallback branch are wrapped in an
    identical retry loop with exponential back-off.  Retry behaviour is
    controlled by three env vars (read fresh on each call so tests can
    monkeypatch them):

    - ``LLM_MAX_RETRIES``          — max retries after initial attempt (default 5)
    - ``LLM_BACKOFF_BASE_SECONDS`` — base delay for exponential back-off (default 2.0)
    - ``LLM_BACKOFF_MAX_SECONDS``  — maximum back-off cap in seconds (default 60.0)

    On the final failure the original exception is re-raised so callers'
    graceful fallbacks (pdf_extract, persona_loop, etc.) still trigger.
    """
    max_retries = _llm_max_retries()
    backoff_base = _llm_backoff_base_seconds()
    backoff_max = _llm_backoff_max_seconds()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]

    if OpenAI is not None:
        # Disable the SDK's built-in retry so our loop is the single retry authority.
        client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=0)
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                should_retry, status_code = _llm_should_retry_exception(exc)
                if not should_retry or attempt >= max_retries:
                    raise
                last_exc = exc
                retry_after = _llm_retry_after_from_exception(exc)
                if retry_after is not None:
                    delay = min(backoff_max, retry_after)
                else:
                    delay = min(backoff_max, backoff_base * (2 ** attempt))
                delay += random.uniform(0, 0.5)
                delay = min(delay, backoff_max)
                logger.warning(
                    "LLM SDK call failed (attempt %d/%d, status=%s, class=%s); retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    status_code,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        # Should never be reached, but satisfy type-checkers.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM SDK retry loop exited unexpectedly")

    # HTTP fallback branch.
    root = clean_text(base_url)
    if not root:
        raise RuntimeError("OpenAI SDK unavailable and no OPENAI_BASE_URL provided for HTTP fallback.")
    url = f"{root.rstrip('/')}/v1/chat/completions"
    last_response = None
    for attempt in range(max_retries + 1):
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        if response.status_code not in _LLM_RETRYABLE_STATUSES:
            # Non-retryable: let raise_for_status handle error codes, or
            # proceed to parse the successful body below.
            break
        if attempt >= max_retries:
            last_response = response
            break
        retry_after_header = (
            response.headers.get("Retry-After") or response.headers.get("retry-after")
        )
        if retry_after_header is not None:
            try:
                delay = min(backoff_max, float(retry_after_header))
            except (TypeError, ValueError):
                delay = min(backoff_max, backoff_base * (2 ** attempt))
        else:
            delay = min(backoff_max, backoff_base * (2 ** attempt))
        delay += random.uniform(0, 0.5)
        delay = min(delay, backoff_max)
        logger.warning(
            "LLM HTTP call failed (attempt %d/%d, status=%d); retrying in %.1fs",
            attempt + 1,
            max_retries + 1,
            response.status_code,
            delay,
        )
        time.sleep(delay)

    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM HTTP fallback returned no choices: {body}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    return clean_text((message or {}).get("content", ""))


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def slugify(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def sanitize_tag(tag: str) -> str:
    return slugify(tag)[:100]


def dedupe_tags(tags: list[str]) -> list[dict[str, str]]:
    out = []
    seen = set()
    for tag in tags:
        normalized = sanitize_tag(tag)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append({"name": normalized})
    return out


def html_to_text(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_source_metadata(url: str, timeout: int = 60) -> dict[str, str]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    html = response.text

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    meta_desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    title = clean_text(title_match.group(1) if title_match else "")
    meta_description = clean_text(meta_desc_match.group(1) if meta_desc_match else "")
    page_text = html_to_text(html)

    excerpt = clean_text(page_text, max_chars=6000)
    return {
        "url": url,
        "title": title,
        "meta_description": meta_description,
        "excerpt": excerpt,
    }


def list_resource_files(resource_dir: Path, max_files: int = 5000) -> list[Path]:
    if not resource_dir.exists():
        raise FileNotFoundError(f"Resource directory does not exist: {resource_dir}")
    files = [path for path in resource_dir.rglob("*") if path.is_file()]
    files = sorted(files)
    if len(files) > max_files:
        files = files[:max_files]
    return files


def resource_title_from_path(path: Path) -> str:
    base = path.stem.replace("_", " ").replace("-", " ")
    return clean_text(base or path.name, max_chars=140)


def human_readable_resource_name(rel: Path) -> str:
    """Human-readable resource title from a relative path.

    Humanizes the folder segments and keeps the real filename (with its
    extension), e.g. ``Model_File/ygjk_tr.dis`` -> ``Model File / ygjk_tr.dis``.
    Unique per relative path (the filename + its extension disambiguates files;
    the folder path disambiguates same-named files in different directories).
    """
    parts = list(rel.parts)
    if not parts:
        return clean_text(str(rel), max_chars=200) or str(rel)
    filename = parts[-1]
    dirs = [p.replace("_", " ").replace("-", " ").strip() for p in parts[:-1]]
    dirs = [d for d in dirs if d]
    if dirs:
        return clean_text(" / ".join(dirs) + " / " + filename, max_chars=200)
    return clean_text(filename, max_chars=200)


# MODFLOW / GAM file-type descriptions (deterministic — no LLM needed).
# Keyed by lowercase extension (no dot).  Covers common MODFLOW-2000/2005/NWT/6
# packages, solvers, and binary outputs, plus the subsidence packages relevant
# to SUBSIDE (.sub, .swt).
_MODFLOW_FILE_DESCRIPTIONS: dict[str, str] = {
    "nam": "MODFLOW name file listing the model's input and output packages.",
    "dis": "Discretization package defining the model grid, layering, and stress periods.",
    "disu": "Unstructured discretization package (MODFLOW-USG) defining the grid.",
    "disv": "Vertex (DISV) discretization package (MODFLOW 6) defining the grid.",
    "bas": "Basic package: active/inactive (IBOUND) cells and starting heads.",
    "ba6": "Basic package: active/inactive (IBOUND) cells and starting heads.",
    "bcf": "Block-Centered Flow package: hydraulic conductivity and storage.",
    "lpf": "Layer-Property Flow package: hydraulic conductivity and storage.",
    "upw": "Upstream-Weighting flow package (MODFLOW-NWT): hydraulic properties.",
    "huf": "Hydrogeologic-Unit Flow package: hydraulic properties.",
    "npf": "Node-Property Flow package (MODFLOW 6): hydraulic conductivity.",
    "sto": "Storage package (MODFLOW 6): specific storage and yield.",
    "ic": "Initial-conditions package (MODFLOW 6): starting heads.",
    "wel": "Well package: pumping/injection rates by stress period.",
    "mnw": "Multi-Node Well package: wells screened across multiple cells.",
    "mnw2": "Multi-Node Well (MNW2) package.",
    "rch": "Recharge package: areal recharge rates.",
    "evt": "Evapotranspiration package: ET rates and extinction depths.",
    "ets": "Evapotranspiration-Segments package.",
    "drn": "Drain package: head-dependent drain boundaries.",
    "riv": "River package: head-dependent river boundaries.",
    "ghb": "General-Head Boundary package.",
    "chd": "Time-Variant Specified-Head (constant-head) package.",
    "str": "Streamflow-routing (STR) package.",
    "sfr": "Streamflow-Routing (SFR) package.",
    "lak": "Lake package.",
    "gag": "Gage package: output at specified locations.",
    "hfb": "Horizontal-Flow-Barrier package.",
    "sub": "Subsidence (SUB) package: interbed compaction and land subsidence.",
    "swt": "Subsidence and Aquifer-System Compaction (SUB-WT) package.",
    "oc": "Output-Control package: head/budget save and print options.",
    "pcg": "Preconditioned Conjugate-Gradient solver configuration.",
    "pcgn": "PCGN solver configuration.",
    "gmg": "Geometric Multigrid solver configuration.",
    "sip": "Strongly Implicit Procedure solver configuration.",
    "de4": "Direct (DE4) solver configuration.",
    "nwt": "Newton (NWT) solver configuration (MODFLOW-NWT).",
    "ims": "Iterative Model Solution (MODFLOW 6) solver configuration.",
    "hob": "Head-Observation package: observed vs. simulated heads.",
    "obs": "Observation package.",
    "zon": "Zone-array file used by parameterization/multipliers.",
    "mlt": "Multiplier-array file used by parameterization.",
    "hds": "Simulated hydraulic-head output (binary).",
    "bhd": "Simulated hydraulic-head output (binary).",
    "ddn": "Simulated drawdown output (binary).",
    "cbb": "Cell-by-cell flow budget output (binary).",
    "cbc": "Cell-by-cell flow budget output (binary).",
    "ccf": "Cell-by-cell flow budget output (binary).",
    "lst": "MODFLOW run listing/log output.",
    "list": "MODFLOW run listing/log output.",
    "glo": "MODFLOW global output file.",
    "out": "Model output file.",
    "in": "Model input file.",
    "inp": "Model input file.",
    "pdf": "Report or documentation (PDF).",
    "doc": "Report or documentation.",
    "docx": "Report or documentation.",
    "txt": "Text documentation or data.",
    "csv": "Tabular data (CSV).",
    "xlsx": "Spreadsheet data.",
    "shp": "Vector GIS data (Esri shapefile).",
    "zip": "Archive of model files.",
}


def describe_resource_file(rel: Path) -> str:
    """Deterministic, domain-aware description for a MODFLOW/GAM resource file.

    Looks up the file extension in :data:`_MODFLOW_FILE_DESCRIPTIONS`; unknown
    extensions get a generic description.  No LLM call (avoids per-file rate
    limits) — file-type meanings are well-defined for MODFLOW models.
    """
    ext = rel.suffix.lower().lstrip(".")
    base = _MODFLOW_FILE_DESCRIPTIONS.get(ext)
    if base is None:
        base = f"{ext.upper()} data file." if ext else "Model data file."
    return clean_text(f"{base} File within the model package: {rel}.", max_chars=3000)


def build_resource_plan(files: list[Path], root: Path, source_url: str) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    used_names = set()

    for path in files:
        rel = path.relative_to(root)
        name = human_readable_resource_name(rel)
        if name in used_names:
            name = f"{name} ({len(used_names)})"
        used_names.add(name)

        suffix = path.suffix.lower().lstrip(".")
        format_guess = suffix.upper() if suffix else "BIN"
        tag_candidates = [

            "aquifer",
            "groundwater",
            "gam",
            "model-files",
            suffix or "binary",
        ]
        plan.append(
            {
                "resource_name": name,
                "resource_title": name,
                "resource_description": describe_resource_file(rel),
                "resource_tags": [sanitize_tag(tag) for tag in tag_candidates if sanitize_tag(tag)],
                "source_url": source_url,
                "local_path": path,
                "relative_path": str(rel),
                "format": format_guess,
            }
        )
    return plan


def summarize_extensions(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        ext = path.suffix.lower() or "<no_ext>"
        counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _mint_headers(api_token: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = clean_text(api_token)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _mint_to_items(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict):
        for key in ("items", "results", "data"):
            value = body.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [body]
    return []


def _mint_first_label(item: dict[str, Any]) -> str:
    value = item.get("label")
    if isinstance(value, list) and value:
        return clean_text(value[0])
    if isinstance(value, str):
        return clean_text(value)
    return ""


def _mint_get(
    base_url: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    api_token: str | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(
        url,
        params=params or {},
        headers=_mint_headers(api_token),
        timeout=timeout,
        verify=verify_ssl,
    )
    response.raise_for_status()
    return response.json()


def _extract_standard_variable_values(values: list[Any]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    labels: list[str] = []
    for value in values:
        if isinstance(value, str):
            text = clean_text(value)
            if not text:
                continue
            if text.startswith("http://") or text.startswith("https://"):
                ids.append(text)
            else:
                labels.append(text)
            continue

        if isinstance(value, dict):
            rid = clean_text(value.get("id") or value.get("@id"))
            if rid:
                ids.append(rid)
            label = value.get("label")
            if isinstance(label, list) and label:
                labels.append(clean_text(label[0]))
            elif isinstance(label, str):
                labels.append(clean_text(label))

    ids = [x for x in dict.fromkeys(ids) if x]
    labels = [x for x in dict.fromkeys(labels) if x]
    return ids, labels


def list_mint_models_by_label(
    label: str,
    *,
    mint_api_base: str = "https://api.models.mint.tacc.utexas.edu/v2.0.0",
    mint_api_token: str | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> list[dict[str, Any]]:
    body = _mint_get(
        mint_api_base,
        "/models",
        params={"label": label},
        api_token=mint_api_token,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    return _mint_to_items(body)


def list_mint_model_standard_variables(
    model_label: str,
    *,
    mint_api_base: str = "https://api.models.mint.tacc.utexas.edu/v2.0.0",
    mint_api_token: str | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> tuple[list[str], list[str]]:
    body = _mint_get(
        mint_api_base,
        "/custom/models/standard_variable",
        params={"label": model_label},
        api_token=mint_api_token,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    items = body if isinstance(body, list) else _mint_to_items(body)
    ids, labels = _extract_standard_variable_values(items)
    return ids, labels


def _dedupe_clean_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _mint_ref_id(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("id") or value.get("@id"))
    return clean_text(value)


def _mint_ref_ids(value: Any) -> list[str]:
    values = value if isinstance(value, list) else ([value] if value else [])
    return _dedupe_clean_text([_mint_ref_id(item) for item in values])


def _mint_item_labels(item: dict[str, Any]) -> list[str]:
    value = item.get("label")
    if isinstance(value, list):
        return _dedupe_clean_text(value)
    return _dedupe_clean_text([value])


def _mint_get_resource(
    collection_path: str,
    resource_id: str,
    *,
    mint_api_base: str,
    mint_api_token: str | None,
    timeout: int,
    verify_ssl: bool,
) -> dict[str, Any] | None:
    rid = clean_text(resource_id)
    if not rid:
        return None
    try:
        body = _mint_get(
            mint_api_base,
            f"{collection_path.rstrip('/')}/{quote(rid, safe='')}",
            api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
    except Exception:
        return None
    return body if isinstance(body, dict) else None


def _mint_query_by_label(
    collection_path: str,
    label: str,
    *,
    mint_api_base: str,
    mint_api_token: str | None,
    timeout: int,
    verify_ssl: bool,
) -> list[dict[str, Any]]:
    query_label = clean_text(label)
    if not query_label:
        return []
    try:
        body = _mint_get(
            mint_api_base,
            collection_path,
            params={"label": query_label},
            api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
    except Exception:
        return []
    return _mint_to_items(body)


def _collect_dataset_refs(item: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for field in ("hasInput", "hasOutput"):
        values = item.get(field)
        if isinstance(values, list):
            refs.extend([value for value in values if isinstance(value, dict)])
        elif isinstance(values, dict):
            refs.append(values)
    return refs


def list_mint_model_configuration_dataset_labels(
    model_label: str,
    *,
    mint_api_base: str = "https://api.models.mint.tacc.utexas.edu/v2.0.0",
    mint_api_token: str | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> list[str]:
    """Return dataset-specification labels reachable from the current MINT model hierarchy."""
    label = clean_text(model_label)
    if not label:
        return []

    version_ids: list[str] = []
    config_ids: list[str] = []
    setup_ids: list[str] = []
    dataset_refs: list[dict[str, Any]] = []
    labels: list[str] = []

    for collection_path in ("/models", "/softwares"):
        for item in _mint_query_by_label(
            collection_path,
            label,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        ):
            version_ids.extend(_mint_ref_ids(item.get("hasVersion") or item.get("has_version")))

    for version_id in _dedupe_clean_text(version_ids):
        version = _mint_get_resource(
            "/softwareversions",
            version_id,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        if version:
            config_ids.extend(_mint_ref_ids(version.get("hasConfiguration") or version.get("has_configuration")))

    label_variants = _dedupe_clean_text(
        [
            label,
            f"{label} default configuration",
            f"{label} default setup",
        ]
    )
    for query_label in label_variants:
        for item in _mint_query_by_label(
            "/modelconfigurations",
            query_label,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        ):
            config_ids.extend(_mint_ref_ids(item))
            setup_ids.extend(_mint_ref_ids(item.get("hasSetup") or item.get("has_setup")))
            dataset_refs.extend(_collect_dataset_refs(item))
        for item in _mint_query_by_label(
            "/modelconfigurationsetups",
            query_label,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        ):
            setup_ids.extend(_mint_ref_ids(item))
            dataset_refs.extend(_collect_dataset_refs(item))

    for config_id in _dedupe_clean_text(config_ids):
        config = (
            _mint_get_resource(
                "/custom/modelconfigurations",
                config_id,
                mint_api_base=mint_api_base,
                mint_api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
            or _mint_get_resource(
                "/modelconfigurations",
                config_id,
                mint_api_base=mint_api_base,
                mint_api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
        )
        if not config:
            continue
        setup_ids.extend(_mint_ref_ids(config.get("hasSetup") or config.get("has_setup")))
        dataset_refs.extend(_collect_dataset_refs(config))

    for setup_id in _dedupe_clean_text(setup_ids):
        setup = (
            _mint_get_resource(
                "/custom/modelconfigurationsetups",
                setup_id,
                mint_api_base=mint_api_base,
                mint_api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
            or _mint_get_resource(
                "/modelconfigurationsetups",
                setup_id,
                mint_api_base=mint_api_base,
                mint_api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
        )
        if setup:
            dataset_refs.extend(_collect_dataset_refs(setup))

    for ref in dataset_refs:
        labels.extend(_mint_item_labels(ref))
        dataset_id = _mint_ref_id(ref)
        dataset = _mint_get_resource(
            "/datasetspecifications",
            dataset_id,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        if dataset:
            labels.extend(_mint_item_labels(dataset))
            for presentation in dataset.get("hasPresentation") or []:
                if isinstance(presentation, dict):
                    labels.extend(_mint_item_labels(presentation))

    return _dedupe_clean_text(labels)


def infer_standard_variable_names_from_mint_dataset_labels(
    dataset_labels: list[str],
    model_label: str = "",
) -> list[str]:
    names: list[str] = []
    model_text = clean_text(model_label).lower()

    for label in dataset_labels:
        text = clean_text(label).lower().replace("_", " ")
        compact = re.sub(r"[^a-z0-9]+", "", text)

        if "simulation archive" in text:
            names.append("groundwater_model__simulation_archive")
        if "name file" in text or "namefile" in text or compact.endswith("nam"):
            if "modflow 6" in model_text or "mf6" in model_text or "simulation name" in text:
                names.append("groundwater_simulation__namefile_specification")
            else:
                names.append("groundwater_model__namefile_specification")
        if "package input" in text or "package input set" in text:
            names.append("aquifer_system__package_input_set")
        if "hydraulic head" in text or compact.endswith("heads"):
            names.append("groundwater__hydraulic_head")
        if "water budget" in text or "cell-by-cell" in text or "cell budget" in text:
            names.append("aquifer_system__volumetric_budget")
        if "drawdown" in text or compact.endswith("ddown"):
            names.append("groundwater__drawdown")
        if "convergence" in text or "solver diagnostic" in text:
            names.append("simulation_run__convergence_diagnostic")
        if "recharge rate" in text:
            names.append("aquifer_system__recharge_rate")
        if "hydraulic propert" in text:
            names.append("aquifer_system__hydraulic_property_set")

    return _dedupe_clean_text(names)


def _find_default_model_yaml_dir() -> Path | None:
    starts = [Path.cwd()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:  # pragma: no cover
        pass

    for start in starts:
        for root in [start, *start.parents]:
            for candidate in (root / "RegisterMintModel" / "model_yamls", root / "model_yamls"):
                if candidate.exists():
                    return candidate
    return None


def load_registered_model_standard_variable_names(
    model_labels: list[str],
    *,
    model_yaml_dir: Path | None = None,
) -> list[str]:
    if yaml is None:
        return []

    yaml_dir = model_yaml_dir or _find_default_model_yaml_dir()
    if yaml_dir is None or not yaml_dir.exists():
        return []

    labels = _dedupe_clean_text(model_labels)
    if not labels:
        return []

    mapping = load_model_standard_variable_names_from_yamls(yaml_dir)
    by_slug = {slugify(label): names for label, names in mapping.items()}
    names: list[str] = []
    for label in labels:
        names.extend(by_slug.get(slugify(label), []))
    return _dedupe_clean_text(names)


def resolve_mint_standard_variable_ids(
    names: list[str],
    *,
    mint_api_base: str = "https://api.models.mint.tacc.utexas.edu/v2.0.0",
    mint_api_token: str | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> tuple[list[str], list[str]]:
    resolved_ids: list[str] = []
    unresolved: list[str] = []

    for name in names:
        label = clean_text(name)
        if not label:
            continue
        try:
            body = _mint_get(
                mint_api_base,
                "/standardvariables",
                params={"label": label},
                api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
            items = _mint_to_items(body)
        except Exception:
            unresolved.append(label)
            continue

        exact_match_id = ""
        fallback_id = ""
        for item in items:
            rid = clean_text(item.get("id"))
            if not rid:
                continue
            item_label = _mint_first_label(item)
            if item_label and item_label.lower() == label.lower():
                exact_match_id = rid
                break
            if not fallback_id:
                fallback_id = rid

        if exact_match_id:
            resolved_ids.append(exact_match_id)
        elif fallback_id:
            resolved_ids.append(fallback_id)
        else:
            unresolved.append(label)

    return [x for x in dict.fromkeys(resolved_ids)], unresolved


def infer_standard_variable_names_from_resource_files(
    resource_files: list[Path],
    model_label: str = "",
) -> list[str]:
    names: list[str] = []
    exts = {path.suffix.lower() for path in resource_files if path.suffix}
    model_text = clean_text(model_label).lower()

    package_exts = {
        ".bas", ".bcf", ".dis", ".drn", ".evt", ".ghb", ".gmg", ".lpf",
        ".oc", ".rch", ".str", ".wel", ".riv", ".upw", ".npf", ".sto",
        ".chd", ".sfr", ".uzf", ".maw", ".buy", ".in", ".ini", ".cfg",
    }
    if exts & package_exts:
        names.append("aquifer_system__package_input_set")

    if ".nam" in exts:
        # MF6 commonly uses simulation namefile term; MODFLOW-2000 commonly uses model namefile term.
        if "modflow 6" in model_text or "mf6" in model_text:
            names.append("groundwater_simulation__namefile_specification")
        else:
            names.append("groundwater_model__namefile_specification")

    if exts & {".zip", ".tar", ".gz", ".tgz", ".7z"}:
        names.append("groundwater_model__simulation_archive")

    if exts & {".hds", ".hed", ".head"}:
        names.append("groundwater__hydraulic_head")

    if exts & {".cbb", ".cbc", ".bud"}:
        names.append("aquifer_system__volumetric_budget")

    if exts & {".ddn"}:
        names.append("groundwater__drawdown")

    if exts & {".lst", ".glo", ".res"}:
        names.append("simulation_run__convergence_diagnostic")

    return [x for x in dict.fromkeys(names) if x]


def resolve_standard_variable_names_in_mint(
    names: list[str],
    *,
    mint_api_base: str = "https://api.models.mint.tacc.utexas.edu/v2.0.0",
    mint_api_token: str | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> tuple[list[str], list[str]]:
    resolved_names: list[str] = []
    unresolved_names: list[str] = []
    for name in names:
        label = clean_text(name)
        if not label:
            continue
        try:
            body = _mint_get(
                mint_api_base,
                "/standardvariables",
                params={"label": label},
                api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
            items = _mint_to_items(body)
        except Exception:
            unresolved_names.append(label)
            continue

        found = False
        for item in items:
            item_label = _mint_first_label(item)
            if item_label and item_label.lower() == label.lower():
                resolved_names.append(label)
                found = True
                break
        if not found:
            unresolved_names.append(label)

    return [x for x in dict.fromkeys(resolved_names)], [x for x in dict.fromkeys(unresolved_names)]


def load_model_standard_variable_names_from_yamls(model_yaml_dir: Path) -> dict[str, list[str]]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load model YAML files.")

    mapping: dict[str, list[str]] = {}
    if not model_yaml_dir.exists():
        return mapping

    for path in sorted(model_yaml_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        model_label = clean_text(payload.get("label") or payload.get("model_name"))
        if not model_label:
            continue
        names: list[str] = []
        for io_key in ("inputs", "outputs"):
            for record in payload.get(io_key, []) or []:
                svo_name = clean_text(record.get("standard_variable_name"))
                if svo_name:
                    names.append(svo_name)
        mapping[model_label] = [x for x in dict.fromkeys(names)]
    return mapping


def load_model_yaml_records_from_yamls(model_yaml_dir: Path) -> dict[str, dict[str, Any]]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load model YAML files.")

    mapping: dict[str, dict[str, Any]] = {}
    if not model_yaml_dir.exists():
        return mapping

    for path in sorted(model_yaml_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        model_label = clean_text(payload.get("label") or payload.get("model_name"))
        if model_label:
            mapping[model_label] = payload
    return mapping


def _resource_looks_non_model_file(relative_path: str) -> bool:
    text = clean_text(relative_path).lower()
    path = Path(text)
    ext = path.suffix.lower()
    name = path.name.lower()
    doc_exts = {
        ".txt",
        ".md",
        ".rst",
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".html",
        ".htm",
    }
    doc_tokens = {
        "readme",
        "license",
        "manual",
        "documentation",
        "summary",
        "report",
        "notes",
        "metadata",
        "about",
    }
    if ext in doc_exts:
        return True
    for token in doc_tokens:
        if token in name:
            return True
    return False


def _score_standard_variable_match(relative_path: str, standard_variable_name: str, io_record: dict[str, Any]) -> int:
    path_text = clean_text(relative_path).lower()
    filename = Path(path_text).name
    ext = Path(path_text).suffix.lower()
    score = 0

    sv_name = clean_text(standard_variable_name).lower()
    alias_texts = []
    alias_texts.extend([clean_text(x).lower() for x in (io_record.get("aliases") or []) if clean_text(x)])
    alias_texts.append(clean_text(io_record.get("name")).lower())
    alias_texts.append(clean_text(io_record.get("label")).lower())
    alias_texts.append(clean_text(io_record.get("quantity_term")).lower())
    alias_texts.append(clean_text(io_record.get("object_term")).lower())

    for alias in alias_texts:
        if alias and alias in path_text:
            score += 3

    if "simulation_archive" in sv_name:
        if ext in {".zip", ".tar", ".gz", ".tgz", ".7z"}:
            score += 8
        if "archive" in path_text:
            score += 2

    if "namefile_specification" in sv_name:
        if ext == ".nam":
            score += 9
        if "namefile" in path_text or "mfsim" in filename:
            score += 4

    if "package_input_set" in sv_name:
        package_exts = {
            ".bas", ".bcf", ".dis", ".drn", ".evt", ".ghb", ".gmg", ".lpf",
            ".oc", ".rch", ".str", ".wel", ".riv", ".upw", ".npf", ".sto",
            ".chd", ".sfr", ".uzf", ".maw", ".buy", ".in", ".ini", ".cfg",
        }
        if ext in package_exts:
            score += 8
        if any(token in filename for token in ["package", "input", "bcf", "bas", "dis", "wel", "rch", "drn", "ghb", "evt", "str"]):
            score += 3

    if "hydraulic_head" in sv_name:
        if ext in {".hds", ".hed", ".head"}:
            score += 9
        if "head" in filename or "hds" in filename:
            score += 3

    if "volumetric_budget" in sv_name:
        if ext in {".cbb", ".cbc", ".bud"}:
            score += 9
        if "budget" in filename or "cbb" in filename or "cbc" in filename:
            score += 3

    if "drawdown" in sv_name:
        if ext in {".ddn"}:
            score += 9
        if "drawdown" in filename or "ddn" in filename:
            score += 3

    if "convergence_diagnostic" in sv_name:
        if ext in {".lst", ".glo", ".res"}:
            score += 7
        if any(token in filename for token in ["solver", "convergence", "residual", "glo", "lst"]):
            score += 3

    return score


def _score_standard_variable_name_only(relative_path: str, standard_variable_name: str) -> int:
    path_text = clean_text(relative_path).lower()
    filename = Path(path_text).name
    ext = Path(path_text).suffix.lower()
    sv_name = clean_text(standard_variable_name).lower()
    score = 0

    # Generic lexical alignment.
    for token in ["simulation_archive", "namefile", "package_input_set", "hydraulic_head", "volumetric_budget", "drawdown", "convergence_diagnostic"]:
        if token in sv_name and token.replace("_", "") in filename.replace("_", ""):
            score += 3

    if "simulation_archive" in sv_name:
        if ext in {".zip", ".tar", ".gz", ".tgz", ".7z"}:
            score += 8
        if "archive" in filename:
            score += 2

    if "namefile_specification" in sv_name:
        if ext == ".nam":
            score += 9
        if "namefile" in filename or "mfsim" in filename:
            score += 4

    if "package_input_set" in sv_name:
        package_exts = {
            ".bas", ".bcf", ".dis", ".drn", ".evt", ".ghb", ".gmg", ".lpf",
            ".oc", ".rch", ".str", ".wel", ".riv", ".upw", ".npf", ".sto",
            ".chd", ".sfr", ".uzf", ".maw", ".buy", ".in", ".ini", ".cfg",
        }
        if ext in package_exts:
            score += 8
        if any(token in filename for token in ["package", "input", "bcf", "bas", "dis", "wel", "rch", "drn", "ghb", "evt", "str"]):
            score += 3

    if "hydraulic_head" in sv_name:
        if ext in {".hds", ".hed", ".head"}:
            score += 9
        if "head" in filename or "hds" in filename:
            score += 3

    if "volumetric_budget" in sv_name:
        if ext in {".cbb", ".cbc", ".bud"}:
            score += 9
        if "budget" in filename or "cbb" in filename or "cbc" in filename:
            score += 3

    if "drawdown" in sv_name:
        if ext in {".ddn"}:
            score += 9
        if "drawdown" in filename or "ddn" in filename:
            score += 3

    if "convergence_diagnostic" in sv_name:
        if ext in {".lst", ".glo", ".res"}:
            score += 7
        if any(token in filename for token in ["solver", "convergence", "residual", "glo", "lst"]):
            score += 3

    return score


def score_standard_variables_for_resource(
    resource_item: dict[str, Any],
    model_yaml_record: dict[str, Any] | None,
    available_standard_variable_names: list[str],
) -> dict[str, int]:
    relative_path = clean_text(resource_item.get("relative_path") or resource_item.get("resource_name"))
    names = [clean_text(x) for x in available_standard_variable_names if clean_text(x)]
    if not relative_path or not names:
        return {name: 0 for name in names}

    # Preferred: score using model YAML IO records.
    if model_yaml_record:
        io_records = []
        for key in ("inputs", "outputs"):
            io_records.extend(model_yaml_record.get(key, []) or [])

        score_by_name: dict[str, int] = {name: 0 for name in names}
        for record in io_records:
            sv_name = clean_text(record.get("standard_variable_name"))
            if not sv_name or sv_name not in score_by_name:
                continue
            score = _score_standard_variable_match(relative_path, sv_name, record)
            score_by_name[sv_name] = max(score_by_name.get(sv_name, 0), score)
        return score_by_name

    # MINT-first fallback: score directly from SVO names only.
    return {name: _score_standard_variable_name_only(relative_path, name) for name in names}


def match_standard_variables_for_resource(
    resource_item: dict[str, Any],
    model_yaml_record: dict[str, Any] | None,
    available_standard_variable_names: list[str],
) -> list[str]:
    relative_path = clean_text(resource_item.get("relative_path") or resource_item.get("resource_name"))
    if not relative_path or _resource_looks_non_model_file(relative_path):
        return []

    names = [clean_text(x) for x in available_standard_variable_names if clean_text(x)]
    if not names:
        return []

    score_by_name = score_standard_variables_for_resource(
        resource_item,
        model_yaml_record,
        names,
    )

    # Keep only available names and only high-confidence matches.
    filtered_scores = {name: score_by_name.get(name, 0) for name in names}
    best = max(filtered_scores.values()) if filtered_scores else 0
    if best < 4:
        return []

    # Return best match; include ties only when equally strong and >= best.
    winners = [name for name, score in filtered_scores.items() if score == best and score >= 4]
    return winners


def infer_model_label_from_resources(
    resource_files: list[Path],
    ext_counts: dict[str, int],
    candidate_model_labels: list[str] | None = None,
    *,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> dict[str, Any]:
    labels = [clean_text(label) for label in (candidate_model_labels or []) if clean_text(label)]
    label_by_normalized = {slugify(label): label for label in labels}

    bag_text = " ".join([p.name.lower() for p in resource_files[:300]])
    bag_text += " " + " ".join([str(p.parent).lower() for p in resource_files[:60]])

    heuristic = ""
    if re.search(r"modflow[-_ ]*2000", bag_text):
        heuristic = "MODFLOW-2000"
    elif re.search(r"modflow[-_ ]*96", bag_text):
        heuristic = "MODFLOW-96"
    elif re.search(r"modflow[-_ ]*6\\b|\\bmf6\\b", bag_text):
        heuristic = "MODFLOW 6"
    elif re.search(r"modflow[-_ ]*usg", bag_text):
        heuristic = "MODFLOW-USG"
    elif "modflow" in bag_text:
        heuristic = "MODFLOW"

    if heuristic:
        key = slugify(heuristic)
        if key in label_by_normalized:
            return {
                "model_label": label_by_normalized[key],
                "method": "heuristic",
                "confidence": "high",
                "rationale": f"Matched model token `{heuristic}` in filenames/paths.",
            }
        for label in labels:
            if key in slugify(label) or slugify(label) in key:
                return {
                    "model_label": label,
                    "method": "heuristic",
                    "confidence": "medium",
                    "rationale": f"Matched approximate model token `{heuristic}` to candidate label `{label}`.",
                }
        return {
            "model_label": heuristic,
            "method": "heuristic",
            "confidence": "medium",
            "rationale": f"Matched model token `{heuristic}` in filenames/paths.",
        }

    missing_reasons: list[str] = []
    if not llm_model:
        missing_reasons.append("llm_model is empty")
    if not clean_text(llm_api_key):
        missing_reasons.append("llm_api_key is empty")
    if OpenAI is None and not clean_text(llm_base_url):
        missing_reasons.append("OpenAI SDK unavailable and llm_base_url is empty")

    if missing_reasons:
        return {
            "model_label": labels[0] if labels else "",
            "method": "fallback",
            "confidence": "low",
            "rationale": "No direct model token found; LLM inference skipped: " + "; ".join(missing_reasons) + ".",
        }

    sample_names = [path.name for path in resource_files[:80]]
    prompt = """Choose the best model label from candidates for this file set.\nReturn strict JSON: {\"model_label\": string, \"confidence\": \"high|medium|low\", \"rationale\": string}."""
    payload = {
        "candidate_model_labels": labels,
        "file_extensions": ext_counts,
        "sample_filenames": sample_names,
    }
    content = _chat_completion_content(
        model=llm_model,
        api_key=llm_api_key or "",
        base_url=llm_base_url,
        system_prompt=prompt,
        user_payload=payload,
        temperature=0,
    )
    parsed = _parse_llm_json(content)
    chosen = clean_text(parsed.get("model_label"))
    if not chosen:
        chosen = labels[0] if labels else ""
    return {
        "model_label": chosen,
        "method": "llm",
        "confidence": clean_text(parsed.get("confidence")) or "low",
        "rationale": clean_text(parsed.get("rationale"), 400),
    }


def get_mint_standard_variables_for_model(
    model_label: str,
    *,
    mint_api_base: str = "https://api.models.mint.tacc.utexas.edu/v2.0.0",
    mint_api_token: str | None = None,
    fallback_model_standard_variable_names: list[str] | None = None,
    fallback_labels: list[str] | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    labels_to_try = [clean_text(model_label)]
    for label in (fallback_labels or []):
        cleaned = clean_text(label)
        if cleaned and cleaned not in labels_to_try:
            labels_to_try.append(cleaned)

    ids: list[str] = []
    names: list[str] = []
    source_parts: list[str] = []
    hierarchy_dataset_labels: list[str] = []

    for label in labels_to_try:
        if not label:
            continue
        label_dataset_labels = list_mint_model_configuration_dataset_labels(
            label,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        if label_dataset_labels:
            hierarchy_dataset_labels = label_dataset_labels
            inferred_names = infer_standard_variable_names_from_mint_dataset_labels(
                hierarchy_dataset_labels,
                model_label=label,
            )
            if inferred_names:
                names.extend(inferred_names)
                source_parts.append("model_configuration_hierarchy")
            break

    for label in labels_to_try:
        if not label:
            continue
        try:
            label_ids, label_names = list_mint_model_standard_variables(
                label,
                mint_api_base=mint_api_base,
                mint_api_token=mint_api_token,
                timeout=timeout,
                verify_ssl=verify_ssl,
            )
        except Exception:
            continue
        if label_ids or label_names:
            ids.extend(label_ids)
            names.extend(label_names)
            source_parts.append("custom/models/standard_variable")
            break

    registered_yaml_names = load_registered_model_standard_variable_names(labels_to_try)
    if registered_yaml_names:
        names.extend(registered_yaml_names)
        source_parts.append("registered_model_yamls")

    if fallback_model_standard_variable_names:
        fallback_names = [clean_text(x) for x in fallback_model_standard_variable_names if clean_text(x)]
        if fallback_names:
            names.extend(fallback_names)
            source_parts.append("fallback_standard_variable_names")

    ids = _dedupe_clean_text(ids)
    names = _dedupe_clean_text(names)

    unresolved_from_names: list[str] = []
    if names:
        resolved_ids, unresolved_from_names = resolve_mint_standard_variable_ids(
            names,
            mint_api_base=mint_api_base,
            mint_api_token=mint_api_token,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        ids = _dedupe_clean_text(resolved_ids + ids)

    return {
        "model_label": clean_text(model_label),
        "standard_variable_ids": ids,
        "standard_variable_names": names,
        "unresolved_standard_variable_names": unresolved_from_names,
        "source": "+".join(_dedupe_clean_text(source_parts)) or "none",
        "mint_dataset_labels": hierarchy_dataset_labels,
    }


def annotate_resource_plan_with_mint_standard_variables(
    resource_plan: list[dict[str, Any]],
    *,
    standard_variable_ids: list[str] | None = None,
    standard_variable_names: list[str] | None = None,
    model_label: str | None = None,
    model_yaml_record: dict[str, Any] | None = None,
    value_mode: str = "name",
    target_field: str = "mint_standard_variables",
    delimiter: str = "; ",
) -> list[dict[str, Any]]:
    ids = [clean_text(x) for x in (standard_variable_ids or []) if clean_text(x)]
    names = [clean_text(x) for x in (standard_variable_names or []) if clean_text(x)]
    mode = clean_text(value_mode).lower() or "name"

    for item in resource_plan:
        score_by_name = score_standard_variables_for_resource(
            item,
            model_yaml_record,
            names or ids,
        )
        matched_names = match_standard_variables_for_resource(
            item,
            model_yaml_record,
            names or ids,
        )
        if matched_names:
            if mode == "id" and ids:
                # Preserve order by matched names when possible.
                name_to_id = {}
                for i, name in enumerate(names):
                    if i < len(ids):
                        name_to_id[name] = ids[i]
                matched_ids = [name_to_id.get(name) for name in matched_names if name_to_id.get(name)]
                payload_value = delimiter.join(matched_ids or matched_names)
            else:
                payload_value = delimiter.join(matched_names)
            item[target_field] = payload_value
        else:
            item.pop(target_field, None)
        item["_mint_svo_match_scores"] = score_by_name
        item["_mint_svo_selected"] = matched_names
        if clean_text(model_label):
            item["mint_model_label"] = clean_text(model_label)
    return resource_plan


def get_tapis_token(username: str, password: str, *, tapis_url: str = DEFAULT_TAPIS_URL) -> str:
    response = requests.post(
        tapis_url,
        data={
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["result"]["access_token"]["access_token"]


def build_ckan_auth_header(
    *,
    auth_mode: str,
    api_token: str | None = None,
    username: str | None = None,
    password: str | None = None,
    tapis_url: str = DEFAULT_TAPIS_URL,
) -> str | None:
    normalized_mode = (auth_mode or "api_token").strip().lower()
    if normalized_mode == "api_token":
        token = (api_token or "").strip()
        return token or None
    if normalized_mode == "tapis_password":
        user = (username or "").strip()
        secret = password or ""
        if not user or not secret:
            raise ValueError("TACC username and password are required for CKAN tapis_password authentication.")
        return f"Bearer {get_tapis_token(user, secret, tapis_url=tapis_url)}"
    raise ValueError(f"Unsupported CKAN auth mode: {auth_mode}")


def auth_headers(auth_header: str | None = None) -> dict[str, str]:
    if not auth_header:
        return {}
    return {"Authorization": auth_header}


def fetch_ckan_dataset(
    base_url: str,
    dataset_name: str,
    auth_header: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}/api/3/action/package_show",
        params={"id": dataset_name},
        headers=auth_headers(auth_header),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise ValueError(f"CKAN package_show failed for {dataset_name}")
    return payload["result"]


def _parse_llm_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    object_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    return {}


def propose_ckan_dataset_metadata_with_llm(
    resource_plan: list[dict[str, Any]],
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    source_metadata_url: str | None = None,
    preferred_dataset_name: str | None = None,
    preferred_dataset_title: str | None = None,
    preferred_dataset_url: str | None = None,
    preferred_dataset_author: str | None = None,
    preferred_dataset_author_email: str | None = None,
    preferred_dataset_maintainer: str | None = None,
    preferred_dataset_maintainer_email: str | None = None,
    preferred_dataset_license_id: str | None = None,
    preferred_dataset_version: str | None = None,
    preferred_dataset_type: str | None = None,
    preferred_dataset_isopen: bool | None = None,
    preferred_dataset_spatial: str | None = None,
    preferred_temporal_coverage_start: str | None = None,
    preferred_temporal_coverage_end: str | None = None,
    preferred_dataset_tags: list[dict[str, str]] | None = None,
    preserve_preferred_values: bool = False,
) -> dict[str, Any]:
    """Propose a CKAN-ready metadata object using LLM plus optional source URL context.

    Returns keys:
    dataset_name, dataset_title, dataset_notes, dataset_url, dataset_author,
    dataset_author_email, dataset_maintainer, dataset_maintainer_email,
    dataset_license_id, dataset_version, dataset_type, dataset_isopen,
    dataset_spatial, temporal_coverage_start, temporal_coverage_end, dataset_tags
    """

    if not resource_plan:
        raise ValueError("resource_plan is empty; cannot propose dataset metadata.")
    if not clean_text(api_key):
        raise ValueError("api_key is required for LLM metadata proposal.")
    if not clean_text(model):
        raise ValueError("model is required for LLM metadata proposal.")

    sample_resources = []
    for resource in resource_plan[:20]:
        sample_resources.append(
            {
                "resource_name": resource.get("resource_name", ""),
                "resource_title": resource.get("resource_title", ""),
                "resource_tags": resource.get("resource_tags", []),
                "resource_description": clean_text(resource.get("resource_description", ""), max_chars=240),
            }
        )

    source_context: dict[str, str] | None = None
    cleaned_source_url = clean_text(source_metadata_url)
    if cleaned_source_url:
        try:
            source_context = fetch_source_metadata(cleaned_source_url)
        except Exception:
            source_context = {"url": cleaned_source_url, "title": "", "meta_description": "", "excerpt": ""}

    prompt =  """
You are generating CKAN dataset-level metadata from:
1. a source website,
2. initial dataset metadata, and
3. resource/file metadata.

Your job is to improve the dataset metadata so it is accurate, concise, useful for discovery, and appropriate for CKAN.

Return STRICT JSON only. Do not include markdown, comments, explanations, or trailing commas.

Required JSON keys:
{
  "dataset_name": string,
  "dataset_title": string,
  "dataset_notes": string,
  "dataset_url": string,
  "dataset_author": string or null,
  "dataset_author_email": string or null,
  "dataset_maintainer": string or null,
  "dataset_maintainer_email": string or null,
  "dataset_license_id": string or null,
  "dataset_version": string or null,
  "dataset_type": string,
  "dataset_isopen": boolean,
  "dataset_spatial": string or null,
  "temporal_coverage_start": string or null,
  "temporal_coverage_end": string or null,
  "dataset_tags": list[string]
}

Field rules:

dataset_name:
- Must be lowercase, URL-safe, and hyphenated.
- Use only letters, numbers, and hyphens.
- Do not include file extensions.
- Prefer a stable descriptive name based on aquifer/model/topic/source.
- Example: "yegua-jackson-aquifer-groundwater-availability-model-files".

dataset_title:
- Use a human-readable title.
- Prefer the official source page title when useful.
- Remove site boilerplate such as agency slogans, browser warnings, navigation labels, and alerts.
- Title case is acceptable.

dataset_notes:
- Write 2–4 concise sentences.
- Describe what the dataset contains, what system/model/project it belongs to, and how it may be used.
- Mention the source organization if known.
- Mention that files/resources are model inputs, outputs, documentation, or supporting files when inferable.
- Include the source URL only if useful, but do not dump raw metadata.
- Do not include phishing alerts, JavaScript warnings, menus, cookie banners, or unrelated website text.
- Do not invent dates, authors, emails, spatial coverage, or temporal coverage.
- If information is not available, leave the relevant field null rather than guessing.

dataset_url:
- Use the canonical source metadata URL when provided.

dataset_author / dataset_author_email:
- Use the original data-producing organization or named author if clearly identified.
- For Texas Water Development Board pages, use "Texas Water Development Board" as author when no more specific author is available.
- Use null for email unless a specific email is provided.

dataset_maintainer / dataset_maintainer_email:
- Preserve the provided maintainer values unless there is clear reason to improve them.
- Do not replace the local CKAN maintainer with the source organization unless explicitly instructed.

dataset_license_id:
- Preserve the provided license_id unless the source clearly states a different license.
- Use CKAN license IDs such as "cc-by" when provided.
- Use null if unknown.

dataset_version:
- Use a model version, report version, publication year, or revision only if clearly stated.
- Otherwise null.

dataset_type:
- Usually "dataset" unless another valid CKAN type is provided.

dataset_isopen:
- Preserve the provided boolean unless the source clearly indicates restricted access.

dataset_spatial:
- Use a concise place name or spatial coverage description if inferable from the dataset title or source.
- For aquifer models, prefer the aquifer name and state/region when clear.
- Example: "Yegua-Jackson Aquifer, Texas".
- Use null if unclear.

temporal_coverage_start / temporal_coverage_end:
- Use ISO date strings: "YYYY-MM-DD" when exact dates are known, or "YYYY" only if the system supports year-only values.
- Only populate when the dataset clearly describes a time period represented by the data/model.
- Do not use the metadata generation date as temporal coverage.
- Use null when unknown.

dataset_tags:
- Return a list of short, useful, lowercase tag strings.
- Tags should support discovery.
- Include subject, geography, source agency, model type, and major data formats when useful.
- Do not include too many file-extension tags; keep only the most informative extensions or resource types.
- Avoid duplicates.
- Avoid overly generic tags unless they are useful, such as "groundwater", "aquifer", "modflow", "gam".
- Prefer 6–15 tags.

Important:
- Improve noisy metadata into clean CKAN metadata.
- Preserve correct provided values.
- Do not hallucinate missing facts.
- When uncertain, use null.
- Output must be valid JSON.
"""
    
    payload = {
        "preferred_dataset_name": (preferred_dataset_name or "") if preserve_preferred_values else "",
        "preferred_dataset_title": (preferred_dataset_title or "") if preserve_preferred_values else "",
        "preferred_dataset_url": (preferred_dataset_url or "") if preserve_preferred_values else "",
        "preferred_dataset_author": (preferred_dataset_author or "") if preserve_preferred_values else "",
        "preferred_dataset_author_email": (preferred_dataset_author_email or "") if preserve_preferred_values else "",
        "preferred_dataset_maintainer": (preferred_dataset_maintainer or "") if preserve_preferred_values else "",
        "preferred_dataset_maintainer_email": (preferred_dataset_maintainer_email or "") if preserve_preferred_values else "",
        "preferred_dataset_license_id": (preferred_dataset_license_id or "") if preserve_preferred_values else "",
        "preferred_dataset_version": (preferred_dataset_version or "") if preserve_preferred_values else "",
        "preferred_dataset_type": (preferred_dataset_type or "") if preserve_preferred_values else "",
        "preferred_dataset_isopen": preferred_dataset_isopen if preserve_preferred_values else None,
        "preferred_dataset_spatial": (preferred_dataset_spatial or "") if preserve_preferred_values else "",
        "preferred_temporal_coverage_start": (preferred_temporal_coverage_start or "") if preserve_preferred_values else "",
        "preferred_temporal_coverage_end": (preferred_temporal_coverage_end or "") if preserve_preferred_values else "",
        "preferred_dataset_tags": [tag.get("name", "") for tag in (preferred_dataset_tags or [])] if preserve_preferred_values else [],
        "source_metadata": source_context or {},
        "resource_count": len(resource_plan),
        "resources": sample_resources,
    }
    content = _chat_completion_content(
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=prompt,
        user_payload=payload,
        temperature=0.1,
    )
    parsed = _parse_llm_json(content)

    def choose_text(parsed_key: str, preferred_value: str | None, fallback_value: str = "") -> str:
        if preserve_preferred_values and clean_text(preferred_value):
            return clean_text(preferred_value)
        return clean_text(parsed.get(parsed_key) or preferred_value or fallback_value)

    def choose_bool(parsed_key: str, preferred_value: bool | None, fallback_value: bool | None = None) -> bool | None:
        if preserve_preferred_values and preferred_value is not None:
            return preferred_value
        parsed_value = parsed.get(parsed_key)
        if isinstance(parsed_value, bool):
            return parsed_value
        if isinstance(parsed_value, str):
            normalized = parsed_value.strip().lower()
            if normalized in {"1", "true", "yes"}:
                return True
            if normalized in {"0", "false", "no"}:
                return False
        if preferred_value is not None:
            return preferred_value
        return fallback_value

    def choose_tags() -> list[dict[str, str]]:
        if preserve_preferred_values and preferred_dataset_tags:
            return dedupe_tags([str(tag.get("name", "")) for tag in preferred_dataset_tags if tag.get("name")])

        parsed_tags = parsed.get("dataset_tags")
        if isinstance(parsed_tags, list):
            parsed_tag_text = [str(tag) for tag in parsed_tags]
        else:
            parsed_tag_text = []

        fallback_tags = [str(tag.get("name", "")) for tag in (preferred_dataset_tags or []) if tag.get("name")]
        if not parsed_tag_text:
            parsed_tag_text = fallback_tags
        return dedupe_tags(parsed_tag_text)

    if preserve_preferred_values:
        title_source = preferred_dataset_title or parsed.get("dataset_title") or "Standalone Dataset"
        name_source = preferred_dataset_name or parsed.get("dataset_name")
    else:
        title_source = parsed.get("dataset_title") or preferred_dataset_title or "Standalone Dataset"
        name_source = parsed.get("dataset_name") or preferred_dataset_name

    inferred_title = clean_text(title_source, max_chars=140)
    inferred_name = slugify(str(name_source or inferred_title)) or "standalone-dataset"
    inferred_notes = clean_text(
        parsed.get("dataset_notes")
        or f"Dataset registered from a local corpus. Resource count: {len(resource_plan)}.",
        max_chars=3000,
    )
    return {
        "dataset_name": inferred_name,
        "dataset_title": inferred_title,
        "dataset_notes": inferred_notes,
        "dataset_url": choose_text("dataset_url", preferred_dataset_url, source_context["url"] if source_context else ""),
        "dataset_author": choose_text("dataset_author", preferred_dataset_author),
        "dataset_author_email": choose_text("dataset_author_email", preferred_dataset_author_email),
        "dataset_maintainer": choose_text("dataset_maintainer", preferred_dataset_maintainer),
        "dataset_maintainer_email": choose_text("dataset_maintainer_email", preferred_dataset_maintainer_email),
        "dataset_license_id": choose_text("dataset_license_id", preferred_dataset_license_id),
        "dataset_version": choose_text("dataset_version", preferred_dataset_version),
        "dataset_type": choose_text("dataset_type", preferred_dataset_type, "dataset"),
        "dataset_isopen": choose_bool("dataset_isopen", preferred_dataset_isopen, True),
        "dataset_spatial": choose_text("dataset_spatial", preferred_dataset_spatial),
        "temporal_coverage_start": choose_text("temporal_coverage_start", preferred_temporal_coverage_start),
        "temporal_coverage_end": choose_text("temporal_coverage_end", preferred_temporal_coverage_end),
        "dataset_tags": choose_tags(),
    }


def _ckan_call_delay_seconds() -> float:
    """Inter-call throttle for CKAN writes/reads.

    Read from ``CKAN_CALL_DELAY_SECONDS`` (default 0.5s) so bulk per-file
    resource uploads don't hammer the CKAN instance.  Set to ``0`` to disable.
    """
    raw = os.getenv("CKAN_CALL_DELAY_SECONDS", "0.5").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.5
    return value if value > 0 else 0.0


_CKAN_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _ckan_max_retries() -> int:
    """Max retries for transient CKAN gateway errors (CKAN_MAX_RETRIES, default 5)."""
    raw = os.getenv("CKAN_MAX_RETRIES", "5").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 5


def ckan_action_post(
    ckan_url: str,
    action: str,
    payload: dict[str, Any],
    auth_header: str | None,
    files: dict[str, Any] | None = None,
    timeout: int = 240,
) -> dict[str, Any]:
    """POST a CKAN action, retrying transient gateway errors (429/500/502/503/504).

    Uploads are sequential and each call is confirmed (the result is returned,
    or a RuntimeError raised) before the caller proceeds to the next resource.
    On a transient failure the call is retried with exponential back-off; any
    file handles in *files* are re-sought to position 0 first, so the FULL body
    is re-sent (a half-consumed handle would upload an empty/partial file).
    """
    url = f"{ckan_url.rstrip('/')}/api/3/action/{action}"
    headers = auth_headers(auth_header)
    max_retries = _ckan_max_retries()
    attempt = 0
    while True:
        # Rewind file handles so a retry re-sends the full content.
        if files:
            for _fobj in files.values():
                try:
                    _fobj.seek(0)
                except (AttributeError, OSError):
                    pass
            kwargs: dict[str, Any] = {
                "headers": headers, "timeout": timeout, "data": payload, "files": files,
            }
        else:
            kwargs = {"headers": headers, "timeout": timeout, "json": payload}

        # Throttle to avoid hammering CKAN during bulk per-file uploads.
        _delay = _ckan_call_delay_seconds()
        if _delay:
            time.sleep(_delay)

        try:
            response = requests.post(url, **kwargs)
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                backoff = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 0.5)
                logger.warning(
                    "CKAN %s connection error (attempt %d/%d): %s; retrying in %.1fs",
                    action, attempt + 1, max_retries, exc, backoff,
                )
                time.sleep(backoff)
                attempt += 1
                continue
            raise

        if response.status_code in _CKAN_RETRYABLE_STATUSES and attempt < max_retries:
            retry_after = response.headers.get("Retry-After")
            try:
                backoff = (
                    min(60.0, float(retry_after)) if retry_after
                    else min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 0.5)
                )
            except (TypeError, ValueError):
                backoff = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 0.5)
            logger.warning(
                "CKAN %s HTTP %s (attempt %d/%d); retrying in %.1fs",
                action, response.status_code, attempt + 1, max_retries, backoff,
            )
            time.sleep(backoff)
            attempt += 1
            continue

        try:
            body = response.json()
        except ValueError:
            body = {}

        if response.status_code >= 400:
            raise RuntimeError(f"CKAN {action} HTTP {response.status_code}: {body or response.text[:1000]}")
        if not body.get("success"):
            raise RuntimeError(f"CKAN {action} failed: {body.get('error')}")
        return body["result"]


def create_or_update_ckan_dataset(
    base_url: str,
    *,
    source_metadata_url: str | None = None,
    dataset_name: str,
    dataset_title: str,
    dataset_notes: str,
    dataset_tags: list[dict[str, str]] | None = None,
    auth_header: str | None = None,
    owner_org: str | None = None,
    private: bool = False,
    dataset_author: str | None = None,
    dataset_author_email: str | None = None,
    dataset_maintainer: str | None = None,
    dataset_maintainer_email: str | None = None,
    dataset_license_id: str | None = None,
    dataset_url: str | None = None,
    dataset_version: str | None = None,
    dataset_type: str | None = "dataset",
    dataset_isopen: bool | None = None,
    dataset_spatial: str | None = None,
    temporal_coverage_start: str | None = None,
    temporal_coverage_end: str | None = None,
    dataset_extras: list[dict[str, str]] | None = None,
    extra_fields: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    create_payload: dict[str, Any] = {
        "name": dataset_name,
        "title": dataset_title,
        "notes": dataset_notes,
        "private": private,
        "tags": dataset_tags or [],
    }
    if owner_org:
        create_payload["owner_org"] = owner_org
    if dataset_type:
        create_payload["type"] = clean_text(dataset_type, max_chars=100)
    if dataset_isopen is not None:
        create_payload["isopen"] = bool(dataset_isopen)

    optional_text_fields = {
        "author": dataset_author,
        "author_email": dataset_author_email,
        "maintainer": dataset_maintainer,
        "maintainer_email": dataset_maintainer_email,
        "license_id": dataset_license_id,
        "url": dataset_url or source_metadata_url,
        "version": dataset_version,
        "spatial": dataset_spatial,
        "temporal_coverage_start": temporal_coverage_start,
        "temporal_coverage_end": temporal_coverage_end,
    }
    for key, value in optional_text_fields.items():
        cleaned = clean_text(value)
        if cleaned:
            create_payload[key] = cleaned

    if dataset_extras:
        create_payload["extras"] = dataset_extras

    if extra_fields:
        for key, value in extra_fields.items():
            if value is not None:
                create_payload[key] = value

    try:
        existing = fetch_ckan_dataset(base_url, dataset_name, auth_header=auth_header, timeout=timeout)
    except Exception:
        return ckan_action_post(
            base_url,
            "package_create",
            create_payload,
            auth_header=auth_header,
            timeout=timeout,
        )

    patch_payload = dict(create_payload)
    patch_payload.pop("owner_org", None)

    desired_owner_org = clean_text(owner_org)
    existing_org = existing.get("organization") if isinstance(existing.get("organization"), dict) else {}
    existing_owner_candidates = {
        clean_text(existing.get("owner_org")).lower(),
        clean_text(existing_org.get("id")).lower(),
        clean_text(existing_org.get("name")).lower(),
        clean_text(existing_org.get("title")).lower(),
    }
    existing_owner_candidates.discard("")

    if desired_owner_org and desired_owner_org.lower() not in existing_owner_candidates:
        ckan_action_post(
            base_url,
            "package_owner_org_update",
            {"id": existing["id"], "organization_id": desired_owner_org},
            auth_header=auth_header,
            timeout=timeout,
        )

    patch_payload["id"] = existing["id"]
    return ckan_action_post(
        base_url,
        "package_patch",
        patch_payload,
        auth_header=auth_header,
        timeout=timeout,
    )


def fetch_existing_dataset_or_none(ckan_url: str, dataset_name: str, auth_header: str | None) -> dict[str, Any] | None:
    try:
        return fetch_ckan_dataset(ckan_url, dataset_name, auth_header=auth_header)
    except Exception:
        return None


def compare_dataset_metadata(existing: dict[str, Any] | None, desired: dict[str, Any]) -> list[dict[str, str]]:
    fields = [
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
    ]
    changes = []
    if existing is None:
        for key in fields:
            changes.append(
                {
                    "field": key,
                    "existing": "<missing dataset>",
                    "desired": clean_text(desired.get(key, ""), 240),
                    "status": "create",
                }
            )
        return changes

    existing_owner = clean_text(existing.get("owner_org") or (existing.get("organization") or {}).get("id"))
    existing_tags = sorted(
        {
            sanitize_tag(str(item.get("name", "")))
            for item in existing.get("tags", [])
            if sanitize_tag(str(item.get("name", "")))
        }
    )
    desired_tags = sorted(
        {
            sanitize_tag(str(item.get("name", "")))
            for item in desired.get("tags", [])
            if sanitize_tag(str(item.get("name", "")))
        }
    )
    existing_values = {
        "title": clean_text(existing.get("title", ""), 240),
        "notes": clean_text(existing.get("notes", ""), 240),
        "url": clean_text(existing.get("url", ""), 240),
        "owner_org": existing_owner,
        "private": str(bool(existing.get("private", False))),
        "author": clean_text(existing.get("author", ""), 240),
        "author_email": clean_text(existing.get("author_email", ""), 240),
        "maintainer": clean_text(existing.get("maintainer", ""), 240),
        "maintainer_email": clean_text(existing.get("maintainer_email", ""), 240),
        "license_id": clean_text(existing.get("license_id", ""), 240),
        "version": clean_text(existing.get("version", ""), 240),
        "type": clean_text(existing.get("type", ""), 240),
        "isopen": str(bool(existing.get("isopen", False))) if existing.get("isopen") is not None else "",
        "spatial": clean_text(existing.get("spatial", ""), 240),
        "temporal_coverage_start": clean_text(existing.get("temporal_coverage_start", ""), 240),
        "temporal_coverage_end": clean_text(existing.get("temporal_coverage_end", ""), 240),
        "tags": ", ".join(existing_tags),
    }
    desired_values = {
        "title": clean_text(desired.get("title", ""), 240),
        "notes": clean_text(desired.get("notes", ""), 240),
        "url": clean_text(desired.get("url", ""), 240),
        "owner_org": clean_text(desired.get("owner_org", ""), 240),
        "private": str(bool(desired.get("private", False))),
        "author": clean_text(desired.get("author", ""), 240),
        "author_email": clean_text(desired.get("author_email", ""), 240),
        "maintainer": clean_text(desired.get("maintainer", ""), 240),
        "maintainer_email": clean_text(desired.get("maintainer_email", ""), 240),
        "license_id": clean_text(desired.get("license_id", ""), 240),
        "version": clean_text(desired.get("version", ""), 240),
        "type": clean_text(desired.get("type", ""), 240),
        "isopen": str(bool(desired.get("isopen", False))) if desired.get("isopen") is not None else "",
        "spatial": clean_text(desired.get("spatial", ""), 240),
        "temporal_coverage_start": clean_text(desired.get("temporal_coverage_start", ""), 240),
        "temporal_coverage_end": clean_text(desired.get("temporal_coverage_end", ""), 240),
        "tags": ", ".join(desired_tags),
    }

    for key in fields:
        before = existing_values[key]
        after = desired_values[key]
        status = "same" if before == after else "update"
        changes.append({"field": key, "existing": before, "desired": after, "status": status})
    return changes


def render_changes_table_markdown(changes: list[dict[str, str]]) -> str:
    lines = [
        "| field | existing | desired | status |",
        "|---|---|---|---|",
    ]
    for row in changes:
        lines.append(
            "| {field} | {existing} | {desired} | {status} |".format(
                field=row["field"],
                existing=str(row["existing"]).replace("|", "\\|"),
                desired=str(row["desired"]).replace("|", "\\|"),
                status=row["status"],
            )
        )
    return "\n".join(lines)


def existing_resources_by_name(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(resource.get("name", "")): resource for resource in dataset.get("resources", []) if resource.get("name")}


def upsert_resources(
    ckan_url: str,
    dataset: dict[str, Any],
    resource_plan: list[dict[str, Any]],
    auth_header: str | None,
    extra_resource_fields: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    existing_by_name = existing_resources_by_name(dataset)
    uploaded = []
    created = 0
    updated = 0
    _total = len(resource_plan)

    for _idx, item in enumerate(resource_plan, 1):
        local_path: Path = item["local_path"]
        payload = {
            "package_id": dataset.get("id") or dataset.get("name"),
            "name": item["resource_name"],
            "description": clean_text(item.get("resource_description", ""), 3000),
            "format": item.get("format", "BIN"),
            "url": item.get("source_url") or "upload",
        }
        mimetype = mimetypes.guess_type(local_path.name)[0]
        if mimetype:
            payload["mimetype"] = mimetype

        existing = existing_by_name.get(item["resource_name"])
        action = "resource_create"
        if existing is not None:
            payload["id"] = existing["id"]
            action = "resource_update"

        explicit_fields = [clean_text(name) for name in (extra_resource_fields or []) if clean_text(name)]
        auto_fields = [key for key in item.keys() if key.startswith("mint_")]
        for field_name in dict.fromkeys(explicit_fields + auto_fields):
            value = item.get(field_name)
            if isinstance(value, list):
                value = "; ".join([clean_text(x) for x in value if clean_text(x)])
            value_text = clean_text(value, 4000)
            if value_text:
                payload[field_name] = value_text
            elif field_name in explicit_fields and action == "resource_update":
                # Explicitly clear stale values on update when this file has no match.
                payload[field_name] = ""

        logger.info(
            "upsert_resources: [%d/%d] %s '%s' (%s)",
            _idx, _total, action, item["resource_name"], local_path.name,
        )
        with local_path.open("rb") as handle:
            result = ckan_action_post(
                ckan_url,
                action,
                payload,
                auth_header=auth_header,
                files={"upload": handle},
            )
            uploaded.append(result)

        if action == "resource_create":
            created += 1
        else:
            updated += 1

    return uploaded, created, updated


def create_link_resources(
    ckan_url: str,
    dataset: dict[str, Any],
    link_resources: list[dict[str, Any]],
    auth_header: str | None,
) -> list[dict[str, Any]]:
    """Create or update url-type (external-link) CKAN resources on *dataset*.

    Each item in *link_resources* must contain ``name`` and ``url`` at minimum.
    Additional SUBSIDE resource-level fields (``abstract``, ``format``,
    ``url_type``, ``description``, etc.) are passed through unchanged.

    Idempotent: if a resource with the same ``name`` already exists on the
    dataset, ``resource_update`` is called; otherwise ``resource_create``.
    No file bytes are uploaded (``url_type="url"``).

    Parameters
    ----------
    ckan_url:
        Base CKAN instance URL (e.g. ``https://ckan.tacc.utexas.edu``).
    dataset:
        CKAN dataset dict (must contain ``id`` or ``name``).
    link_resources:
        List of resource dicts with at minimum ``name`` and ``url``.
    auth_header:
        Authorization header value (Bearer token or API key).

    Returns
    -------
    list[dict]
        CKAN result dicts for each resource_create / resource_update call.
    """
    existing_by_name = existing_resources_by_name(dataset)
    results: list[dict[str, Any]] = []
    package_id = dataset.get("id") or dataset.get("name")

    for item in link_resources:
        name = clean_text(item.get("name", ""))
        url = clean_text(item.get("url", ""))
        if not name or not url:
            continue  # Skip incomplete link resources.

        payload: dict[str, Any] = {
            "package_id": package_id,
            "name": name,
            "url": url,
            "url_type": item.get("url_type", "url"),
            "format": clean_text(item.get("format", ""), 100) or "",
            "description": clean_text(item.get("description", ""), 3000) or "",
        }
        # Pass through any additional SUBSIDE resource-level fields.
        _pass_through_keys = {
            "abstract", "program_area", "data_contact_email", "caveats_usage",
            "categories", "primary_tags", "secondary_tags", "collection_method",
            "quality_control_level", "spatial",
        }
        for key in _pass_through_keys:
            if key in item and item[key] is not None:
                payload[key] = item[key]

        existing = existing_by_name.get(name)
        action = "resource_create"
        if existing is not None:
            payload["id"] = existing["id"]
            action = "resource_update"

        result = ckan_action_post(ckan_url, action, payload, auth_header=auth_header)
        results.append(result)

    return results


def remove_stale_resources(
    ckan_url: str,
    dataset: dict[str, Any],
    planned_resource_names: set[str],
    auth_header: str | None,
) -> int:
    existing = dataset.get("resources", [])
    removed = 0
    for resource in existing:
        name = str(resource.get("name", ""))
        resource_id = resource.get("id")
        if not name or not resource_id:
            continue
        if name not in planned_resource_names:
            ckan_action_post(
                ckan_url,
                "resource_delete",
                {"id": resource_id},
                auth_header=auth_header,
            )
            removed += 1
    return removed
