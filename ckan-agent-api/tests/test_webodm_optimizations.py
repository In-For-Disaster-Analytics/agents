"""Tests for the WebODM agent optimizations.

Covers:
  1. _invalid_args_error — URL-in-local-path-tool + placeholder detection
  2. fetch_remote_pdf handler — URL validation (non-PDF, non-HTTPS)
  3. Engine loop — pdf_summarize(url) rejected without calling executor.invoke
  4. Engine loop — fetch_remote_pdf(zip_url) returns not_a_pdf_url from handler
  5. Engine loop — model calls fetch_remote_pdf(report.pdf) in round 1, converges
  6. LLM_CALL_DELAY_SECONDS default is 1.0 (was 5.0)
  7. End-to-end mocked: WebODM payload → CKAN metadata, url field is null

Run from ckan-agent-api/:
    pytest tests/test_webodm_optimizations.py -v
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.personas import PersonaRegistry, run_persona_metadata_loop
from app.personas.engine import (
    STOP_CONVERGED,
    _invalid_args_error,
)
from app.schemas import SchemaRegistry
from app.settings import PROJECT_ROOT, get_settings
from app.tools.handlers.remote import fetch_remote_pdf

SEED_PERSONAS_DIR = PROJECT_ROOT / "app" / "personas"
SEED_SCHEMAS_DIR = PROJECT_ROOT / "app" / "schemas"

WEBODM_HOST = "https://webodm.tacc.utexas.edu"
PROJECT_ID = 6
TASK_ID = "fe1a55e1-fe4b-417a-98fc-94d11cff2fde"

REPORT_PDF_URL = f"{WEBODM_HOST}/api/projects/{PROJECT_ID}/tasks/{TASK_ID}/download/report.pdf"
ALL_ZIP_URL = f"{WEBODM_HOST}/api/projects/{PROJECT_ID}/tasks/{TASK_ID}/download/all.zip"
ORTHO_URL = f"{WEBODM_HOST}/api/projects/{PROJECT_ID}/tasks/{TASK_ID}/download/orthophoto.tif"
MAP_URL = f"{WEBODM_HOST}/public/task/{TASK_ID}/map/"

REMOTE_RESOURCES = [
    {"url": ORTHO_URL, "name": "Orthophoto (GeoTIFF)", "format": "GTiff"},
    {"url": f"{WEBODM_HOST}/api/projects/{PROJECT_ID}/tasks/{TASK_ID}/download/dsm.tif", "name": "Digital Surface Model", "format": "GTiff"},
    {"url": REPORT_PDF_URL, "name": "Processing Report (PDF)", "format": "PDF"},
    {"url": ALL_ZIP_URL, "name": "All Outputs (ZIP)", "format": "ZIP"},
    {"url": MAP_URL, "name": "Web Map Viewer", "format": "HTML"},
    {"url": f"{WEBODM_HOST}/public/task/{TASK_ID}/3d/", "name": "3D Model Viewer", "format": "HTML"},
]

GOOD_CANDIDATE = {
    "title": "TACC Multispectral Orthophoto and SfM Point Cloud Survey (2026-02-11)",
    "name": "tacc-multispectral-orthophoto-sfm-2026",
    "notes": "Multispectral orthophoto, SfM point cloud, DSM, and DTM of the TACC campus. Processed 2026-02-11.",
    "tag_string": "drone,dsm,dtm,elevation,multispectral,orthophoto,sfm,uas",
    "temporal_coverage_start": "2026-02-11",
    "temporal_coverage_end": "2026-02-11",
    "author": "wmobley",
    "author_email": "wmobley@tacc.utexas.edu",
    "maintainer": "William Mobley",
    "maintainer_email": "wmobley@tacc.utexas.edu",
    "license_id": "cc-by",
    "owner_org": "tacc",
    "url": None,
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _personas():
    reg = PersonaRegistry(SEED_PERSONAS_DIR)
    return reg.author(), reg.evaluators()


def _generic():
    return SchemaRegistry(SEED_SCHEMAS_DIR).get("generic_ckan")


def _pass_eval(system: str, payload: str) -> str:
    return json.dumps({"verdict": "pass", "questions": [], "recommendations": []})


# ── 1. _invalid_args_error ────────────────────────────────────────────────────


class TestInvalidArgsError:
    def test_url_to_pdf_summarize_returns_redirect_message(self):
        err = _invalid_args_error("pdf_summarize", {"path": REPORT_PDF_URL})
        assert err is not None
        assert "fetch_remote_pdf" in err

    def test_url_to_file_read_text(self):
        err = _invalid_args_error("file_read_text", {"path": ORTHO_URL})
        assert err is not None
        assert "fetch_remote_pdf" in err

    def test_url_to_gdal_info(self):
        err = _invalid_args_error("gdal_info", {"path": ORTHO_URL})
        assert err is not None

    def test_valid_local_path_passes(self):
        assert _invalid_args_error("pdf_summarize", {"path": "/tmp/report.pdf"}) is None

    def test_placeholder_path_detected(self):
        err = _invalid_args_error("pdf_summarize", {"path": "path_to_pdf_file"})
        assert err is not None
        assert "placeholder" in err

    def test_placeholder_query_detected(self):
        err = _invalid_args_error("ckan_package_search", {"query": "your_query"})
        assert err is not None

    def test_fetch_remote_pdf_with_valid_url_passes(self):
        # fetch_remote_pdf is NOT in _LOCAL_PATH_TOOLS — URL args are valid
        assert _invalid_args_error("fetch_remote_pdf", {"url": REPORT_PDF_URL}) is None

    def test_http_url_caught_for_local_path_tool(self):
        err = _invalid_args_error("pdf_summarize", {"path": "http://example.com/x.pdf"})
        assert err is not None  # http:// is a URL; pdf_summarize is a local-path tool


# ── 2. fetch_remote_pdf handler URL validation ───────────────────────────────


class TestFetchRemotePdfValidation:
    def test_rejects_zip_url(self):
        result = fetch_remote_pdf({"url": ALL_ZIP_URL})
        assert result.get("error") == "not_a_pdf_url"
        assert "report.pdf" in result["message"]

    def test_rejects_tif_url(self):
        result = fetch_remote_pdf({"url": ORTHO_URL})
        assert result.get("error") == "not_a_pdf_url"

    def test_rejects_html_viewer_url(self):
        result = fetch_remote_pdf({"url": MAP_URL})
        assert result.get("error") == "not_a_pdf_url"

    def test_rejects_http(self):
        result = fetch_remote_pdf({"url": "http://example.com/report.pdf"})
        assert result.get("error") == "invalid_url"

    def test_valid_pdf_url_passes_validation_fails_on_network(self):
        # Proper PDF URL should pass URL validation; only fail on the (non-existent in CI) server.
        result = fetch_remote_pdf({"url": REPORT_PDF_URL})
        assert result.get("error") in ("download_failed", "not_a_pdf_url"), result
        assert result.get("error") != "invalid_url"


# ── 3. Engine: pdf_summarize(url) rejected before executor.invoke ─────────────


def test_engine_url_to_local_path_tool_not_forwarded_to_executor():
    """_invalid_args_error must intercept pdf_summarize(url) before executor.invoke."""
    author, evaluators = _personas()
    invocations: list[tuple[str, dict]] = []

    class _TrackingExec:
        def invoke(self, name: str, args: dict) -> dict:
            invocations.append((name, dict(args)))
            return {"success": True, "tool": name, "result": {}}

    pdf_spec = {"type": "function", "function": {
        "name": "pdf_summarize",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }}
    turns: list[int] = []

    def tool_chat(messages, tools):
        turns.append(len(turns))
        if len(turns) == 1:
            return {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "pdf_summarize", "arguments": {"path": REPORT_PDF_URL}}],
                "raw_message": {"role": "assistant", "content": None},
            }
        return {"content": json.dumps(GOOD_CANDIDATE), "tool_calls": [], "raw_message": {}}

    run_persona_metadata_loop(
        {"user_message": "test"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_generic(),
        chat_fn=_pass_eval,
        tool_executor=_TrackingExec(),
        author_tool_specs=[pdf_spec],
        tool_chat_fn=tool_chat,
        max_tool_calls=4,
        max_rounds=1,
        delay_seconds=0.0,
    )

    assert invocations == [], \
        f"executor.invoke must NOT be called when _invalid_args_error fires. Got: {invocations}"


# ── 4. Engine: fetch_remote_pdf(zip_url) handler returns not_a_pdf_url ────────


def test_handler_rejects_zip_url_with_not_a_pdf_url():
    """fetch_remote_pdf handler rejects the ZIP URL immediately."""
    result = fetch_remote_pdf({"url": ALL_ZIP_URL})
    assert result["error"] == "not_a_pdf_url", f"Expected not_a_pdf_url, got: {result}"
    assert "report.pdf" in result["message"]


def test_engine_fetch_remote_pdf_zip_url_feeds_error_back():
    """Engine: fetch_remote_pdf(zip_url) is invoked, returns error, model sees it."""
    author, evaluators = _personas()
    call_log: list[tuple[str, str]] = []

    class _RealFetchExec:
        def invoke(self, name: str, args: dict) -> dict:
            url = args.get("url") or args.get("path") or ""
            call_log.append((name, url))
            if name == "fetch_remote_pdf":
                return {"success": True, "tool": name, "result": fetch_remote_pdf(args)}
            return {"success": True, "tool": name, "result": {}}

    fetch_spec = {"type": "function", "function": {
        "name": "fetch_remote_pdf",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }}
    turns: list[int] = []

    def tool_chat(messages, tools):
        turns.append(len(turns))
        if len(turns) == 1:
            return {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "fetch_remote_pdf", "arguments": {"url": ALL_ZIP_URL}}],
                "raw_message": {"role": "assistant", "content": None},
            }
        # Model received the not_a_pdf_url error and now produces final metadata
        return {"content": json.dumps(GOOD_CANDIDATE), "tool_calls": [], "raw_message": {}}

    result = run_persona_metadata_loop(
        {"user_message": "test"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_generic(),
        chat_fn=_pass_eval,
        tool_executor=_RealFetchExec(),
        author_tool_specs=[fetch_spec],
        tool_chat_fn=tool_chat,
        max_tool_calls=4,
        max_rounds=1,
        delay_seconds=0.0,
    )

    assert any(name == "fetch_remote_pdf" and ALL_ZIP_URL in url
               for name, url in call_log), f"ZIP URL call not found: {call_log}"
    assert result.proposed_metadata is not None


# ── 5. Engine: model reads report.pdf in round 1 and converges ───────────────


def test_engine_fetch_remote_pdf_round_1_converges():
    """fetch_remote_pdf called on report.pdf → converges in round 1."""
    author, evaluators = _personas()
    invocations: list[tuple[str, str]] = []

    class _PDFExec:
        def invoke(self, name: str, args: dict) -> dict:
            invocations.append((name, args.get("url") or args.get("path") or ""))
            if name == "fetch_remote_pdf":
                text = "TACC campus survey. GSD 2.1cm. 896 images. EPSG:4326."
                return {"success": True, "tool": name, "result": {
                    "text": text, "page_count": 5, "pages_read": [0, 1],
                    "characters_returned": len(text), "truncated": False,
                    "source_url": args["url"],
                }}
            return {"success": True, "tool": name, "result": {}}

    fetch_spec = {"type": "function", "function": {
        "name": "fetch_remote_pdf",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }}
    turns: list[int] = []

    def tool_chat(messages, tools):
        turns.append(len(turns))
        if len(turns) == 1:
            return {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "fetch_remote_pdf", "arguments": {"url": REPORT_PDF_URL}}],
                "raw_message": {"role": "assistant", "content": None},
            }
        return {"content": json.dumps(GOOD_CANDIDATE), "tool_calls": [], "raw_message": {}}

    result = run_persona_metadata_loop(
        {"user_message": "Analyze WebODM outputs"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_generic(),
        chat_fn=_pass_eval,
        tool_executor=_PDFExec(),
        author_tool_specs=[fetch_spec],
        tool_chat_fn=tool_chat,
        max_tool_calls=4,
        max_rounds=3,
        delay_seconds=0.0,
    )

    assert result.stop_reason == STOP_CONVERGED, f"Expected converged, got {result.stop_reason}"
    assert result.rounds == 1
    assert any(name == "fetch_remote_pdf" and REPORT_PDF_URL in url
               for name, url in invocations), f"fetch_remote_pdf not called: {invocations}"
    assert result.proposed_metadata is not None
    assert result.proposed_metadata.get("url") is None, \
        f"url must be null, got {result.proposed_metadata.get('url')!r}"


# ── 6. LLM_CALL_DELAY_SECONDS default ─────────────────────────────────────────


def test_default_llm_call_delay_is_1_second():
    """Throttle was reduced from 5s to 1s to stay well under 100 RPM server limit."""
    settings = get_settings()
    assert settings.llm_call_delay_seconds == 1.0, (
        f"Expected 1.0s default, got {settings.llm_call_delay_seconds}s. "
        "LLM_CALL_DELAY_SECONDS may be set in .env — unset it to test the default."
    )


# ── 7. End-to-end mocked: full WebODM payload ─────────────────────────────────


def test_e2e_webodm_payload_converges_with_null_url():
    """Full mocked run with a WebODM-like payload.

    Verifies:
    - fetch_remote_pdf called on report.pdf (not all.zip, not ortho.tif)
    - No local-path tools called with URLs
    - url field is null in final metadata
    - Converges in 1 round
    - Runs fast without the throttle
    """
    author, evaluators = _personas()

    file_lines = "\n".join(
        f"  - {r['name']} ({r['format']}): {r['url']}"
        for r in REMOTE_RESOURCES
    )
    user_message = (
        "Analyze these WebODM outputs and propose CKAN dataset metadata.\n\n"
        "Remote downloads (use fetch_remote_pdf for PDFs, NOT pdf_summarize):\n"
        f"{file_lines}\n\n"
        f'Your FIRST tool call MUST be: fetch_remote_pdf({{"url": "{REPORT_PDF_URL}"}})\n\n'
        "IMPORTANT — dataset `url` field: Set it to null."
    )

    call_log: list[tuple[str, str]] = []

    class _MockExec:
        def invoke(self, name: str, args: dict) -> dict:
            url_or_path = args.get("url") or args.get("path") or ""
            call_log.append((name, url_or_path))
            if name == "fetch_remote_pdf":
                text = "TACC campus survey, 896 images, GSD 2.1cm, EPSG:4326."
                return {"success": True, "tool": name, "result": {
                    "text": text, "page_count": 8, "pages_read": list(range(8)),
                    "characters_returned": len(text), "truncated": False,
                    "source_url": args["url"],
                }}
            return {"success": True, "tool": name, "result": {}}

    fetch_spec = {"type": "function", "function": {
        "name": "fetch_remote_pdf",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "HTTPS PDF URL"},
        }, "required": ["url"]},
    }}

    turns: list[int] = []

    def tool_chat(messages, tools):
        turns.append(len(turns))
        if tools is not None and len(turns) == 1:
            return {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "fetch_remote_pdf", "arguments": {"url": REPORT_PDF_URL}}],
                "raw_message": {"role": "assistant", "content": None},
            }
        return {"content": json.dumps(GOOD_CANDIDATE), "tool_calls": [], "raw_message": {}}

    t0 = time.perf_counter()
    result = run_persona_metadata_loop(
        {
            "user_message": user_message,
            "dataset_overrides": {
                "title": "TACC",
                "notes": "Multispectral UAS survey.",
                "author": "wmobley",
                "author_email": "wmobley@tacc.utexas.edu",
            },
            "source_urls": [],
            "file_heads": [],
            "file_reports": [],
        },
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_generic(),
        chat_fn=_pass_eval,
        tool_executor=_MockExec(),
        author_tool_specs=[fetch_spec],
        tool_chat_fn=tool_chat,
        max_tool_calls=4,
        max_rounds=3,
        delay_seconds=0.0,
    )
    elapsed = time.perf_counter() - t0

    assert result.stop_reason == STOP_CONVERGED, f"stop_reason={result.stop_reason}"
    assert result.rounds == 1

    assert any(name == "fetch_remote_pdf" and REPORT_PDF_URL in url
               for name, url in call_log), f"fetch_remote_pdf not called on report.pdf. Log: {call_log}"

    bad_calls = [(n, u) for n, u in call_log if n in ("pdf_summarize", "file_extract_pdf_text")]
    assert not bad_calls, f"Local-path PDF tools called with URLs: {bad_calls}"

    assert result.proposed_metadata is not None
    assert result.proposed_metadata.get("url") is None, \
        f"url field must be null, got {result.proposed_metadata.get('url')!r}"

    assert elapsed < 5.0, f"Mocked run should be fast, took {elapsed:.2f}s"
    print(f"\n  Elapsed (no throttle, mocked LLM+tools): {elapsed:.3f}s")
