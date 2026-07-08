"""Guardrail tests for the legacy standalone ``ckan_agent.py`` registration module.

That module lives in a *sibling* project (``agents/ckan-registration/ckan_agent.py``), not in this
repo — ``LegacyCkanWorker`` still loads it at runtime as the live registration backend. These
tests therefore only run where that sibling project is checked out alongside this one; otherwise
they skip. (Retiring the legacy module + these tests is tracked by the CKAN MCP-write re-point,
GitHub issue #1.)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_legacy_worker_module():
    module_path = Path(__file__).resolve().parents[2] / "ckan-registration" / "ckan_agent.py"
    if not module_path.exists():
        pytest.skip(
            f"legacy ckan_agent.py not present at {module_path} — the legacy registration backend "
            "is an external sibling project; run these where it is checked out (see issue #1)."
        )
    spec = importlib.util.spec_from_file_location("_test_ckan_agent_guardrails", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyze_rejects_no_existing_dataset_clarification_as_metadata(tmp_path: Path) -> None:
    worker = load_legacy_worker_module()

    result = worker.analyze(
        {
            "session_id": "guardrail-test",
            "message": "I don't have an existing dataset",
            "debug_trace": True,
        },
        tmp_path,
        use_llm=False,
    )

    assert result["ok"] is False
    assert result["status"] == "needs_input"
    assert result["issue"]["code"] == "insufficient_dataset_input"
    assert "I don't have an existing dataset" not in result["review_markdown"]
    assert any(event["step"] == "analyze.preflight.needs_input" for event in result["trace"])
    assert not (tmp_path / "guardrail-test.json").exists()


def test_analyze_rejects_attachment_reference_without_readable_files(tmp_path: Path) -> None:
    worker = load_legacy_worker_module()

    result = worker.analyze(
        {
            "session_id": "missing-files-test",
            "message": "I built a dataset using the following jupyter notebook and base data.",
            "debug_trace": True,
        },
        tmp_path,
        use_llm=False,
    )

    assert result["ok"] is False
    assert result["status"] == "needs_input"
    assert result["issue"]["code"] == "missing_readable_files"
    assert "readable file path or upload directory" in result["review_markdown"]
    assert not (tmp_path / "missing-files-test.json").exists()


def test_analyze_allows_intentional_metadata_only_registration(tmp_path: Path) -> None:
    worker = load_legacy_worker_module()

    result = worker.analyze(
        {
            "session_id": "metadata-only-test",
            "message": (
                "Dataset title: Demo metadata-only dataset. "
                "Description: A deliberate metadata-only CKAN record."
            ),
            "allow_metadata_only": True,
            "debug_trace": True,
        },
        tmp_path,
        use_llm=False,
    )

    assert result["ok"] is True
    assert result["resource_count"] == 0
    assert (tmp_path / "metadata-only-test.json").exists()
