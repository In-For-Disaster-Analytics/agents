"""Tests for orchestrate.py — end-to-end GAM registration pipeline.

Tests use mocks/fixtures for all external calls (LLM, CKAN, filesystem).
No live network calls or CKAN writes are made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure src/ is on path so gam_registration package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.orchestrate as orchestrate  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_record(tmp_path: Path) -> dict:
    """Return a minimal manifest model record pointing to tmp_path."""
    return {
        "package_id": "test-aquifer-gam",
        "package_folder": str(tmp_path),
        "title": "Test Aquifer GAM",
        "twdb_page_url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
        "report_url": "",
        "boundary_bbox_geojson": "",
        "dataset_spatial": "",
    }


def _make_minimal_file(tmp_path: Path) -> Path:
    """Create a minimal .nam file in tmp_path so list_resource_files returns something."""
    f = tmp_path / "test.nam"
    f.write_text("# test MODFLOW namefile\n")
    return f


def _mock_persona_result(converged: bool = True) -> MagicMock:
    """Return a mock PersonaLoopResult."""
    from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound
    evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
    loop_round = LoopRound(
        round_number=1,
        candidate_metadata={"dataset_title": "Test GAM", "notes": "A test."},
        fair_evaluator=evaluator,
        usability_evaluator=evaluator,
        converged=converged,
    )
    return PersonaLoopResult(
        converged=converged,
        rounds=1,
        proposed_metadata={"dataset_title": "Test GAM", "notes": "A test."},
        transcript=[loop_round],
        stop_reason="converged" if converged else "max_rounds",
        model_id="test-aquifer-gam",
        timestamp="20260625_120000",
    )


# ---------------------------------------------------------------------------
# RegistrationResult dataclass
# ---------------------------------------------------------------------------

class TestRegistrationResultDataclass:
    def test_defaults_ok(self):
        r = orchestrate.RegistrationResult(model_id="test")
        assert r.ok is True
        assert r.apply_result is None
        assert r.error == ""

    def test_error_path(self):
        r = orchestrate.RegistrationResult(model_id="test", ok=False, error="boom")
        assert r.ok is False
        assert r.error == "boom"


# ---------------------------------------------------------------------------
# No-PDF backward-compatible path
# ---------------------------------------------------------------------------

class TestNoPdfPath:
    """With report_pdf_url absent, the pipeline uses the landing-page-only proposal."""

    def test_no_pdf_calls_propose_once(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        model_record["report_url"] = ""  # No PDF

        proposed_metadata = {
            "dataset_name": "test-aquifer-gam",
            "dataset_title": "Test Aquifer GAM",
            "dataset_notes": "A groundwater availability model.",
            "dataset_tags": ["groundwater"],
        }
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files") as mock_list, \
             patch("gam_registration.orchestrate._u.build_resource_plan") as mock_build, \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables") as mock_mint, \
             patch("gam_registration.orchestrate._u.fetch_source_metadata") as mock_fetch, \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None) as mock_discover, \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value=proposed_metadata) as mock_propose, \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result) as mock_loop, \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset") as mock_map, \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]) as mock_links, \
             patch("gam_registration.orchestrate.time.sleep"):

            mock_list.return_value = []
            mock_build.return_value = []
            mock_mint.return_value = []
            mock_fetch.return_value = {"title": "Test", "excerpt": "some text", "url": "", "meta_description": ""}
            mock_map.return_value = {"type": "subside_dataset", "name": "test-aquifer-gam"}

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="test-model",
                llm_api_key="test-key",
            )

        assert result.ok is True
        mock_propose.assert_called_once()
        mock_loop.assert_called_once()
        # PDF enrichment should not have been called.
        mock_discover.assert_called()

    def test_no_pdf_result_has_correct_type(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        model_record["report_url"] = ""

        proposed_metadata = {"dataset_name": "test-gam", "dataset_title": "Test"}
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value=proposed_metadata), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.dry_run_summary["package_body"]["type"] == "subside_dataset"


# ---------------------------------------------------------------------------
# Dry-run surfaces link resources
# ---------------------------------------------------------------------------

class TestDryRunSurfacesLinkResources:
    """Dry-run output must include link resource plan with name + URL."""

    def test_link_resource_plan_present_in_dry_run(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        model_record["report_url"] = "https://twdb.example.com/report.pdf"

        link_resources = [
            {"name": "twdb-landing-page", "url": "https://twdb.example.com/test", "format": "HTML"},
            {"name": "gam-report-pdf", "url": "https://twdb.example.com/report.pdf", "format": "PDF"},
        ]
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value="https://twdb.example.com/report.pdf"), \
             patch("gam_registration.orchestrate._run_pdf_enrichment", return_value={"dataset_title": "Test"}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=link_resources), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.ok is True
        plan = result.dry_run_summary.get("link_resource_plan", [])
        assert len(plan) == 2
        names = [lr["name"] for lr in plan]
        assert "twdb-landing-page" in names
        assert "gam-report-pdf" in names

    def test_link_resource_urls_in_dry_run(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)

        link_resources = [
            {"name": "twdb-landing-page", "url": "https://twdb.example.com/test", "format": "HTML"},
        ]
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=link_resources), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        plan = result.dry_run_summary.get("link_resource_plan", [])
        assert plan[0]["url"] == "https://twdb.example.com/test"


# ---------------------------------------------------------------------------
# Apply gate
# ---------------------------------------------------------------------------

class TestApplyGate:
    """Apply must only be called when approval='REGISTER'."""

    def test_no_apply_without_approval(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate._u.create_or_update_ckan_dataset") as mock_apply, \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                approval="",  # No approval
            )

        assert result.apply_result is None
        mock_apply.assert_not_called()

    def test_apply_called_with_register_approval(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        persona_result = _mock_persona_result(converged=True)

        mock_dataset = {"id": "abc123", "name": "test-aquifer-gam", "resources": []}

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-aquifer-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate._u.create_or_update_ckan_dataset", return_value=mock_dataset) as mock_apply, \
             patch("gam_registration.orchestrate._u.upsert_resources", return_value=([], 0, 0)), \
             patch("gam_registration.orchestrate._u.create_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                approval="REGISTER",
            )

        assert result.apply_result is not None
        assert result.apply_result["dataset_name"] == "test-aquifer-gam"
        mock_apply.assert_called_once()

    def test_apply_sets_subside_dataset_type(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        persona_result = _mock_persona_result(converged=True)

        mock_dataset = {"id": "abc123", "name": "test-aquifer-gam", "resources": []}
        captured_kwargs: dict = {}

        def capture_apply(base_url, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_dataset

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-aquifer-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate._u.create_or_update_ckan_dataset", side_effect=capture_apply), \
             patch("gam_registration.orchestrate._u.upsert_resources", return_value=([], 0, 0)), \
             patch("gam_registration.orchestrate._u.create_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                approval="REGISTER",
            )

        assert captured_kwargs.get("dataset_type") == "subside_dataset"


# ---------------------------------------------------------------------------
# Persona loop convergence fields in dry-run summary
# ---------------------------------------------------------------------------

class TestPersonaLoopInDryRun:
    def test_converged_field_in_summary(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.dry_run_summary["persona_loop_converged"] is True
        assert result.dry_run_summary["persona_loop_rounds"] == 1
        assert result.dry_run_summary["outstanding_questions"] == []
        # extras key must always be present in dry_run_summary.
        assert "extras" in result.dry_run_summary
        assert isinstance(result.dry_run_summary["extras"], list)

    def test_extras_in_dry_run_summary_reflects_package_extras(self, tmp_path):
        """dry_run_summary['extras'] mirrors package_body['extras']."""
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        persona_result = _mock_persona_result(converged=True)
        pkg_with_extras = {
            "type": "subside_dataset",
            "name": "test-gam",
            "extras": [
                {"key": "collection_method", "value": "Model Output"},
                {"key": "categories", "value": '["Groundwater"]'},
            ],
        }

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value=pkg_with_extras), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        extras = result.dry_run_summary["extras"]
        assert len(extras) == 2
        keys = [e["key"] for e in extras]
        assert "collection_method" in keys
        assert "categories" in keys

    def test_unconverged_has_questions(self, tmp_path):
        """Unconverged persona loop surfaces outstanding questions in dry-run."""
        from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound

        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)

        revise_evaluator = EvaluatorVerdict(
            verdict="revise",
            questions=["What is the spatial extent?", "Is a license available?"],
            recommendations=[],
        )
        pass_evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
        loop_round = LoopRound(
            round_number=3,
            candidate_metadata={"dataset_title": "Test"},
            fair_evaluator=revise_evaluator,
            usability_evaluator=pass_evaluator,
            converged=False,
        )
        persona_result_unconverged = PersonaLoopResult(
            converged=False,
            rounds=3,
            proposed_metadata={"dataset_title": "Test"},
            transcript=[loop_round],
            stop_reason="max_rounds",
            model_id="test-aquifer-gam",
            timestamp="20260625_120000",
        )

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result_unconverged), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.dry_run_summary["persona_loop_converged"] is False
        questions = result.dry_run_summary["outstanding_questions"]
        assert "What is the spatial extent?" in questions
        assert "Is a license available?" in questions


# ---------------------------------------------------------------------------
# Error handling — outer loop continues on failure
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_package_folder_returns_failed_result(self, tmp_path):
        model_record = {"package_id": "test-gam"}  # No package_folder

        result = orchestrate.run_registration(
            model_record,
            ckan_url="https://ckan.example.com",
            llm_model="m",
            llm_api_key="k",
        )

        assert result.ok is False
        assert "package_folder" in result.error.lower() or result.error

    def test_nonexistent_package_folder_returns_failed_result(self, tmp_path):
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path / "nonexistent"),
        }

        result = orchestrate.run_registration(
            model_record,
            ckan_url="https://ckan.example.com",
            llm_model="m",
            llm_api_key="k",
        )

        assert result.ok is False

    def test_manifest_loop_continues_after_model_failure(self, tmp_path):
        """run_manifest_registration continues processing remaining models on failure."""
        _make_minimal_file(tmp_path)
        model_record_good = _make_model_record(tmp_path)
        model_record_bad = {"package_id": "bad-model"}  # missing package_folder

        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"), \
             patch("pathlib.Path.mkdir"):  # suppress runs/ creation

            results = orchestrate.run_manifest_registration(
                [model_record_bad, model_record_good],
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert len(results) == 2
        # First model (bad) should fail.
        assert results[0].ok is False
        # Second model (good) should succeed.
        assert results[1].ok is True


# ---------------------------------------------------------------------------
# create_link_resources in utils.py
# ---------------------------------------------------------------------------

class TestCreateLinkResourcesInUtils:
    """Test the create_link_resources helper added to utils.py."""

    def test_create_called_for_new_resource(self):
        import gam_registration.utils as u

        mock_dataset = {"id": "pkg123", "resources": []}
        link = [{"name": "twdb-landing-page", "url": "https://twdb.example.com/", "format": "HTML"}]

        with patch("gam_registration.utils.ckan_action_post") as mock_post:
            mock_post.return_value = {"id": "res123", "name": "twdb-landing-page"}
            results = u.create_link_resources("https://ckan.example.com", mock_dataset, link, None)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][1] == "resource_create"
        assert call_args[0][2]["name"] == "twdb-landing-page"
        assert call_args[0][2]["url"] == "https://twdb.example.com/"
        assert call_args[0][2]["url_type"] == "url"

    def test_update_called_for_existing_resource(self):
        import gam_registration.utils as u

        existing_resource = {"id": "res999", "name": "twdb-landing-page"}
        mock_dataset = {"id": "pkg123", "resources": [existing_resource]}
        link = [{"name": "twdb-landing-page", "url": "https://twdb.example.com/v2", "format": "HTML"}]

        with patch("gam_registration.utils.ckan_action_post") as mock_post:
            mock_post.return_value = {"id": "res999", "name": "twdb-landing-page"}
            results = u.create_link_resources("https://ckan.example.com", mock_dataset, link, None)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][1] == "resource_update"
        assert call_args[0][2]["id"] == "res999"

    def test_skips_incomplete_link_resources(self):
        import gam_registration.utils as u

        mock_dataset = {"id": "pkg123", "resources": []}
        # Missing URL — should be skipped.
        link = [{"name": "incomplete-link", "url": "", "format": "HTML"}]

        with patch("gam_registration.utils.ckan_action_post") as mock_post:
            results = u.create_link_resources("https://ckan.example.com", mock_dataset, link, None)

        mock_post.assert_not_called()
        assert results == []

    def test_both_link_resources_created(self):
        import gam_registration.utils as u

        mock_dataset = {"id": "pkg123", "resources": []}
        links = [
            {"name": "twdb-landing-page", "url": "https://twdb.example.com/", "format": "HTML"},
            {"name": "gam-report-pdf", "url": "https://twdb.example.com/report.pdf", "format": "PDF"},
        ]

        with patch("gam_registration.utils.ckan_action_post") as mock_post:
            mock_post.side_effect = [
                {"id": "res1", "name": "twdb-landing-page"},
                {"id": "res2", "name": "gam-report-pdf"},
            ]
            results = u.create_link_resources("https://ckan.example.com", mock_dataset, links, None)

        assert mock_post.call_count == 2
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Enhancement A: dataset url populated from twdb_page_url
# ---------------------------------------------------------------------------

def _run_with_model_record(model_record: dict, tmp_path: Path, proposed_metadata: dict | None = None):
    """Helper: run registration with full mocks, returning the result.

    The persona loop mock returns *proposed_metadata* as its proposed_metadata
    so that orchestrate's Step 5 sees the same dict the test provided.
    """
    from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound

    if proposed_metadata is None:
        proposed_metadata = {
            "dataset_name": model_record.get("package_id", "test-gam"),
            "dataset_title": model_record.get("title", "Test GAM"),
        }

    # Build a persona result that carries proposed_metadata as given.
    evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
    loop_round = LoopRound(
        round_number=1,
        candidate_metadata=proposed_metadata,
        fair_evaluator=evaluator,
        usability_evaluator=evaluator,
        converged=True,
    )
    persona_result = PersonaLoopResult(
        converged=True,
        rounds=1,
        proposed_metadata=proposed_metadata,
        transcript=[loop_round],
        stop_reason="converged",
        model_id=model_record.get("package_id", "test-gam"),
        timestamp="20260625_120000",
    )

    # Capture the proposed_metadata dict passed into map_to_subside_dataset.
    captured = {}

    def capture_map(proposed, **kwargs):
        captured["proposed"] = dict(proposed)
        # Return a basic package body so the rest of the pipeline completes.
        return {
            "type": "subside_dataset",
            "name": proposed.get("dataset_name") or proposed.get("name") or "test-gam",
            "url": proposed.get("url") or proposed.get("dataset_url") or "",
            "extras": [],
        }

    with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
         patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
         patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
         patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
         patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
         patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value=proposed_metadata), \
         patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
         patch("gam_registration.orchestrate._sm.map_to_subside_dataset", side_effect=capture_map), \
         patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
         patch("gam_registration.orchestrate.time.sleep"):

        result = orchestrate.run_registration(
            model_record,
            ckan_url="https://ckan.example.com",
            llm_model="m",
            llm_api_key="k",
        )

    return result, captured


class TestEnhancementAUrlBackfill:
    """url in the package body is populated from twdb_page_url when LLM leaves it empty."""

    def test_url_backfilled_when_proposed_metadata_has_no_url(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "yegua-jackson-aquifer-gam",
            "package_folder": str(tmp_path),
            "title": "Yegua-Jackson Aquifer GAM",
            "twdb_page_url": "https://www.twdb.texas.gov/groundwater/models/gam/ygjk/ygjk.asp",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
        }
        # LLM produces metadata with no url/dataset_url.
        proposed = {
            "dataset_name": "yegua-jackson-aquifer-gam",
            "dataset_title": "Yegua-Jackson Aquifer GAM",
        }
        result, captured = _run_with_model_record(model_record, tmp_path, proposed_metadata=proposed)

        assert result.ok is True
        proposed_sent = captured.get("proposed", {})
        assert proposed_sent.get("url") == "https://www.twdb.texas.gov/groundwater/models/gam/ygjk/ygjk.asp"

    def test_url_not_overwritten_when_llm_already_set_url(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
        }
        # LLM already produced a url.
        proposed = {
            "dataset_name": "test-gam",
            "dataset_title": "Test GAM",
            "url": "https://llm-supplied.example.com/test",
        }
        result, captured = _run_with_model_record(model_record, tmp_path, proposed_metadata=proposed)

        assert result.ok is True
        proposed_sent = captured.get("proposed", {})
        # LLM's url is preserved; twdb_page_url is not used to overwrite.
        assert proposed_sent.get("url") == "https://llm-supplied.example.com/test"

    def test_url_not_set_when_twdb_page_url_also_empty(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
        }
        proposed = {"dataset_name": "test-gam"}
        result, captured = _run_with_model_record(model_record, tmp_path, proposed_metadata=proposed)

        assert result.ok is True
        proposed_sent = captured.get("proposed", {})
        # Neither url nor dataset_url should be set.
        assert not proposed_sent.get("url") and not proposed_sent.get("dataset_url")


# ---------------------------------------------------------------------------
# Enhancement B: coordinate_system injected from model_record into package extras
# ---------------------------------------------------------------------------

class TestEnhancementBCoordinateSystem:
    """coordinate_system from discovery is injected into proposed metadata -> extras."""

    def test_coordinate_system_injected_when_model_record_has_it(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
            "coordinate_system": "EPSG:32614",
        }
        proposed = {"dataset_name": "test-gam", "dataset_title": "Test GAM"}
        result, captured = _run_with_model_record(model_record, tmp_path, proposed_metadata=proposed)

        assert result.ok is True
        proposed_sent = captured.get("proposed", {})
        assert proposed_sent.get("coordinate_system") == "EPSG:32614"

    def test_coordinate_system_in_package_extras(self, tmp_path):
        """End-to-end: coordinate_system from model_record appears in package extras."""
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
            "coordinate_system": "EPSG:32614",
        }
        proposed = {"dataset_name": "test-gam", "dataset_title": "Test GAM"}
        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value=proposed), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.ok is True
        extras = result.dry_run_summary.get("extras", [])
        crs_extras = [e for e in extras if e.get("key") == "coordinate_system"]
        assert len(crs_extras) == 1
        assert crs_extras[0]["value"] == "EPSG:32614"

    def test_coordinate_system_not_overwritten_when_llm_set_it(self, tmp_path):
        """If LLM already set coordinate_system, model_record does not overwrite it."""
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
            "coordinate_system": "EPSG:32614",
        }
        # LLM produced a different coordinate_system.
        proposed = {
            "dataset_name": "test-gam",
            "coordinate_system": "EPSG:4326",
        }
        result, captured = _run_with_model_record(model_record, tmp_path, proposed_metadata=proposed)

        assert result.ok is True
        proposed_sent = captured.get("proposed", {})
        # LLM's value is preserved; discovery value is NOT injected.
        assert proposed_sent.get("coordinate_system") == "EPSG:4326"

    def test_no_coordinate_system_when_model_record_lacks_it(self, tmp_path):
        """When model_record has no coordinate_system, it is not injected."""
        _make_minimal_file(tmp_path)
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "",
            "report_url": "",
            "boundary_bbox_geojson": "",
            "dataset_spatial": "",
            # no coordinate_system key
        }
        proposed = {"dataset_name": "test-gam"}
        result, captured = _run_with_model_record(model_record, tmp_path, proposed_metadata=proposed)

        assert result.ok is True
        proposed_sent = captured.get("proposed", {})
        assert "coordinate_system" not in proposed_sent


# ---------------------------------------------------------------------------
# File inventory: built from model files and passed to persona loop
# ---------------------------------------------------------------------------

class TestFileInventoryPassedToPersonaLoop:
    """file_inventory must be constructed from model files and forwarded to run_persona_metadata_loop."""

    def _run_with_files(self, tmp_path, files_list, model_record=None):
        """Helper: run registration with a given mock file list, capture persona loop call args."""
        if model_record is None:
            model_record = _make_model_record(tmp_path)

        persona_result = _mock_persona_result(converged=True)
        captured_kwargs: dict = {}

        def capture_persona_loop(consolidated_inputs, **kwargs):
            captured_kwargs.update(kwargs)
            return persona_result

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=files_list), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.summarize_extensions", wraps=orchestrate._u.summarize_extensions), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", side_effect=capture_persona_loop) as mock_loop, \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        return result, captured_kwargs, mock_loop

    def test_file_inventory_kwarg_present(self, tmp_path):
        """run_persona_metadata_loop must be called with a file_inventory kwarg."""
        _make_minimal_file(tmp_path)
        files = [tmp_path / "test.nam", tmp_path / "test.dis"]
        result, captured, mock_loop = self._run_with_files(tmp_path, files)

        assert result.ok is True
        mock_loop.assert_called_once()
        assert "file_inventory" in captured, (
            f"Expected 'file_inventory' kwarg; got keys={list(captured.keys())}"
        )

    def test_file_inventory_has_required_keys(self, tmp_path):
        """file_inventory must contain file_count, extension_counts, and filenames."""
        _make_minimal_file(tmp_path)
        files = [tmp_path / "test.nam", tmp_path / "test.dis"]
        result, captured, _ = self._run_with_files(tmp_path, files)

        inv = captured["file_inventory"]
        assert "file_count" in inv
        assert "extension_counts" in inv
        assert "filenames" in inv

    def test_file_inventory_file_count_matches_files(self, tmp_path):
        """file_count must equal the number of files from list_resource_files."""
        _make_minimal_file(tmp_path)
        files = [tmp_path / "a.nam", tmp_path / "b.dis", tmp_path / "c.bas"]
        result, captured, _ = self._run_with_files(tmp_path, files)

        assert captured["file_inventory"]["file_count"] == 3

    def test_file_inventory_extension_counts_correct(self, tmp_path):
        """extension_counts must reflect the extensions of the model files."""
        _make_minimal_file(tmp_path)
        files = [tmp_path / "a.nam", tmp_path / "b.dis", tmp_path / "c.dis"]
        result, captured, _ = self._run_with_files(tmp_path, files)

        ext_counts = captured["file_inventory"]["extension_counts"]
        assert ext_counts.get(".dis") == 2
        assert ext_counts.get(".nam") == 1

    def test_file_inventory_filenames_are_basenames(self, tmp_path):
        """filenames in the inventory must be file name strings (stem+ext), not full paths."""
        _make_minimal_file(tmp_path)
        files = [tmp_path / "ygjk_tr.nam", tmp_path / "1980_1999.dis"]
        result, captured, _ = self._run_with_files(tmp_path, files)

        filenames = captured["file_inventory"]["filenames"]
        assert "ygjk_tr.nam" in filenames
        assert "1980_1999.dis" in filenames

    def test_file_inventory_cap_at_200(self, tmp_path):
        """filenames list must be capped at 200; filenames_truncated=True when more exist."""
        _make_minimal_file(tmp_path)
        # Simulate 250 files.
        files = [tmp_path / f"file_{i:04d}.txt" for i in range(250)]
        result, captured, _ = self._run_with_files(tmp_path, files)

        inv = captured["file_inventory"]
        assert inv["file_count"] == 250
        assert len(inv["filenames"]) == 200
        assert inv.get("filenames_truncated") is True

    def test_file_inventory_no_truncation_flag_when_under_cap(self, tmp_path):
        """filenames_truncated must NOT be present when file count <= 200."""
        _make_minimal_file(tmp_path)
        files = [tmp_path / f"file_{i}.nam" for i in range(5)]
        result, captured, _ = self._run_with_files(tmp_path, files)

        inv = captured["file_inventory"]
        assert len(inv["filenames"]) == 5
        assert "filenames_truncated" not in inv

    def test_file_inventory_empty_files_list(self, tmp_path):
        """With no files, file_inventory must have file_count=0 and empty filenames/extension_counts."""
        model_record = _make_model_record(tmp_path)
        result, captured, _ = self._run_with_files(tmp_path, [])

        inv = captured["file_inventory"]
        assert inv["file_count"] == 0
        assert inv["filenames"] == []
        assert inv["extension_counts"] == {}
        assert "filenames_truncated" not in inv


# ---------------------------------------------------------------------------
# Local-first PDF enrichment
# ---------------------------------------------------------------------------

class TestLocalFirstPdfEnrichment:
    """When a local report PDF is found, the pipeline uses it without downloading."""

    def _base_patches(self, tmp_path):
        """Return the common patch context manager kwargs (without PDF-specific patches)."""
        persona_result = _mock_persona_result(converged=True)
        return persona_result

    def test_local_pdf_uses_run_pdf_map_reduce_not_fetch(self, tmp_path):
        """When a local PDF is found, run_pdf_map_reduce is called; fetch_pdf_to_temp is NOT."""
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)

        # Create a fake local report PDF.
        report_dir = tmp_path / "Report"
        report_dir.mkdir()
        local_pdf = report_dir / "gam_report.pdf"
        local_pdf.write_bytes(b"%PDF-1.4 local report content")

        persona_result = _mock_persona_result(converged=True)
        enriched_meta = {"dataset_title": "GAM from local PDF", "notes": "enriched"}

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "excerpt", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url") as mock_discover, \
             patch("gam_registration.pdf_extract.fetch_pdf_to_temp") as mock_fetch, \
             patch("gam_registration.pdf_extract.run_pdf_map_reduce", return_value=enriched_meta) as mock_map_reduce, \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-aquifer-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]) as mock_links, \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="test-model",
                llm_api_key="test-key",
            )

        assert result.ok is True
        # run_pdf_map_reduce must have been called with the local path.
        mock_map_reduce.assert_called_once()
        call_args = mock_map_reduce.call_args
        called_path = call_args[0][0]  # first positional arg
        assert called_path == local_pdf, f"Expected local_pdf {local_pdf}, got {called_path}"
        # fetch_pdf_to_temp must NOT have been called.
        mock_fetch.assert_not_called()
        # discover_report_url must NOT have been called (local PDF was found first).
        mock_discover.assert_not_called()

    def test_local_pdf_no_report_url_link_resource(self, tmp_path):
        """When a local PDF is used, build_link_resources is called with report_url=None."""
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)

        report_dir = tmp_path / "Report"
        report_dir.mkdir()
        local_pdf = report_dir / "gam_report.pdf"
        local_pdf.write_bytes(b"%PDF-1.4 local report")

        persona_result = _mock_persona_result(converged=True)

        captured_link_args: dict = {}

        def capture_link_resources(landing_url, report_url):
            captured_link_args["landing_url"] = landing_url
            captured_link_args["report_url"] = report_url
            return []

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url"), \
             patch("gam_registration.pdf_extract.run_pdf_map_reduce", return_value={"dataset_title": "T"}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", side_effect=capture_link_resources), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.ok is True
        # report_url must be None when local PDF was used.
        assert captured_link_args.get("report_url") is None, (
            f"Expected report_url=None, got {captured_link_args.get('report_url')!r}"
        )
        # landing_page_url must still be forwarded.
        assert captured_link_args.get("landing_url") == model_record["twdb_page_url"]

    def test_fallback_to_url_when_no_local_pdf(self, tmp_path):
        """When no local PDF exists, discover_report_url is called (existing URL path)."""
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        # No Report/ directory created — no local PDF.

        persona_result = _mock_persona_result(converged=True)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value="https://twdb.texas.gov/report.pdf") as mock_discover, \
             patch("gam_registration.orchestrate._run_pdf_enrichment", return_value={"dataset_title": "URL PDF"}) as mock_enrich, \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]) as mock_links, \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.ok is True
        # discover_report_url MUST have been called (no local PDF found).
        mock_discover.assert_called()
        # URL-based enrichment MUST have been used.
        mock_enrich.assert_called_once()

    def test_url_path_report_link_resource_included(self, tmp_path):
        """When the report came from a URL, build_link_resources receives the URL."""
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)
        # No local PDF.

        persona_result = _mock_persona_result(converged=True)
        captured_link_args: dict = {}

        def capture_link_resources(landing_url, report_url):
            captured_link_args["landing_url"] = landing_url
            captured_link_args["report_url"] = report_url
            return []

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value="https://twdb.texas.gov/report.pdf"), \
             patch("gam_registration.orchestrate._run_pdf_enrichment", return_value={"dataset_title": "URL PDF"}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", side_effect=capture_link_resources), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
            )

        assert result.ok is True
        assert captured_link_args.get("report_url") == "https://twdb.texas.gov/report.pdf"


# ---------------------------------------------------------------------------
# org_defaults: Fix A — org defaults seeded into persona loop and final package
# ---------------------------------------------------------------------------

_ORG_DEFAULTS_SAMPLE = {
    "license_id": "cc-by",
    "author": "Texas Water Development Board",
    "author_email": "groundwater@twdb.texas.gov",
    "maintainer": "TWDB Groundwater Division",
    "maintainer_email": "gam@twdb.texas.gov",
    "owner_org": "twdb",
    "data_contact_email": "groundwater@twdb.texas.gov",
}


def _run_with_org_defaults(
    tmp_path,
    org_defaults=None,
    owner_org=None,
    proposed_metadata=None,
):
    """Helper: run registration with org_defaults, capture persona loop call kwargs
    and the proposed_metadata dict passed into map_to_subside_dataset."""
    from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound

    if proposed_metadata is None:
        proposed_metadata = {
            "dataset_name": "test-aquifer-gam",
            "dataset_title": "Test Aquifer GAM",
        }

    evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
    loop_round = LoopRound(
        round_number=1,
        candidate_metadata=proposed_metadata,
        fair_evaluator=evaluator,
        usability_evaluator=evaluator,
        converged=True,
    )
    persona_result = PersonaLoopResult(
        converged=True,
        rounds=1,
        proposed_metadata=proposed_metadata,
        transcript=[loop_round],
        stop_reason="converged",
        model_id="test-aquifer-gam",
        timestamp="20260625_120000",
    )

    captured_loop_kwargs: dict = {}
    captured_map_proposed: dict = {}

    def capture_loop(consolidated_inputs, **kwargs):
        captured_loop_kwargs.update(kwargs)
        return persona_result

    def capture_map(proposed, **kwargs):
        captured_map_proposed.update(proposed)
        return {
            "type": "subside_dataset",
            "name": proposed.get("name") or proposed.get("dataset_name") or "test-gam",
            "license_id": proposed.get("license_id"),
            "author": proposed.get("author"),
            "maintainer": proposed.get("maintainer"),
            "owner_org": proposed.get("owner_org"),
            "extras": [],
        }

    model_record = _make_model_record(tmp_path)
    _make_minimal_file(tmp_path)

    with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
         patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[]), \
         patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
         patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
         patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
         patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value=proposed_metadata), \
         patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", side_effect=capture_loop), \
         patch("gam_registration.orchestrate._sm.map_to_subside_dataset", side_effect=capture_map), \
         patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
         patch("gam_registration.orchestrate.time.sleep"):

        result = orchestrate.run_registration(
            model_record,
            ckan_url="https://ckan.example.com",
            llm_model="m",
            llm_api_key="k",
            owner_org=owner_org,
            org_defaults=org_defaults,
        )

    return result, captured_loop_kwargs, captured_map_proposed


class TestOrgDefaultsPersonaLoopKwarg:
    """org_defaults → organizational_metadata kwarg in run_persona_metadata_loop."""

    def test_organizational_metadata_passed_when_org_defaults_provided(self, tmp_path):
        result, loop_kwargs, _ = _run_with_org_defaults(
            tmp_path, org_defaults=_ORG_DEFAULTS_SAMPLE
        )
        assert result.ok is True
        assert "organizational_metadata" in loop_kwargs, (
            f"Expected 'organizational_metadata' kwarg; got keys={list(loop_kwargs.keys())}"
        )
        org_meta = loop_kwargs["organizational_metadata"]
        assert org_meta["license_id"] == "cc-by"
        assert org_meta["author"] == "Texas Water Development Board"
        assert org_meta["owner_org"] == "twdb"

    def test_organizational_metadata_absent_when_no_org_defaults(self, tmp_path):
        result, loop_kwargs, _ = _run_with_org_defaults(tmp_path, org_defaults=None)
        assert result.ok is True
        # With org_defaults=None and owner_org=None, none of the org-DEFAULT keys
        # should be present. (url / coordinate_system may still be seeded from the
        # model record itself — that is expected and independent of org_defaults.)
        org_meta = loop_kwargs.get("organizational_metadata") or {}
        for _k in ("license_id", "author", "author_email", "maintainer",
                   "maintainer_email", "data_contact_email", "owner_org"):
            assert _k not in org_meta, (
                f"{_k} should be absent without org_defaults; got {org_meta!r}"
            )

    def test_explicit_owner_org_wins_over_org_defaults_in_loop(self, tmp_path):
        """Explicit owner_org param must override org_defaults.owner_org in the loop kwarg."""
        defaults_with_org = dict(_ORG_DEFAULTS_SAMPLE)
        defaults_with_org["owner_org"] = "twdb-default"

        result, loop_kwargs, _ = _run_with_org_defaults(
            tmp_path,
            org_defaults=defaults_with_org,
            owner_org="twdb-explicit",  # explicit param
        )
        assert result.ok is True
        org_meta = loop_kwargs.get("organizational_metadata", {})
        assert org_meta.get("owner_org") == "twdb-explicit", (
            f"Expected explicit owner_org to win; got {org_meta.get('owner_org')!r}"
        )

    def test_org_defaults_drops_empty_values(self, tmp_path):
        """Empty string values in org_defaults must not appear in organizational_metadata."""
        sparse_defaults = {
            "license_id": "cc-by",
            "author": "",          # empty — should be dropped
            "maintainer": None,    # None — should be dropped
            "owner_org": "twdb",
        }
        result, loop_kwargs, _ = _run_with_org_defaults(
            tmp_path, org_defaults=sparse_defaults
        )
        assert result.ok is True
        org_meta = loop_kwargs.get("organizational_metadata", {})
        assert org_meta.get("license_id") == "cc-by"
        assert org_meta.get("owner_org") == "twdb"
        assert "author" not in org_meta, "Empty author should not appear in org_meta"
        assert "maintainer" not in org_meta, "None maintainer should not appear in org_meta"


class TestOrgDefaultsFinalPackageBackfill:
    """org_defaults values are backfilled into the final proposed_metadata when author left them null."""

    def test_license_id_backfilled_from_org_defaults(self, tmp_path):
        # Author's proposed_metadata has no license_id (simulating _gap_ situation).
        proposed = {"dataset_name": "test-gam", "dataset_title": "Test GAM"}
        result, _, captured_proposed = _run_with_org_defaults(
            tmp_path, org_defaults=_ORG_DEFAULTS_SAMPLE, proposed_metadata=proposed
        )
        assert result.ok is True
        assert captured_proposed.get("license_id") == "cc-by", (
            f"Expected license_id='cc-by' from org_defaults; got {captured_proposed.get('license_id')!r}"
        )

    def test_maintainer_backfilled_from_org_defaults(self, tmp_path):
        proposed = {"dataset_name": "test-gam"}
        result, _, captured_proposed = _run_with_org_defaults(
            tmp_path, org_defaults=_ORG_DEFAULTS_SAMPLE, proposed_metadata=proposed
        )
        assert result.ok is True
        assert captured_proposed.get("maintainer") == "TWDB Groundwater Division"

    def test_data_contact_email_backfilled_from_org_defaults(self, tmp_path):
        proposed = {"dataset_name": "test-gam"}
        result, _, captured_proposed = _run_with_org_defaults(
            tmp_path, org_defaults=_ORG_DEFAULTS_SAMPLE, proposed_metadata=proposed
        )
        assert result.ok is True
        assert captured_proposed.get("data_contact_email") == "groundwater@twdb.texas.gov"

    def test_author_not_overwritten_when_already_set_by_llm(self, tmp_path):
        """If the author LLM already set author, org_defaults must not overwrite it."""
        proposed = {
            "dataset_name": "test-gam",
            "author": "LLM-supplied Author",  # author already set
        }
        result, _, captured_proposed = _run_with_org_defaults(
            tmp_path, org_defaults=_ORG_DEFAULTS_SAMPLE, proposed_metadata=proposed
        )
        assert result.ok is True
        assert captured_proposed.get("author") == "LLM-supplied Author", (
            f"Expected LLM author to be preserved; got {captured_proposed.get('author')!r}"
        )

    def test_explicit_owner_org_wins_in_final_package(self, tmp_path):
        """Explicit owner_org param must appear in final proposed_metadata regardless of org_defaults."""
        proposed = {"dataset_name": "test-gam"}
        defaults_with_org = dict(_ORG_DEFAULTS_SAMPLE)
        defaults_with_org["owner_org"] = "twdb-from-defaults"

        result, _, captured_proposed = _run_with_org_defaults(
            tmp_path,
            org_defaults=defaults_with_org,
            owner_org="twdb-explicit",
            proposed_metadata=proposed,
        )
        assert result.ok is True
        assert captured_proposed.get("owner_org") == "twdb-explicit", (
            f"Expected explicit owner_org; got {captured_proposed.get('owner_org')!r}"
        )

    def test_no_backfill_when_org_defaults_is_none(self, tmp_path):
        """When org_defaults is None, no fields are backfilled."""
        proposed = {"dataset_name": "test-gam"}
        result, _, captured_proposed = _run_with_org_defaults(
            tmp_path, org_defaults=None, proposed_metadata=proposed
        )
        assert result.ok is True
        # No org fields should appear since there is nothing to backfill.
        assert captured_proposed.get("license_id") is None
        assert captured_proposed.get("maintainer") is None


class TestRunManifestRegistrationOrgDefaults:
    """run_manifest_registration forwards org_defaults to each model's run_registration call."""

    def test_org_defaults_forwarded_to_all_models(self, tmp_path):
        _make_minimal_file(tmp_path)
        model_record = _make_model_record(tmp_path)

        persona_result = _mock_persona_result(converged=True)
        captured_org_defaults: list = []

        def capture_run_registration(rec, **kwargs):
            captured_org_defaults.append(kwargs.get("org_defaults"))
            # Return a successful result
            return orchestrate.RegistrationResult(
                model_id=rec.get("package_id", "test"),
                ok=True,
                dry_run_summary={"package_body": {}, "extras": [], "link_resource_plan": [],
                                  "persona_loop_converged": True, "persona_loop_rounds": 1,
                                  "persona_loop_stop_reason": "converged",
                                  "outstanding_questions": []},
            )

        with patch("gam_registration.orchestrate.run_registration", side_effect=capture_run_registration), \
             patch("pathlib.Path.mkdir"):

            results = orchestrate.run_manifest_registration(
                [model_record, model_record],
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                org_defaults=_ORG_DEFAULTS_SAMPLE,
            )

        assert len(results) == 2
        assert len(captured_org_defaults) == 2
        for od in captured_org_defaults:
            assert od == _ORG_DEFAULTS_SAMPLE, (
                f"Expected org_defaults forwarded unchanged; got {od!r}"
            )


def test_is_geodatabase_path_excludes_gdb_and_geodatabase_folder():
    import gam_registration.orchestrate as orch
    from pathlib import Path
    assert orch._is_geodatabase_path(Path("/x/Yegua_GAM/Geodatabase/ygjk.gdb/a00000001.gdbtable")) is True
    assert orch._is_geodatabase_path(Path("/x/Yegua_GAM/Geodatabase/info.xml")) is True
    assert orch._is_geodatabase_path(Path("/x/Yegua_GAM/model.gdb/contents")) is True
    assert orch._is_geodatabase_path(Path("/x/Yegua_GAM/Model_File/ygjk_tr.nam")) is False
    assert orch._is_geodatabase_path(Path("/x/Yegua_GAM/Report/report.pdf")) is False
