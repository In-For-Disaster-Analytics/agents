"""Tests for resource_review.py — batched LLM resource-description review.

No live LLM calls.  All LLM interactions are mocked via
``unittest.mock.patch``.

Test classes
------------
TestNoOpCases
    Empty api_key or empty plan → returns 0, no LLM call.
TestDescriptionsUpdated
    LLM returns improved descriptions → resource_plan mutated correctly;
    resource_name / resource_title unchanged.
TestBatching
    60 resources with batch_size=25 → exactly 3 LLM calls; sleep called
    between batches.
TestFallback
    LLM raises / returns unparseable / omits items → those keep
    deterministic descriptions, no exception, correct count returned.
TestMaxBatches
    max_batches cap → only first N batches processed, warning logged,
    remainder keep deterministic descriptions.
TestOrchestrateWiring
    When review_resources=True, resource_review is invoked and the plan
    flows to mapping; when False, it is NOT invoked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure src/ is on path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.resource_review as rr  # noqa: E402
import gam_registration.orchestrate as orchestrate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(n: int) -> list[dict]:
    """Return a resource plan with *n* entries."""
    return [
        {
            "resource_name": f"path/to/file_{i:03d}.nam",
            "resource_title": f"path/to/file_{i:03d}.nam",
            "resource_description": f"Original description for file {i}.",
            "relative_path": f"path/to/file_{i:03d}.nam",
            "local_path": Path(f"/tmp/file_{i:03d}.nam"),
        }
        for i in range(n)
    ]


def _llm_response_for_batch(batch_resources: list[dict]) -> str:
    """Return a JSON string that improves every file's description."""
    resources = []
    for entry in batch_resources:
        fn = entry.get("file_name") or entry.get("relative_path") or entry.get("resource_name")
        resources.append({
            "file_name": fn,
            "description": f"Improved: {fn} is a MODFLOW input file.",
        })
    return json.dumps({"resources": resources})


# ---------------------------------------------------------------------------
# TestNoOpCases
# ---------------------------------------------------------------------------

class TestNoOpCases:
    def test_empty_api_key_returns_zero(self):
        plan = _make_plan(3)
        result = rr.review_resource_descriptions(
            plan,
            llm_model="test-model",
            llm_api_key="",
        )
        assert result == 0

    def test_falsy_api_key_returns_zero(self):
        plan = _make_plan(3)
        result = rr.review_resource_descriptions(
            plan,
            llm_model="test-model",
            llm_api_key=None,
        )
        assert result == 0

    def test_empty_api_key_no_llm_call(self):
        plan = _make_plan(3)
        with patch("gam_registration.utils._chat_completion_content") as mock_llm:
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="",
            )
        mock_llm.assert_not_called()

    def test_empty_plan_returns_zero(self):
        result = rr.review_resource_descriptions(
            [],
            llm_model="test-model",
            llm_api_key="real-key",
        )
        assert result == 0

    def test_empty_plan_no_llm_call(self):
        with patch("gam_registration.utils._chat_completion_content") as mock_llm:
            rr.review_resource_descriptions(
                [],
                llm_model="test-model",
                llm_api_key="real-key",
            )
        mock_llm.assert_not_called()

    def test_whitespace_api_key_returns_zero(self):
        plan = _make_plan(2)
        result = rr.review_resource_descriptions(
            plan,
            llm_model="test-model",
            llm_api_key="   ",
        )
        assert result == 0


# ---------------------------------------------------------------------------
# TestDescriptionsUpdated
# ---------------------------------------------------------------------------

class TestDescriptionsUpdated:
    def test_descriptions_updated_by_llm(self):
        plan = _make_plan(3)
        original_descs = [p["resource_description"] for p in plan]

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 3
        for item in plan:
            assert item["resource_description"].startswith("Improved:")
            assert item["resource_description"] != original_descs[plan.index(item)]

    def test_resource_name_and_title_unchanged(self):
        plan = _make_plan(3)
        original_names = [p["resource_name"] for p in plan]
        original_titles = [p["resource_title"] for p in plan]

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        for i, item in enumerate(plan):
            assert item["resource_name"] == original_names[i], (
                f"resource_name changed: {item['resource_name']!r} != {original_names[i]!r}"
            )
            assert item["resource_title"] == original_titles[i], (
                f"resource_title changed: {item['resource_title']!r} != {original_titles[i]!r}"
            )

    def test_description_trimmed_to_2000_chars(self):
        plan = _make_plan(1)
        long_desc = "X" * 3000

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            fn = payload["resources"][0]["file_name"]
            return json.dumps({"resources": [{"file_name": fn, "description": long_desc}]})

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        # clean_text with max_chars=2000 truncates and appends "..."
        assert len(plan[0]["resource_description"]) <= 2000

    def test_dataset_context_passed_in_payload(self):
        plan = _make_plan(2)
        captured = {}

        def fake_llm(**kwargs):
            captured.update(kwargs)
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                dataset_context="Yegua-Jackson Aquifer GAM",
            )

        assert captured["user_payload"]["dataset_context"] == "Yegua-Jackson Aquifer GAM"

    def test_returns_count_of_updated_descriptions(self):
        plan = _make_plan(5)

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 5


# ---------------------------------------------------------------------------
# TestBatching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_60_resources_3_llm_calls(self):
        plan = _make_plan(60)
        call_count = [0]

        def fake_llm(**kwargs):
            call_count[0] += 1
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
            )

        assert call_count[0] == 3, (
            f"Expected 3 LLM calls for 60 resources batch_size=25, got {call_count[0]}"
        )

    def test_sleep_between_batches_not_after_last(self):
        plan = _make_plan(60)
        sleep_calls = []

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
            )

        # 3 batches → 2 inter-batch sleeps (not after last).
        assert len(sleep_calls) == 2, (
            f"Expected 2 sleeps for 3 batches, got {len(sleep_calls)}"
        )

    def test_single_batch_no_sleep(self):
        plan = _make_plan(10)
        sleep_calls = []

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
            )

        assert sleep_calls == [], "No sleep for a single-batch run"

    def test_25_resources_1_llm_call(self):
        plan = _make_plan(25)
        call_count = [0]

        def fake_llm(**kwargs):
            call_count[0] += 1
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
            )

        assert call_count[0] == 1

    def test_all_60_descriptions_updated(self):
        plan = _make_plan(60)

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
            )

        assert count == 60


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------

class TestFallback:
    def test_llm_raises_keeps_deterministic_descriptions(self):
        plan = _make_plan(3)
        original_descs = [p["resource_description"] for p in plan]

        with patch(
            "gam_registration.utils._chat_completion_content",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 0
        for i, item in enumerate(plan):
            assert item["resource_description"] == original_descs[i], (
                "Description should be unchanged after LLM error"
            )

    def test_llm_raises_no_exception_propagated(self):
        plan = _make_plan(3)

        with patch(
            "gam_registration.utils._chat_completion_content",
            side_effect=Exception("catastrophic failure"),
        ):
            # Must not raise.
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 0

    def test_unparseable_json_keeps_deterministic_descriptions(self):
        plan = _make_plan(3)
        original_descs = [p["resource_description"] for p in plan]

        with patch(
            "gam_registration.utils._chat_completion_content",
            return_value="this is not json at all!!!",
        ):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 0
        for i, item in enumerate(plan):
            assert item["resource_description"] == original_descs[i]

    def test_missing_resources_key_keeps_deterministic(self):
        plan = _make_plan(3)
        original_descs = [p["resource_description"] for p in plan]

        with patch(
            "gam_registration.utils._chat_completion_content",
            return_value=json.dumps({"something_else": []}),
        ):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 0
        for i, item in enumerate(plan):
            assert item["resource_description"] == original_descs[i]

    def test_partial_response_only_updates_present_items(self):
        """LLM only returns 2 of 3 files; 3rd keeps deterministic description."""
        plan = _make_plan(3)
        original_desc_2 = plan[2]["resource_description"]

        def partial_llm(**kwargs):
            payload = kwargs["user_payload"]
            # Only return first 2 resources.
            resources = [
                {
                    "file_name": payload["resources"][0]["file_name"],
                    "description": "Improved first file.",
                },
                {
                    "file_name": payload["resources"][1]["file_name"],
                    "description": "Improved second file.",
                },
            ]
            return json.dumps({"resources": resources})

        with patch("gam_registration.utils._chat_completion_content", side_effect=partial_llm):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 2
        assert plan[0]["resource_description"] == "Improved first file."
        assert plan[1]["resource_description"] == "Improved second file."
        assert plan[2]["resource_description"] == original_desc_2

    def test_unknown_file_name_in_response_ignored(self):
        """LLM returns a file_name not in the batch; it is ignored without error."""
        plan = _make_plan(2)
        original_descs = [p["resource_description"] for p in plan]

        def fake_llm(**kwargs):
            return json.dumps({
                "resources": [
                    {"file_name": "unknown/ghost_file.xyz", "description": "Ghost file."},
                ]
            })

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
            )

        assert count == 0
        for i, item in enumerate(plan):
            assert item["resource_description"] == original_descs[i]

    def test_fallback_does_not_affect_other_batches(self):
        """If batch 1 fails, batch 2 still runs and updates its descriptions."""
        plan = _make_plan(30)  # 2 batches of 15
        original_descs_0_14 = [p["resource_description"] for p in plan[:15]]
        call_count = [0]

        def mixed_llm(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("batch 1 fails")
            # batch 2 succeeds.
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=mixed_llm), \
             patch("time.sleep"):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=15,
            )

        # Only second batch updated.
        assert count == 15
        for i in range(15):
            assert plan[i]["resource_description"] == original_descs_0_14[i]
        for i in range(15, 30):
            assert plan[i]["resource_description"].startswith("Improved:")


# ---------------------------------------------------------------------------
# TestMaxBatches
# ---------------------------------------------------------------------------

class TestMaxBatches:
    def test_max_batches_limits_llm_calls(self):
        plan = _make_plan(60)
        call_count = [0]

        def fake_llm(**kwargs):
            call_count[0] += 1
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
                max_batches=2,
            )

        assert call_count[0] == 2, (
            f"Expected 2 LLM calls with max_batches=2, got {call_count[0]}"
        )

    def test_max_batches_remainder_keeps_deterministic(self):
        plan = _make_plan(60)  # 3 batches of 25 (10 in last)
        original_descs_50_59 = [p["resource_description"] for p in plan[50:]]

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
                max_batches=2,
            )

        # First 50 updated, last 10 not.
        assert count == 50
        for i, item in enumerate(plan[50:]):
            assert item["resource_description"] == original_descs_50_59[i]

    def test_max_batches_warning_logged(self, caplog):
        import logging
        plan = _make_plan(60)

        def fake_llm(**kwargs):
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with caplog.at_level(logging.WARNING, logger="gam_registration.resource_review"), \
             patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
                max_batches=2,
            )

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("max_batches" in m.lower() for m in warning_msgs), (
            f"Expected warning about max_batches; got: {warning_msgs}"
        )
        assert any("10" in m for m in warning_msgs), (
            f"Expected '10' (skipped resources count) in warning; got: {warning_msgs}"
        )

    def test_max_batches_equal_to_total_batches_processes_all(self):
        plan = _make_plan(60)
        call_count = [0]

        def fake_llm(**kwargs):
            call_count[0] += 1
            payload = kwargs["user_payload"]
            return _llm_response_for_batch(payload["resources"])

        with patch("gam_registration.utils._chat_completion_content", side_effect=fake_llm), \
             patch("time.sleep"):
            count = rr.review_resource_descriptions(
                plan,
                llm_model="test-model",
                llm_api_key="key",
                batch_size=25,
                max_batches=3,  # exactly 3 batches
            )

        assert call_count[0] == 3
        assert count == 60


# ---------------------------------------------------------------------------
# TestOrchestrateWiring
# ---------------------------------------------------------------------------

class TestOrchestrateWiring:
    """Verify review_resources flag is wired correctly in orchestrate."""

    def _make_model_record(self, tmp_path: Path) -> dict:
        return {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "",
            "report_url": "",
            "boundary_bbox_geojson": "",
        }

    def _make_persona_result(self):
        from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound
        ev = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
        lr = LoopRound(
            round_number=1,
            candidate_metadata={"dataset_title": "Test"},
            fair_evaluator=ev,
            usability_evaluator=ev,
            converged=True,
        )
        return PersonaLoopResult(
            converged=True,
            rounds=1,
            proposed_metadata={"dataset_title": "Test"},
            transcript=[lr],
            stop_reason="converged",
            model_id="test-gam",
            timestamp="20260625_120000",
        )

    def _common_patches(self, tmp_path, plan=None):
        if plan is None:
            plan = []
        persona_result = self._make_persona_result()
        patches = {
            "gam_registration.orchestrate._u.list_resource_files": MagicMock(return_value=[]),
            "gam_registration.orchestrate._u.build_resource_plan": MagicMock(return_value=plan),
            "gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables": MagicMock(side_effect=lambda rp, **kw: rp),
            "gam_registration.orchestrate._u.fetch_source_metadata": MagicMock(return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}),
            "gam_registration.orchestrate._twdb.discover_report_url": MagicMock(return_value=None),
            "gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm": MagicMock(return_value={}),
            "gam_registration.orchestrate._pl.run_persona_metadata_loop": MagicMock(return_value=persona_result),
            "gam_registration.orchestrate._sm.map_to_subside_dataset": MagicMock(return_value={"type": "subside_dataset", "name": "test-gam"}),
            "gam_registration.orchestrate._twdb.build_link_resources": MagicMock(return_value=[]),
            "gam_registration.orchestrate.time.sleep": MagicMock(),
        }
        return patches

    def test_review_resources_true_invokes_review(self, tmp_path):
        """When review_resources=True and api_key set, _rr.review_resource_descriptions is called."""
        model_record = self._make_model_record(tmp_path)
        f = tmp_path / "test.nam"
        f.write_text("test")
        plan = [
            {
                "resource_name": "test.nam",
                "resource_title": "test.nam",
                "resource_description": "original",
                "relative_path": "test.nam",
            }
        ]
        persona_result = self._make_persona_result()

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=plan), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"), \
             patch("gam_registration.orchestrate._rr.review_resource_descriptions", return_value=1) as mock_review:

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                review_resources=True,
            )

        assert result.ok is True
        mock_review.assert_called_once()
        call_kwargs = mock_review.call_args
        # First positional arg should be the resource_plan.
        assert call_kwargs[0][0] is plan

    def test_review_resources_false_does_not_invoke_review(self, tmp_path):
        """When review_resources=False, _rr.review_resource_descriptions is NOT called."""
        model_record = self._make_model_record(tmp_path)
        persona_result = self._make_persona_result()

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
             patch("gam_registration.orchestrate._rr.review_resource_descriptions") as mock_review:

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                review_resources=False,
            )

        assert result.ok is True
        mock_review.assert_not_called()

    def test_review_resources_default_is_false(self, tmp_path):
        """Default call (no review_resources kwarg) must NOT invoke review."""
        model_record = self._make_model_record(tmp_path)
        persona_result = self._make_persona_result()

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
             patch("gam_registration.orchestrate._rr.review_resource_descriptions") as mock_review:

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                # review_resources not passed → default False
            )

        assert result.ok is True
        mock_review.assert_not_called()

    def test_review_resources_plan_flows_to_mapping(self, tmp_path):
        """After review_resources, the (mutated) resource_plan flows to persona loop."""
        model_record = self._make_model_record(tmp_path)
        plan = [
            {
                "resource_name": "test.nam",
                "resource_title": "test.nam",
                "resource_description": "original",
                "relative_path": "test.nam",
            }
        ]
        persona_result = self._make_persona_result()
        captured_plan = {}

        def capture_persona(consolidated, **kwargs):
            captured_plan["rp"] = kwargs.get("resource_plan")
            return persona_result

        def mock_review(rp, **kwargs):
            # Simulate description update.
            rp[0]["resource_description"] = "Improved by LLM."
            return 1

        patches = self._common_patches(tmp_path)

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=plan), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", side_effect=capture_persona), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam"}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate.time.sleep"), \
             patch("gam_registration.orchestrate._rr.review_resource_descriptions", side_effect=mock_review):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                review_resources=True,
            )

        assert result.ok is True
        # The persona loop should see the mutated plan.
        rp_seen = captured_plan.get("rp")
        assert rp_seen is not None
        assert rp_seen[0]["resource_description"] == "Improved by LLM."

    def test_manifest_registration_forwards_review_resources(self, tmp_path):
        """run_manifest_registration forwards review_resources to each run_registration call."""
        f = tmp_path / "test.nam"
        f.write_text("test")
        model_record = self._make_model_record(tmp_path)
        manifest = [model_record]

        persona_result = self._make_persona_result()
        patches = self._common_patches(tmp_path)

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
             patch("gam_registration.orchestrate._rr.review_resource_descriptions", return_value=0) as mock_review, \
             patch("pathlib.Path.mkdir"):

            results = orchestrate.run_manifest_registration(
                manifest,
                ckan_url="https://ckan.example.com",
                llm_model="m",
                llm_api_key="k",
                review_resources=True,
            )

        assert len(results) == 1
        assert results[0].ok is True
        mock_review.assert_called_once()
