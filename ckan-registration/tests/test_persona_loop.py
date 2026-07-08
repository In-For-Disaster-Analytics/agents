"""Tests for persona_loop.py — Capability D (three-persona metadata evaluation loop).

All LLM calls are mocked; no live LLM is invoked.
Audit trail writes use tmp_path so the working tree is not polluted.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so gam_registration package is importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Suppress delay in all tests by default.
os.environ.setdefault("LLM_CALL_DELAY_SECONDS", "0")

import gam_registration.persona_loop as pl  # noqa: E402 — import after sys.path fix


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

MINIMAL_INPUTS: dict[str, Any] = {
    "title": "Test Aquifer GAM",
    "notes": "Model files for the Test Aquifer.",
    "url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
}

SAMPLE_RESOURCE_PLAN: list[dict[str, Any]] = [
    {"resource_name": "test.nam", "format": "NAM", "relative_path": "test.nam", "mint_standard_variables": ""},
    {"resource_name": "test.dis", "format": "DIS", "relative_path": "test.dis", "mint_standard_variables": ""},
]

PASS_VERDICT_PAYLOAD = json.dumps({
    "verdict": "pass",
    "questions": [],
    "recommendations": ["Consider adding more tags."],
})

REVISE_VERDICT_PAYLOAD = json.dumps({
    "verdict": "revise",
    "questions": ["What is the temporal coverage of this model?"],
    "recommendations": [],
})

AUTHOR_CANDIDATE = json.dumps({
    "title": "Test Aquifer Groundwater Availability Model",
    "name": "test-aquifer-gam",
    "notes": "MODFLOW model for the Test Aquifer in Texas.",
    "url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
    "collection_method": "Model Output",
    "categories": ["Groundwater"],
    "temporal_coverage_start": None,
    "_gap_temporal_coverage_start": "no date found in sources",
})

AUTHOR_CANDIDATE_ROUND2 = json.dumps({
    "title": "Test Aquifer Groundwater Availability Model",
    "name": "test-aquifer-gam",
    "notes": "MODFLOW model for the Test Aquifer in Texas. Simulated period: 1980-2000.",
    "url": "https://www.twdb.texas.gov/groundwater/models/gam/test/test.asp",
    "collection_method": "Model Output",
    "categories": ["Groundwater"],
    "temporal_coverage_start": "1980",
    "temporal_coverage_end": "2000",
})


def _make_loop_kwargs(runs_dir: Path, **overrides: Any) -> dict[str, Any]:
    """Build a minimal valid kwargs dict for run_persona_metadata_loop."""
    defaults: dict[str, Any] = {
        "consolidated_inputs": MINIMAL_INPUTS,
        "resource_plan": SAMPLE_RESOURCE_PLAN,
        "mint_standard_variables": None,
        "bbox_geojson": None,
        "subside_schema_fields": [],
        "gam_defaults": {"collection_method": "Model Output", "categories": ["Groundwater"]},
        "max_rounds": 3,
        "llm_model": "test-model",
        "llm_api_key": "test-key",
        "llm_base_url": "https://test.example.com",
        "model_id": "test-aquifer-gam",
        "run_timestamp": "20260625_120000",
        "runs_dir": runs_dir,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Test: import smoke test (module is importable without errors)
# ---------------------------------------------------------------------------

def test_module_importable() -> None:
    """persona_loop must be importable without side effects."""
    assert hasattr(pl, "run_persona_metadata_loop")
    assert hasattr(pl, "EvaluatorVerdict")
    assert hasattr(pl, "PersonaLoopResult")
    assert hasattr(pl, "LoopRound")
    assert hasattr(pl, "DOMAIN_EXPERT_PROMPT")
    assert hasattr(pl, "DATA_CURATOR_PROMPT")
    assert hasattr(pl, "DATA_SCIENTIST_PROMPT")


# ---------------------------------------------------------------------------
# Test 1: Convergence on round 1 (both evaluators pass immediately)
# ---------------------------------------------------------------------------

def test_convergence_round1(tmp_path: Path) -> None:
    """Both evaluators pass on round 1 — loop converges immediately."""
    kwargs = _make_loop_kwargs(tmp_path)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        # Call order: author → evaluator1 (parallel) / evaluator2 (parallel).
        # The parallel calls may arrive in any order; the mock returns the same
        # pass verdict for both evaluators.
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,    # round 1 author
            PASS_VERDICT_PAYLOAD,  # evaluator 1 (fair or usability — order non-deterministic)
            PASS_VERDICT_PAYLOAD,  # evaluator 2
        ]

        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    assert result.rounds == 1
    assert result.stop_reason == "converged"
    assert len(result.transcript) == 1
    assert result.transcript[0].converged is True
    assert result.transcript[0].fair_evaluator.verdict == "pass"
    assert result.transcript[0].usability_evaluator.verdict == "pass"
    # The proposed_metadata must match the author's output.
    assert result.proposed_metadata.get("name") == "test-aquifer-gam"


# ---------------------------------------------------------------------------
# Test 2: Convergence after a revise round (round 1 revise, round 2 pass)
# ---------------------------------------------------------------------------

def test_convergence_after_revise(tmp_path: Path) -> None:
    """One evaluator requests revision on round 1; both pass on round 2."""
    kwargs = _make_loop_kwargs(tmp_path)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,       # round 1 author
            REVISE_VERDICT_PAYLOAD, # round 1 evaluator (fair or usability — revise)
            PASS_VERDICT_PAYLOAD,   # round 1 evaluator (the other — pass)
            AUTHOR_CANDIDATE_ROUND2,  # round 2 author (revision)
            PASS_VERDICT_PAYLOAD,   # round 2 evaluator 1
            PASS_VERDICT_PAYLOAD,   # round 2 evaluator 2
        ]

        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    assert result.rounds == 2
    assert result.stop_reason == "converged"
    assert len(result.transcript) == 2
    assert result.transcript[0].converged is False
    assert result.transcript[1].converged is True


# ---------------------------------------------------------------------------
# Test 3: Non-convergence hitting the 3-round cap (stop_reason = max_rounds)
# ---------------------------------------------------------------------------

def test_max_rounds_cap(tmp_path: Path) -> None:
    """Loop hits max_rounds=3 without convergence; returns max_rounds stop_reason."""
    kwargs = _make_loop_kwargs(tmp_path, max_rounds=3)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        # 3 rounds × (1 author + 2 evaluators) = 9 LLM calls; all evaluators revise.
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,       # r1 author
            REVISE_VERDICT_PAYLOAD, # r1 eval1
            REVISE_VERDICT_PAYLOAD, # r1 eval2
            AUTHOR_CANDIDATE,       # r2 author
            REVISE_VERDICT_PAYLOAD, # r2 eval1
            REVISE_VERDICT_PAYLOAD, # r2 eval2
            AUTHOR_CANDIDATE,       # r3 author
            REVISE_VERDICT_PAYLOAD, # r3 eval1
            REVISE_VERDICT_PAYLOAD, # r3 eval2
        ]

        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is False
    assert result.stop_reason == "max_rounds"
    assert result.rounds == 3
    assert len(result.transcript) == 3
    # All rounds should be non-converged.
    for round_entry in result.transcript:
        assert round_entry.converged is False
    # Outstanding questions should appear in the transcript.
    last_round = result.transcript[-1]
    assert len(last_round.fair_evaluator.questions) > 0 or len(last_round.usability_evaluator.questions) > 0


# ---------------------------------------------------------------------------
# Test 4: LLM failure path — stop_reason = llm_error, no exception propagates
# ---------------------------------------------------------------------------

def test_llm_failure_round1_author(tmp_path: Path) -> None:
    """Author LLM call fails in round 1; returns llm_error, best-available=consolidated_inputs."""
    kwargs = _make_loop_kwargs(tmp_path)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        mock_llm.side_effect = RuntimeError("Connection refused")

        # Must NOT raise — the outer model loop must continue.
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is False
    assert result.stop_reason == "llm_error"
    # Best-available prior state when round 1 author fails is the consolidated input.
    assert result.proposed_metadata == MINIMAL_INPUTS
    assert result.rounds == 0


def test_llm_failure_round2_evaluator(tmp_path: Path) -> None:
    """Evaluator fails in round 2; returns llm_error, proposed_metadata = last candidate."""
    kwargs = _make_loop_kwargs(tmp_path)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,       # r1 author — succeeds
            REVISE_VERDICT_PAYLOAD, # r1 eval1 — revise
            PASS_VERDICT_PAYLOAD,   # r1 eval2 — pass (mixed round)
            AUTHOR_CANDIDATE_ROUND2,  # r2 author — succeeds
            RuntimeError("Timeout"), # r2 evaluators — fail
        ]

        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is False
    assert result.stop_reason == "llm_error"
    # Best-available is the round-2 candidate (last successful author output).
    assert result.proposed_metadata.get("name") == "test-aquifer-gam"
    # No exception must escape.


def test_llm_failure_no_exception_propagation(tmp_path: Path) -> None:
    """Verify that any LLM exception is caught and never propagates to the caller."""
    kwargs = _make_loop_kwargs(tmp_path)

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=Exception("catastrophic")):
        try:
            result = pl.run_persona_metadata_loop(**kwargs)
        except Exception as exc:
            pytest.fail(f"run_persona_metadata_loop raised unexpectedly: {exc}")

    assert result.stop_reason == "llm_error"


# ---------------------------------------------------------------------------
# Test 5: Re-raise prevention — prior _gap is injected into round-2 evaluator prompts
# ---------------------------------------------------------------------------

def test_reraise_prevention_gap_injected_in_round2(tmp_path: Path) -> None:
    """A _gap_ annotation in round-1 candidate must appear in round-2 evaluator prompts."""
    kwargs = _make_loop_kwargs(tmp_path)

    captured_payloads: list[dict[str, Any]] = []

    def capture_llm(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        captured_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(captured_payloads) - 1
        # Call order: r1-author(0), r1-eval1(1), r1-eval2(2), r2-author(3), r2-eval1(4), r2-eval2(5)
        if idx == 0:
            return AUTHOR_CANDIDATE       # r1 author — has _gap_temporal_coverage_start
        elif idx in (1, 2):
            return REVISE_VERDICT_PAYLOAD  # r1 both evaluators revise
        elif idx == 3:
            return AUTHOR_CANDIDATE_ROUND2  # r2 author — addresses feedback
        else:
            return PASS_VERDICT_PAYLOAD    # r2 both evaluators pass

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_llm):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True, f"Expected convergence; got stop_reason={result.stop_reason}"

    # Round-2 evaluator prompts (indices 4 and 5) must contain the resolved gap section.
    # The gap was: _gap_temporal_coverage_start = "no date found in sources"
    # The injected text should appear in the user_payload of round-2 evaluator calls.
    round2_eval_payloads = [p["user_payload"] for p in captured_payloads[4:6]]
    for payload in round2_eval_payloads:
        resolved_instruction = payload.get("resolved_gaps_instruction", "")
        assert "temporal_coverage_start" in resolved_instruction, (
            f"Expected 'temporal_coverage_start' in resolved_gaps_instruction; "
            f"got: {resolved_instruction!r}"
        )
        assert "no date found in sources" in resolved_instruction, (
            f"Expected gap reason in resolved_gaps_instruction; got: {resolved_instruction!r}"
        )


def test_reraise_prevention_same_question_not_reraised(tmp_path: Path) -> None:
    """Re-raise prevention: a question already addressed via _gap must not appear again.

    The evaluator prompt includes 'Previously resolved gaps — do not re-raise'.
    We verify the text appears in the captured system prompt or user payload.
    """
    kwargs = _make_loop_kwargs(tmp_path, max_rounds=2)

    captured_system_prompts: list[str] = []

    def capture_system(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        captured_system_prompts.append(system_prompt)
        idx = len(captured_system_prompts) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE
        elif idx in (1, 2):
            return REVISE_VERDICT_PAYLOAD
        elif idx == 3:
            return AUTHOR_CANDIDATE_ROUND2
        else:
            return PASS_VERDICT_PAYLOAD

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_system):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True

    # The DATA_CURATOR_PROMPT and DATA_SCIENTIST_PROMPT both include the re-raise
    # instruction ("CRITICAL CONSTRAINT — Re-raise prevention").
    # Verify this text is present in the evaluator system prompts.
    evaluator_prompts = [p for p in captured_system_prompts if "Re-raise prevention" in p or "do not re-raise" in p.lower()]
    assert len(evaluator_prompts) >= 2, (
        "Expected at least 2 evaluator prompts containing re-raise prevention text; "
        f"found {len(evaluator_prompts)}"
    )


# ---------------------------------------------------------------------------
# Test 6: Evaluators run via ThreadPoolExecutor
# ---------------------------------------------------------------------------

def test_evaluators_run_via_threadpoolexecutor(tmp_path: Path) -> None:
    """Verify that evaluator calls are submitted to a ThreadPoolExecutor."""
    kwargs = _make_loop_kwargs(tmp_path)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,
            PASS_VERDICT_PAYLOAD,
            PASS_VERDICT_PAYLOAD,
        ]

        with patch("gam_registration.persona_loop.ThreadPoolExecutor", wraps=pl.ThreadPoolExecutor) as mock_executor_cls:
            result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    # ThreadPoolExecutor was instantiated at least once.
    assert mock_executor_cls.called, "ThreadPoolExecutor was not used for evaluators"


def test_evaluators_submitted_concurrently(tmp_path: Path) -> None:
    """Both evaluators are submitted to the same ThreadPoolExecutor context."""
    kwargs = _make_loop_kwargs(tmp_path)

    submit_calls: list[str] = []

    original_run_evaluators = pl._run_evaluators_parallel

    def capturing_run_evaluators(**ev_kwargs: Any) -> tuple[pl.EvaluatorVerdict, pl.EvaluatorVerdict]:
        """Wrap the real function and record that it was called."""
        submit_calls.append("called")
        return original_run_evaluators(**ev_kwargs)

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,
            PASS_VERDICT_PAYLOAD,
            PASS_VERDICT_PAYLOAD,
        ]
        with patch("gam_registration.persona_loop._run_evaluators_parallel", side_effect=capturing_run_evaluators):
            result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    assert len(submit_calls) == 1, "Expected _run_evaluators_parallel called once for round 1"


# ---------------------------------------------------------------------------
# Test 7: Audit trail file is written to runs/ (tmp_path)
# ---------------------------------------------------------------------------

def test_audit_trail_written(tmp_path: Path) -> None:
    """Audit trail JSON is written to runs_dir/<model_id>_<timestamp>.json."""
    kwargs = _make_loop_kwargs(
        tmp_path,
        model_id="audit-test-gam",
        run_timestamp="20260625_120000",
    )

    with patch("gam_registration.persona_loop._chat_completion_content") as mock_llm:
        mock_llm.side_effect = [
            AUTHOR_CANDIDATE,
            PASS_VERDICT_PAYLOAD,
            PASS_VERDICT_PAYLOAD,
        ]

        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True

    audit_files = list(tmp_path.glob("*.json"))
    assert len(audit_files) == 1, f"Expected 1 audit file, found: {audit_files}"

    content = json.loads(audit_files[0].read_text())
    assert content["model_id"] == "audit-test-gam"
    assert content["converged"] is True
    assert content["stop_reason"] == "converged"
    assert "transcript" in content
    assert len(content["transcript"]) == 1


def test_audit_trail_written_on_llm_failure(tmp_path: Path) -> None:
    """Audit trail is also written when the loop exits with llm_error."""
    kwargs = _make_loop_kwargs(tmp_path, model_id="fail-gam", run_timestamp="20260625_130000")

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=RuntimeError("down")):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.stop_reason == "llm_error"

    audit_files = list(tmp_path.glob("*.json"))
    assert len(audit_files) == 1
    content = json.loads(audit_files[0].read_text())
    assert content["stop_reason"] == "llm_error"
    assert content["converged"] is False


# ---------------------------------------------------------------------------
# Test 8: EvaluatorVerdict / PersonaLoopResult dataclass sanity checks
# ---------------------------------------------------------------------------

def test_evaluator_verdict_defaults() -> None:
    """EvaluatorVerdict can be instantiated with just verdict."""
    v = pl.EvaluatorVerdict(verdict="pass")
    assert v.questions == []
    assert v.recommendations == []


def test_persona_loop_result_rounds_to_converge() -> None:
    """rounds_to_converge property returns rounds when converged, None otherwise."""
    converged_result = pl.PersonaLoopResult(
        converged=True, rounds=2, proposed_metadata={}, transcript=[], stop_reason="converged"
    )
    assert converged_result.rounds_to_converge == 2

    unconverged_result = pl.PersonaLoopResult(
        converged=False, rounds=3, proposed_metadata={}, transcript=[], stop_reason="max_rounds"
    )
    assert unconverged_result.rounds_to_converge is None


def test_evaluator_result_alias() -> None:
    """EvaluatorResult is an exported alias for EvaluatorVerdict (spec D3 compat)."""
    assert pl.EvaluatorResult is pl.EvaluatorVerdict


# ---------------------------------------------------------------------------
# Test 9: _extract_resolved_gaps and _format_resolved_gaps_section
# ---------------------------------------------------------------------------

def test_extract_resolved_gaps_empty() -> None:
    metadata = {"title": "Test", "notes": "Something."}
    gaps = pl._extract_resolved_gaps(metadata)
    assert gaps == {}


def test_extract_resolved_gaps_populated() -> None:
    metadata = {
        "title": "Test",
        "temporal_coverage_start": None,
        "_gap_temporal_coverage_start": "no date found in sources",
        "author": None,
        "_gap_author": "no author identified",
    }
    gaps = pl._extract_resolved_gaps(metadata)
    assert gaps == {
        "temporal_coverage_start": "no date found in sources",
        "author": "no author identified",
    }


def test_format_resolved_gaps_section_empty() -> None:
    section = pl._format_resolved_gaps_section({})
    assert section == ""


def test_format_resolved_gaps_section_nonempty() -> None:
    section = pl._format_resolved_gaps_section(
        {"temporal_coverage_start": "no date found in sources"}
    )
    assert "Previously resolved gaps" in section
    assert "temporal_coverage_start" in section
    assert "no date found in sources" in section


# ---------------------------------------------------------------------------
# Test 10: Prompt templates contain required guidance text
# ---------------------------------------------------------------------------

def test_domain_expert_prompt_contains_required_guidance() -> None:
    """Domain Expert prompt must instruct the author to mark unknowns (anti-hallucination)."""
    assert "MARK UNKNOWNS" in pl.DOMAIN_EXPERT_PROMPT or "_gap" in pl.DOMAIN_EXPERT_PROMPT
    assert "Model Output" in pl.DOMAIN_EXPERT_PROMPT
    assert "Groundwater" in pl.DOMAIN_EXPERT_PROMPT
    assert "null" in pl.DOMAIN_EXPERT_PROMPT


def test_evaluator_prompts_contain_reraise_prevention() -> None:
    """Both evaluator prompts must contain re-raise prevention instructions."""
    assert "_gap" in pl.DATA_CURATOR_PROMPT or "do not re-raise" in pl.DATA_CURATOR_PROMPT.lower()
    assert "_gap" in pl.DATA_SCIENTIST_PROMPT or "do not re-raise" in pl.DATA_SCIENTIST_PROMPT.lower()


def test_evaluator_prompts_contain_fair_and_usability_criteria() -> None:
    """Prompts must contain FAIR and usability-specific criteria."""
    assert "Findable" in pl.DATA_CURATOR_PROMPT
    assert "Accessible" in pl.DATA_CURATOR_PROMPT
    assert "Interoperable" in pl.DATA_CURATOR_PROMPT
    assert "Reusable" in pl.DATA_CURATOR_PROMPT

    assert "acronym" in pl.DATA_SCIENTIST_PROMPT.lower() or "GAM" in pl.DATA_SCIENTIST_PROMPT
    assert "temporal" in pl.DATA_SCIENTIST_PROMPT.lower()
    assert "spatial" in pl.DATA_SCIENTIST_PROMPT.lower()


# ---------------------------------------------------------------------------
# Test 11: file_inventory is threaded to the Domain Expert author user_payload
# ---------------------------------------------------------------------------

SAMPLE_FILE_INVENTORY: dict[str, Any] = {
    "file_count": 3,
    "extension_counts": {".nam": 1, ".dis": 1, ".bas": 1},
    "filenames": ["ygjk_tr.nam", "1980_1999.dis", "calibration.bas"],
}


def test_file_inventory_included_in_author_payload(tmp_path: Path) -> None:
    """When file_inventory is passed, it must appear in the Domain Expert author user_payload."""
    kwargs = _make_loop_kwargs(tmp_path, file_inventory=SAMPLE_FILE_INVENTORY)

    captured_author_payloads: list[dict[str, Any]] = []

    def capture_llm(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        # Capture author calls (system_prompt contains DOMAIN_EXPERT_PROMPT marker).
        if "Domain Expert" in system_prompt or "file_inventory" in system_prompt or "MANDATORY RULES" in system_prompt:
            captured_author_payloads.append(user_payload)
        idx = len(captured_author_payloads) - 1 if captured_author_payloads else -1
        # Always return pass for evaluators, author candidate for author.
        if "verdict" not in system_prompt and "MANDATORY RULES" in system_prompt:
            return AUTHOR_CANDIDATE
        return PASS_VERDICT_PAYLOAD

    # Use a simpler approach: capture all calls and inspect the first (author) call.
    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE  # round 1 author
        return PASS_VERDICT_PAYLOAD  # both evaluators pass

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True

    # First call is always the author (Domain Expert).
    author_payload = all_payloads[0]["user_payload"]
    assert "file_inventory" in author_payload, (
        f"Expected 'file_inventory' in author user_payload; keys={list(author_payload.keys())}"
    )
    inv = author_payload["file_inventory"]
    assert inv["file_count"] == 3
    assert inv["extension_counts"] == {".nam": 1, ".dis": 1, ".bas": 1}
    assert "ygjk_tr.nam" in inv["filenames"]


def test_file_inventory_omitted_when_not_provided(tmp_path: Path) -> None:
    """When file_inventory is not passed (None), 'file_inventory' key must NOT appear in author payload."""
    # _make_loop_kwargs does not pass file_inventory, so it defaults to None.
    kwargs = _make_loop_kwargs(tmp_path)  # no file_inventory kwarg

    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE
        return PASS_VERDICT_PAYLOAD

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True

    author_payload = all_payloads[0]["user_payload"]
    assert "file_inventory" not in author_payload, (
        f"Expected 'file_inventory' absent from author payload when not provided; "
        f"keys={list(author_payload.keys())}"
    )


def test_file_inventory_omitted_when_empty_dict(tmp_path: Path) -> None:
    """When file_inventory={} (empty), 'file_inventory' key must NOT appear in author payload."""
    kwargs = _make_loop_kwargs(tmp_path, file_inventory={})

    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE
        return PASS_VERDICT_PAYLOAD

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    author_payload = all_payloads[0]["user_payload"]
    assert "file_inventory" not in author_payload


def test_domain_expert_prompt_contains_file_inventory_guidance() -> None:
    """DOMAIN_EXPERT_PROMPT must contain file_inventory guidance text."""
    prompt = pl.DOMAIN_EXPERT_PROMPT
    assert "file_inventory" in prompt.lower() or "FILE INVENTORY" in prompt
    assert "temporal_coverage_start" in prompt
    assert "temporal_coverage_end" in prompt
    # Anti-hallucination rules must remain intact.
    assert "MARK UNKNOWNS" in prompt or "_gap_" in prompt
    assert "do NOT fabricate" in prompt or "Do NOT guess" in prompt


# ---------------------------------------------------------------------------
# Test 12: organizational_metadata threading (Fix A — persona_loop side)
# ---------------------------------------------------------------------------

SAMPLE_ORG_METADATA: dict[str, Any] = {
    "license_id": "cc-by",
    "author": "Texas Water Development Board",
    "author_email": "groundwater@twdb.texas.gov",
    "maintainer": "TWDB Groundwater Division",
    "maintainer_email": "gam@twdb.texas.gov",
    "owner_org": "twdb",
    "data_contact_email": "groundwater@twdb.texas.gov",
}


def test_organizational_metadata_included_in_author_payload(tmp_path: Path) -> None:
    """When organizational_metadata is passed, it must appear in the Domain Expert author user_payload."""
    kwargs = _make_loop_kwargs(tmp_path, organizational_metadata=SAMPLE_ORG_METADATA)

    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE  # round 1 author
        return PASS_VERDICT_PAYLOAD  # both evaluators pass

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True

    # First call is the author (Domain Expert).
    author_payload = all_payloads[0]["user_payload"]
    assert "organizational_metadata" in author_payload, (
        f"Expected 'organizational_metadata' in author user_payload; keys={list(author_payload.keys())}"
    )
    org_meta = author_payload["organizational_metadata"]
    assert org_meta["license_id"] == "cc-by"
    assert org_meta["author"] == "Texas Water Development Board"
    assert org_meta["owner_org"] == "twdb"


def test_organizational_metadata_in_author_payload_on_every_round(tmp_path: Path) -> None:
    """organizational_metadata must appear in the author payload on every round (round 1 and round 2)."""
    kwargs = _make_loop_kwargs(tmp_path, organizational_metadata=SAMPLE_ORG_METADATA, max_rounds=3)

    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE       # round 1 author
        elif idx in (1, 2):
            return REVISE_VERDICT_PAYLOAD  # round 1 evaluators — revise
        elif idx == 3:
            return AUTHOR_CANDIDATE_ROUND2  # round 2 author
        else:
            return PASS_VERDICT_PAYLOAD    # round 2 evaluators — pass

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    assert result.rounds == 2

    # Call indices 0 and 3 are the author calls (round 1 and round 2).
    for author_idx in (0, 3):
        author_payload = all_payloads[author_idx]["user_payload"]
        assert "organizational_metadata" in author_payload, (
            f"Expected 'organizational_metadata' in author payload at call index {author_idx}; "
            f"keys={list(author_payload.keys())}"
        )
        assert author_payload["organizational_metadata"]["license_id"] == "cc-by"


def test_organizational_metadata_omitted_when_none(tmp_path: Path) -> None:
    """When organizational_metadata is None (default), it must NOT appear in author payload."""
    kwargs = _make_loop_kwargs(tmp_path)  # no organizational_metadata

    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE
        return PASS_VERDICT_PAYLOAD

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    author_payload = all_payloads[0]["user_payload"]
    assert "organizational_metadata" not in author_payload, (
        f"Expected 'organizational_metadata' absent when not provided; "
        f"keys={list(author_payload.keys())}"
    )


def test_organizational_metadata_omitted_when_empty_dict(tmp_path: Path) -> None:
    """When organizational_metadata={} (empty), it must NOT appear in author payload."""
    kwargs = _make_loop_kwargs(tmp_path, organizational_metadata={})

    all_payloads: list[dict[str, Any]] = []

    def capture_all(
        *,
        model: str,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> str:
        all_payloads.append({"system_prompt": system_prompt, "user_payload": user_payload})
        idx = len(all_payloads) - 1
        if idx == 0:
            return AUTHOR_CANDIDATE
        return PASS_VERDICT_PAYLOAD

    with patch("gam_registration.persona_loop._chat_completion_content", side_effect=capture_all):
        result = pl.run_persona_metadata_loop(**kwargs)

    assert result.converged is True
    author_payload = all_payloads[0]["user_payload"]
    assert "organizational_metadata" not in author_payload


# ---------------------------------------------------------------------------
# Test 13: Prompt content checks — Fix A (org-metadata guidance) and Fix B
# ---------------------------------------------------------------------------

def test_domain_expert_prompt_contains_organizational_metadata_guidance() -> None:
    """DOMAIN_EXPERT_PROMPT must contain the ORGANIZATIONAL METADATA section guidance."""
    prompt = pl.DOMAIN_EXPERT_PROMPT
    assert "organizational_metadata" in prompt.lower() or "ORGANIZATIONAL METADATA" in prompt
    assert "license_id" in prompt
    assert "data_contact_email" in prompt
    # Must instruct author to use these as authoritative values and drop _gap_ for them.
    assert "AUTHORITATIVE" in prompt or "authoritative" in prompt
    # Anti-hallucination rules must remain intact.
    assert "MARK UNKNOWNS" in prompt or "_gap_" in prompt
    assert "do NOT fabricate" in prompt or "Do NOT guess" in prompt


def test_fair_evaluator_prompt_contains_strengthened_gap_handling() -> None:
    """DATA_CURATOR_PROMPT must contain the strengthened gap-handling instruction."""
    prompt = pl.DATA_CURATOR_PROMPT
    # Must contain the "genuinely unavailable" language.
    assert "genuinely unavailable" in prompt or "ACKNOWLEDGE it as unavailable" in prompt
    # Must say not to re-raise _gap_ fields.
    assert "do not re-raise" in prompt.lower() or "_gap_" in prompt
    # Must contain the format note about CKAN assigning format from extensions.
    assert "file extension" in prompt.lower() or "file extensions" in prompt.lower()
    # Must retain FAIR criteria.
    assert "Findable" in prompt
    assert "Accessible" in prompt


def test_usability_evaluator_prompt_contains_strengthened_gap_handling() -> None:
    """DATA_SCIENTIST_PROMPT must contain the strengthened gap-handling instruction."""
    prompt = pl.DATA_SCIENTIST_PROMPT
    # Must contain the "genuinely unavailable" language.
    assert "genuinely unavailable" in prompt or "ACKNOWLEDGE it as unavailable" in prompt
    # Must say not to re-raise _gap_ fields.
    assert "do not re-raise" in prompt.lower() or "_gap_" in prompt
    # Must contain the format note.
    assert "file extension" in prompt.lower() or "file extensions" in prompt.lower()
    # Must retain usability criteria.
    assert "temporal" in prompt.lower()
    assert "spatial" in prompt.lower()
