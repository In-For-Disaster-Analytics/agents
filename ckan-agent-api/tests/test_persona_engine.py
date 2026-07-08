"""Engine tests with a deterministic injected chat_fn (no real LLM calls).

The fake chat routes on the persona system prompt: the seed personas contain the
literal strings "Domain Expert", "Data Curator", and "Data Scientist", so the stub
can return canned author / evaluator responses per round.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.personas import PersonaRegistry, run_persona_metadata_loop
from app.personas.engine import (
    STOP_CONVERGED,
    STOP_MAX_ROUNDS,
    STOP_NEEDS_CLARIFICATION,
)
from app.schemas import SchemaRegistry
from app.settings import PROJECT_ROOT

SEED_PERSONAS_DIR = PROJECT_ROOT / "app" / "personas"
SEED_SCHEMAS_DIR = PROJECT_ROOT / "app" / "schemas"


def _personas():
    reg = PersonaRegistry(SEED_PERSONAS_DIR)
    return reg.author(), reg.evaluators()


def _subside():
    return SchemaRegistry(SEED_SCHEMAS_DIR).get("subside")


def _is_author(system: str) -> bool:
    return "Domain Expert" in system


def _run(chat_fn, **kw):
    author, evaluators = _personas()
    return run_persona_metadata_loop(
        {"source": "x"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_subside(),
        chat_fn=chat_fn,
        max_rounds=kw.pop("max_rounds", 3),
        **kw,
    )


CANDIDATE = {"title": "Test GAM", "name": "test-gam", "notes": "A model.", "collection_method": "Model Output"}


def test_converges_when_all_evaluators_pass():
    def chat(system, payload):
        if _is_author(system):
            return json.dumps(CANDIDATE)
        return json.dumps({"verdict": "pass", "questions": [], "recommendations": []})

    result = _run(chat)
    assert result.converged is True
    assert result.stop_reason == STOP_CONVERGED
    assert result.rounds == 1
    assert result.proposed_metadata["name"] == "test-gam"


def test_eager_escalation_on_requires_human():
    def chat(system, payload):
        if _is_author(system):
            return json.dumps(CANDIDATE)
        if "Data Curator" in system:
            return json.dumps(
                {
                    "verdict": "revise",
                    "questions": [
                        {
                            "field": "license_id",
                            "question": "What license applies?",
                            "requires_human": True,
                            "reason_not_derivable": "no license in any source",
                        }
                    ],
                    "recommendations": [],
                }
            )
        return json.dumps({"verdict": "pass", "questions": []})

    result = _run(chat)
    assert result.stop_reason == STOP_NEEDS_CLARIFICATION
    assert result.converged is False
    assert result.rounds == 1  # eager: did not loop
    assert result.clarification_questions[0]["field"] == "license_id"


def test_source_derivable_revise_then_converge():
    calls = {"author": 0}

    def chat(system, payload):
        if _is_author(system):
            calls["author"] += 1
            return json.dumps(CANDIDATE)
        if "Data Curator" in system and calls["author"] == 1:
            # round 1: an ACTIONABLE source-derivable revise (a real, empty, non-narrative
            # field) -> the author should get another round to address it.
            return json.dumps(
                {
                    "verdict": "revise",
                    "questions": [{"field": "url", "question": "Add the source URL.", "requires_human": False}],
                }
            )
        return json.dumps({"verdict": "pass", "questions": []})

    result = _run(chat)
    assert result.stop_reason == STOP_CONVERGED
    assert result.rounds == 2
    assert calls["author"] == 2


def test_string_questions_are_coerced_and_do_not_escalate():
    def chat(system, payload):
        if _is_author(system):
            return json.dumps(CANDIDATE)
        if "Data Curator" in system:
            return json.dumps({"verdict": "revise", "questions": ["Improve the notes."]})
        return json.dumps({"verdict": "pass", "questions": []})

    # curator revises with a bare-string, field-less question (coerced to requires_human=False) ->
    # it does NOT escalate to the user (no human question), but a "revise" loops the AUTHOR for
    # improvement until max_rounds (the curator never passes in this stub).
    result = _run(chat, max_rounds=2)
    assert result.stop_reason == STOP_MAX_ROUNDS
    assert result.clarification_questions == []


def test_schema_tokens_render_into_author_prompt():
    captured = {}

    def chat(system, payload):
        if _is_author(system):
            captured["author_system"] = system
            return json.dumps(CANDIDATE)
        return json.dumps({"verdict": "pass", "questions": []})

    _run(chat)
    # defaults + controlled vocab from subside.yaml must be rendered into the prompt.
    assert "Model Output" in captured["author_system"]
    assert "Groundwater" in captured["author_system"]
    assert "{{schema_fields}}" not in captured["author_system"]


def test_audit_trail_written_to_runs_dir(tmp_path: Path):
    def chat(system, payload):
        if _is_author(system):
            return json.dumps(CANDIDATE)
        return json.dumps({"verdict": "pass", "questions": []})

    _run(chat, runs_dir=tmp_path, model_id="test-gam", run_timestamp="20260626_000000")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["stop_reason"] == STOP_CONVERGED
    assert saved["model_id"] == "test-gam"


def test_engine_never_raises_on_llm_error():
    def chat(system, payload):
        raise RuntimeError("boom")

    result = _run(chat)
    assert result.stop_reason == "llm_error"
    assert result.converged is False
