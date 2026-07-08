"""Persona subgraph tests: real LangGraph interrupt/resume with a fake engine."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.ckan_registration.persona_nodes import (
    CLARIFICATION_CAP,
    _config_org_defaults,
    _valid_email,
    build_persona_subgraph,
    route_after_persona,
)
from app.settings import get_settings


@dataclasses.dataclass
class _FakeResult:
    proposed_metadata: dict
    clarification_questions: list
    stop_reason: str
    transcript: list = dataclasses.field(default_factory=list)


def _settings(tmp_path: Path, *, ask_schema: bool = False):
    # personas_dir / schemas_dir already default to the real seed dirs.
    # ask_schema defaults False here so persona/clarify tests don't hit the schema prompt first.
    return dataclasses.replace(
        get_settings(), state_dir=tmp_path / "state", runs_dir=tmp_path / "runs", ask_schema=ask_schema
    )


def _config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


def _converged_engine(consolidated_inputs, **kw):
    return _FakeResult(
        proposed_metadata={
            "title": "Test GAM",
            "name": "test-gam",
            "notes": "A model.",
            "collection_method": "Model Output",  # schema-default
            "license_id": "cc-by",
        },
        clarification_questions=[],
        stop_reason="converged",
    )


def test_converged_path_proposes_with_field_origins(tmp_path: Path):
    settings = _settings(tmp_path)
    graph = build_persona_subgraph(settings, engine=_converged_engine, checkpointer=InMemorySaver())
    # Use the subside profile explicitly so collection_method is a schema default to label.
    out = graph.invoke(
        {"thread_id": "t1", "request": {"session_id": "t1", "message": "register my model", "schema": "subside"}},
        _config("t1"),
    )

    result = out["result"]
    assert result["status"] == "analyzed"
    assert result["desired_dataset_payload"]["name"] == "test-gam"
    # R6 field-origin labeling: schema default vs llm-derived.
    assert result["field_origins"]["collection_method"] == "schema-default"
    assert result["field_origins"]["title"] == "llm-derived"
    assert "user-supplied" not in result["review_markdown"] or "llm-derived" in result["review_markdown"]
    # legacy-compatible analyzed state written for the dry-run/apply path.
    assert (settings.state_dir / "t1.json").exists()


def test_clarify_interrupt_then_resume_splits_org_and_dataset(tmp_path: Path):
    """Engine asks for spatial (not config-seeded) until provided; resume supplies it +
    an org-level field. Uses fields NOT pre-filled by config org-defaults."""

    def engine(consolidated_inputs, *, organizational_metadata=None, **kw):
        org = organizational_metadata or {}
        if not org.get("spatial"):
            return _FakeResult(
                proposed_metadata={"title": "T", "name": "t", "_gap_spatial": "no geometry in sources"},
                clarification_questions=[
                    {"field": "spatial", "question": "What is the spatial extent?", "requires_human": True}
                ],
                stop_reason="needs_clarification",
            )
        return _FakeResult(
            proposed_metadata={"title": "T", "name": "t", "spatial": org["spatial"]},
            clarification_questions=[],
            stop_reason="converged",
        )

    settings = _settings(tmp_path)
    graph = build_persona_subgraph(settings, engine=engine, checkpointer=InMemorySaver())

    first = graph.invoke({"thread_id": "t2", "request": {"session_id": "t2"}}, _config("t2"))
    assert "__interrupt__" in first
    interrupt_value = first["__interrupt__"][0].value
    assert interrupt_value["type"] == "metadata_clarification_required"
    assert interrupt_value["questions"][0]["field"] == "spatial"

    resumed = graph.invoke(
        Command(resume={"clarifications": {"spatial": "Yegua-Jackson Aquifer, Texas", "data_contact_email": "x@y.org"}}),
        _config("t2"),
    )
    assert resumed["result"]["status"] == "analyzed"
    assert resumed["result"]["desired_dataset_payload"]["spatial"] == "Yegua-Jackson Aquifer, Texas"
    # R3 split: spatial is dataset-specific; data_contact_email is org-level (thread-sticky).
    assert resumed["dataset_clarifications"] == {"spatial": "Yegua-Jackson Aquifer, Texas"}
    assert resumed["org_metadata"] == {"data_contact_email": "x@y.org"}
    assert resumed["clarification_round"] == 1


def test_one_question_at_a_time_and_free_text_resume(tmp_path: Path):
    """clarify presents a single question (with its text); a plain free-text reply answers it."""

    def engine(consolidated_inputs, *, organizational_metadata=None, **kw):
        org = organizational_metadata or {}
        if not org.get("spatial"):
            return _FakeResult(
                {"title": "T", "name": "t"},
                [
                    {"field": "spatial", "question": "What is the spatial extent?", "requires_human": True},
                    {"field": "coordinate_system", "question": "What CRS?", "requires_human": True},
                ],
                "needs_clarification",
            )
        return _FakeResult({"title": "T", "name": "t", "spatial": org["spatial"]}, [], "converged")

    settings = _settings(tmp_path)
    graph = build_persona_subgraph(settings, engine=engine, checkpointer=InMemorySaver())

    first = graph.invoke({"thread_id": "t7", "request": {"session_id": "t7"}}, _config("t7"))
    iv = first["__interrupt__"][0].value
    assert len(iv["questions"]) == 1  # one question at a time
    # coordinate_system (priority 5) sorts before spatial (no priority → 99)
    assert "What CRS?" in iv["message"]
    assert iv["field"] == "coordinate_system"
    assert bool(graph.get_state(_config("t7")).next)  # the pending-interrupt signal the runner uses

    # After answering coordinate_system the engine still needs spatial, so one more round.
    mid = graph.invoke(Command(resume={"message": "EPSG:4326"}), _config("t7"))
    iv2 = mid["__interrupt__"][0].value
    assert iv2["field"] == "spatial"
    assert "What is the spatial extent?" in iv2["message"]

    resumed = graph.invoke(Command(resume={"message": "Yegua-Jackson Aquifer, Texas"}), _config("t7"))
    assert resumed["result"]["status"] == "analyzed"
    assert resumed["dataset_clarifications"] == {"spatial": "Yegua-Jackson Aquifer, Texas"}


def test_runner_pending_interrupt_false_for_unknown_thread():
    from app.agents.ckan_registration.graph import CkanRegistrationRunner

    runner = CkanRegistrationRunner(get_settings())
    assert runner.pending_interrupt("does-not-exist-xyz") is False


def test_schema_select_asks_then_uses_choice(tmp_path: Path):
    settings = _settings(tmp_path, ask_schema=True)
    graph = build_persona_subgraph(settings, engine=_converged_engine, checkpointer=InMemorySaver())

    first = graph.invoke({"thread_id": "s1", "request": {"session_id": "s1"}}, _config("s1"))
    iv = first["__interrupt__"][0].value
    assert iv["type"] == "schema_selection_required"
    assert set(iv["options"]) == {"generic_ckan", "subside"}

    resumed = graph.invoke(Command(resume={"message": "subside"}), _config("s1"))
    assert resumed["schema_profile"] == "subside"
    assert resumed["result"]["status"] == "analyzed"


def test_schema_select_skipped_when_explicit(tmp_path: Path):
    settings = _settings(tmp_path, ask_schema=True)
    graph = build_persona_subgraph(settings, engine=_converged_engine, checkpointer=InMemorySaver())
    out = graph.invoke(
        {"thread_id": "s2", "request": {"session_id": "s2", "schema": "generic_ckan"}}, _config("s2")
    )
    assert "__interrupt__" not in out
    assert out["schema_profile"] == "generic_ckan"
    assert out["result"]["status"] == "analyzed"


def test_match_profile_maps_reply():
    from app.agents.ckan_registration.persona_nodes import _match_profile

    profiles = [
        {"name": "generic_ckan", "when_to_use": "arbitrary files"},
        {"name": "subside", "when_to_use": "groundwater subsidence models"},
    ]
    assert _match_profile({"message": "subside"}, profiles, "generic_ckan") == "subside"
    assert _match_profile({"message": "use the groundwater one"}, profiles, "generic_ckan") == "subside"
    assert _match_profile({"message": "no idea"}, profiles, "generic_ckan") == "generic_ckan"


def test_route_after_persona_caps_clarifications():
    base = {"persona_stop_reason": "needs_clarification", "clarification_questions": [{"q": 1}]}
    assert route_after_persona({**base, "clarification_round": 0}) == "clarify"
    assert route_after_persona({**base, "clarification_round": CLARIFICATION_CAP}) == "propose"


def test_route_after_persona_proposes_when_converged():
    assert route_after_persona({"persona_stop_reason": "converged"}) == "propose"


def test_build_graph_wires_persona_path_when_flag_enabled():
    from app.agents.ckan_registration.graph import build_graph

    graph = build_graph(dataclasses.replace(get_settings(), persona_chat_enabled=True))
    names = set(graph.get_graph().nodes)
    assert {"persona", "clarify", "propose"} <= names
    assert "metadata" not in names


def test_build_graph_uses_single_pass_when_flag_disabled():
    from app.agents.ckan_registration.graph import build_graph

    graph = build_graph(dataclasses.replace(get_settings(), persona_chat_enabled=False))
    names = set(graph.get_graph().nodes)
    assert "metadata" in names
    assert "persona" not in names


def test_proposal_carries_and_reviews_all_schema_fields(tmp_path: Path):
    """Schema-specific fields the author populated are kept in the payload AND shown in the
    review (not dropped to a fixed CKAN-core subset); empty schema fields are listed too."""

    def engine(consolidated_inputs, **kw):
        return _FakeResult(
            proposed_metadata={
                "title": "Yegua-Jackson GAM",
                "name": "yegua-jackson-gam",
                "notes": "A groundwater availability model.",
                "coordinate_system": "EPSG:3081",
                "program_area": "Groundwater Resources",
                "tag_string": "groundwater, texas",
                # categories / collection_method come from schema defaults
                "_gap_temporal_coverage_start": "no date found in sources",
            },
            clarification_questions=[],
            stop_reason="converged",
        )

    settings = _settings(tmp_path)
    graph = build_persona_subgraph(settings, engine=engine, checkpointer=InMemorySaver())
    out = graph.invoke(
        {"thread_id": "tf", "request": {"session_id": "tf", "schema": "subside"}}, _config("tf")
    )

    payload = out["result"]["desired_dataset_payload"]
    # Schema-specific fields are preserved (previously dropped) ...
    assert payload["coordinate_system"] == "EPSG:3081"
    assert payload["program_area"] == "Groundwater Resources"
    # ... and required subside defaults are applied.
    assert payload["collection_method"] == "Model Output"
    assert payload["categories"] == ["Groundwater"]

    md = out["result"]["review_markdown"]
    assert "coordinate_system" in md and "program_area" in md
    assert "collection_method" in md and "categories" in md
    # Empty schema fields surface for the user, with the author's gap reason.
    assert "Not set / needs input" in md
    assert "no date found in sources" in md


def test_valid_email_accepts_addresses_and_rejects_bare_usernames():
    assert _valid_email("wmobley@tacc.utexas.edu")
    assert not _valid_email("wmobley")
    assert not _valid_email("a@b")  # no TLD


def test_config_defaults_seed_contact_email_and_crs(tmp_path: Path):
    base = _settings(tmp_path)
    # data_contact_email only seeds when explicitly configured — no fallback to author_email
    # (personal identity fields are intentionally left for the LLM to ask).
    seeded = dataclasses.replace(
        base, ckan_data_contact_email="team@x.org", ckan_dataset_author_email="auth@x.org", ckan_coordinate_system="EPSG:3081"
    )
    defaults = _config_org_defaults(seeded)
    assert defaults["data_contact_email"] == "team@x.org"
    assert defaults["coordinate_system"] == "EPSG:3081"
    # author_email not in .env → not seeded; LLM will ask.
    assert "author" not in defaults
    assert "author_email" not in defaults
    assert "maintainer" not in defaults
    # CRS not configured → not seeded (still asked).
    assert "coordinate_system" not in _config_org_defaults(dataclasses.replace(base, ckan_coordinate_system=""))
    # data_contact_email empty → not seeded (no author_email fallback).
    assert "data_contact_email" not in _config_org_defaults(dataclasses.replace(base, ckan_data_contact_email=""))


def test_crs_answer_is_thread_sticky(tmp_path: Path):
    """A CRS answer lands in org_metadata (reused for the rest of the thread), not the
    per-dataset bucket."""

    def engine(consolidated_inputs, *, organizational_metadata=None, **kw):
        org = organizational_metadata or {}
        if not org.get("coordinate_system"):
            return _FakeResult(
                {"title": "T", "name": "t"},
                [{"field": "coordinate_system", "question": "What CRS?", "requires_human": True}],
                "needs_clarification",
            )
        return _FakeResult({"title": "T", "name": "t", "coordinate_system": org["coordinate_system"]}, [], "converged")

    settings = _settings(tmp_path)
    graph = build_persona_subgraph(settings, engine=engine, checkpointer=InMemorySaver())
    graph.invoke({"thread_id": "tc", "request": {"session_id": "tc"}}, _config("tc"))
    resumed = graph.invoke(Command(resume={"message": "EPSG:3842"}), _config("tc"))
    assert resumed["result"]["status"] == "analyzed"
    assert resumed["org_metadata"] == {"coordinate_system": "EPSG:3842"}
    assert resumed.get("dataset_clarifications", {}) == {}


def test_invalid_contact_email_rejected_then_accepted(tmp_path: Path):
    """A bare username triggers an inline re-interrupt with an error message; a valid address is stored."""

    def engine(consolidated_inputs, *, organizational_metadata=None, **kw):
        org = organizational_metadata or {}
        if not _valid_email(str(org.get("data_contact_email", ""))):
            return _FakeResult(
                {"title": "T", "name": "t"},
                [{"field": "data_contact_email", "question": "Contact email?", "requires_human": True}],
                "needs_clarification",
            )
        return _FakeResult({"title": "T", "name": "t", "data_contact_email": org["data_contact_email"]}, [], "converged")

    # Clear the config email defaults so the field is actually asked.
    settings = dataclasses.replace(
        _settings(tmp_path), ckan_data_contact_email="", ckan_dataset_author_email="", ckan_dataset_maintainer_email=""
    )
    graph = build_persona_subgraph(settings, engine=engine, checkpointer=InMemorySaver())
    graph.invoke({"thread_id": "te", "request": {"session_id": "te"}}, _config("te"))

    # Invalid email → inline re-interrupt within the same clarify node (no persona round-trip).
    retry = graph.invoke(Command(resume={"message": "wmobley"}), _config("te"))
    assert "__interrupt__" in retry  # still interrupted — asking again
    retry_msg = retry["__interrupt__"][0].value["message"]
    assert "wmobley" in retry_msg   # shows what was wrong
    assert "Contact email?" in retry_msg  # repeats the original question

    # Valid email on the inline retry → clarify node completes, persona converges.
    done = graph.invoke(Command(resume={"message": "wmobley@tacc.utexas.edu"}), _config("te"))
    assert done["result"]["status"] == "analyzed"
    assert done["org_metadata"] == {"data_contact_email": "wmobley@tacc.utexas.edu"}
