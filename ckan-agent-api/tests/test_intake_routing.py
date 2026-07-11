"""
Intake routing tests — two distinct layers:

  Layer 1 (pytest): normalize_action fast-path alias table.
  Pure deterministic logic, no LLM. Always run in CI.

  Layer 2 (script): simulate_conversation().
  Runs a full registration session with the real LLM router and a fake persona
  engine. Phase 1 bootstraps to a proposal using the persona subgraph directly
  (writes the state file). Phase 2 drives a goal-directed adaptive conversation:
  each scenario specifies WHAT to accomplish; the driver LLM generates natural,
  synonym-varied phrasing on every run, reading the actual agent response for
  context before generating its next message.

  Run the simulation:

      # as a pytest test (no hard assertions on LLM routing quality):
      pytest tests/test_intake_routing.py::test_conversation_simulation -v -s

      # directly as a script:
      python tests/test_intake_routing.py
"""

from __future__ import annotations

import dataclasses
import json
import re
import textwrap
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from app.agents.ckan_registration.nodes import normalize_action
from app.settings import get_settings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _settings(tmp_path: Path, *, ask_schema: bool = False):
    return dataclasses.replace(
        get_settings(),
        state_dir=tmp_path / "state",
        runs_dir=tmp_path / "runs",
        ask_schema=ask_schema,
    )


_INLINE_FILES = [
    {
        "filename": "report.txt",
        "content": textwrap.dedent("""\
            # Bethel, Alaska Orthophoto and 3D Model Survey
            Captured: 2025-08-01
            Location: Bethel, Alaska (60.7966N, 161.7786W)
            Instrument: DJI Mavic 3
            Software: OpenDroneMap 3.6.0
            GSD: 0.8 cm/pixel, 757 features, horizontal accuracy 0.024 m
        """),
    },
    {
        "filename": "coords.txt",
        "content": "W=-161.77915 E=-161.77832 S=60.79659 N=60.79688\n",
    },
]

_KNOWN_ORGS    = ["upstream", "dso-internal", "tacc", "twdb-gams", "planet-texas-2050"]
_KNOWN_SCHEMAS = ["generic_ckan", "subside"]


# ---------------------------------------------------------------------------
# Scenario definitions — each entry is (label, driver_goal, expected_routing).
#
# driver_goal  is the instruction passed to the driver LLM.  The LLM generates
# natural, synonym-varied phrasing every run; the goal just specifies WHAT to
# accomplish, not HOW to phrase it.
# ---------------------------------------------------------------------------

_SCENARIOS: list[tuple[str, str, str]] = [
    # ── Focused field show ────────────────────────────────────────────────
    ("show:title",
     "Ask to see only the dataset title — use a casual synonym like 'what's the title' or 'show me the name'.",
     "show → focused title"),

    ("show:notes",
     "Ask what the dataset description says. Do NOT say 'notes' — use synonyms like "
     "'description', 'abstract', 'summary', or 'what does it say about the data'.",
     "show → focused notes"),

    ("show:org",
     "Ask which organization or group this dataset belongs to. Use synonyms: 'org', 'organization', 'owner'.",
     "show → focused owner_org"),

    ("show:full",
     "Ask to see everything — all fields at once. Phrase it like 'show me the full metadata' "
     "or 'what do we have so far' or 'can I see all of it'.",
     "show → full dump"),

    # ── Dry-run synonyms ──────────────────────────────────────────────────
    ("dry-run:validate",
     "Ask to validate or check the metadata before submitting. Use a short synonym: "
     "'validate', 'verify', 'preview', or just 'check it'.",
     "dry-run node"),

    ("dry-run:synonyms",
     "Ask to run a dry run but phrase it differently than the last time — try 'test this', "
     "'run a check', 'preview the submission', or 'see if it passes'.",
     "dry-run node (LLM)"),

    # ── Single-field corrections ──────────────────────────────────────────
    ("revise:title",
     "Ask to update or change the title to something more specific. Suggest an actual new title "
     "value in your message, e.g. 'Alaska Orthophoto 2025' or similar.",
     "revise-field title"),

    ("revise:zoom-in",
     "Ask to zoom in or be more specific about the location in the title. "
     "Don't say 'zoom' — use phrases like 'more specific location', 'narrow it down', 'be more precise'.",
     "revise-field title"),

    ("revise:add-tag",
     "Ask to add a keyword or tag. Use synonyms: 'keyword', 'tag', 'label'. "
     "Suggest 'aerial-survey' or 'drone' as the value.",
     "revise-field tags"),

    ("revise:maintainer",
     "Correct the agent — you are NOT the maintainer. Ask to remove or clear that field. "
     "Use natural phrasing like 'that's not me' or 'I'm not responsible for maintaining this'.",
     "revise-field maintainer"),

    # ── Skepticism / implicit corrections ────────────────────────────────
    ("skeptic:wrong",
     "Express skepticism about a value that looks questionable — something doesn't seem right. "
     "Be vague (don't name the field); use phrases like 'that doesn't look right', "
     "'I'm not sure about that', or 'that seems off to me'.",
     "revise-field (any field)"),

    ("skeptic:date",
     "Challenge a date that appears to be in the future or implausible. "
     "Ask how it could have been processed or captured after a certain year. "
     "Use phrasing like 'that date can't be right' or 'how is that possible if it's still 2025'.",
     "revise-field notes/date"),

    ("skeptic:location",
     "Dispute the location — express doubt that the place name or coordinates are correct. "
     "Use phrasing like 'I don't think that location is right' or 'the location seems off'.",
     "revise-field title or spatial"),

    # ── Apply affirmations ────────────────────────────────────────────────
    ("apply:affirm",
     "Express approval and ask to go ahead and submit or register the dataset. "
     "Use phrases like 'looks good', 'go ahead', 'submit it', 'I approve', or 'create it'.",
     "apply or approval gate"),
]


# ---------------------------------------------------------------------------
# Driver LLM persona
# ---------------------------------------------------------------------------

_DRIVER_SYSTEM = """\
You are a cautious data scientist reviewing CKAN metadata before publication.
You speak casually and naturally, like someone typing in a chat window.

Key synonym rules — ALWAYS use these instead of technical field names:
  notes        → say "description", "abstract", or "summary"
  owner_org    → say "organization", "org", or "group"
  license_id   → say "license"
  tag_string   → say "tags" or "keywords"
  temporal_coverage_start → say "start date" or "begin date"
  temporal_coverage_end   → say "end date"

You occasionally make small typos (e.g. "tittle" instead of "title").
You are skeptical of values that look wrong or suspiciously generic.
Keep every reply to ONE short sentence — no pleasantries, no explanations.
"""


# ---------------------------------------------------------------------------
# Layer 1: normalize_action — fast-path alias table (deterministic, CI-safe)
# ---------------------------------------------------------------------------

class TestNormalizeAction:
    @pytest.mark.parametrize("raw", [
        "validate", "Validate", "VALIDATE",
        "preview", "Preview",
        "dry-run", "dry run", "dryrun",
        "DRY-RUN", "DRY RUN",
    ])
    def test_dry_run_synonyms(self, raw):
        assert normalize_action(raw) == "dry-run"

    @pytest.mark.parametrize("raw", ["register", "Register", "REGISTER"])
    def test_register_synonyms(self, raw):
        assert normalize_action(raw) == "apply"

    @pytest.mark.parametrize("raw", ["show", "status", "inspect"])
    def test_show_synonyms(self, raw):
        assert normalize_action(raw) == "show"

    @pytest.mark.parametrize("raw", ["revise-field", "revise_field"])
    def test_revise_field(self, raw):
        assert normalize_action(raw) == "revise-field"

    @pytest.mark.parametrize("raw", [None, "", "analyze", "revise", "something-else"])
    def test_passthrough(self, raw):
        result = normalize_action(raw)
        expected = str(raw or "").strip().lower().replace("_", "-")
        assert result == expected


# ---------------------------------------------------------------------------
# Layer 2: Conversation simulation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FakePersonaResult:
    proposed_metadata: dict
    clarification_questions: list
    stop_reason: str
    transcript: list = dataclasses.field(default_factory=list)


def _fake_engine(consolidated_inputs, *, organizational_metadata=None, **kw):
    """Fake persona engine: asks for author once, then converges."""
    org    = (organizational_metadata or {}).get("owner_org", "upstream")
    author = (organizational_metadata or {}).get("author", "")
    email  = (organizational_metadata or {}).get("author_email", "")
    if not author:
        return _FakePersonaResult(
            proposed_metadata={"title": "T", "name": "t"},
            clarification_questions=[{
                "field": "author",
                "question": "Who is the author of this dataset?",
                "requires_human": True,
                "reason_not_derivable": "not in sources",
            }],
            stop_reason="needs_clarification",
        )
    return _FakePersonaResult(
        proposed_metadata={
            "title": "Bethel, Alaska Orthophoto and 3D Model (2025)",
            "name": "bethel-alaska-orthophoto-3d-2025",
            "notes": (
                "Orthophoto and 3D model of Bethel, Alaska captured August 2025 "
                "using OpenDroneMap. GSD 0.8 cm, horizontal accuracy 0.024 m."
            ),
            "owner_org": org,
            "author": author,
            "author_email": email,
            "license_id": "cc-by",
            "tags": [{"name": "orthophoto"}, {"name": "alaska"}, {"name": "odm"}],
            "spatial": (
                '{"type":"Polygon","coordinates":[[[-161.779,60.797],[-161.778,60.797],'
                '[-161.778,60.796],[-161.779,60.796],[-161.779,60.797]]]}'
            ),
            "temporal_coverage_start": "2025-08-01",
            "temporal_coverage_end": "2025-08-01",
        },
        clarification_questions=[],
        stop_reason="converged",
    )


def _detect_response_type(text: str) -> str:
    lo = text.lower()
    if "schema profile" in lo or "which schema" in lo:
        return "schema_question"
    if ("which ckan organization" in lo or "which organization" in lo
            or ("owner" in lo and "?" in lo)):
        return "org_question"
    if "author" in lo and "?" in lo:
        return "clarification_question"
    if any(k in lo for k in ("clarification", "what is", "email", "license")) and "?" in lo:
        return "clarification_question"
    if "proposed ckan metadata" in lo or "files reviewed" in lo or "current metadata" in lo:
        return "proposal"
    if "dry run" in lo or "validated" in lo or "✓" in lo or "✗" in lo or "validation" in lo:
        return "dry_run_result"
    if "updated:" in lo or "make more changes" in lo:
        return "revise_result"
    if re.search(r"\*\*\w+\*\* \(`", text):
        return "show_result"
    return "unknown"


def _auto_respond(resp_type: str, text: str) -> str | None:
    if resp_type == "schema_question":
        for s in _KNOWN_SCHEMAS:
            if s in text.lower():
                return s
        return _KNOWN_SCHEMAS[0]
    if resp_type == "org_question":
        for o in _KNOWN_ORGS:
            if o in text.lower():
                return o
        return "upstream"
    if resp_type == "clarification_question":
        lo = text.lower()
        if "author"     in lo: return "Will Mobley"
        if "email"      in lo: return "wmobley@tacc.utexas.edu"
        if "license"    in lo: return "cc-by"
        if "spatial"    in lo: return "Bethel, Alaska"
        if "crs"        in lo or "coordinate" in lo: return "EPSG:4326"
        return "not applicable"
    return None


def _bootstrap_phase(session_id: str, settings, verbose: bool = True) -> bool:
    """
    Use build_persona_subgraph (with the fake engine) to drive the session
    through schema → org → clarification → proposal, writing the state file.
    Returns True if a state file was produced.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from app.agents.ckan_registration.persona_nodes import build_persona_subgraph

    graph = build_persona_subgraph(settings, engine=_fake_engine, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": session_id}}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  PHASE 1  Bootstrap   (session={session_id})")
        print(f"{'─'*60}")

    state = graph.invoke(
        {
            "thread_id": session_id,
            "request": {
                "session_id": session_id,
                "inline_files": _INLINE_FILES,
            },
        },
        config,
    )

    for _ in range(12):
        iv = state.get("__interrupt__")
        result_status = (state.get("result") or {}).get("status")

        if not iv and result_status in {"analyzed", "dry_run"}:
            break

        if not iv:
            if verbose:
                print(f"   ⚠ No interrupt, status={result_status!r}. Stopping bootstrap.")
            break

        interrupt_val = iv[0].value if isinstance(iv, (list, tuple)) else iv
        msg   = interrupt_val.get("message", "") if isinstance(interrupt_val, dict) else str(interrupt_val)
        itype = interrupt_val.get("type", "")    if isinstance(interrupt_val, dict) else ""

        resp_type = _detect_response_type(msg) if msg else "unknown"
        if itype == "schema_selection_required":
            resp_type = "schema_question"
        elif itype == "metadata_clarification_required":
            resp_type = "clarification_question"

        reply = _auto_respond(resp_type, msg)
        if reply is None:
            if verbose:
                print(f"   ⚠ Cannot auto-respond to interrupt type={resp_type!r}. Bootstrap stuck.")
            break

        if verbose:
            print(f"   → auto ({resp_type}): {reply!r}")
        state = graph.invoke(Command(resume={"message": reply}), config)

    produced = (settings.state_dir / f"{session_id}.json").exists()
    if verbose:
        if produced:
            print(f"   ✓ State file written: {session_id}.json")
        else:
            print(f"   ⚠ No state file produced — scenario phase will show 'No saved state'")
    return produced


def _driver_turn(
    settings,
    goal: str,
    conversation: list[dict],
    initial_context: str,
) -> str:
    """
    Call the driver LLM to generate one natural user message that achieves `goal`.

    The driver sees the recent conversation history and the goal instruction so it
    can reference real values from the agent's latest response.
    """
    from app.agents.ckan_registration.nodes import _invoke_openai_chat

    if conversation:
        history_lines = []
        for turn in conversation[-8:]:
            if turn["role"] == "human":
                history_lines.append(f"You: {turn['text'][:200]}")
            else:
                # Show only the first 300 chars of agent response
                history_lines.append(f"Agent: {turn['text'][:300]}")
        history_block = "\n".join(history_lines)
    else:
        history_block = f"[Initial metadata proposal]\n{initial_context[:600]}"

    prompt = (
        f"{history_block}\n\n"
        f"Your goal for this reply: {goal}\n\n"
        "Write exactly ONE short sentence (under 20 words) that achieves this goal. "
        "Use casual, natural phrasing and synonyms — never technical CKAN field names. "
        "Output only the sentence, nothing else."
    )

    try:
        return _invoke_openai_chat(
            settings,
            [
                {"role": "system", "content": _DRIVER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.8,
            max_tokens=60,
            timeout=20,
        ).strip().strip('"').strip("'")
    except Exception as exc:
        # Fall back to the goal text itself so the test keeps running
        return f"(driver LLM failed: {exc}) — {goal[:60]}"


def _adaptive_scenario_phase(
    session_id: str,
    settings,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Drive a goal-directed adaptive conversation against the full runner.

    For each scenario the driver LLM generates a natural message that achieves the
    stated goal, informed by whatever the agent actually said in prior turns.
    The real LLM router handles action selection; CKAN MCP is stubbed.
    """
    from app.agents.ckan_registration.graph import CkanRegistrationRunner, build_graph
    from app.agents.ckan_registration.schemas import CkanRunRequest

    graph  = build_graph(settings)
    runner = CkanRegistrationRunner.__new__(CkanRegistrationRunner)
    runner.settings = settings
    runner.graph    = graph

    # Load proposal from state file so the driver has real field values to reference.
    state_file = settings.state_dir / f"{session_id}.json"
    initial_context = ""
    if state_file.exists():
        try:
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            proposal = saved.get("desired_dataset_payload") or {}
            initial_context = json.dumps(proposal, indent=2)
        except Exception:
            pass

    # Stub MCP dry-run with a realistic passing response so the driver LLM
    # can read "✓ Valid" and know to proceed toward approval.
    mcp_stub = {
        "ok": True, "valid": True, "errors": [], "warnings": [],
        "status": "dry_run", "command": "dry-run",
        "session_id": session_id, "resource_count": 1,
        "review_markdown": (
            "## CKAN Dry-Run Preview\n\n"
            "**Validation**: ✓ Valid\n\n"
            "Dry-run passed. Send `REGISTER` to create this dataset and upload resources."
        ),
    }

    conversation: list[dict] = []   # {"role": "human"|"agent", "text": str}
    results: list[dict[str, Any]] = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  PHASE 2  Adaptive Scenario Conversation ({len(_SCENARIOS)} scenarios)")
        print(f"{'─'*60}")

    with patch("app.agents.ckan_registration.nodes._mcp_dry_run",
               return_value=mcp_stub), \
         patch("app.agents.ckan_registration.nodes._update_field_with_llm",
               return_value="(updated by simulation)"):

        for label, goal, hope in _SCENARIOS:
            # Generate a natural message that achieves this scenario's goal.
            driver_msg = _driver_turn(settings, goal, conversation, initial_context)

            if verbose:
                print(f"\n  [{label}]")
                print(f"   goal: {goal[:80]}...")
                print(f"   → user: {driver_msg!r}")
                print(f"   hope: {hope}")

            conversation.append({"role": "human", "text": driver_msg})

            try:
                resp  = runner.invoke(CkanRunRequest(session_id=session_id, message=driver_msg))
                md    = (resp.result or {}).get("review_markdown", "") or resp.error or ""
                rtype = _detect_response_type(md)

                if verbose:
                    print(f"   ← command={resp.command!r}  status={resp.status!r}  type={rtype!r}")
                    first = md.splitlines()[0] if md else "(no markdown)"
                    print(f"   ← {first[:120]}")

                conversation.append({"role": "agent", "text": md or f"[{resp.command}:{resp.status}]"})

                results.append({
                    "label": label, "goal": goal, "hope": hope,
                    "driver_message": driver_msg,
                    "command": resp.command, "status": resp.status,
                    "response_type": rtype, "ok": resp.ok, "error": resp.error,
                })

            except Exception as exc:
                if verbose:
                    print(f"   ✗ Exception: {exc}")
                conversation.append({"role": "agent", "text": f"[error: {exc}]"})
                results.append({
                    "label": label, "goal": goal, "hope": hope,
                    "driver_message": driver_msg,
                    "command": None, "status": "error",
                    "response_type": "error", "ok": False, "error": str(exc),
                })

    return results


def _print_summary(results: list[dict[str, Any]]):
    print(f"\n{'─'*60}")
    print("  SUMMARY")
    print(f"{'─'*60}")
    for r in results:
        ok   = r["ok"] and r["status"] != "error"
        icon = "✓" if ok else "✗"
        print(
            f"  {icon}  {r['label'].ljust(22)}"
            f"  sent: {r['driver_message'][:40].ljust(42)}"
            f"  cmd={str(r['command']).ljust(14)}"
            f"  type={r['response_type']}"
        )
    errors = [r for r in results if r["status"] == "error"]
    print(f"\n  {len(results) - len(errors)}/{len(results)} scenarios ran without exceptions")


def run_conversation(tmp_path: Path, verbose: bool = True) -> list[dict[str, Any]]:
    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    settings   = _settings(tmp_path, ask_schema=True)

    _bootstrap_phase(session_id, settings, verbose=verbose)
    results = _adaptive_scenario_phase(session_id, settings, verbose=verbose)
    if verbose:
        _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Pytest entry point for Layer 2
# ---------------------------------------------------------------------------

def test_conversation_simulation(tmp_path: Path):
    """
    Run the full adaptive conversation simulation and print a log.
    No hard assertions on LLM routing quality — fails only if the runner crashes.
    """
    results = run_conversation(tmp_path, verbose=True)
    errors  = [r for r in results if r["status"] == "error"]
    assert len(errors) < len(results), (
        f"All {len(results)} scenarios threw exceptions — something is fundamentally broken.\n"
        + "\n".join(f"  {e['label']}: {e['error']}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Direct script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run_conversation(Path(d), verbose=True)
