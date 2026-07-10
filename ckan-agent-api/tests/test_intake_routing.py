"""
Intake routing tests — two distinct layers:

  Layer 1 (pytest): normalize_action fast-path alias table.
  Pure deterministic logic, no LLM. Always run in CI.

  Layer 2 (script): simulate_conversation().
  Runs a full registration session with the real LLM router and a fake persona
  engine. Phase 1 bootstraps to a proposal using the persona subgraph directly
  (writes the state file). Phase 2 exercises synonym/correction/question phrases
  against the full runner and prints a human-readable conversation log.

  Run the simulation:

      # as a pytest test (no hard assertions on LLM routing quality):
      pytest tests/test_intake_routing.py::test_conversation_simulation -v -s

      # directly as a script:
      python tests/test_intake_routing.py
"""

from __future__ import annotations

import dataclasses
import json
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

_KNOWN_ORGS   = ["upstream", "dso-internal", "tacc", "twdb-gams", "planet-texas-2050"]
_KNOWN_SCHEMAS = ["generic_ckan", "subside"]

# Scenarios to exercise once we have a live proposal in state.
# (label, message, what we hope routing does)
_SCENARIOS: list[tuple[str, str, str]] = [
    # Focused field show
    ("show:title",           "whats the title?",                             "show → focused title"),
    ("show:notes",           "what is the description?",                     "show → focused notes"),
    ("show:org",             "what organization is set?",                    "show → focused owner_org"),
    ("show:full",            "show me everything",                           "show → full dump"),
    # Dry-run synonyms
    ("dry-run:validate",     "validate",                                     "dry-run node (fast-path)"),
    ("dry-run:preview",      "preview",                                      "dry-run node (fast-path)"),
    ("dry-run:check",        "check it",                                     "dry-run node (LLM)"),
    ("dry-run:test",         "test the metadata",                            "dry-run node (LLM)"),
    # Single-field corrections
    ("revise:title",         "update the title to Alaska Orthophoto 2025",   "revise-field title"),
    ("revise:zoom-in",       "can we zoom in on the location?",              "revise-field title"),
    ("revise:add-tag",       "add a tag for aerial-survey",                  "revise-field tags"),
    ("revise:maintainer",    "I'm not the maintainer, remove that",          "revise-field maintainer"),
    # Skepticism / implicit corrections
    ("skeptic:wrong",        "that doesn't look right",                      "revise-field (any field)"),
    ("skeptic:date",         "how could it be processed after 2025?",        "revise-field notes"),
    ("skeptic:location",     "the location doesn't look right",              "revise-field title or spatial"),
    # Apply affirmations (currently require exact REGISTER — documents the gap)
    ("apply:looks-good",     "looks good, go ahead",                         "apply or approval gate"),
    ("apply:proceed",        "proceed",                                      "apply or approval gate"),
    ("apply:submit",         "submit this dataset",                          "apply or approval gate"),
]


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
    org   = (organizational_metadata or {}).get("owner_org", "upstream")
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
    if "which ckan organization" in lo or "which organization" in lo or "owner" in lo and "?" in lo:
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
    if re.search(r"\*\*\w+\*\* \(`", text):      # field (origin): value pattern
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
        if "author" in lo:     return "Will Mobley"
        if "email" in lo:      return "wmobley@tacc.utexas.edu"
        if "license" in lo:    return "cc-by"
        if "spatial" in lo:    return "Bethel, Alaska"
        if "crs" in lo or "coordinate" in lo: return "EPSG:4326"
        return "not applicable"
    return None


import re  # needed by _detect_response_type


class ConversationLog:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: list[dict[str, Any]] = []

    def record(self, role: str, text: str, **meta):
        self.turns.append({"role": role, "text": text, "meta": meta})

    def print(self):
        w = 80
        print("\n" + "=" * w)
        print(f"  SESSION  {self.session_id}")
        print("=" * w)
        for t in self.turns:
            prefix = "→" if t["role"] == "user" else "←"
            meta   = t["meta"]
            tag    = f"[{meta.get('command','?')} / {meta.get('status','?')}]" if t["role"] == "agent" else ""
            print(f"\n{prefix} {t['role'].upper()} {tag}")
            for line in t["text"].splitlines()[:6]:
                print(f"   {line}")
            extra = len(t["text"].splitlines()) - 6
            if extra > 0:
                print(f"   ... ({extra} more lines)")
        print("=" * w + "\n")


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

    # Initial invoke — provides files, may hit schema interrupt first
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
            # No interrupt and not yet done — something unexpected
            if verbose:
                print(f"   ⚠ No interrupt, status={result_status!r}. Stopping bootstrap.")
            break

        interrupt_val = iv[0].value if isinstance(iv, (list, tuple)) else iv
        msg  = interrupt_val.get("message", "") if isinstance(interrupt_val, dict) else str(interrupt_val)
        itype = interrupt_val.get("type", "") if isinstance(interrupt_val, dict) else ""

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


def _scenario_phase(
    session_id: str,
    settings,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Invoke the full CkanRegistrationRunner for each scenario message and record results.
    The real LLM router decides routing; CKAN MCP is stubbed.
    """
    from app.agents.ckan_registration.graph import CkanRegistrationRunner, build_graph
    from app.agents.ckan_registration.schemas import CkanRunRequest

    graph = build_graph(settings)
    runner = CkanRegistrationRunner.__new__(CkanRegistrationRunner)
    runner.settings = settings
    runner.graph = graph

    results: list[dict[str, Any]] = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  PHASE 2  Scenario Exercises")
        print(f"{'─'*60}")

    mcp_stub = {"ok": True, "valid": True, "errors": [], "warnings": []}

    for label, message, hope in _SCENARIOS:
        if verbose:
            print(f"\n  [{label}]")
            print(f"   user: {message!r}")
            print(f"   hope: {hope}")

        with patch("app.agents.ckan_registration.nodes._mcp_dry_run", return_value=mcp_stub), \
             patch("app.agents.ckan_registration.nodes._update_field_with_llm",
                   return_value="(updated by simulation)"):
            try:
                resp = runner.invoke(CkanRunRequest(session_id=session_id, message=message))
                md   = (resp.result or {}).get("review_markdown", "") or resp.error or ""
                rtype = _detect_response_type(md)

                if verbose:
                    print(f"   ← command={resp.command!r}  status={resp.status!r}  type={rtype!r}")
                    first = md.splitlines()[0] if md else "(no markdown)"
                    print(f"   ← {first[:100]}")

                results.append({
                    "label": label, "message": message, "hope": hope,
                    "command": resp.command, "status": resp.status,
                    "response_type": rtype, "ok": resp.ok, "error": resp.error,
                })
            except Exception as exc:
                if verbose:
                    print(f"   ✗ Exception: {exc}")
                results.append({
                    "label": label, "message": message, "hope": hope,
                    "command": None, "status": "error", "response_type": "error",
                    "ok": False, "error": str(exc),
                })

    return results


def _print_summary(results: list[dict[str, Any]]):
    print(f"\n{'─'*60}")
    print("  SUMMARY")
    print(f"{'─'*60}")
    for r in results:
        ok    = r["ok"] and r["status"] != "error"
        icon  = "✓" if ok else "✗"
        label = r["label"].ljust(25)
        print(f"  {icon}  {label}  cmd={str(r['command']).ljust(15)}  "
              f"status={str(r['status']).ljust(15)}  type={r['response_type']}")
    errors = [r for r in results if r["status"] == "error"]
    print(f"\n  {len(results) - len(errors)}/{len(results)} scenarios ran without exceptions")


def run_conversation(tmp_path: Path, verbose: bool = True) -> list[dict[str, Any]]:
    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    settings   = _settings(tmp_path, ask_schema=True)

    _bootstrap_phase(session_id, settings, verbose=verbose)
    results = _scenario_phase(session_id, settings, verbose=verbose)
    if verbose:
        _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Pytest entry point for Layer 2
# ---------------------------------------------------------------------------

def test_conversation_simulation(tmp_path: Path):
    """
    Run the full conversation simulation and print a log.
    No hard assertions on LLM routing quality — fails only if the runner crashes.
    """
    results = run_conversation(tmp_path, verbose=True)
    errors = [r for r in results if r["status"] == "error"]
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
