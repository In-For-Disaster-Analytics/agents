"""Tests for grounding owner_org in the live CKAN organization list (spec 2026-06-30)."""

from __future__ import annotations

from dataclasses import replace

from app.agents.ckan_registration import persona_nodes
from app.agents.ckan_registration.org_grounding import (
    resolve_license_id,
    resolve_owner_org_choice,
)
from app.settings import Settings

TWDB = {"id": "org-1", "name": "twdb-gam", "title": "TWDB GAM"}
DSO = {"id": "org-2", "name": "dso-institute", "title": "DSO Institute"}


def _ground(settings, state):
    """Run both org-level groundings the way schema_select does; return the merged org_metadata."""
    org_meta = dict(state.get("org_metadata") or {})
    persona_nodes._ground_owner_org_field(settings, state, org_meta)
    persona_nodes._ground_license_field(settings, org_meta)
    return org_meta


# ── pure resolver ──────────────────────────────────────────────────────────


def test_resolver_configured_match_by_name():
    resolved, ambiguous, options = resolve_owner_org_choice([TWDB, DSO], "dso-institute")
    assert resolved == "dso-institute" and ambiguous is False and options == []


def test_resolver_configured_match_by_title_case_insensitive():
    resolved, ambiguous, _ = resolve_owner_org_choice([TWDB, DSO], "TWDB GAM")
    assert resolved == "twdb-gam" and ambiguous is False


def test_resolver_single_org_used_when_configured_absent():
    resolved, ambiguous, _ = resolve_owner_org_choice([TWDB], "DSO-Institute")
    assert resolved == "twdb-gam" and ambiguous is False  # the local twdb-gam case


def test_resolver_multiple_no_match_is_ambiguous():
    resolved, ambiguous, options = resolve_owner_org_choice([TWDB, DSO], "DSO-Institute-typo")
    assert resolved is None and ambiguous is True
    assert {o["name"] for o in options} == {"twdb-gam", "dso-institute"}


def test_resolver_no_orgs_returns_none():
    assert resolve_owner_org_choice([], "anything") == (None, False, [])


# ── _ground_owner_org node helper ───────────────────────────────────────────


def test_ground_uses_single_local_org(monkeypatch):
    settings = replace(Settings(), ckan_owner_org="DSO-Institute")
    monkeypatch.setattr("app.agents.ckan_registration.org_grounding.fetch_orgs", lambda s: [TWDB])
    assert _ground(settings, {"thread_id": "t"}) == {"owner_org": "twdb-gam"}


def test_ground_noops_when_configured_matches(monkeypatch):
    settings = replace(Settings(), ckan_owner_org="dso-institute")
    monkeypatch.setattr(
        "app.agents.ckan_registration.org_grounding.fetch_orgs", lambda s: [TWDB, DSO]
    )
    assert _ground(settings, {"thread_id": "t"}) == {}  # configured org exists → no override


def test_ground_skips_when_already_chosen(monkeypatch):
    called = {"n": 0}

    def _fetch(_s):
        called["n"] += 1
        return [TWDB, DSO]

    monkeypatch.setattr("app.agents.ckan_registration.org_grounding.fetch_orgs", _fetch)
    org_meta = {"owner_org": "twdb-gam"}
    persona_nodes._ground_owner_org_field(replace(Settings(), ckan_owner_org="x"), {}, org_meta)
    assert org_meta == {"owner_org": "twdb-gam"} and called["n"] == 0  # short-circuits, no CKAN call


def test_ground_degrades_on_ckan_error(monkeypatch):
    def _boom(_s):
        raise RuntimeError("ckan down")

    monkeypatch.setattr("app.agents.ckan_registration.org_grounding.fetch_orgs", _boom)
    assert _ground(replace(Settings(), ckan_owner_org="DSO-Institute"), {}) == {}  # never blocks


def test_ground_ambiguous_interrupts_and_uses_choice(monkeypatch):
    settings = replace(Settings(), ckan_owner_org="not-a-real-org")
    monkeypatch.setattr(
        "app.agents.ckan_registration.org_grounding.fetch_orgs", lambda s: [TWDB, DSO]
    )
    # Simulate the human picking from the interrupt options.
    monkeypatch.setattr(persona_nodes, "interrupt", lambda payload: {"message": "twdb-gam"})
    assert _ground(settings, {"thread_id": "t"}) == {"owner_org": "twdb-gam"}


# ── license_id grounding ─────────────────────────────────────────────────────

LICENSES = [
    {"id": "cc-by-4.0", "title": "Creative Commons Attribution 4.0"},
    {"id": "notspecified", "title": "License not specified"},
]


def test_resolve_license_exact_id():
    assert resolve_license_id(LICENSES, "cc-by-4.0") == "cc-by-4.0"


def test_resolve_license_normalized_match():
    # configured "cc-by" normalizes to match "cc-by-4.0"? No — different; title/id normalized match.
    assert resolve_license_id([{"id": "CC-BY", "title": "Attribution"}], "cc-by") == "CC-BY"


def test_resolve_license_no_match_returns_none():
    assert resolve_license_id(LICENSES, "some-unknown-license") is None
    assert resolve_license_id([], "cc-by") is None


def test_ground_license_maps_to_portal_id(monkeypatch):
    settings = replace(Settings(), ckan_owner_org="dso-institute", ckan_dataset_license_id="CC-BY")
    monkeypatch.setattr("app.agents.ckan_registration.org_grounding.fetch_orgs", lambda s: [DSO])
    monkeypatch.setattr(
        "app.agents.ckan_registration.org_grounding.fetch_licenses",
        lambda s: [{"id": "cc-by", "title": "Attribution"}],
    )
    out = _ground(settings, {"thread_id": "t"})
    assert out["license_id"] == "cc-by"  # canonical portal id


def test_controlled_vocab_violations_flags_off_vocab_values():
    vocab = {"categories": ["Groundwater", "Planning"], "collection_method": ["Model Output"]}
    # list field with one bad value, string field with a bad value
    payload = {"categories": ["Groundwater", "Volcanoes"], "collection_method": "Guesswork"}
    v = persona_nodes._controlled_vocab_violations(payload, vocab)
    assert v["categories"]["invalid"] == ["Volcanoes"]
    assert v["collection_method"]["invalid"] == ["Guesswork"]


def test_controlled_vocab_violations_clean_and_case_insensitive():
    vocab = {"collection_method": ["Model Output"]}
    assert persona_nodes._controlled_vocab_violations({"collection_method": "model output"}, vocab) == {}
    assert persona_nodes._controlled_vocab_violations({}, vocab) == {}  # unset field is not a violation


def test_match_owner_org_fallbacks():
    names = {"twdb-gam", "dso-institute"}
    assert persona_nodes._match_owner_org("TWDB-GAM", names, "d") == "twdb-gam"  # case-insensitive
    assert persona_nodes._match_owner_org("use twdb-gam please", names, "d") == "twdb-gam"  # substring
    assert persona_nodes._match_owner_org("", names, "default-org") == "default-org"  # fallback
