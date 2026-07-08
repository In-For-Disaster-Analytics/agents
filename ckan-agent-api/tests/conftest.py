"""Shared test fixtures for the agent test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_org_grounding(monkeypatch):
    """Keep tests hermetic: owner_org grounding (spec 2026-06-30) must never reach a live CKAN
    portal during tests. Default to an empty org list (grounding no-ops, preserving prior
    behavior). Tests that exercise grounding override ``org_grounding.fetch_orgs`` themselves.
    """
    monkeypatch.setattr(
        "app.agents.ckan_registration.org_grounding.fetch_orgs",
        lambda settings: [],
        raising=False,
    )
    monkeypatch.setattr(
        "app.agents.ckan_registration.org_grounding.fetch_licenses",
        lambda settings: [],
        raising=False,
    )
