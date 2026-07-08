"""Tests for the gated geo transform path (spec 2026-06-30).

Covers the proposal contract (build_tool_call + approval_payload), the GeoTransformRunner
(submit→poll, token injection, RUNNING-on-timeout), and the geo-approval/geo-apply nodes
(approval gate, per-session cap, transform-status, token never written to state, graceful
"not configured" handling). All mocked — no live Tapis/CKAN.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from app.agents.ckan_registration.geo_transform import (
    TransformProposalError,
    approval_payload,
    build_tool_call,
)
from app.agents.ckan_registration.nodes import make_geo_apply_node, make_geo_approval_node
from app.settings import Settings
from app.tools.executor import GeoTransformRunner

_PROP = {"operation": "reproject", "resource_id": "r", "output_name": "o.tif", "target_crs": 4326}


def _geo_settings(**kw: Any) -> Settings:
    base = {"geo_mcp_enabled": True, "geo_mcp_tapis_token": "TOK"}
    base.update(kw)
    return replace(Settings(), **base)


# ── proposal contract ────────────────────────────────────────────────────


def test_build_tool_call_maps_operations():
    tool, args = build_tool_call(
        {"operation": "reproject", "resource_id": "r1", "output_name": "o.tif", "target_crs": 4326}
    )
    assert tool == "reproject_raster"
    assert args == {"resource_id": "r1", "output_name": "o.tif", "target_crs": 4326}

    tool, args = build_tool_call(
        {"operation": "clip", "resource_id": "r", "output_name": "o.tif",
         "clip_geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}
    )
    assert tool == "clip_raster"
    assert args["clip_geometry"]["type"] == "Polygon"


def test_build_tool_call_rejects_bad_proposal():
    with pytest.raises(TransformProposalError):
        build_tool_call({"operation": "nope", "resource_id": "r", "output_name": "o.tif"})
    with pytest.raises(TransformProposalError):
        build_tool_call({"operation": "reproject", "output_name": "o.tif"})  # missing resource_id


def test_approval_payload_surfaces_destination_and_bbox():
    payload = approval_payload(
        {"operation": "clip", "resource_id": "r", "output_name": "o.tif",
         "clip_geometry": {"type": "Polygon", "coordinates": [[[-100, 30], [-99, 30], [-99, 31], [-100, 30]]]}},
        thread_id="t1",
    )
    assert payload["required_approval"] == "REGISTER"
    assert "source resource's dataset" in payload["message"]  # default destination surfaced
    assert "bbox lon[-100.0000, -99.0000]" in payload["message"]

    payload2 = approval_payload(
        {"operation": "reproject", "resource_id": "r", "output_name": "o.tif",
         "register_to_dataset": "ds-99", "target_crs": 3857}
    )
    assert "ds-99" in payload2["message"]
    assert payload2["register_to_dataset"] == "ds-99"


# ── GeoTransformRunner ───────────────────────────────────────────────────


class _FakeGeoClient:
    """Scripts submit + status with a configurable number of RUNNING polls."""

    def __init__(self, running_polls: int = 1):
        self.running_polls = running_polls
        self.polls = 0
        self.submit_token = None
        self.status_calls = 0

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "get_execution_status":
            self.status_calls += 1
            self.polls += 1
            if self.polls <= self.running_polls:
                return {"execution_id": args["execution_id"], "status": "RUNNING"}
            return {"execution_id": args["execution_id"], "status": "COMPLETE",
                    "result": {"operation": "reproject"}, "registered": {"resource": {"id": "new-res"}}}
        # a transform submit
        self.submit_token = args.get("tapis_token")
        return {"execution_id": "exec-9", "status": "SUBMITTED"}


def test_runner_injects_token_and_polls_to_complete():
    client = _FakeGeoClient(running_polls=1)
    runner = GeoTransformRunner(client, token_value="TOK", poll_timeout=10.0, sleep=lambda s: None)
    out = runner.run("reproject_raster", {"resource_id": "r", "target_crs": 4326, "output_name": "o.tif"})
    assert out["success"] is True
    assert out["result"]["status"] == "COMPLETE"
    assert out["result"]["registered"]["resource"]["id"] == "new-res"
    assert client.submit_token == "TOK"  # token injected server-side


def test_runner_returns_running_on_timeout():
    client = _FakeGeoClient(running_polls=10**9)  # never reaches terminal within the timeout
    runner = GeoTransformRunner(client, token_value="TOK", poll_timeout=0.05, sleep=lambda s: None)
    out = runner.run("reproject_raster", {"resource_id": "r", "target_crs": 4326, "output_name": "o.tif"})
    assert out["success"] is True
    assert out["result"]["status"] == "RUNNING"
    assert out["result"]["execution_id"] == "exec-9"


# ── geo-approval / geo-apply nodes ─────────────────────────────────────────


def test_geo_approval_passthrough_when_already_approved():
    node = make_geo_approval_node()
    state = {
        "action": "geo-transform",
        "request": {"approval": "REGISTER"},
        "transform_request": {"operation": "reproject", "resource_id": "r", "output_name": "o.tif", "target_crs": 4326},
    }
    out = node(state)
    assert out["status"] == "approved"


def test_geo_apply_blocks_without_approval():
    node = make_geo_apply_node(_geo_settings())
    out = node({"action": "geo-transform", "request": {}, "transform_request": _PROP})
    assert out["status"] == "approval_missing"


def test_geo_apply_enforces_session_cap():
    node = make_geo_apply_node(_geo_settings(geo_max_transforms_per_session=2))
    state = {
        "action": "geo-transform",
        "request": {"approval": "REGISTER"},
        "transform_request": _PROP,
        "transforms_submitted": 2,
    }
    out = node(state)
    assert out["status"] == "limit_reached"


def test_geo_apply_unconfigured_is_graceful():
    node = make_geo_apply_node(Settings())  # geo_mcp_enabled False
    out = node({"action": "geo-transform", "request": {"approval": "REGISTER"}, "transform_request": _PROP})
    assert out["status"] == "error"
    assert "unavailable" in out["error"].lower()


def test_geo_apply_runs_via_runner_and_scrubs_token(monkeypatch):
    settings = _geo_settings(geo_mcp_tapis_token="SECRET_TOK", geo_max_transforms_per_session=5)
    client = _FakeGeoClient(running_polls=0)
    runner = GeoTransformRunner(
        client, token_value=settings.geo_mcp_tapis_token, poll_timeout=10.0, sleep=lambda s: None
    )
    monkeypatch.setattr("app.agents.ckan_registration.nodes._geo_runner", lambda s: runner)

    node = make_geo_apply_node(settings)
    state = {
        "action": "geo-transform",
        "request": {"approval": "REGISTER"},
        "transform_request": _PROP,
        "transforms_submitted": 0,
    }
    out = node(state)
    assert out["status"] == "ok"
    assert out["transforms_submitted"] == 1
    assert out["result"]["registered"]["resource"]["id"] == "new-res"
    # the injected token must never appear in the state written back
    assert "SECRET_TOK" not in str(out)


def test_route_from_intake_geo_transform():
    from app.agents.ckan_registration.nodes import route_from_intake

    assert route_from_intake({"action": "geo-transform"}) == "geo-transform"
    assert route_from_intake({"action": "transform-status"}) == "geo-transform"
    assert route_from_intake({"action": "apply"}) == "apply"


def test_route_after_propose():
    from app.agents.ckan_registration.persona_nodes import route_after_propose

    assert route_after_propose({"transform_request": {"operation": "reproject"}}) == "geo-approval"
    assert route_after_propose({}) == "END"
