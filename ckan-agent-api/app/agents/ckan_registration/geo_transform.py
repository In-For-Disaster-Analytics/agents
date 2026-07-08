"""Geo transform proposal helpers (spec 2026-06-30).

A persona may *propose* a geo transform; a human approves it; the ``geo-apply`` node executes it.
This module owns the proposal contract: mapping a proposed ``operation`` to its MCP tool name,
validating + assembling the tool args, and rendering the human-facing approval payload (which
surfaces ``register_to_dataset`` and a readable clip bbox so the approver sees exactly what runs).

Nothing here touches a token — the token is injected server-side by ``GeoTransformRunner`` at the
gated node, never from a proposal or from state.
"""

from __future__ import annotations

from typing import Any

# Proposed operation -> MCP geo tool name.
OPERATION_TOOLS: dict[str, str] = {
    "reproject": "reproject_raster",
    "cog": "convert_to_cog",
    "clip": "clip_raster",
    "overviews": "build_overviews",
}


class TransformProposalError(ValueError):
    """Raised when a transform proposal is missing/invalid (no submission is made)."""


def _require(proposal: dict[str, Any], key: str) -> Any:
    value = proposal.get(key)
    if value in (None, ""):
        raise TransformProposalError(f"transform proposal missing required field: {key!r}")
    return value


def build_tool_call(proposal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate a proposal and return ``(tool_name, args)`` ready for the runner.

    Args carry only operation inputs — never a token. ``register_to_dataset`` is passed through
    as-is (the server resolves None to the source dataset). Heavy validation (CRS range, geometry
    bounds, output-name regex, SSRF) is enforced server-side; here we fail fast on shape only.
    """
    operation = str(_require(proposal, "operation")).strip().lower()
    tool = OPERATION_TOOLS.get(operation)
    if tool is None:
        raise TransformProposalError(
            f"unknown transform operation {operation!r}; expected one of {sorted(OPERATION_TOOLS)}"
        )
    resource_id = str(_require(proposal, "resource_id")).strip()
    output_name = str(_require(proposal, "output_name")).strip()
    args: dict[str, Any] = {"resource_id": resource_id, "output_name": output_name}
    register_to = proposal.get("register_to_dataset")
    if register_to:
        args["register_to_dataset"] = str(register_to).strip()

    if operation == "reproject":
        args["target_crs"] = int(_require(proposal, "target_crs"))
    elif operation == "cog":
        if proposal.get("compression"):
            args["compression"] = str(proposal["compression"]).strip()
    elif operation == "clip":
        geometry = _require(proposal, "clip_geometry")
        if not isinstance(geometry, dict):
            raise TransformProposalError("clip_geometry must be a GeoJSON object")
        args["clip_geometry"] = geometry
    elif operation == "overviews":
        if proposal.get("overview_levels") is not None:
            levels = proposal["overview_levels"]
            if not isinstance(levels, list):
                raise TransformProposalError("overview_levels must be a list of integers")
            args["overview_levels"] = [int(x) for x in levels]
    return tool, args


def _bbox_summary(geometry: dict[str, Any]) -> str:
    """Return a compact lon/lat bounding-box summary for a GeoJSON Polygon/MultiPolygon."""
    coords = geometry.get("coordinates")
    xs: list[float] = []
    ys: list[float] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and all(isinstance(v, (int, float)) for v in node):
                xs.append(float(node[0]))
                ys.append(float(node[1]))
            else:
                for child in node:
                    _walk(child)

    _walk(coords)
    if not xs or not ys:
        return "geometry (unparsed)"
    return f"bbox lon[{min(xs):.4f}, {max(xs):.4f}] lat[{min(ys):.4f}, {max(ys):.4f}]"


def approval_payload(proposal: dict[str, Any], *, thread_id: str | None = None) -> dict[str, Any]:
    """Build the human-facing ``geo-approval`` interrupt payload.

    Surfaces the operation, source resource, and — explicitly — the destination dataset
    (incl. the source-dataset default) and a readable clip bbox, so the approver authorizes the
    exact effect (security High 4a/4b). Resume with ``approval == "REGISTER"`` to run it.
    """
    operation = str(proposal.get("operation") or "").strip().lower()
    register_to = proposal.get("register_to_dataset")
    destination = str(register_to) if register_to else "→ source resource's dataset (default)"
    lines = [
        "## Geo transform — approval required",
        f"- Operation: `{operation}`",
        f"- Source resource: `{proposal.get('resource_id', '<missing>')}`",
        f"- Output name: `{proposal.get('output_name', '<missing>')}`",
        f"- Registers output to dataset: {destination}",
    ]
    if operation == "reproject" and proposal.get("target_crs"):
        lines.append(f"- Target CRS: `EPSG:{proposal['target_crs']}`")
    if operation == "clip" and isinstance(proposal.get("clip_geometry"), dict):
        lines.append(f"- Clip extent: {_bbox_summary(proposal['clip_geometry'])}")
    lines.append("")
    lines.append("This spends Tapis Abaco compute and creates a new CKAN resource. "
                 "Reply `REGISTER` to authorize, or anything else to cancel.")
    return {
        "type": "geo_transform_approval_required",
        "message": "\n".join(lines),
        "required_approval": "REGISTER",
        "operation": operation,
        "resource_id": proposal.get("resource_id"),
        "register_to_dataset": register_to,
        "review_markdown": "\n".join(lines),
        "thread_id": thread_id,
    }
