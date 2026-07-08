# Human-approval gated node: geo transforms + CKAN MCP writes

## Status

Implemented (geo transform gated path). **CKAN MCP-write re-point deferred** — see deviation.

### Implementation summary (2026-06-30)

On branch `feat/mcp-tool-integration`:

- **Geo transform gated path (new, additive):** `geo-approval` (interrupt; surfaces operation,
  destination dataset incl. source default, clip bbox; resumes on `REGISTER`) → `geo-apply`
  (per-session cap, server-side token injection via `GeoTransformRunner`, bounded poll, RUNNING
  on timeout, token never written to state). New `geo-transform` / `transform-status` actions and
  routing. `GeoTransformRunner` (executor.py) + `geo_transform.py` (proposal contract +
  approval payload). State: `transform_request`, `transform_execution_id`, `transforms_submitted`.
- **Persona-proposed trigger:** the `propose` node detects an author `_transform_proposal`,
  shape-validates it, and routes to `geo-approval` (`route_after_propose`). The model only
  proposes; transforms stay hard-blocked in the tool loop; a human authorizes via `REGISTER`.
- **12 gated-node tests** (proposal contract, runner submit/poll/timeout + token injection, node
  approval gate, per-session cap, token-scrub, graceful unconfigured, routing). Graph builds in
  both `persona_chat` states.

**Deviation / deferred — CKAN MCP-write re-point.** Not done. The live CKAN `apply` path
(`make_safe_apply_node` → `LegacyCkanWorker`) **delegates to an external legacy script**
(`ckan-registration/ckan_agent.py`) that handles multi-resource registration + uploads. Faithfully
re-pointing that to the MCP `schema_create_*` write tools (saved-dry-run-state → MCP metadata
mapping, per-resource create/upload parity) is a subproject of its own and exactly what the
security review said needs a dedicated pass. It is **not safe to bolt a partial replacement onto
the proven registration path** at the tail of this work. The legacy apply path is unchanged;
CKAN live writes still flow through it. Recommended as its own next increment.

---

Approved — implementing (user, 2026-06-30).

### Locked decisions (user, 2026-06-30)

- **Build both**: the geo transform gated node **and** the additive CKAN MCP-write branch (legacy
  worker stays the default/fallback — proven path not deleted).
- **Transform entry = persona-proposed.** A persona may emit a structured **transform proposal**
  (it never executes — transforms stay hard-blocked from the tool loop). The proposal becomes the
  `geo-approval` interrupt payload; a human confirms with `REGISTER`; only then does `geo-apply`
  run it. The model proposes; the human authorizes; the node executes.
- **Long-run = bounded poll + status follow-up.** `geo-apply` polls to terminal up to a bounded
  timeout; if still running it returns the `execution_id` and a `transform-status` action to poll
  later — never blocks indefinitely.

## Objective

Deliver the human-approval **gated execution** path so the side-effectful MCP tools become
usable — strictly behind explicit approval, never from the autonomous persona loop:

1. **Geo transforms** (`reproject_raster` / `convert_to_cog` / `clip_raster` / `build_overviews`)
   — spend Tapis Abaco compute and register a new CKAN resource.
2. **CKAN live writes** via the MCP write tools (`schema_create_package` / `schema_update_package`
   / `schema_create_resource`) — the deferred "Fork A" re-point from the CKAN increment.

This completes the gated half of the dso-geo integration
([2026-06-30-integrate-dso-geo-mcp.md](2026-06-30-integrate-dso-geo-mcp.md)) and the deferred
Fork A from the CKAN increment ([2026-06-29-integrate-dso-ckan-mcp.md](2026-06-29-integrate-dso-ckan-mcp.md)).

## Current system summary

- **Graph** (`graph.py`): `intake → route_from_intake → {metadata | dry-run | apply | show}`;
  `apply` routes through `approval` (a LangGraph `interrupt()` requiring `approval == "REGISTER"`)
  then `make_safe_apply_node`, which runs `LegacyCkanWorker.run("apply", request)` (the proven
  multi-resource registration path; requires saved state `status == "dry_run"`).
- **Actions** (`normalize_action`): `analyze, revise, dry-run, apply, show` (+ aliases). No
  transform action exists.
- **State** (`CkanRegistrationState`, TypedDict): `thread_id, action, request, result, status,
  error, …`. Persisted by the SQLite checkpointer.
- **Agent tool layer** (already landed): `MCPClient`, `MCPToolExecutor` (hard-blocks transforms +
  CKAN live writes; `token_arg` injection), `GeoSyncExecutor`, multi-server router. The
  `geo_mcp_*` / `mcp_*` settings exist, incl. `geo_max_transforms_per_session`.
- **MCP write tools**: `schema_create_package(dataset_type, metadata, tapis_token, dry_run)`,
  `schema_update_package(id, metadata_updates, …)`, `schema_create_resource(package_id,
  resource_metadata, upload_file, …)`. Geo transforms: submit → `execution_id` → poll.

## Proposed design

Two **separate** nodes sharing the `interrupt()` + token-injection + scrub pattern (architect O3).

### A. Geo transform node (new, additive)

- **New action `geo-transform`** (+ alias `transform`). `route_from_intake` routes it to a new
  `geo-approval` node → `geo-apply` node.
- **Request shape** (in `request`): `{operation: reproject|cog|clip|overviews, resource_id,
  output_name, register_to_dataset?, target_crs?|compression?|clip_geometry?|overview_levels?}`.
- **`geo-approval`** — `interrupt()` whose payload surfaces, for human confirmation: the
  operation, source `resource_id`, **`register_to_dataset`** (explicitly, incl. the
  "→ source dataset `<id>`" default), and a **human-readable clip bbox** when clipping. Resumes
  only on `approval == "REGISTER"` (reuse `APPLY_APPROVAL`).
- **`geo-apply`** — enforces the per-session cap (`geo_max_transforms_per_session`) and refuses if
  an earlier transform for the session is still in flight; injects `geo_mcp_tapis_token`
  server-side into the tool args (never from `request`/model); calls the geo transform tool via
  the geo `MCPClient.call_tool` **directly** (bypassing the executor block, which is the
  persona-loop guard); then polls `get_execution_status` to terminal (bounded) and returns the
  `registered` CKAN resource. The token is **stripped from any state/result written to the
  checkpointer** and never logged (security High 2a/2b).
- **State fields** (new): `transform_request`, `transform_execution_id`, `transforms_submitted`.

### B. CKAN MCP write path (additive re-point of Fork A)

- `make_safe_apply_node` gains an **MCP-write branch** used when `mcp_enabled`: after approval +
  dry-run state, it issues the live writes through the MCP CKAN write tools (`dry_run=False`),
  reusing the saved dry-run metadata. **The `LegacyCkanWorker` path remains the default/fallback**
  when `mcp_enabled` is off or the MCP server is unreachable — we do **not** delete the proven
  path (Fork B philosophy). Same approval gate, same `status == "dry_run"` precondition.
- Token: `mcp_tapis_token` via the client header (CKAN), already supported; never in args/state.

### Routing / graph changes

```
intake → route_from_intake
  ├ geo-transform → geo-approval → geo-apply → END
  └ apply         → approval     → apply (MCP-write branch when mcp_enabled, else legacy)
```

## Files likely affected

| Path | Change |
|---|---|
| `app/agents/ckan_registration/nodes.py` | `normalize_action` + `route_from_intake` (geo-transform); `make_geo_approval_node`, `make_geo_apply_node`; MCP-write branch in `make_safe_apply_node` |
| `app/agents/ckan_registration/graph.py` | register + wire the two geo nodes |
| `app/agents/ckan_registration/state.py` | `transform_request`, `transform_execution_id`, `transforms_submitted` |
| `app/agents/ckan_registration/schemas.py` | optional transform fields on the run request |
| `app/tools/…` | a small node-side helper to run+poll a geo transform with token injection + scrub (reuse `GeoSyncExecutor` internals) |
| tests | geo-approval interrupt + cap + token-scrub; MCP-write apply branch (mocked); legacy fallback intact; routing |
| docs | README transform/approval flow |

## Risks and tradeoffs

- **R1 — re-pointing the proven CKAN write path.** Mitigation: additive (MCP branch opt-in,
  legacy default/fallback); same gate + dry-run precondition; mocked tests + legacy path untouched.
- **R2 — token into checkpointer/logs (security 2a/2b).** Mitigation: inject at the node, scrub
  from all state/results before persistence; never log args.
- **R3 — compute spend / runaway transforms (security 5).** Mitigation: per-session cap +
  in-flight check; approval required per transform; approval payload states the cost.
- **R4 — wrong-target writes via `register_to_dataset` (security 4a).** Mitigation: surfaced
  explicitly in the approval payload; node forwards only human-confirmed args (no augmentation
  beyond the token).
- **R5 — long Abaco runs blocking the node.** Mitigation: bounded poll; on timeout return the
  `execution_id` + a `transform-status` follow-up action rather than blocking indefinitely.

## Test plan

- Routing: `geo-transform` → `geo-approval`; approval interrupt fires without `REGISTER`.
- `geo-apply` (mocked geo client): injects token server-side, polls to terminal, returns
  `registered`; **token absent from returned state**; per-session cap refuses the N+1th; in-flight
  refusal.
- CKAN MCP-write apply branch (mocked MCP client): live write only after approval + dry-run;
  **legacy path used when `mcp_enabled` off** (regression: existing apply tests stay green).
- Approval payload includes `register_to_dataset` (+ source default) and clip bbox summary.

## Open questions / forks

1. **Fork A scope — CKAN re-point now or geo-only?** Recommend: build the geo transform node now
   **and** the additive CKAN MCP-write branch (legacy default). Alternative: geo-only this round,
   CKAN re-point later.
2. **Transform entry point** — a first-class `geo-transform` action via the API/intent (recommended,
   structured request), vs. letting the persona *propose* a transform that becomes the approval
   payload. Recommend the explicit action for v1 (deterministic, no model in the write decision).
3. **Long-run handling** — bounded block + `transform-status` follow-up action (recommended) vs.
   fire-and-return `execution_id` immediately.

## Decisions

- 2026-06-30: separate geo + CKAN nodes (architect O3); transforms/writes gated by `interrupt()`,
  token injected server-side + scrubbed, per-session cap, `register_to_dataset` surfaced.

## User feedback / decisions

- 2026-06-30: User approved building the gated-node increment after the geo foundation. _Awaiting
  approval of this spec (and the forks above) before implementation._
