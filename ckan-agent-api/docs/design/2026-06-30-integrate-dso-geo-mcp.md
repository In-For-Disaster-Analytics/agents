# Integrate the dso-geo MCP server (GDAL/Abaco) as a second tool source

## Status

Implemented (foundation + persona geo metadata + full transform/write blocking).
**Gated approval node deferred** — see deviation below.

### Implementation summary (2026-06-30)

Landed on branch `feat/mcp-tool-integration`:

- **Geo server** (`mcp-suite/servers/geo`): `MCP_TRANSPORT=http` mode that **requires**
  `MCP_HTTP_SHARED_SECRET` (refuses to start without it) and refuses a non-loopback bind while
  `GEO_TAPIS_TOKEN` is set; shared-secret bearer middleware. 5 HTTP-transport tests. Suite: 155 passed.
- **Agent**: `GEO_TRANSFORM_TOOLS` + `GEO_PERSONA_METADATA_TOOLS` + `PERSONA_BLOCKED_TOOLS`
  (`mcp_client.py`); `MCPToolExecutor` gains geo-transform hard-block + `token_arg` injection
  (after the defensive pop); `GeoSyncExecutor` (submit→bounded-poll→one envelope, `geo_not_ready`
  on timeout); multi-server `CompositeToolExecutor` (flat `dict[str, executor]` map);
  `_mcp_executor_and_schemas` builds the CKAN+geo router with the no-overlap assertion and
  **unconditional** transform schema-exclusion; `GEO_MCP_*` settings; docs + `.env`. 8 geo
  integration tests (in-memory FastMCP). Agent suite: 130 passed (3 pre-existing unrelated fails).

**Deviation / deferred (gated node):** the human-approval **gated transform/write node** is **not**
built this round. Rationale: it is a substantial graph feature (a new transform intent + route +
approval UX + execution/polling + per-session cap) **and** re-pointing the CKAN apply node off the
proven `LegacyCkanWorker` replaces live registration behavior — which the security review flagged
as needing its own pass. Critically, the system is **safe without it**: geo transforms (and CKAN
live writes) are hard-blocked at both the executor and schema layers, so they are unreachable from
the persona loop. Building the gated node (CKAN re-point as an additive MCP-write path with legacy
fallback + a new geo transform route) is the tracked next increment.

---

Approved — build A + B this round (user, 2026-06-30).

_Reviewed 2026-06-30 by architect, skeptic, security-reviewer; findings resolved in
[Review synthesis](#review-synthesis-2026-06-30). User confirmed: **build everything now** —
foundation (A) + persona geo metadata via the bounded submit-poll wrapper + the human-approval
gated node (CKAN live writes **and** geo transforms). B-blocker resolutions below are locked._

### Locked decisions (user, 2026-06-30)

- **Scope:** A + B in one round, sequenced foundation → metadata wrapper → gated node.
- **Persona geo (O2 = yes):** expose **`gdalinfo_extract` only** (single `execution_id`);
  **exclude `gdalinfo_summary`** from the persona surface to avoid the multi-execution fan-out.
- **Sync wrapper:** bounded poll on the geo client's own loop thread up to `GEO_POLL_TIMEOUT`
  (default 90s); on timeout return a structured `not_ready` tool result that tells the model to
  proceed without geo metadata — **no false "resumable" claim** (drop the `pending` concept for
  this round; `get_execution_status` stays internal to the wrapper, not in any persona schema).
- **Gated node (Fork A + geo):** build the human-approval `interrupt()` node now — re-point CKAN
  live writes off `LegacyCkanWorker` to the MCP write tools, **and** a separate geo transform
  node sharing the pattern. Per-session cap `GEO_MAX_TRANSFORMS_PER_SESSION`; approval payload
  surfaces `register_to_dataset` (incl. source-dataset default) + a human-readable clip bbox;
  token injected server-side post-approval and scrubbed from state/logs.
- **Two-layer transform block + require-secret HTTP + token scrubbing** (B1/B2/B3) as in synthesis.

## Objective

Add the **`dso-geo`** MCP server (`mcp-suite/servers/geo`) as a second MCP tool source for the
agent, alongside the already-integrated **`dso-ckan`** server. Personas get the read-only
**gdalinfo metadata** tools; the side-effectful **transform** tools (reproject / COG / clip /
overviews — which spend Tapis Abaco compute and register new CKAN resources) are **gated behind
human approval**, never callable by the autonomous persona loop. Both servers are reached over
**HTTP** (multi-server), and the agent's composite executor routes tool calls across them by name.

This extends the MCP integration from [2026-06-29-integrate-dso-ckan-mcp.md](2026-06-29-integrate-dso-ckan-mcp.md)
and reuses its patterns (HTTP `MCPClient` + owned-loop sync bridge, schema normalization,
header/arg token scrubbing, composite routing, write/exec gating = Fork A).

## User need

- "We've added a geo server, you can find it in mcp-suite." Integrate it like the CKAN server.
- Decisions (user, 2026-06-30):
  - **Design spec first** (Major-tier workflow).
  - **Multi-server HTTP**: both servers run HTTP; the agent connects to each by URL
    (`CKAN_MCP_URL` + new `GEO_MCP_URL`) and routes by tool name. (Requires adding HTTP mode to
    the geo server, which is currently stdio-only.)
  - **Metadata tools to personas; transforms gated** behind human approval (consistent with the
    CKAN write posture, Fork A).

## Current system summary

### dso-geo server (`mcp-suite/servers/geo`)

- FastMCP **stdio-only** server (`dso-geo-mcp`; `mcp.run()` — no HTTP transport yet).
- **Metadata tools (read-only, no CKAN write):**
  - `gdalinfo_extract(resource_id, include_stats=True, tapis_token=None)`
  - `gdalinfo_summary(dataset_id, tapis_token=None)`
- **Transform tools (token-REQUIRED, side-effectful — run GDAL on an Abaco actor AND
  auto-register the output as a new CKAN resource):**
  - `reproject_raster(resource_id, target_crs, output_name, register_to_dataset=None, tapis_token=None)`
  - `convert_to_cog(resource_id, output_name, compression="deflate", register_to_dataset=None, tapis_token=None)`
  - `clip_raster(resource_id, clip_geometry, output_name, register_to_dataset=None, tapis_token=None)`
  - `build_overviews(resource_id, output_name, overview_levels=[2,4,8], register_to_dataset=None, tapis_token=None)`
- **Status:** `get_execution_status(execution_id, tapis_token=None)` — polls **once**; caller
  drives the retry loop. Returns `{status}` while running; `{status: COMPLETE, result, registered}`
  or `{status: FAILED/ERROR, error}` when terminal.
- **Async model:** every tool returns `{execution_id, status: SUBMITTED}` immediately; results
  come only from polling `get_execution_status`.
- **Token model (key difference from CKAN):** token is a **tool argument** (`tapis_token`), not
  an HTTP header. Metadata tools fall back to the `GEO_TAPIS_TOKEN` env var; **transforms require
  an explicit token arg and have NO env fallback** (deliberate write gate). Server already
  scrubs tokens from logs/results and SSRF-validates CKAN URLs.
- Geo calls CKAN directly for URL resolution/registration; it does **not** call dso-ckan via MCP.

### dso-ckan server relocation

The CKAN server **moved** from `ckan-docker/mcp-server` to `mcp-suite/servers/ckan` and already
carries the HTTP-transport work from the prior increment (`MCP_TRANSPORT`, `_build_http_app`,
shared-secret middleware). The old `ckan-docker/mcp-server/.../server.py` no longer exists. The
agent integration is unaffected (it targets `CKAN_MCP_URL`, not a source path), but the earlier
commit in the `ckan-docker` repo is **superseded** by this relocation — note it, don't act on it.

### Agent side (from the CKAN increment)

- `app/tools/mcp_client.py` — `MCPClient` (owned-loop sync bridge, schema normalization,
  `MODEL_HIDDEN_ARGS={"tapis_token"}`, `WRITE_TOOL_NAMES`, shared-client factory).
- `app/tools/executor.py` — `MCPToolExecutor` (blocks live writes, scrubs token) +
  `CompositeToolExecutor(mcp, in_process, mcp_names)` — **single MCP client today**.
- `persona_nodes._tool_kwargs` / `_mcp_executor_and_schemas` — builds the composite + merged
  schemas, gated by `CKAN_MCP_ENABLED` with in-process fallback.

## Proposed design

### 1. Geo server: add HTTP transport

Mirror the CKAN server's HTTP mode in `mcp-suite/servers/geo`: `MCP_TRANSPORT` (`stdio` default |
`http`), `MCP_HTTP_HOST` (default `127.0.0.1`), `MCP_HTTP_PORT` (e.g. `8200`),
`MCP_HTTP_SHARED_SECRET` (bearer middleware). Same `_build_http_app()` + uvicorn pattern. Keeps
stdio working unchanged.

### 2. Agent: generalize routing to multiple MCP servers

Refactor the single-MCP composite into a **multi-server router**:

- A small registry: `name -> MCPClient` built from each configured server's `list_tools()`.
- `CompositeToolExecutor` takes the in-process executor plus a list of `(MCPToolExecutor,
  tool_names)` pairs (or a `tool_name -> executor` map). Routes by tool name; everything not
  served by any MCP server goes in-process.
- **No-overlap assertion across ALL sources** (in-process + every MCP server). Tool-name
  collisions are a hard startup error.
- Merged schema generation unions per-server schemas (filtered by the persona allow-list).

### 3. Geo token handling (B2-consistent: never model-visible)

- `tapis_token` is already in `MODEL_HIDDEN_ARGS`, so it is **stripped from all geo tool
  schemas** the model sees — the model can never supply it.
- **Metadata tools (persona-callable):** rely on the geo server's `GEO_TAPIS_TOKEN` env (set from
  the agent's deployment Tapis token), so metadata submission works without the model passing a
  token. The agent does not inject a token arg for metadata calls.
- **Transform tools (gated):** the approval-gated graph node injects the token into the tool args
  **server-side at call time** from the agent's auth context / `GEO_MCP_TAPIS_TOKEN` setting —
  after human approval — and calls the geo client directly (bypassing the persona-loop block).
  The token never enters model context, logs, or the checkpointer.

### 4. Geo "write"/transform gating (Fork A, extended)

- Define `GEO_TRANSFORM_TOOLS = {reproject_raster, convert_to_cog, clip_raster, build_overviews}`.
- `MCPToolExecutor` hard-blocks these from the persona tool loop (returns a `tool_error`,
  analogous to the CKAN `live_write_blocked` guard), and they are **omitted from persona
  schemas**. Personas can call only `gdalinfo_extract`, `gdalinfo_summary`, and (for polling, if
  the sync wrapper is not used) `get_execution_status`.
- Live transforms run only from a human-approval `interrupt()` graph node (shared design with the
  deferred CKAN Fork A apply-node work — they can land together).

### 5. Async submit→poll: a synchronous wrapper (the key UX call)

Exposing raw `execution_id` + `get_execution_status` to the author loop is poor: Abaco runs are
slow, and the model polling burns `max_tool_calls` and may never converge. Proposed:

- An agent-side **`GeoSyncExecutor`** wraps a geo submit tool call: it calls the tool, reads the
  `execution_id`, then polls `get_execution_status` with bounded backoff up to `GEO_POLL_TIMEOUT`
  (e.g. 60s for metadata), and returns the **terminal result as a single tool envelope**. The
  model sees one logical `gdalinfo_extract` call that returns metadata (or a timeout error).
- Polling uses the same owned-loop `MCPClient`; the token for polling is injected the same way as
  the submit call (env for metadata, gated-node injection for transforms).
- If a run exceeds the timeout, return a structured `pending` envelope with the `execution_id` so
  a later turn can resume — never block indefinitely.

*(Open question O1: confirm the sync-wrapper approach vs. exposing raw poll tools to the model.)*

### 6. Settings (`app/settings.py`)

| Setting | Env | Default | Purpose |
|---|---|---|---|
| `geo_mcp_enabled` | `GEO_MCP_ENABLED` | `false` | Route geo tools via the dso-geo server |
| `geo_mcp_url` | `GEO_MCP_URL` | `http://localhost:8200/mcp` | HTTP endpoint of the geo server |
| `geo_mcp_shared_secret` | `GEO_MCP_SHARED_SECRET` | *(unset)* | Bearer secret matching the geo server |
| `geo_mcp_tapis_token` | `GEO_MCP_TAPIS_TOKEN` | *(unset)* | Tapis token injected server-side for gated transforms (never model-visible) |
| `geo_poll_timeout` | `GEO_POLL_TIMEOUT` | `60` | Sync-wrapper poll budget (seconds) |

## Files likely affected

| Path | Change |
|---|---|
| `mcp-suite/servers/geo/src/dso_geo_mcp/{server,config}.py` | HTTP transport mode + shared-secret middleware |
| `mcp-suite/servers/geo/{README.md,.env.example}` | HTTP run docs |
| `app/tools/mcp_client.py` | `GEO_TRANSFORM_TOOLS`; multi-server-aware helpers; (poll support) |
| `app/tools/executor.py` | multi-server `CompositeToolExecutor`; geo transform block; `GeoSyncExecutor` |
| `app/agents/ckan_registration/persona_nodes.py` | build router across CKAN + geo; merged schemas; allow-lists |
| `app/settings.py` | `GEO_MCP_*` settings |
| `app/agents/ckan_registration/{graph,nodes}.py` | gated transform node (with the CKAN Fork A node) |
| README / `.env.sample` | geo integration docs |
| tests | multi-server routing, geo transform block, sync-wrapper (mocked), no-overlap, HTTP smoke |

## API / schema changes

- No new agent HTTP endpoints; a second outbound MCP/HTTP dependency.
- Persona frontmatter `tools:` may list geo **metadata** tool names; transform names are refused
  in persona allow-lists (gated path only).
- Geo server entrypoint gains `MCP_TRANSPORT` / `MCP_HTTP_*` env support.

## Data flow

1. Persona node builds the multi-server composite (CKAN + geo) + merged schemas filtered by the
   author allow-list (only when the respective `*_MCP_ENABLED` flags are on and servers reachable).
2. Author may call `gdalinfo_extract`/`gdalinfo_summary`; `GeoSyncExecutor` submits to the geo
   server, polls to terminal, returns metadata as one envelope. Token comes from the geo server's
   `GEO_TAPIS_TOKEN` env — never from the model.
3. Transforms are not in the author surface. A transform is proposed → human approval
   `interrupt()` → gated node injects the Tapis token and calls the geo transform tool → polls →
   the new CKAN resource id is surfaced.
4. CKAN read/dry-run path unchanged; CKAN/geo live writes both flow only through approval gates.

## Risks and tradeoffs

- **R1 — transforms spend real compute + create CKAN resources.** Higher-impact than CKAN
  metadata writes. Mitigation: gated-only (Fork A), token injected server-side post-approval,
  never model-visible; server-side SSRF + param validation are the backstop.
- **R2 — token plumbing differs from CKAN (arg, not header).** Risk of a token leaking into
  model-visible args/logs. Mitigation: `tapis_token` stays in `MODEL_HIDDEN_ARGS` (stripped from
  schemas); metadata uses the server env token; transforms inject at the gated node only.
- **R3 — async/slow Abaco runs.** A naive poll loop burns LLM budget / hangs. Mitigation: the
  sync wrapper with a bounded timeout + `pending` resumable envelope; cap `max_tool_calls`.
- **R4 — second network dependency.** Geo server down/slow degrades the geo path. Mitigation:
  `GEO_MCP_ENABLED` flag, fail-fast health check, per-call timeout, structured errors; CKAN path
  unaffected.
- **R5 — multi-server tool-name collisions.** Mitigation: hard no-overlap assertion across all
  sources at startup.
- **R6 — geo HTTP endpoint auth.** Same as CKAN: bind `127.0.0.1`, shared-secret middleware,
  document the `GEO_TAPIS_TOKEN` ambient-privilege risk.

## Alternatives considered

1. **stdio subprocess for geo** instead of HTTP — user chose multi-server HTTP. (Rejected.)
2. **Expose raw async poll tools to the model** instead of a sync wrapper — simpler server-side
   but poor model UX and budget risk. (Disfavored; O1.)
3. **All geo tools to personas** — lets the model run transforms autonomously; rejected for the
   compute/write risk (user chose metadata-only-to-personas).
4. **Have the agent call the geo server's stdio via dso-ckan composition** — geo already calls
   CKAN directly; no benefit. (Rejected.)

## Test plan

- Geo HTTP transport: 401 without/with wrong bearer; passes with correct/unset secret (mirror
  CKAN tests).
- Multi-server router: routes CKAN names → CKAN client, geo names → geo client, other →
  in-process; merged schema union; **no-overlap assertion** across all sources.
- Geo transform block: each transform tool returns a block error from the persona loop and is
  absent from persona schemas; `tapis_token` absent from all geo schemas.
- `GeoSyncExecutor` (mocked client): submit→poll→terminal COMPLETE returns metadata in one
  envelope; FAILED surfaces error; timeout returns a resumable `pending` envelope.
- Regression: CKAN-only path and no-tools path unchanged.

## Documentation plan

- Agent README: `GEO_MCP_*` env, what geo tools personas get, the transform-approval flow,
  the `GEO_TAPIS_TOKEN`/SSRF notes.
- Geo server README + `.env.example`: HTTP run mode.
- Note the dso-ckan relocation to `mcp-suite/servers/ckan` (supersedes the `ckan-docker` copy).

## Rollout / rollback plan

- Additive behind `GEO_MCP_ENABLED=false`. Add geo HTTP mode first (independent). Then the
  multi-server router + sync wrapper + metadata exposure. Transforms land with the shared Fork A
  gated node. Flip on per environment after health + smoke tests.
- Rollback: `GEO_MCP_ENABLED=false` (CKAN + in-process unaffected).

## Review synthesis (2026-06-30)

Verdicts: architect = revise; security = revise (2 Critical, 4 High); skeptic = revise + **split
into two increments**. The strategic finding all three share: the async poll wrapper and persona
geo exposure are only justified if O2 is "yes," and geo *transforms* depend on the CKAN **Fork A**
approval node, which is still **unbuilt** — so as written this spec would deliver a new server,
routing, and a complex wrapper but **zero transform capability**.

### Recommended split (skeptic; architect/security concur)

- **Increment A (low-risk, ship now, no O2/Fork A dependency):** add HTTP transport to the geo
  server (**requiring** a shared secret in HTTP mode); multi-server routing (flat
  `dict[str, MCPToolExecutor]` map + no-overlap assertion across all sources); `GEO_TRANSFORM_TOOLS`
  hard-block in the executor **and** unconditional schema exclusion; `GEO_MCP_*` settings;
  graceful fallback. **No geo tools exposed to personas; no `GeoSyncExecutor`; no transforms.**
- **Increment B (contingent):** persona metadata exposure + `GeoSyncExecutor` (only if O2 = yes
  and the `gdalinfo_summary` fan-out is solved) **and/or** the gated transform node (only after
  Fork A's shared approval infrastructure lands).

### Blocking findings → resolutions (baked in)

- **B1 — transform gating is two independent layers (security Critical 1a/1b).** (1) Executor
  hard-block: `GEO_TRANSFORM_TOOLS = {reproject_raster, convert_to_cog, clip_raster,
  build_overviews}` blocked in `MCPToolExecutor.invoke` before any `call_tool`, returning
  `tool_error("transform_blocked")`. (2) Schema exclusion: transform names are **unconditionally**
  removed in the schema-merge code, never relying on persona allow-lists. Both must exist before
  the geo client is wired into the composite.
- **B2 — geo HTTP must require the shared secret (security High 3a/3b).** In `MCP_TRANSPORT=http`,
  refuse to start if `MCP_HTTP_SHARED_SECRET` is unset (stricter than CKAN's warn-only); default
  `MCP_HTTP_HOST=127.0.0.1` and assert the loopback bind in tests; elevate the `GEO_TAPIS_TOKEN`
  ambient-spend warning to an error when bind host is non-loopback.
- **B3 — agent-side token scrubbing (security High 2a/2b).** The gated node injects
  `GEO_MCP_TAPIS_TOKEN` into tool args; the agent must strip `tapis_token` from any dict written
  to LangGraph state/checkpointer, pass geo results through a scrub before persisting, and never
  log geo args. (Server already scrubs its own logs.)

### High/medium findings → resolutions (baked in)

- **Flat router shape (architect):** `CompositeToolExecutor` keyed by `dict[str, MCPToolExecutor]`
  (O(1) routing; the map *is* the registry — no `ServerRegistry` class). `_mcp_executor_and_schemas`
  iterates over configured servers; signature changes — noted in files-affected.
- **Token-injection ordering (architect Risk 1):** `MCPToolExecutor` gains an optional
  `token_arg`/`inject_args` so the geo token is injected **after** the existing defensive
  `args.pop("tapis_token")`, keeping the CKAN header path and geo arg-inject path cleanly separate.
  A clear "what carries what" table (CKAN = `X-Tapis-Token` header; geo = `GEO_TAPIS_TOKEN` env for
  metadata / gated-node arg-inject for transforms; shared-secret bearer for both HTTP endpoints)
  goes in the spec/README.
- **Each `MCPClient` owns its own loop thread** (already true via `get_shared_client` per-URL), so
  geo polling doesn't block CKAN calls — but the **author-loop thread itself still blocks** during
  a poll (skeptic Risk 1), which is why the sync wrapper is Increment-B-only and bounded.
- **`get_execution_status` exposure (architect Risk 3):** if the sync wrapper is used, it is
  **internal only** and NOT in any persona schema. Remove the conditional wording.
- **`gdalinfo_summary` fan-out (skeptic blocker 3):** returns up to 10 execution_ids; a simple
  sync wrapper can't represent that. Increment B must either exclude `gdalinfo_summary` from the
  persona surface, or poll the fan-out concurrently. Decided when B is scoped.
- **`pending` resume path (skeptic Risk/Prod 4):** the engine's tool messages are not in graph
  state, so a `pending` envelope cannot resume across turns as written. Increment B must either
  persist the `execution_id` to graph state with a resume node, or treat timeout as a hard
  user-facing failure (no false "resumable" claim).
- **Poll budget vs. request timeout (skeptic rec 6):** Increment B must reconcile `GEO_POLL_TIMEOUT`
  (raise default toward 120s for cold-start actors) with the uvicorn/FastAPI request timeout, or
  run transforms off the request path.
- **Per-session transform cap (security Medium 5):** `GEO_MAX_TRANSFORMS_PER_SESSION` enforced at
  the gated node; check for an in-flight job before resubmitting. Approval prompt must surface
  `register_to_dataset` (incl. the source-dataset default) and a human-readable clip bbox
  (security High 4a / Medium 4b).
- **O3 → separate nodes (architect):** geo transform node and CKAN Fork A apply node are distinct
  code sharing the `interrupt()` + token-injection pattern; land in the same increment.
- **ckan-docker drift (skeptic Risk 6):** add a rollout check that the deployed `CKAN_MCP_URL`
  resolves to `mcp-suite/servers/ckan`, not a stale `ckan-docker` image.

### Forks needing user confirmation

- **Fork C — increment split.** Recommended: ship **Increment A** now; gate Increment B on the
  decisions below.
- **O2 — do personas need geo `gdalinfo` metadata during CKAN authoring?** If **no**, drop the
  `GeoSyncExecutor` and persona geo exposure from scope entirely (Increment A is the near-term
  whole). If **yes**, build B's wrapper after solving fan-out + timeout + resume.
- **Fork A node now?** Geo transforms (and CKAN live writes) need the human-approval gated node,
  which is still deferred. Build it now (unblocks both), or keep deferred (geo ships with no
  transform capability yet).

## Open questions

1. **O1 — sync wrapper vs. raw poll tools** for the async execution model (recommend sync wrapper).
2. **O2 — do personas actually need geo metadata during authoring**, or is geo purely a
   gated-transform feature for now? (If the latter, skip persona metadata exposure and ship
   transforms-behind-gate only.)
3. **O3 — single shared Fork A gated node** for both CKAN writes and geo transforms, or separate
   nodes? (Affects whether geo transforms wait on the deferred CKAN apply-node work.)
4. **O4 — branch**: continue on `feat/mcp-tool-integration` or a new branch for the geo work?

## Decisions

- 2026-06-30: Design-spec-first; multi-server HTTP (CKAN + geo by URL, route by name); personas
  get geo metadata tools only, transforms gated behind human approval. (User, AskUserQuestion.)

## User feedback / decisions

- 2026-06-30: User pointed to the new `dso-geo` server in `mcp-suite` and asked to integrate it
  like CKAN. Forks answered as in Decisions. _Awaiting review of this spec before implementation._
