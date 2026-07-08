# CKAN Agent API

FastAPI and LangGraph service for CKAN registration workflows.

This project is intentionally separate from `../ckan-registration`. The first
version wraps that existing worker for behavior parity, while the API, schemas,
prompts, and graph live here so additional registration agents can be added
without expanding the original notebook-oriented folder.

## What It Exposes

- `GET /health`
- `GET /openapi.json`
- `GET /schemas/openai-tools.json`
- `GET /schemas/chatgpt-actions.json`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/ckan-registration/runs`
- `POST /v1/ckan-registration/runs/{thread_id}/resume`
- `GET /v1/ckan-registration/runs/{thread_id}`
- `POST /v1/ckan-registration/tools/{tool_name}`
- `POST /v1/uploads`

## Conda Setup

Use conda to create the local development environment:

```bash
cd ckan-agent-api
conda env create -f environment.yml
conda activate ckan-agent-api
cp .env.sample .env
uvicorn app.main:app --reload --port 8787
```

Update an existing environment after dependency changes:

```bash
cd ckan-agent-api
conda env update -f environment.yml --prune
conda activate ckan-agent-api
```

Run tests:

```bash
cd ckan-agent-api
conda activate ckan-agent-api
pytest
```

## Container Build

Build from the repository root so the compatibility backend in
`ckan-registration/` is included:

```bash
docker build -f ckan-agent-api/Dockerfile -t ckan-agent-api .
docker run --rm -p 8787:8787 --env-file ckan-agent-api/.env ckan-agent-api
```

## CKAN And Tapis Auth

The default CKAN auth mode is `tapis_password`.

`CKAN_USERNAME` and `CKAN_PASSWORD` are pass-through credentials used to request
a Tapis access token from `CKAN_TAPIS_URL`. CKAN calls then use:

```text
Authorization: Bearer <tapis access token>
```

`CKAN_API_TOKEN` is still supported for API-token mode, but it is not the
preferred path for this service.

## MCP tool integration (dso_ckan_mcp)

When `CKAN_MCP_ENABLED=true`, the tool-calling personas get their CKAN tools from the standalone
[`dso-ckan`](../../modflow-suite/mcp-suite/servers/ckan) MCP server over HTTP instead of the
in-repo handlers. File-extractor tools (`file_*`, `pdf_*`) always stay in-process; a composite
executor routes by tool name.

| Variable | Default | Description |
|---|---|---|
| `CKAN_MCP_ENABLED` | `false` | Route CKAN tools through the MCP server |
| `CKAN_MCP_URL` | `http://localhost:8100/mcp` | HTTP endpoint of the running MCP server |
| `CKAN_MCP_TIMEOUT` | `30` | Per-call timeout (seconds) |
| `CKAN_MCP_SHARED_SECRET` | *(unset)* | Bearer secret sent to the MCP server (must match its `MCP_HTTP_SHARED_SECRET`) |
| `CKAN_MCP_TAPIS_TOKEN` | *(unset)* | Tapis write token, sent as the `X-Tapis-Token` **HTTP header** — never a tool argument |

Notes:

- **Graceful fallback (Fork B):** if MCP is disabled *or* the server is unreachable at request
  time, the agent falls back to the in-repo CKAN **read** tools — it never ends up with no CKAN
  tools. Start the MCP server before the agent for the MCP path.
- **Writes are gated (Fork A):** CKAN write tools are never advertised to the autonomous persona
  loop. The `MCPToolExecutor` hard-blocks any live write (`dry_run=False`) and scrubs the token
  arg; the write-tool schemas the model sees are dry-run-only. Live writes are issued solely from
  the human-approval graph gate. *(Re-pointing that gate's apply node from the legacy worker to
  the MCP write tools is a tracked follow-up increment — see the design spec.)*
- The Tapis token is short-lived; refresh `CKAN_MCP_TAPIS_TOKEN` when it expires.

See `docs/design/2026-06-29-integrate-dso-ckan-mcp.md` for the full design and review.

### Geo MCP server (dso-geo)

`GEO_MCP_ENABLED=true` adds the `dso-geo` server as a **second** MCP source (a flat
tool-name router sends CKAN names to the CKAN server, geo names to the geo server, and
everything else in-process). Geo runs GDAL on Tapis Abaco actors.

| Variable | Default | Description |
|---|---|---|
| `GEO_MCP_ENABLED` | `false` | Add the geo server as a tool source |
| `GEO_MCP_URL` | `http://localhost:8200/mcp` | HTTP endpoint of the running geo server |
| `GEO_MCP_SHARED_SECRET` | *(unset)* | Bearer secret (must match the geo server's `MCP_HTTP_SHARED_SECRET`) |
| `GEO_MCP_TAPIS_TOKEN` | *(unset)* | Tapis token injected **server-side** into geo tool args — never model-visible |
| `GEO_POLL_TIMEOUT` | `90` | Seconds the submit-poll wrapper waits for a metadata execution |
| `GEO_MAX_TRANSFORMS_PER_SESSION` | `5` | Cap for gated transform submissions per session |

Notes:

- **Personas get `gdalinfo_extract` only.** Because geo tools are async (submit →
  `execution_id` → poll), a synchronous wrapper submits and polls to completion, returning
  metadata as one tool result (or a `geo_not_ready` signal on timeout). `gdalinfo_summary`
  (multi-execution fan-out) and `get_execution_status` are not exposed to personas.
- **Transforms are gated.** `reproject_raster` / `convert_to_cog` / `clip_raster` /
  `build_overviews` spend Abaco compute and register new CKAN resources; they are hard-blocked
  from the persona loop (executor **and** schema layers). They run only via the gated graph path:
  a persona may *propose* a transform (it never executes one) → the `geo-approval` node interrupts
  and surfaces the operation, destination dataset, and clip extent → on `REGISTER` the `geo-apply`
  node injects the Tapis token server-side, submits to the Abaco actor, and polls to completion
  (or returns the `execution_id` for a later `transform-status` check). A per-session cap
  (`GEO_MAX_TRANSFORMS_PER_SESSION`) bounds compute spend. See
  `docs/design/2026-06-30-gated-write-transform-node.md`.
- **Token plumbing differs from CKAN:** geo takes the token as a tool *arg*, so the agent injects
  `GEO_MCP_TAPIS_TOKEN` server-side (never in model-visible args); metadata can also use the geo
  server's `GEO_TAPIS_TOKEN` env. CKAN uses an `X-Tapis-Token` header.

See `docs/design/2026-06-30-integrate-dso-geo-mcp.md` for the full design and review.

## ChatGPT Actions

Register this schema URL in a custom GPT Action:

```text
https://<host>/schemas/chatgpt-actions.json
```

The action should call only the `/v1/ckan-registration/*` endpoints. `apply`
requires exact approval text:

```text
REGISTER
```

Stale resource deletion additionally requires:

```text
DELETE_STALE_RESOURCES
```

## Chatbox / OpenAI-Compatible Clients

For clients that only understand OpenAI-compatible chat endpoints, point them at
the service base URL and use:

```text
GET /v1/models
POST /v1/chat/completions
```

Clients that already append `/models` and `/chat/completions` to the configured
base URL can either use `http://<host>/v1` as the base URL or call the root-level
compatibility aliases:

```text
GET /models
POST /chat/completions
```

Model id:

```text
ckan-registration-agent
```

The compatibility endpoint accepts both non-streaming requests and
OpenAI-style `stream: true` requests. Streaming responses are Server-Sent Events
that emit the completed agent response once the synchronous CKAN workflow
finishes.

Plain greetings or messages without CKAN workflow details return brief usage
guidance instead of creating a registration run. Analyze responses prefer the
human-readable review markdown when the worker provides it.

For follow-up requests such as dry-run, revise, status, or register, the chat
compatibility endpoint reuses a prior thread id from the conversation history
when the client does not send `metadata.session_id`.

## Audit Trace

The worker saves a structured audit trace in each registration state. This is
not raw model/private reasoning; it records observable inputs and decisions:
chat metadata received, file references supplied, resources discovered,
fallback versus LLM metadata selection, dataset overrides, and CKAN dry-run
comparison details.

To include the trace in an action response, send:

```json
{
  "action": "analyze",
  "message": "Analyze this dataset",
  "upload_dir": "/path/to/staged/files",
  "debug_trace": true
}
```

For chat-compatible clients, put the same flag in `metadata.debug_trace`.
Without that flag, inspect the saved trace with:

```text
GET /v1/ckan-registration/runs/{thread_id}
```

If a notebook or CSV was expected but omitted, the trace will show
`upload_dir_count: 0`, `file_ref_count: 0`, and `resource_count: 0`.

The trace is intended for debugging, but it is still an audit trace rather than
raw private model reasoning. It records observable inputs, branch decisions,
short rationale summaries, and safe next steps.

Metadata-only proposals are blocked when the request has no files, source URL,
existing CKAN entry, or explicit dataset fields unless the caller sets:

```json
{
  "allow_metadata_only": true
}
```

## Prompt Files

Prompts live under:

```text
app/prompts/ckan_registration/
```

They are plain Markdown files so the prompts can be tuned and tested without
editing graph or API code.
