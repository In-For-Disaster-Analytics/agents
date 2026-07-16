# DSO Agent API — Component Reference for WebODM Integration

This document describes the internal architecture of `dso-agent-api` — the FastAPI + LangGraph service that mediates all CKAN publishing in the DSO stack — from the perspective of a WebODM plugin developer. It covers the graph structure, API surface, request/response schemas, workflow state machine, and the exact call sequence the WebODM CKAN plugin uses.

**Source:** `ckan-agent-api/` in this repo (`In-For-Disaster-Analytics/agents`)
**Design spec:** `odm-suite/WebODM/docs/design/2026-07-08-publish-to-ckan.md`

---

## 1. Service overview

`dso-agent-api` is a FastAPI application that wraps a LangGraph state machine. It orchestrates CKAN dataset registration end-to-end: it accepts a dataset source (file paths, URLs, pasted metadata, or pre-specified remote resource URLs), infers CKAN metadata via an LLM, validates against CKAN with a dry-run diff, and on explicit approval creates the package and registers all resources.

The service exposes two API styles:

- **Structured registration API** (`/v1/ckan-registration/…`) — the primary API used by the WebODM plugin. Stateful, thread-based.
- **OpenAI-compatible chat API** (`/v1/chat/completions`) — conversational interface for notebook / UI clients.

Both styles use the same underlying LangGraph runner.

**Deployed pod:** `https://dso-agent-api.pods.portals.tapis.io`
**Port (local):** `8787`

---

## 2. Authentication

All requests must carry a **Tapis JWT** in the `Authorization` header:

```
Authorization: Bearer <tapis_jwt>
```

**Obtaining a JWT via the agent** (recommended for WebODM, avoids calling Tapis directly):

```
POST /v1/auth/login
Content-Type: application/json

{"username": "<tapis_username>", "password": "<tapis_password>"}
```

Response: `{"access_token": "<jwt>", "expires_in": 21600}` (6-hour TTL)

**Direct Tapis token exchange** (alternative):

```
POST https://portals.tapis.io/v3/oauth2/tokens
Content-Type: application/json

{"grant_type": "password", "client_id": "webodm-localhost-dev",
 "username": "<user>", "password": "<pass>"}
```

**Access control:** Every request to the registration API runs `require_ckan_org_access` — a FastAPI dependency that calls `organization_list_for_user` on CKAN. If the caller is not a member of any CKAN organization, the agent rejects the request with HTTP 403. Network failures are treated as uncertain and allowed through.

Set `CKAN_AUTH_BYPASS=1` in `.env` to skip this check in local development. Never set it in production.

---

## 3. API endpoints

Base URL: `https://dso-agent-api.pods.portals.tapis.io`

All registration endpoints require `Authorization: Bearer <jwt>` and CKAN org membership.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/auth/login` | Exchange Tapis credentials for a JWT |
| `POST` | `/v1/ckan-registration/runs` | Start a new registration run |
| `POST` | `/v1/ckan-registration/runs/{thread_id}/resume` | Resume a paused run |
| `GET`  | `/v1/ckan-registration/runs/{thread_id}` | Read current run state |
| `POST` | `/v1/ckan-registration/tools/{tool_name}` | Invoke a single CKAN tool directly |
| `POST` | `/v1/chat/completions` | OpenAI-compatible conversational endpoint |
| `GET`  | `/v1/schemas` | List available CKAN metadata schemas |

### Tool names (for the direct tool endpoint)

| Tool | Action | Description |
|------|--------|-------------|
| `ckan_analyze` | `analyze` | Build proposed metadata and resource plan from source |
| `ckan_revise` | `revise` | Revise a saved proposal without writing to CKAN |
| `ckan_dry_run` | `dry-run` | Compare proposal against CKAN without writing |
| `ckan_apply` | `apply` | Create/update package and register resources (requires `approval: "REGISTER"`) |
| `ckan_show` | `show` | Return saved session state for debugging |

### Organization list (used by the WebODM org selector)

```
POST /v1/ckan-registration/tools/organization_list
Authorization: Bearer <jwt>
Content-Type: application/json

{"arguments": {}}
```

Returns the list of CKAN organizations the authenticated user can write to.

---

## 4. Request and response schemas

All schemas live in `app/agents/ckan_registration/schemas.py`.

### `CkanRunRequest` — start a new run

`CkanResumeRequest` is identical (an alias). Use the same body shape for both start and resume calls.

```python
class CkanRunRequest(BaseModel):
    action: str | None          # "analyze" | "revise" | "dry-run" | "apply" | "show"
                                # If omitted, the agent infers it. Always pass it explicitly.
    message: str | None         # Free-form user instruction
    session_id: str | None      # Stable thread ID for a resumable session

    # Dataset source — pick whichever applies
    upload_dir: str | None      # Server-local directory path
    files: list[FileReference | str] | None  # Server-local file paths
    source_url: str | None      # Primary metadata URL
    remote_resources: list[RemoteResource] | None  # Pre-specified URL resources ← WebODM uses this

    # Metadata seed — agent fills in what's missing
    dataset: CkanDatasetOverride | None  # Known metadata fields

    # Apply gate
    approval: str | None        # Must be exactly "REGISTER" to allow CKAN writes
```

### `RemoteResource` — URL-type CKAN resource

WebODM passes one `RemoteResource` per available output file:

```python
class RemoteResource(BaseModel):
    url: str          # Public download URL (WebODM asset endpoint)
    name: str | None  # Display name, e.g. "Orthophoto (GeoTIFF)"
    format: str | None  # CKAN format label, e.g. "GTiff", "LAZ", "GeoJSON"
    description: str | None
```

Resources are registered as URL links — no file download or upload occurs. This bypasses CKAN's 100 MB upload limit.

### `CkanDatasetOverride` — seed metadata fields

Pass known fields here; the agent fills in the rest via LLM analysis:

```python
class CkanDatasetOverride(BaseModel):
    title: str | None           # Human-readable title
    notes: str | None           # Dataset description
    owner_org: str | None       # CKAN org name or ID
    name: str | None            # CKAN slug (auto-generated if omitted)
    spatial: str | None         # WKT geometry (bounding box)
    temporal_coverage_start: str | None
    temporal_coverage_end: str | None
    tags: list[str] | None      # Tag names
    private: bool | None
    author: str | None
    author_email: str | None
    license_id: str | None      # e.g. "cc-by"
```

### `AgentRunResponse` — every endpoint returns this shape

```python
class AgentRunResponse(BaseModel):
    ok: bool
    thread_id: str              # Use this as session_id for subsequent resume calls
    command: str | None         # Last action executed
    status: str | None          # Workflow state (see section 6)
    result: dict                # Action-specific payload (see below)
    requires_action: dict | None  # Set when the graph is paused for human input
    error: str | None
```

**Key `result` fields by status:**

| Status | `result` fields of interest |
|--------|------------------------------|
| `metadata_report` | `review_markdown` — formatted metadata proposal |
| `dry_run` | `review_markdown` — CKAN diff |
| `applied` | `dataset_url` — the created CKAN package URL |
| `needs_input` | `message` — what the agent needs |
| `error` | `error` — error message |

---

## 5. LangGraph architecture

The agent is a `StateGraph` compiled from `CkanRegistrationState` with a SQLite checkpointer (falls back to `InMemorySaver`). State is persisted across API calls by `thread_id`.

### Graph nodes

```
START → intake → [route] → metadata / schema_select → persona → clarify(interrupt) → propose → END
                           ↘ dry-run → END
                           ↘ approval → apply → END
                           ↘ show → END
                           ↘ revise-field → END
                           ↘ geo-approval(interrupt) → geo-apply → END
```

| Node | Role |
|------|------|
| `intake` | Receives the request, normalizes the action, routes to the correct next node |
| `metadata` | Single-pass: analyzes files/URLs, calls LLM, writes a metadata proposal and resource plan |
| `schema_select` | (persona path) Picks the CKAN schema profile for this dataset type |
| `persona` | (persona path) Runs the author + evaluator loop; may emit clarification questions |
| `clarify` | (persona path) Emits a LangGraph `interrupt()` so the API caller can send answers back |
| `propose` | (persona path) Formats the final metadata proposal as review markdown |
| `dry-run` | Calls CKAN's read API to diff the proposal against any existing package |
| `approval` | Validates `approval == "REGISTER"` before allowing the write |
| `apply` | Creates/updates the CKAN package and registers all resources; sets `result.dataset_url` |
| `show` | Returns the current persisted state as review markdown |
| `revise-field` | LLM-targeted field edit on a specific metadata field |
| `geo-approval` | Emits `interrupt()` for human sign-off before a geo transform |
| `geo-apply` | Submits the approved geo transform to `dso-geo` MCP |

### State schema (`CkanRegistrationState`)

```python
class CkanRegistrationState(TypedDict, total=False):
    thread_id: str
    action: str
    request: dict       # Raw request payload
    result: dict        # Output from the last node
    status: str         # Current workflow state label
    error: str

    # Persona-chat state (only populated when CKAN_PERSONA_CHAT=true)
    schema_profile: str
    candidate_metadata: dict
    clarification_questions: list[dict]
    clarification_round: int
    org_metadata: dict          # Thread-sticky org-level answers
    dataset_clarifications: dict
    llm_locked_fields: dict     # Fields locked after first LLM derivation
    declined_fields: list[str]  # Fields user declined to answer

    # Geo transform state
    transform_request: dict
    transform_execution_id: str
    transforms_submitted: int

    # Targeted field edit
    revise_field_target: dict   # {"field": str, "instruction": str}
    show_target: dict           # {"field": str, "question": str}
```

### Checkpointing

State is persisted to a SQLite file (`/tmp/ckan-agent-api/checkpoints.sqlite` by default). `thread_id` is the LangGraph checkpoint key. Resume calls use `Command(resume=payload)` to continue from a `interrupt()` pause point. `get_state(config)` can check whether a thread is paused.

---

## 6. Workflow state machine

The `status` field in `AgentRunResponse` reflects the current position in the workflow:

| Status | Meaning | Next call |
|--------|---------|-----------|
| `metadata_report` | Agent has proposed metadata; waiting for user review | `resume` with corrections, or `resume` with `action: "apply"` |
| `needs_dataset_intent` | Agent needs to know: new dataset or update existing? | `resume` with `message: "new dataset"` or `"update <name>"` |
| `needs_existing_dataset_choice` | Multiple CKAN matches found; choose one | `resume` with `message: "update <exact-name>"` |
| `needs_dry_run` | Dry-run required before apply is allowed | `resume` with `action: "dry-run"` |
| `dry_run` | Dry-run complete; CKAN diff shown | `resume` with `action: "apply", approval: "REGISTER"` |
| `applied` | CKAN write complete; `result.dataset_url` is set | Done |
| `needs_input` | Agent needs more information | `resume` with the requested information |
| `error` | Something failed | Retry or inspect `result.error` |

**Action aliases recognized by the intake node:**

`validate`, `preview` → `dry-run`
`status`, `inspect` → `show`
`register` → `apply`

---

## 7. WebODM plugin call sequence

This is the exact sequence the WebODM CKAN plugin (`coreplugins/ckan/`) follows:

### Step 1: Start — analyze and propose metadata

```python
POST /v1/ckan-registration/runs
Authorization: Bearer <tapis_jwt>

{
  "action": "analyze",
  "message": "Analyze these WebODM outputs and propose CKAN dataset metadata.",
  "dataset": {
    "title": "<task.name>",
    "notes": "<processing summary>",
    "spatial": "<WKT bounding box from task.orthophoto_extent>"
  },
  "remote_resources": [
    {"url": "https://<wo-host>/api/projects/<p>/tasks/<t>/download/orthophoto.tif",
     "name": "Orthophoto (GeoTIFF)", "format": "GTiff"},
    {"url": "https://<wo-host>/api/projects/<p>/tasks/<t>/download/dsm.tif",
     "name": "Digital Surface Model (GeoTIFF)", "format": "GTiff"},
    {"url": "https://<wo-host>/api/projects/<p>/tasks/<t>/download/dtm.tif",
     "name": "Digital Terrain Model (GeoTIFF)", "format": "GTiff"},
    ...
  ]
}
```

Response: `{thread_id, status: "metadata_report", result: {review_markdown: "..."}}`

Save `thread_id` — it is required for all subsequent calls.

### Step 2: User corrections — resume with a message

```python
POST /v1/ckan-registration/runs/{thread_id}/resume
Authorization: Bearer <tapis_jwt>

{
  "message": "Change the title to 'Hurricane Maria Assessment — Sector 4'"
}
```

Response: `{thread_id, status: "metadata_report", result: {review_markdown: "..."}}`

Repeat for each user correction. The agent maintains the running metadata plan in its checkpointed state.

### Step 3: Confirm — apply with REGISTER approval

```python
POST /v1/ckan-registration/runs/{thread_id}/resume
Authorization: Bearer <tapis_jwt>

{
  "action": "apply",
  "approval": "REGISTER"
}
```

Response: `{thread_id, status: "applied", result: {dataset_url: "https://ckan.tacc.utexas.edu/dataset/..."}}`

Extract `result.dataset_url` and store it on the WebODM task (`task.ckan_url`).

### Handling interrupts (persona chat mode)

When `CKAN_PERSONA_CHAT=true`, the graph may pause mid-analyze to ask the user clarifying questions. The response will have `requires_action` set:

```json
{
  "ok": true,
  "thread_id": "abc-123",
  "requires_action": {
    "message": "What is the coordinate reference system for these outputs?",
    "thread_id": "abc-123"
  }
}
```

Resume with the user's answer:

```python
POST /v1/ckan-registration/runs/{thread_id}/resume

{"message": "EPSG:32614 — UTM Zone 14N"}
```

The `pending_interrupt(thread_id)` method on the runner checks `graph.get_state()` to detect a paused thread before routing.

---

## 8. APPLY approval gate

The `apply` node will refuse to write to CKAN unless `approval` is exactly the string `"REGISTER"`. This guard lives in `nodes.py`:

```python
APPLY_APPROVAL = "REGISTER"
```

The approval check is also surfaced in the schema:

```python
class CkanApplyInput(CkanRegistrationBaseInput):
    action: Literal["apply"] = "apply"
    approval: str | None  # Must be exactly "REGISTER"
    delete_approval: str | None  # Must be "DELETE_STALE_RESOURCES" to delete stale resources
```

**Result field:** On success, `result.dataset_url` contains the canonical CKAN dataset URL (confirmed from `nodes.py`, apply output). The WebODM Celery task extracts this and stores it as `task.ckan_url`.

---

## 9. Configuration reference

Environment variables loaded by `Settings.from_env()` in `app/settings.py`:

### Agent behaviour

| Variable | Default | Description |
|----------|---------|-------------|
| `CKAN_PERSONA_CHAT` | `false` | Enable persona-driven clarification loop (LangGraph interrupts) |
| `CKAN_ASK_SCHEMA` | `true` | Prompt user to select CKAN schema profile |
| `CKAN_DEFAULT_SCHEMA` | `generic_ckan` | Schema profile when not selected interactively |
| `CKAN_MAX_TOOL_CALLS` | `6` | Max LLM tool calls per analyze round |
| `CKAN_AUTH_BYPASS` | `false` | Skip org membership check (dev only) |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://ai.tejas.tacc.utexas.edu` | LLM API base (OpenAI-compatible) |
| `OPENAI_API_KEY` | — | LLM API key |
| `CKAN_LLM_MODEL` | `Meta-Llama-3.3-70B-Instruct` | Model name |
| `LLM_CALL_DELAY_SECONDS` | `5` | Throttle between LLM calls |

### CKAN connection

| Variable | Default | Description |
|----------|---------|-------------|
| `CKAN_URL` | `https://ckan.tacc.utexas.edu` | Target CKAN portal |
| `CKAN_AUTH_MODE` | `tapis_password` | Auth mode (`tapis_password` or `api_token`) |
| `CKAN_USERNAME` | — | Tapis username (service account) |
| `CKAN_PASSWORD` | — | Tapis password (service account) |
| `CKAN_TAPIS_URL` | `https://portals.tapis.io/v3/oauth2/tokens` | Tapis token endpoint |
| `CKAN_OWNER_ORG` | `DSO-Institute` | Default CKAN organization |

### MCP integration

| Variable | Default | Description |
|----------|---------|-------------|
| `CKAN_MCP_ENABLED` | `false` | Route CKAN tools through `dso-mcp` over HTTP |
| `CKAN_MCP_URL` | `http://localhost:8100/mcp` | `dso-mcp` endpoint |
| `CKAN_MCP_SHARED_SECRET` | — | Shared secret for `dso-mcp` |
| `GEO_MCP_ENABLED` | `false` | Enable `dso-geo` geo transform tools |
| `GEO_MCP_URL` | `http://localhost:8200/mcp` | `dso-geo` endpoint |

### State and storage

| Variable | Default | Description |
|----------|---------|-------------|
| `CKAN_AGENT_STATE_DIR` | `/tmp/ckan-agent-api/ckan-registration` | Legacy JSON state files |
| `CKAN_AGENT_CHECKPOINT_DB` | `/tmp/ckan-agent-api/checkpoints.sqlite` | LangGraph SQLite checkpointer |
| `CKAN_AGENT_UPLOAD_ROOT` | `/tmp/ckan-upload` | Uploaded file staging area |

---

## 10. Source file map

```
ckan-agent-api/
  app/
    main.py                        FastAPI application entry point
    settings.py                    Settings dataclass + env loader
    auth_context.py                Per-request Tapis JWT contextvar
    llm.py                         Shared LLM helper with throttle/retry
    api/
      routes_agent.py              Registration + chat endpoints (FastAPI router)
      routes_auth.py               /v1/auth/login
      security.py                  merge_secret_headers helper
    agents/ckan_registration/
      graph.py                     LangGraph StateGraph + CkanRegistrationRunner
      nodes.py                     All graph node factory functions; APPLY_APPROVAL constant
      persona_nodes.py             Persona-chat subgraph (schema_select→persona→clarify→propose)
      state.py                     CkanRegistrationState TypedDict
      schemas.py                   All Pydantic request/response models
      tools.py                     TOOL_SPECS registry (tool_name → action mapping)
      ckan_client.py               Thin CKAN REST client
      auth.py                      Tapis JWT exchange helpers
      org_grounding.py             Organization membership checks
      geo_transform.py             Geo transform submission/polling
    personas/
      engine.py                    Author + evaluator persona loop
      registry.py                  Persona YAML registry
    schemas/
      registry.py                  CKAN schema profile loader
    tools/
      executor.py                  Single-tool invocation (used by direct tool endpoint)
      mcp_client.py                HTTP MCP client for dso-mcp / dso-geo
      handlers/
        ckan.py                    CKAN read/write tool handlers
        gdal.py                    GDAL metadata tool handlers
    files/
      analyze.py                   File inventory builder
      extractors/                  Per-format metadata extractors
```
