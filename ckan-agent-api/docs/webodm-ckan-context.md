# DSO CKAN Publishing Infrastructure — Context for WebODM Integration

This document describes the two deployed services that form the DSO CKAN publishing stack and how they could be used to automate publishing WebODM outputs to CKAN.

---

## 1. What we built

Two services are deployed as Tapis pods on `portals.tapis.io`:

### dso-agent-api

A FastAPI + LangGraph conversational agent that orchestrates CKAN dataset registration end-to-end.

- **Pod:** `dso-agent-api` on `portals.tapis.io`
- **Image:** `ghcr.io/in-for-disaster-analytics/agents/dso-agent-api:latest`
- **Port:** 8787
- **Source:** [`In-For-Disaster-Analytics/agents`](https://github.com/In-For-Disaster-Analytics/agents), `ckan-agent-api/`

The agent accepts a dataset source (file path, URL, uploaded file, or pasted metadata), infers CKAN metadata via LLM, runs a dry-run validation against CKAN, and on approval creates the package and uploads all resources. It exposes both a structured REST API and an OpenAI-compatible `/v1/chat/completions` endpoint for conversational use.

### dso-mcp (CKAN MCP server)

A standalone Model Context Protocol (MCP) server that wraps the CKAN API as callable tools.

- **Pod:** `dso-mcp` on `portals.tapis.io`
- **Image:** `ghcr.io/in-for-disaster-analytics/mcp-suite/ckan-mcp:latest`
- **Port:** 8100
- **Source:** [`In-For-Disaster-Analytics/mcp-suite`](https://github.com/In-For-Disaster-Analytics/mcp-suite), `servers/ckan/`

The MCP server exposes tools including `schema_create_package`, `schema_upsert_package`, `schema_update_package`, and `resource_create`. The agent calls these tools over HTTP. Any other agent or script that speaks MCP over HTTP can call the same tools directly.

---

## 2. How they fit together

```
Client (browser / script / WebODM)
       |
       |  Authorization: Bearer <tapis_jwt>
       v
 dso-agent-api  (FastAPI + LangGraph)
       |
       |  tapis_token injected per tool call
       v
 dso-mcp  (CKAN MCP server)
       |
       |  X-Tapis-Token: <jwt>  →  CKAN API write
       v
 CKAN portal  (ckan.tacc.utexas.edu)
```

The agent never stores a long-lived credential. Every request carries the caller's Tapis JWT (6-hour TTL), which flows through the stack and is used for the actual CKAN write. This means the CKAN package is created as the authenticated user's account, under an organization they belong to.

---

## 3. Authentication

All authentication is Tapis OAuth2 JWT. There are no static CKAN API tokens.

**To obtain a token:**
```
POST https://portals.tapis.io/v3/oauth2/tokens
Content-Type: application/json

{
  "grant_type": "password",
  "client_id": "webodm-localhost-dev",
  "username": "<user>",
  "password": "<pass>"
}
```

**To call the agent:**
```
Authorization: Bearer <tapis_jwt>
```

The agent API also exposes `/v1/auth/login` which accepts a Tapis username/password and returns a JWT, so clients that cannot reach Tapis directly can exchange credentials through the agent.

**Access control:** The agent rejects callers who are not members of at least one CKAN organization (HTTP 403). This ensures only authorized data publishers can write to CKAN.

---

## 4. Agent API surface

Base URL: `https://dso-agent-api.pods.portals.tapis.io` *(once pod is spun up)*

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/auth/login` | Exchange Tapis credentials for a JWT |
| `POST` | `/v1/ckan-registration/runs` | Start a new dataset registration run |
| `POST` | `/v1/ckan-registration/runs/{thread_id}/resume` | Resume a paused run (e.g. after human review) |
| `GET`  | `/v1/ckan-registration/runs/{thread_id}` | Retrieve run state |
| `POST` | `/v1/ckan-registration/tools/{tool_name}` | Invoke a single CKAN tool directly |
| `POST` | `/v1/chat/completions` | OpenAI-compatible conversational endpoint |
| `GET`  | `/v1/schemas` | List available CKAN metadata schemas |

### Run request body

```json
{
  "message": "Register this WebODM output",
  "action": "apply",
  "source_url": "https://...",
  "upload_dir": "/path/to/outputs",
  "files": [{"path": "/outputs/odm_orthophoto.tif", "name": "Orthophoto"}],
  "metadata": {
    "title": "Site A Orthophoto — July 2026",
    "owner_org": "DSO-Institute",
    "dataset_type": "dataset"
  }
}
```

### Workflow states

```
analyze  →  metadata_report  →  (human review)  →  dry_run  →  (human review)  →  apply  →  applied
```

The dry-run step validates the proposed package and resource plan against CKAN without writing anything. The final `apply` step creates the package and uploads all resources. Both steps can be triggered programmatically (no human in the loop) by setting `action: "apply"` directly.

---

## 5. MCP server tools (direct access)

Base URL: `https://dso-mcp.pods.portals.tapis.io` *(once pod is spun up)*

The MCP server speaks the [MCP streamable HTTP transport](https://modelcontextprotocol.io). Tools can be called directly without going through the agent.

| Tool | Description |
|------|-------------|
| `schema_create_package` | Create a new CKAN package (validated against schema) |
| `schema_upsert_package` | Create or update a package by name |
| `schema_update_package` | Update fields on an existing package |
| `resource_create` | Upload a file or register a URL as a CKAN resource |
| `package_show` | Read an existing package |
| `organization_list` | List available CKAN organizations |

All write tools accept `tapis_token` as a call argument. The token is forwarded to CKAN as `X-Tapis-Token`.

---

## 6. WebODM outputs and CKAN mapping

WebODM processing tasks produce a set of standard output files. These map naturally to a single CKAN package with multiple resources:

| WebODM output | File | CKAN resource name |
|---|---|---|
| Orthophoto | `odm_orthophoto/odm_orthophoto.tif` | Orthophoto (GeoTIFF) |
| Digital Surface Model | `odm_dem/dsm.tif` | Digital Surface Model (GeoTIFF) |
| Digital Terrain Model | `odm_dem/dtm.tif` | Digital Terrain Model (GeoTIFF) |
| Point Cloud | `odm_pointcloud/cloud.laz` | 3D Point Cloud (LAZ) |
| 3D Mesh | `odm_texturing/odm_textured_model.obj` | Textured 3D Mesh (OBJ) |
| Contours | `odm_georeferencing/odm_georef_model.contours.geojson` | Contour Lines (GeoJSON) |
| Report | `odm_report/report.pdf` | Processing Report (PDF) |

CKAN metadata that can be auto-populated from WebODM task metadata:

| CKAN field | WebODM source |
|---|---|
| `title` | Task name |
| `notes` | Processing options summary |
| `spatial` | Computed bounding box from the orthophoto/point cloud |
| `temporal_coverage_start/end` | Image capture date range (from EXIF) |
| `coordinate_system` | Output CRS from ODM processing log |
| `owner_org` | Configured per deployment or per project |

---

## 7. CI/CD status

Both pods are deployed via GitHub Actions on every push to `main`:

| Repo | Workflow | Image | Status |
|---|---|---|---|
| `In-For-Disaster-Analytics/agents` | `build-agent-api.yml` | `ghcr.io/.../dso-agent-api:latest` | **Awaiting GHCR image whitelisting** |
| `In-For-Disaster-Analytics/mcp-suite` | `build-ckan-mcp.yml` | `ghcr.io/.../ckan-mcp:latest` | **Awaiting GHCR image whitelisting** |

**Blocker:** GHCR packages must be set to Public before Tapis pods can pull them without image-pull credentials. Once each package's visibility is changed to Public, the next push to `main` will automatically build and deploy the pod.

**GitHub Secrets still needed:**

| Repo | Secrets |
|---|---|
| `agents` | `TAPIS_USERNAME`, `TAPIS_PASSWORD`, `OPENAI_API_KEY`, `CKAN_PORTAL_URL` |
| `mcp-suite` | `TAPIS_USERNAME`, `TAPIS_PASSWORD`, `CKAN_BASE_URL`, `MCP_HTTP_SHARED_SECRET` |

---

## 8. What a WebODM → CKAN publish flow would look like

The simplest path uses the agent's structured API directly from WebODM's post-processing webhook or a sidecar script:

1. **WebODM task completes** — outputs written to a shared volume or object store
2. **Fetch a Tapis JWT** — `POST /v3/oauth2/tokens` with the service account credentials
3. **POST to `/v1/ckan-registration/runs`** — include `upload_dir` (or individual file paths) and any known metadata fields (task name, capture date, CRS, bounding box)
4. **Agent analyzes outputs** — infers metadata, maps files to resource types
5. **Skip dry-run, go straight to apply** — set `action: "apply"` for fully automated publishing
6. **Agent returns CKAN package URL** — store the URL back in the WebODM task as a custom attribute

For deployments that want human review before publishing, the dry-run → apply two-step workflow is already built in and can be surfaced in a simple UI.
