# Tapis Storage System Setup for Register-by-Reference Mode

This document explains how to register a Tapis v3 LINUX storage system pointing
at the Corral GAM root directory, grant the required Files permissions, and
understand how postits are minted and redeemed by the register-by-reference
registration mode.

---

## Overview

Register-by-reference mode eliminates byte-uploads to CKAN.  Instead of
uploading each model file, the pipeline mints a **Tapis postit** per file — a
redeemable HTTP URL — and registers a CKAN `url`-type resource pointing at it.
Files remain on Corral storage.

Requirements:
- A Tapis v3 storage system registered in your tenant pointing at the Corral
  GAM root directory.
- Files API permission granted on the system to the user whose JWT will be used.
- `CKAN_AUTH_MODE=tapis_password` in your environment so `auth_header` is a
  Bearer JWT that also authenticates against Tapis Files.

---

## Step 1 — Register the Tapis Storage System

Use the Tapis Systems API (v3) or the Tapis CLI to create a LINUX storage system
that maps the Corral GAM root directory as the system `rootDir`.

Minimal required fields:

| Field | Description |
|---|---|
| `id` | Unique system ID in your tenant (e.g. `corral-gam`). |
| `systemType` | Must be `LINUX`. |
| `host` | Hostname or IP of the Corral storage node (ask your TACC admin). |
| `rootDir` | Absolute path on the host that becomes the Tapis file-system root.  All Tapis paths are relative to this. |
| `defaultAuthnMethod` | Authentication method — typically `PASSWORD` or `PKI_KEYS`. |
| `effectiveUserId` | The OS user on the host under which Tapis will read files. |

See `tapis-system-definition.template.json` for a JSON template with placeholder
values.  Replace all `<PLACEHOLDER>` values before submitting.

### CLI example

```bash
tapis systems create -F tapis-system-definition.template.json
```

Or via the REST API:

```bash
curl -X POST https://<TAPIS_TENANT>/v3/systems \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d @tapis-system-definition.template.json
```

---

## Step 2 — Grant Files Permissions

After creating the system, grant READ permission to the user (or service account)
that will mint postits:

```bash
# CLI
tapis systems perms grant -s corral-gam -u <USERNAME> READ

# REST
curl -X POST https://<TAPIS_TENANT>/v3/systems/perms/corral-gam \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"userName": "<USERNAME>", "permission": "READ"}'
```

Postit minting also requires Files API access.  Verify by listing files:

```bash
curl https://<TAPIS_TENANT>/v3/files/ops/corral-gam/ \
  -H "Authorization: Bearer <JWT>"
```

---

## Step 3 — Configure Environment Variables

Add to your `.env` (see `.env.sample` for all options):

```
CKAN_AUTH_MODE=tapis_password
CKAN_USERNAME=<your-tacc-username>
CKAN_PASSWORD=<your-tacc-password>

REGISTER_BY_REFERENCE=true
TAPIS_SYSTEM_ID=corral-gam
TAPIS_SYSTEM_ROOTDIR=<ABSOLUTE_PATH_ON_HOST_THAT_IS_rootDir>
TAPIS_FILES_BASE_URL=https://<TAPIS_TENANT>
```

`TAPIS_SYSTEM_ROOTDIR` is the absolute local path on the storage host that
corresponds to the system's `rootDir` field.  The pipeline strips this prefix
from each file's absolute path to compute the Tapis-relative path used when
minting postits.

---

## How Postits Are Minted and Redeemed

The pipeline calls:

```
POST {TAPIS_FILES_BASE_URL}/v3/files/postits/{TAPIS_SYSTEM_ID}/{tapis_path}
    ?allowedUses=-1&validSeconds=3153600000
Authorization: Bearer <jwt>
```

The response JSON contains either:
- `result.redeemUrl` — the ready-made redeem URL (preferred), or
- `result.id` — used to construct `{base}/v3/files/postits/redeem/{id}`.

The redeem URL is what gets stored in CKAN as the resource `url`.

---

## Postit Longevity and Refresh

The pipeline requests `validSeconds=3153600000` (~100 years) and
`allowedUses=-1` (unlimited redemptions) by default.  However, **Tapis tenants
may cap `validSeconds` to a lower value** (e.g. 90 days).  If the tenant cap
is lower than requested, postits will expire and CKAN resource links will break.

To refresh expired postits:

```python
from gam_registration.tapis_links import refresh_postit_urls

fresh_urls = refresh_postit_urls(
    [("corral-gam", "ygjk/Model_File/ygjk.nam"), ...],
    base_url="https://portals.tapis.io",
    jwt="<fresh-jwt>",
)
```

Then update the corresponding CKAN resources using `utils.create_link_resources`
with the refreshed URLs.

You can check your tenant's postit TTL cap by consulting the Tapis admin or by
inspecting the `result.expiry` field in a postit creation response.

---

## Tapis System Definition Template

See `tapis-system-definition.template.json` in the repo root for a ready-to-fill
JSON definition.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `RuntimeError: register_by_reference=True but TAPIS_SYSTEM_ID …` | `TAPIS_SYSTEM_ID` or `TAPIS_SYSTEM_ROOTDIR` env var not set. |
| `RuntimeError: register_by_reference=True requires a Bearer JWT auth_header` | `CKAN_AUTH_MODE` is not `tapis_password`. |
| `HTTP 401` when minting postits | JWT expired or wrong tenant.  Re-authenticate. |
| `ValueError: … is not under system_root_dir` | `TAPIS_SYSTEM_ROOTDIR` does not match the actual file paths.  Check that it is the absolute path on the storage host matching the system's `rootDir`. |
| Postit links expire quickly | Tenant has capped `validSeconds`.  Use `refresh_postit_urls` periodically. |
