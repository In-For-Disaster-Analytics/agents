#!/usr/bin/env bash
#
# setup_tapis_system.sh — create a Tapis LINUX storage system over Corral and
# verify it, so CKAN register-by-reference (REGISTER_BY_REFERENCE=true) can mint
# postit URLs per file instead of uploading bytes.
#
# Uses ONLY curl + python3 (no `tapis` CLI, no `jq`).
#
# It will:
#   1. Mint a Tapis JWT from your TACC username/password (password grant).
#   2. Create the storage system (idempotent — skips if it already exists).
#   3. Register the login credential for the system.
#   4. Smoke-test: list a folder + mint a test postit + download it back.
#
# Reads defaults from ./.env (CKAN_USERNAME, CKAN_PASSWORD, CKAN_TAPIS_URL,
# TAPIS_SYSTEM_ID, TAPIS_SYSTEM_ROOTDIR, TAPIS_FILES_BASE_URL). Override any
# value by exporting it before running, e.g.:
#
#   TAPIS_SYSTEM_HOST=cloud.corral.tacc.utexas.edu bash scripts/setup_tapis_system.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate repo root and load .env (without echoing secrets)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.env"
  set +a
fi

# ---------------------------------------------------------------------------
# Config (override via environment)
# ---------------------------------------------------------------------------
TACC_USER="${TAPIS_USERNAME:-${CKAN_USERNAME:-}}"
TACC_PASS="${TAPIS_PASSWORD:-${CKAN_PASSWORD:-}}"
SYSTEM_ID="${TAPIS_SYSTEM_ID:-ptdatax272-corral}"
ROOT_DIR="${TAPIS_SYSTEM_ROOTDIR:-/corral-repl/tacc/aci/PT2050/projects/PTDATAX-272/}"
# Host that Tapis uses to reach Corral. VERIFY this for your tenant/allocation.
HOST="${TAPIS_SYSTEM_HOST:-cloud.corral.tacc.utexas.edu}"
EFFECTIVE_USER="${TAPIS_EFFECTIVE_USER:-${TACC_USER}}"
# A path UNDER rootDir to use for the smoke test (folder to list / file to postit).
TEST_DIR="${TAPIS_TEST_DIR:-Yegua-Jackson_Aquifer_GAM}"
VALID_SECONDS="${POSTIT_VALID_SECONDS:-3153600000}"
ALLOWED_USES="${POSTIT_ALLOWED_USES:--1}"
TOKEN_URL="${CKAN_TAPIS_URL:-https://portals.tapis.io/v3/oauth2/tokens}"

fail() { echo "ERROR: $*" >&2; exit 1; }
command -v curl >/dev/null || fail "curl is required."
command -v python3 >/dev/null || fail "python3 is required."

# ---- JSON helpers (pass data as ARGV — never via stdin, to avoid heredoc clashes) ----
# jget <json-string> <dotted.path>  -> prints the value (or empty)
jget() {
  python3 -c '
import json, sys
try:
    cur = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)
for p in sys.argv[2].split("."):
    if isinstance(cur, dict) and p in cur:
        cur = cur[p]
    else:
        sys.exit(0)
sys.stdout.write("" if cur is None else str(cur))
' "$1" "$2"
}
# jbuild <key1> <val1> [<key2> <val2> ...] -> prints a JSON object
jbuild() {
  python3 -c '
import json, sys
a = sys.argv[1:]
sys.stdout.write(json.dumps({a[i]: a[i+1] for i in range(0, len(a), 2)}))
' "$@"
}

# Derive tenant base (scheme://host) from the token URL, unless overridden.
TENANT_BASE="${TAPIS_FILES_BASE_URL:-$(python3 -c '
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1]); sys.stdout.write(f"{u.scheme}://{u.netloc}")
' "$TOKEN_URL")}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
[ -n "$TACC_USER" ] || fail "TACC username not set (CKAN_USERNAME in .env or TAPIS_USERNAME)."
[ -n "$TACC_PASS" ] || fail "TACC password not set (CKAN_PASSWORD in .env or TAPIS_PASSWORD)."
[ -n "$HOST" ] || fail "TAPIS_SYSTEM_HOST not set — the host Tapis uses to reach Corral."

echo "=== Tapis system setup ==="
echo "  tenant base : ${TENANT_BASE}"
echo "  system id   : ${SYSTEM_ID}"
echo "  host        : ${HOST}"
echo "  rootDir     : ${ROOT_DIR}"
echo "  login user  : ${EFFECTIVE_USER}"
echo "  smoke test  : ${TEST_DIR}"
echo

# ---------------------------------------------------------------------------
# 1. Mint a Tapis JWT (password grant)
# ---------------------------------------------------------------------------
echo "[1/4] Requesting Tapis token ..."
TOKEN_BODY="$(jbuild username "$TACC_USER" password "$TACC_PASS" grant_type password)"
TOKEN_RESP="$(curl -sS -X POST "$TOKEN_URL" -H "Content-Type: application/json" -d "$TOKEN_BODY")"
JWT="$(jget "$TOKEN_RESP" "result.access_token.access_token")"
[ -n "$JWT" ] || { echo "$TOKEN_RESP" >&2; fail "Could not obtain Tapis token (check username/password/tenant)."; }
echo "      token OK (len ${#JWT})."
auth=(-H "X-Tapis-Token: ${JWT}" -H "Authorization: Bearer ${JWT}")

# ---------------------------------------------------------------------------
# 2. Create the system (skip if it already exists)
# ---------------------------------------------------------------------------
echo "[2/4] Creating/updating storage system '${SYSTEM_ID}' ..."
AUTHN_METHOD="${TAPIS_AUTHN_METHOD:-PASSWORD}"
# Build the system definition. include_id=yes for POST (create), no for PUT (update).
build_sys() {  # $1 = yes|no (include id)
  python3 -c '
import json, sys
inc, sid, host, euser, root, authn = sys.argv[1:7]
d = {
    "description": "Corral storage for SUBSIDE GAM files (register-by-reference)",
    "systemType": "LINUX",
    "host": host,
    "effectiveUserId": euser,
    "defaultAuthnMethod": authn,
    "rootDir": root,
    "canExec": False,
    "canRunBatch": False,
}
if inc == "yes":
    d = {"id": sid, **d}
sys.stdout.write(json.dumps(d))
' "$1" "$SYSTEM_ID" "$HOST" "$EFFECTIVE_USER" "$ROOT_DIR" "$AUTHN_METHOD"
}
EXISTING_CODE="$(curl -sS -o /dev/null -w "%{http_code}" "${auth[@]}" "${TENANT_BASE}/v3/systems/${SYSTEM_ID}")"
if [ "$EXISTING_CODE" = "200" ]; then
  echo "      system exists — updating host/rootDir/authn via PUT ..."
  SYS_RESP="$(curl -sS -w $'\n%{http_code}' -X PUT "${TENANT_BASE}/v3/systems/${SYSTEM_ID}" \
    "${auth[@]}" -H "Content-Type: application/json" -d "$(build_sys no)")"
else
  echo "      creating system ..."
  SYS_RESP="$(curl -sS -w $'\n%{http_code}' -X POST "${TENANT_BASE}/v3/systems" \
    "${auth[@]}" -H "Content-Type: application/json" -d "$(build_sys yes)")"
fi
SYS_CODE="$(printf '%s' "$SYS_RESP" | tail -n1)"
SYS_BODY="$(printf '%s' "$SYS_RESP" | sed '$d')"
if [ "$SYS_CODE" = "200" ] || [ "$SYS_CODE" = "201" ]; then
  echo "      system OK (host=${HOST}, authn=${AUTHN_METHOD})."
else
  echo "$SYS_BODY" >&2
  fail "system create/update failed (HTTP ${SYS_CODE})."
fi

# ---------------------------------------------------------------------------
# 3. Register login credential (PASSWORD authn) for the effective user
# ---------------------------------------------------------------------------
echo "[3/4] Registering credential for user '${EFFECTIVE_USER}' ..."
if [ "${SKIP_CRED_REGISTER:-}" = "true" ] || [ "${SKIP_CRED_REGISTER:-}" = "1" ]; then
  echo "      SKIP_CRED_REGISTER set — skipping credential registration entirely."
  echo "      (Step 4 will only work if a usable credential already exists on the system.)"
else
  CRED_BODY="$(jbuild password "$TACC_PASS")"
  CRED_URL="${TENANT_BASE}/v3/systems/credential/${SYSTEM_ID}/user/${EFFECTIVE_USER}"
  if [ "${SKIP_CRED_CHECK:-}" = "true" ] || [ "${SKIP_CRED_CHECK:-}" = "1" ]; then
    CRED_URL="${CRED_URL}?skipCredentialCheck=true"
    echo "      (skipCredentialCheck=true — registering WITHOUT the SSH validation)"
  fi
  CRED_RESP="$(curl -sS -w $'\n%{http_code}' -X POST \
    "$CRED_URL" \
    "${auth[@]}" -H "Content-Type: application/json" -d "$CRED_BODY")"
  CRED_CODE="$(printf '%s' "$CRED_RESP" | tail -n1)"
  if [ "$CRED_CODE" = "200" ] || [ "$CRED_CODE" = "201" ]; then
    echo "      credential registered."
  else
    printf '%s\n' "$CRED_RESP" | sed '$d' >&2
    fail "credential registration failed (HTTP ${CRED_CODE})."
  fi
fi

# ---------------------------------------------------------------------------
# 4. Smoke test: list folder + mint a test postit + download it
# ---------------------------------------------------------------------------
echo "[4/4] Smoke test ..."
LIST_RESP="$(curl -sS -w $'\n%{http_code}' "${auth[@]}" \
  "${TENANT_BASE}/v3/files/ops/${SYSTEM_ID}/${TEST_DIR}?limit=5")"
LIST_CODE="$(printf '%s' "$LIST_RESP" | tail -n1)"
if [ "$LIST_CODE" != "200" ]; then
  printf '%s\n' "$LIST_RESP" | sed '$d' >&2
  fail "files list failed (HTTP ${LIST_CODE}) — check rootDir/host/credential/Files perms."
fi
echo "      list OK."

# pick the first FILE under TEST_DIR (recurse) for the postit test
LIST_ALL="$(curl -sS "${auth[@]}" \
  "${TENANT_BASE}/v3/files/ops/${SYSTEM_ID}/${TEST_DIR}?recurse=true&limit=200")"
FIRST_FILE="$(python3 -c '
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)
for item in data.get("result", []):
    if (item.get("type") or "").lower() == "file":
        sys.stdout.write(item.get("path") or item.get("name") or "")
        break
' "$LIST_ALL")"

if [ -z "$FIRST_FILE" ]; then
  echo "      (no file found under ${TEST_DIR} to test a postit; system + creds verified)"
  echo
  echo "DONE. System '${SYSTEM_ID}' is set up. Set in .env: REGISTER_BY_REFERENCE=true"
  exit 0
fi
FIRST_FILE="${FIRST_FILE#/}"
echo "      test file: ${FIRST_FILE}"

POSTIT_RESP="$(curl -sS -w $'\n%{http_code}' -X POST \
  "${TENANT_BASE}/v3/files/postits/${SYSTEM_ID}/${FIRST_FILE}?allowedUses=${ALLOWED_USES}&validSeconds=${VALID_SECONDS}" \
  "${auth[@]}")"
POSTIT_CODE="$(printf '%s' "$POSTIT_RESP" | tail -n1)"
POSTIT_BODY="$(printf '%s' "$POSTIT_RESP" | sed '$d')"
if [ "$POSTIT_CODE" != "200" ] && [ "$POSTIT_CODE" != "201" ]; then
  echo "$POSTIT_BODY" >&2
  fail "postit create failed (HTTP ${POSTIT_CODE}) — token may lack Files/postit perms."
fi
REDEEM_URL="$(jget "$POSTIT_BODY" "result.redeemUrl")"
POSTIT_ID="$(jget "$POSTIT_BODY" "result.id")"
POSTIT_EXPIRES="$(jget "$POSTIT_BODY" "result.expiration")"
[ -n "$REDEEM_URL" ] || REDEEM_URL="${TENANT_BASE}/v3/files/postits/redeem/${POSTIT_ID}"
echo "      postit OK."
echo "        redeem URL : ${REDEEM_URL}"
echo "        expiration : ${POSTIT_EXPIRES:-<none reported>}   <-- if this is soon, the tenant caps TTL"

DL_CODE="$(curl -sS -L -o /dev/null -w "%{http_code}" "$REDEEM_URL")"
if [ "$DL_CODE" = "200" ]; then
  echo "      download via postit OK (HTTP 200)."
else
  echo "      WARNING: postit download returned HTTP ${DL_CODE} (link created but did not resolve)." >&2
fi

echo
echo "DONE. System '${SYSTEM_ID}' is ready."
echo "Next: in .env set ->  REGISTER_BY_REFERENCE=true"
echo "                      TAPIS_SYSTEM_ID=${SYSTEM_ID}"
echo "                      TAPIS_SYSTEM_ROOTDIR=${ROOT_DIR}"
echo "                      TAPIS_FILES_BASE_URL=${TENANT_BASE}"
echo "Then run discovery + Section 5.2 with SINGLE_MODEL_APPLY=True."
