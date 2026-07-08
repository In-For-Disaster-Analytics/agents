#!/usr/bin/env bash
#
# list_tapis_systems.sh — list Tapis storage systems visible in your tenant,
# showing host / systemType / rootDir / defaultAuthnMethod. Use this to find
# how Tapis is configured to reach Corral (or an existing system you can reuse)
# so you can set TAPIS_SYSTEM_HOST / auth correctly for setup_tapis_system.sh.
#
# curl + python3 only (no `tapis` CLI, no `jq`). Reads creds from ./.env.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then set -a; . "${REPO_ROOT}/.env"; set +a; fi

TACC_USER="${TAPIS_USERNAME:-${CKAN_USERNAME:-}}"
TACC_PASS="${TAPIS_PASSWORD:-${CKAN_PASSWORD:-}}"
TOKEN_URL="${CKAN_TAPIS_URL:-https://portals.tapis.io/v3/oauth2/tokens}"
TENANT_BASE="${TAPIS_FILES_BASE_URL:-$(python3 -c 'import sys;from urllib.parse import urlparse as u;x=u(sys.argv[1]);sys.stdout.write(f"{x.scheme}://{x.netloc}")' "$TOKEN_URL")}"
FILTER="${1:-}"   # optional: only show systems whose id/host/rootDir contains this substring

[ -n "$TACC_USER" ] && [ -n "$TACC_PASS" ] || { echo "Set CKAN_USERNAME/CKAN_PASSWORD in .env" >&2; exit 1; }

TOKEN_RESP="$(curl -sS -X POST "$TOKEN_URL" -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys;sys.stdout.write(json.dumps({"username":sys.argv[1],"password":sys.argv[2],"grant_type":"password"}))' "$TACC_USER" "$TACC_PASS")")"
JWT="$(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);sys.stdout.write(d.get("result",{}).get("access_token",{}).get("access_token",""))' "$TOKEN_RESP")"
[ -n "$JWT" ] || { echo "$TOKEN_RESP" >&2; echo "token failed" >&2; exit 1; }

SYS_RESP="$(curl -sS -H "X-Tapis-Token: ${JWT}" -H "Authorization: Bearer ${JWT}" \
  "${TENANT_BASE}/v3/systems?limit=500&select=id,host,systemType,rootDir,defaultAuthnMethod,effectiveUserId,owner")"

python3 -c '
import json, sys
data = json.loads(sys.argv[1]); flt = (sys.argv[2] or "").lower()
rows = data.get("result", [])
print(f"{\"id\":40} {\"systemType\":10} {\"authn\":10} {\"host\":36} rootDir")
print("-"*150)
n=0
for s in rows:
    blob = f"{s.get(\"id\",\"\")} {s.get(\"host\",\"\")} {s.get(\"rootDir\",\"\")}".lower()
    if flt and flt not in blob: continue
    n+=1
    print(f"{str(s.get(\"id\",\"\"))[:40]:40} {str(s.get(\"systemType\",\"\"))[:10]:10} {str(s.get(\"defaultAuthnMethod\",\"\"))[:10]:10} {str(s.get(\"host\",\"\"))[:36]:36} {s.get(\"rootDir\",\"\")}")
print(f"\n{n} system(s)" + (f" matching {sys.argv[2]!r}" if flt else "") + f" of {len(rows)} total.")
print("Look for one whose host reaches /corral-repl and note its host + authn method.")
' "$SYS_RESP" "$FILTER"
