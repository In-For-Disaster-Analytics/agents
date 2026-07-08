"""Deploy or restart a Tapis pod after a successful image build.

Uses tapipy for authentication (handles the v3 token response format
correctly), then calls the Pods API directly with requests so we are
not dependent on tapipy's auto-generated method names.

Environment variables set by the workflow:
  TAPIS_USERNAME, TAPIS_PASSWORD  — Tapis credentials (from secrets)
  POD_ID                          — Tapis pod identifier
  IMAGE                           — GHCR image name (without tag)
  OPENAI_API_KEY                  — forwarded to pod env
  CKAN_URL                        — forwarded to pod env
"""
import json
import os
import sys

import requests
from tapipy.tapis import Tapis

base_url   = "https://portals.tapis.io"
username   = os.environ["TAPIS_USERNAME"]
password   = os.environ["TAPIS_PASSWORD"]
pod_id     = os.environ["POD_ID"]
image      = os.environ["IMAGE"].lower() + ":latest"
openai_key = os.environ.get("OPENAI_API_KEY", "")
ckan_url   = os.environ.get("CKAN_URL", "")

print(f"Authenticating to {base_url} as {username}")
t = Tapis(base_url=base_url, username=username, password=password)
t.get_tokens()
token = t.access_token.access_token
print("Token obtained.")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# ── 1. Check whether the pod already exists ───────────────────────
resp = requests.get(f"{base_url}/v3/pods/{pod_id}", headers=headers, timeout=30)

if resp.status_code == 200:
    print(f"Pod {pod_id} exists — restarting to pick up new image")
    r = requests.post(f"{base_url}/v3/pods/{pod_id}/restart_pod", headers=headers, timeout=30)
    r.raise_for_status()
    print("Restart requested.")

elif resp.status_code == 404:
    print(f"Pod {pod_id} not found — creating")
    pod_body = {
        "pod_id": pod_id,
        "image": image,
        "description": "CKAN Agent API (FastAPI + LangGraph)",
        "environment_variables": {
            "CKAN_AGENT_API_HOST": "0.0.0.0",
            "CKAN_AGENT_API_PORT": "8787",
            "OPENAI_API_KEY": openai_key,
            "CKAN_URL": ckan_url,
        },
        "networking": {"protocol": "http", "port": 8787},
    }
    r = requests.post(f"{base_url}/v3/pods", headers=headers,
                      data=json.dumps(pod_body), timeout=30)
    if not r.ok:
        print(f"Pod creation failed (HTTP {r.status_code}): {r.text}", file=sys.stderr)
        sys.exit(1)
    print(f"Pod {pod_id} created.")

else:
    print(f"Unexpected status {resp.status_code} checking pod: {resp.text}", file=sys.stderr)
    sys.exit(1)
