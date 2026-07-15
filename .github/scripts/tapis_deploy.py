"""Deploy or restart a Tapis pod after a successful image build.

Environment variables set by the workflow:
  TAPIS_USERNAME, TAPIS_PASSWORD  — Tapis credentials (from secrets)
  POD_ID                          — Tapis pod identifier
  IMAGE                           — GHCR image name (without tag)
  OPENAI_API_KEY                  — forwarded to pod env
  CKAN_URL                        — forwarded to pod env
"""
import os
import sys

from tapipy.tapis import Tapis

base_url   = "https://portals.tapis.io"
username   = os.environ["TAPIS_USERNAME"]
password   = os.environ["TAPIS_PASSWORD"]
pod_id     = os.environ["POD_ID"]
image      = os.environ["IMAGE"].lower() + ":latest"
openai_key = os.environ.get("OPENAI_API_KEY", "")
ckan_url   = os.environ.get("CKAN_URL", "")
ckan_mcp_url = os.environ.get("CKAN_MCP_URL", "https://dsomcp.pods.portals.tapis.io/mcp")
ckan_mcp_shared_secret = os.environ.get("CKAN_MCP_SHARED_SECRET", "")
langchain_api_key = os.environ.get("LANGCHAIN_API_KEY", "")
langchain_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "true")

print(f"Authenticating to {base_url} as {username}")
t = Tapis(base_url=base_url, username=username, password=password)
t.get_tokens()
print("Token obtained.")

# ── Check whether the pod already exists ──────────────────────────
pod_exists = False
try:
    t.pods.get_pod(pod_id=pod_id)
    pod_exists = True
except Exception as e:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 404:
        pod_exists = False
    else:
        print(f"Error checking pod: {e}", file=sys.stderr)
        sys.exit(1)

pod_env = {
    "CKAN_AGENT_API_HOST": "0.0.0.0",
    "CKAN_AGENT_API_PORT": "8787",
    "OPENAI_API_KEY": openai_key,
    "CKAN_URL": ckan_url,
    "CKAN_MCP_URL": ckan_mcp_url,
    "CKAN_MCP_SHARED_SECRET": ckan_mcp_shared_secret,
    "CKAN_MCP_ENABLED": "1",
    "CKAN_PERSONA_CHAT": "1",
    "CKAN_PERSONA_TOOLS": "1",
    "LANGCHAIN_TRACING_V2": langchain_tracing,
    "LANGCHAIN_API_KEY": langchain_api_key,
}

if pod_exists:
    print(f"Pod {pod_id} exists — updating env vars then restarting")
    t.pods.update_pod(pod_id=pod_id, environment_variables=pod_env)
    t.pods.restart_pod(pod_id=pod_id)
    print("Update and restart requested.")
else:
    print(f"Pod {pod_id} not found — creating")
    t.pods.create_pod(
        pod_id=pod_id,
        image=image,
        description="CKAN Agent API (FastAPI + LangGraph)",
        environment_variables=pod_env,
        networking={"default": {"protocol": "http", "port": 8787}},
    )
    print(f"Pod {pod_id} created.")
