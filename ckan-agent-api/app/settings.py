from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def _clean(value: str | None) -> str:
    return str(value or "").strip()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    repo_root: Path = REPO_ROOT
    host: str = "0.0.0.0"
    port: int = 8787
    public_base_url: str = ""
    state_dir: Path = Path("/tmp/ckan-agent-api/ckan-registration")
    checkpoint_db: Path = Path("/tmp/ckan-agent-api/checkpoints.sqlite")
    upload_root: Path = Path("/tmp/ckan-upload")
    legacy_ckan_registration_dir: Path = REPO_ROOT / "ckan-registration"
    prompt_dir: Path = PROJECT_ROOT / "app" / "prompts"
    personas_dir: Path = PROJECT_ROOT / "app" / "personas"
    schemas_dir: Path = PROJECT_ROOT / "app" / "schemas"
    tools_dir: Path = PROJECT_ROOT / "app" / "tools" / "catalog"
    runs_dir: Path = Path("/tmp/ckan-agent-api/runs")
    default_schema_profile: str = "generic_ckan"
    ask_schema: bool = True
    persona_chat_enabled: bool = False
    persona_tools_enabled: bool = False
    max_tool_calls: int = 4
    # MCP integration (spec 2026-06-29): when enabled, CKAN tools are served by the standalone
    # dso_ckan_mcp server over HTTP; the in-repo CKAN read tools remain a disabled-by-default
    # fallback used when MCP is off or unreachable (Fork B). The Tapis write token is sent as an
    # HTTP header by the client layer — never as a model-visible tool argument (B2).
    mcp_enabled: bool = False
    mcp_server_url: str = "http://localhost:8100/mcp"
    mcp_timeout: float = 30.0
    mcp_shared_secret: str = ""
    mcp_tapis_token: str = ""
    # Geo MCP integration (dso-geo, spec 2026-06-30). Second MCP source. Personas get the
    # gdalinfo_extract metadata tool only (via a bounded submit-poll wrapper); transforms are
    # gated behind human approval. The Tapis token is injected server-side, never model-visible.
    geo_mcp_enabled: bool = False
    geo_mcp_url: str = "http://localhost:8200/mcp"
    geo_mcp_shared_secret: str = ""
    geo_mcp_tapis_token: str = ""
    geo_poll_timeout: float = 90.0
    geo_max_transforms_per_session: int = 5
    # Throttle real LLM round-trips so the persona loop (esp. the author tool loop)
    # stops hammering a rate-limited model group; retry 429s with backoff instead of
    # failing the whole drafting round. Applied centrally in app/llm.py.
    llm_call_delay_seconds: float = 5.0
    llm_max_retries: int = 4
    # Upload / zip-extraction safety caps.
    max_upload_bytes: int = 200 * 1024 * 1024
    max_zip_uncompressed_bytes: int = 500 * 1024 * 1024
    max_zip_members: int = 2000
    max_file_bytes: int = 50 * 1024 * 1024
    # When <= this many files are supplied, fully analyze them up front; above it, give the
    # author cheap head previews and let it deep-review the most informative ones via tools.
    deep_review_threshold: int = 3

    ckan_url: str = "https://ckan.tacc.utexas.edu"
    ckan_auth_mode: str = "tapis_password"
    ckan_api_token: str = ""
    ckan_username: str = ""
    ckan_password: str = ""
    ckan_tapis_url: str = "https://portals.tapis.io/v3/oauth2/tokens"
    ckan_owner_org: str = "DSO-Institute"
    ckan_dataset_author: str = "William Mobley"
    ckan_dataset_author_email: str = "wmobley@tacc.utexas.edu"
    ckan_dataset_maintainer: str = "William Mobley"
    ckan_dataset_maintainer_email: str = "wmobley@tacc.utexas.edu"
    ckan_dataset_license_id: str = "cc-by"
    # Deployment-wide defaults for the two fields the persona loop otherwise re-asks on every
    # dataset (contact email is thread-sticky but had no default; CRS is usually constant per
    # deployment). Empty means "ask". data_contact_email falls back to the author email below.
    ckan_data_contact_email: str = ""
    ckan_coordinate_system: str = ""
    ckan_dataset_type: str = "dataset"
    ckan_dataset_isopen: str = "true"
    ckan_dataset_version: str = ""
    ckan_dataset_spatial: str = ""
    ckan_temporal_coverage_start: str = ""
    ckan_temporal_coverage_end: str = ""

    # Set CKAN_AUTH_BYPASS=1 in .env to skip the per-request CKAN org check for local dev.
    # Never set this in production — it allows unauthenticated access to the agent API.
    auth_bypass: bool = False

    openai_base_url: str = "https://ai.tejas.tacc.utexas.edu"
    openai_api_key: str = ""
    ckan_llm_model: str = "Meta-Llama-3.3-70B-Instruct"

    @classmethod
    def from_env(cls) -> Settings:
        _load_env_file(PROJECT_ROOT / ".env")

        def path_env(name: str, default: Path) -> Path:
            raw = _clean(os.getenv(name))
            if not raw:
                return default
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()
            return path

        return cls(
            host=_clean(os.getenv("CKAN_AGENT_API_HOST")) or "0.0.0.0",
            port=int(_clean(os.getenv("CKAN_AGENT_API_PORT")) or "8787"),
            public_base_url=_clean(os.getenv("CKAN_AGENT_PUBLIC_BASE_URL")),
            state_dir=path_env("CKAN_AGENT_STATE_DIR", Path("/tmp/ckan-agent-api/ckan-registration")),
            checkpoint_db=path_env("CKAN_AGENT_CHECKPOINT_DB", Path("/tmp/ckan-agent-api/checkpoints.sqlite")),
            upload_root=path_env("CKAN_AGENT_UPLOAD_ROOT", Path("/tmp/ckan-upload")),
            legacy_ckan_registration_dir=path_env("CKAN_AGENT_LEGACY_DIR", REPO_ROOT / "ckan-registration"),
            personas_dir=path_env("CKAN_AGENT_PERSONAS_DIR", PROJECT_ROOT / "app" / "personas"),
            schemas_dir=path_env("CKAN_AGENT_SCHEMAS_DIR", PROJECT_ROOT / "app" / "schemas"),
            tools_dir=path_env("CKAN_AGENT_TOOLS_DIR", PROJECT_ROOT / "app" / "tools" / "catalog"),
            runs_dir=path_env("CKAN_AGENT_RUNS_DIR", Path("/tmp/ckan-agent-api/runs")),
            default_schema_profile=_clean(os.getenv("CKAN_DEFAULT_SCHEMA")) or "generic_ckan",
            ask_schema=_clean(os.getenv("CKAN_ASK_SCHEMA") or "true").lower() in {"1", "true", "yes", "on"},
            persona_chat_enabled=_clean(os.getenv("CKAN_PERSONA_CHAT")).lower() in {"1", "true", "yes", "on"},
            persona_tools_enabled=_clean(os.getenv("CKAN_PERSONA_TOOLS")).lower() in {"1", "true", "yes", "on"},
            max_tool_calls=int(_clean(os.getenv("CKAN_MAX_TOOL_CALLS")) or "4"),
            mcp_enabled=_clean(os.getenv("CKAN_MCP_ENABLED")).lower() in {"1", "true", "yes", "on"},
            mcp_server_url=_clean(os.getenv("CKAN_MCP_URL")) or "http://localhost:8100/mcp",
            mcp_timeout=float(_clean(os.getenv("CKAN_MCP_TIMEOUT")) or "30"),
            mcp_shared_secret=_clean(os.getenv("CKAN_MCP_SHARED_SECRET")),
            mcp_tapis_token=os.getenv("CKAN_MCP_TAPIS_TOKEN") or "",
            geo_mcp_enabled=_clean(os.getenv("GEO_MCP_ENABLED")).lower() in {"1", "true", "yes", "on"},
            geo_mcp_url=_clean(os.getenv("GEO_MCP_URL")) or "http://localhost:8200/mcp",
            geo_mcp_shared_secret=_clean(os.getenv("GEO_MCP_SHARED_SECRET")),
            geo_mcp_tapis_token=os.getenv("GEO_MCP_TAPIS_TOKEN") or "",
            geo_poll_timeout=float(_clean(os.getenv("GEO_POLL_TIMEOUT")) or "90"),
            geo_max_transforms_per_session=int(_clean(os.getenv("GEO_MAX_TRANSFORMS_PER_SESSION")) or "5"),
            llm_call_delay_seconds=float(_clean(os.getenv("LLM_CALL_DELAY_SECONDS")) or "5"),
            llm_max_retries=int(_clean(os.getenv("LLM_MAX_RETRIES")) or "4"),
            max_upload_bytes=int(_clean(os.getenv("CKAN_MAX_UPLOAD_BYTES")) or str(200 * 1024 * 1024)),
            max_zip_uncompressed_bytes=int(_clean(os.getenv("CKAN_MAX_ZIP_UNCOMPRESSED_BYTES")) or str(500 * 1024 * 1024)),
            max_zip_members=int(_clean(os.getenv("CKAN_MAX_ZIP_MEMBERS")) or "2000"),
            max_file_bytes=int(_clean(os.getenv("CKAN_AGENT_MAX_FILE_BYTES")) or str(50 * 1024 * 1024)),
            deep_review_threshold=int(_clean(os.getenv("CKAN_DEEP_REVIEW_THRESHOLD")) or "3"),
            ckan_url=_clean(os.getenv("CKAN_URL")) or "https://ckan.tacc.utexas.edu",
            ckan_auth_mode=_clean(os.getenv("CKAN_AUTH_MODE")) or "tapis_password",
            ckan_api_token=_clean(os.getenv("CKAN_API_TOKEN") or os.getenv("CKAN_API_KEY")),
            ckan_username=_clean(os.getenv("CKAN_USERNAME")),
            ckan_password=os.getenv("CKAN_PASSWORD") or "",
            ckan_tapis_url=_clean(os.getenv("CKAN_TAPIS_URL")) or "https://portals.tapis.io/v3/oauth2/tokens",
            ckan_owner_org=_clean(os.getenv("CKAN_OWNER_ORG") or os.getenv("CKAN_OWNER_ORG_ID")) or "DSO-Institute",
            ckan_dataset_author=_clean(os.getenv("CKAN_DATASET_AUTHOR")) or "William Mobley",
            ckan_dataset_author_email=_clean(os.getenv("CKAN_DATASET_AUTHOR_EMAIL")) or "wmobley@tacc.utexas.edu",
            ckan_dataset_maintainer=_clean(os.getenv("CKAN_DATASET_MAINTAINER")) or "William Mobley",
            ckan_dataset_maintainer_email=_clean(os.getenv("CKAN_DATASET_MAINTAINER_EMAIL")) or "wmobley@tacc.utexas.edu",
            ckan_dataset_license_id=_clean(os.getenv("CKAN_DATASET_LICENSE_ID")) or "cc-by",
            ckan_data_contact_email=_clean(os.getenv("CKAN_DATA_CONTACT_EMAIL")),
            ckan_coordinate_system=_clean(os.getenv("CKAN_COORDINATE_SYSTEM")),
            ckan_dataset_type=_clean(os.getenv("CKAN_DATASET_TYPE")) or "dataset",
            ckan_dataset_isopen=_clean(os.getenv("CKAN_DATASET_ISOPEN")) or "true",
            ckan_dataset_version=_clean(os.getenv("CKAN_DATASET_VERSION")),
            ckan_dataset_spatial=_clean(os.getenv("CKAN_DATASET_SPATIAL")),
            ckan_temporal_coverage_start=_clean(os.getenv("CKAN_TEMPORAL_COVERAGE_START")),
            ckan_temporal_coverage_end=_clean(os.getenv("CKAN_TEMPORAL_COVERAGE_END")),
            auth_bypass=_clean(os.getenv("CKAN_AUTH_BYPASS")).lower() in {"1", "true", "yes", "on"},
            openai_base_url=_clean(os.getenv("OPENAI_BASE_URL")) or "https://ai.tejas.tacc.utexas.edu",
            openai_api_key=os.getenv("OPENAI_API_KEY") or "",
            ckan_llm_model=_clean(os.getenv("CKAN_LLM_MODEL")) or "Meta-Llama-3.3-70B-Instruct",
        )

    def legacy_env(self) -> dict[str, str]:
        values = {
            "CKAN_URL": self.ckan_url,
            "CKAN_AUTH_MODE": self.ckan_auth_mode,
            "CKAN_API_TOKEN": self.ckan_api_token,
            "CKAN_USERNAME": self.ckan_username,
            "CKAN_PASSWORD": self.ckan_password,
            "CKAN_TAPIS_URL": self.ckan_tapis_url,
            "CKAN_OWNER_ORG": self.ckan_owner_org,
            "CKAN_DATASET_AUTHOR": self.ckan_dataset_author,
            "CKAN_DATASET_AUTHOR_EMAIL": self.ckan_dataset_author_email,
            "CKAN_DATASET_MAINTAINER": self.ckan_dataset_maintainer,
            "CKAN_DATASET_MAINTAINER_EMAIL": self.ckan_dataset_maintainer_email,
            "CKAN_DATASET_LICENSE_ID": self.ckan_dataset_license_id,
            "CKAN_DATA_CONTACT_EMAIL": self.ckan_data_contact_email,
            "CKAN_COORDINATE_SYSTEM": self.ckan_coordinate_system,
            "CKAN_DATASET_TYPE": self.ckan_dataset_type,
            "CKAN_DATASET_ISOPEN": self.ckan_dataset_isopen,
            "CKAN_DATASET_VERSION": self.ckan_dataset_version,
            "CKAN_DATASET_SPATIAL": self.ckan_dataset_spatial,
            "CKAN_TEMPORAL_COVERAGE_START": self.ckan_temporal_coverage_start,
            "CKAN_TEMPORAL_COVERAGE_END": self.ckan_temporal_coverage_end,
            "CKAN_AGENT_STATE_DIR": str(self.state_dir),
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_API_KEY": self.openai_api_key,
            "CKAN_LLM_MODEL": self.ckan_llm_model,
        }
        return {key: value for key, value in values.items() if value != ""}


@lru_cache
def get_settings() -> Settings:
    settings = Settings.from_env()
    # Activate the shared LLM throttle/retry once, at first settings load.
    from app import llm

    llm.configure_throttle(
        delay_seconds=settings.llm_call_delay_seconds, max_retries=settings.llm_max_retries
    )
    return settings
