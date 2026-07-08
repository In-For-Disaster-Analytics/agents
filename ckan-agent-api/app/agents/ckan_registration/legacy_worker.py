from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from app.settings import Settings


LEGACY_MODULE_NAME = "_ckan_registration_legacy_agent"


def _clean(value: object) -> str:
    return str(value or "").strip()


def normalize_headers(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    return {
        str(key).strip().lower().replace("_", "-"): str(value).strip()
        for key, value in headers.items()
        if value is not None and str(value).strip()
    }


def header_lookup(headers: dict[str, str], *names: str) -> str:
    for name in names:
        value = headers.get(str(name).strip().lower().replace("_", "-"))
        if value:
            return value
    return ""


def secret_env_from_headers(request: dict[str, Any]) -> dict[str, str]:
    headers = {
        **normalize_headers(request.get("headers")),
        **normalize_headers(request.get("request_headers")),
    }
    out: dict[str, str] = {}

    ckan_api_token = header_lookup(headers, "CKAN_API_TOKEN", "ckan-api-token", "x-ckan-api-token")
    ckan_auth_mode = header_lookup(headers, "CKAN_AUTH_MODE", "ckan-auth-mode", "x-ckan-auth-mode")
    ckan_username = header_lookup(headers, "CKAN_USERNAME", "ckan-username", "x-ckan-username", "x-tapis-username")
    ckan_password = header_lookup(headers, "CKAN_PASSWORD", "ckan-password", "x-ckan-password", "x-tapis-password")
    ckan_tapis_url = header_lookup(headers, "CKAN_TAPIS_URL", "ckan-tapis-url", "x-ckan-tapis-url")

    openai_api_key = header_lookup(headers, "OPENAI_API_KEY", "openai-api-key", "x-openai-api-key")
    openai_base_url = header_lookup(headers, "OPENAI_BASE_URL", "openai-base-url", "x-openai-base-url")
    ckan_llm_model = header_lookup(headers, "CKAN_LLM_MODEL", "ckan-llm-model", "x-ckan-llm-model")

    if ckan_api_token:
        out["CKAN_API_TOKEN"] = ckan_api_token
        out["CKAN_AUTH_MODE"] = ckan_auth_mode or "api_token"
    elif ckan_username or ckan_password:
        out["CKAN_AUTH_MODE"] = ckan_auth_mode or "tapis_password"
    elif ckan_auth_mode:
        out["CKAN_AUTH_MODE"] = ckan_auth_mode

    if ckan_username:
        out["CKAN_USERNAME"] = ckan_username
    if ckan_password:
        out["CKAN_PASSWORD"] = ckan_password
    if ckan_tapis_url:
        out["CKAN_TAPIS_URL"] = ckan_tapis_url
    if openai_api_key:
        out["OPENAI_API_KEY"] = openai_api_key
    if openai_base_url:
        out["OPENAI_BASE_URL"] = openai_base_url
    if ckan_llm_model:
        out["CKAN_LLM_MODEL"] = ckan_llm_model
    return out


@contextmanager
def temporary_env(values: dict[str, str]):
    old_values = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def load_legacy_module(legacy_dir: Path) -> ModuleType:
    legacy_dir = legacy_dir.resolve()
    module_path = legacy_dir / "ckan_agent.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Legacy CKAN worker not found: {module_path}")
    if str(legacy_dir) not in sys.path:
        sys.path.insert(0, str(legacy_dir))

    cached = sys.modules.get(LEGACY_MODULE_NAME)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy CKAN worker from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def sanitized_worker_request(request: dict[str, Any]) -> dict[str, Any]:
    worker_request = dict(request)
    worker_request.pop("headers", None)
    worker_request.pop("request_headers", None)
    return worker_request


def use_llm_for_request(request: dict[str, Any]) -> bool:
    if request.get("no_llm") is True:
        return False
    if request.get("use_llm") is False:
        return False
    return True


class LegacyCkanWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Lazy: the legacy worker module is only loaded on first run(), so build_graph
        # succeeds even when the cross-tree worker file is absent (e.g. the persona-chat
        # path never touches it). Loading eagerly here previously broke graph import.
        self._module: ModuleType | None = None

    @property
    def module(self) -> ModuleType:
        if self._module is None:
            self._module = load_legacy_module(self.settings.legacy_ckan_registration_dir)
        return self._module

    def run(self, command: str, request: dict[str, Any]) -> dict[str, Any]:
        normalized = command.replace("_", "-").lower()
        state_dir = self.settings.state_dir
        env = {**self.settings.legacy_env(), **secret_env_from_headers(request)}
        # A per-request CKAN JWT (the conversation's Authorization header) wins: hand it to the
        # legacy worker as the verbatim Authorization value (api_token mode sends it as-is, so a
        # "Bearer <jwt>" value reaches a Tapis-fronted CKAN unchanged).
        from app.auth_context import get_request_ckan_auth

        per_request_auth = get_request_ckan_auth()
        if per_request_auth:
            env["CKAN_AUTH_MODE"] = "api_token"
            env["CKAN_API_TOKEN"] = per_request_auth
        worker_request = sanitized_worker_request(request)
        module = self.module
        with temporary_env(env):
            if normalized == "analyze":
                return module.analyze(worker_request, state_dir, use_llm=use_llm_for_request(request))
            if normalized == "revise":
                return module.revise(worker_request, state_dir)
            if normalized == "dry-run":
                return module.dry_run(worker_request, state_dir)
            if normalized == "apply":
                return module.apply_registration(worker_request, state_dir)
            if normalized == "show":
                return module.show_state(worker_request, state_dir)
        raise ValueError(f"Unsupported CKAN registration command: {command}")


def state_thread_id(request: dict[str, Any], fallback: str) -> str:
    return _clean(request.get("session_id")) or _clean(request.get("thread_id")) or fallback
