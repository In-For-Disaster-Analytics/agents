from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Meta-Llama-3.3-70B-Instruct"

load_dotenv(PROJECT_ROOT / ".env")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def required_env(name: str) -> str:
    value = env(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.sample to .env and fill it in.")
    return value


def env_float(name: str, default: float) -> float:
    value = env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    value = env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@lru_cache
def ckan_base_url() -> str:
    return env("CKAN_BASE_URL", "https://ckan.tacc.utexas.edu").rstrip("/")


@lru_cache
def ckan_openapi_url() -> str:
    return env("CKAN_OPENAPI_URL", "http://localhost:5001/api-specs/ckan-openapi.json")


@lru_cache
def ckan_api_token() -> str:
    return env("CKAN_API_TOKEN")

