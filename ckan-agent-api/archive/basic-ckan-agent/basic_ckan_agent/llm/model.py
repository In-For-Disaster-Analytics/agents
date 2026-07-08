from __future__ import annotations

from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from basic_ckan_agent.settings import DEFAULT_MODEL, env, env_float, env_int, required_env


@lru_cache
def build_rate_limiter() -> InMemoryRateLimiter:
    # Throttle outgoing LLM calls to stay under the provider's rate limit and
    # avoid bursts of 429 Too Many Requests. Tunable via .env.
    return InMemoryRateLimiter(
        requests_per_second=env_float("CKAN_LLM_REQUESTS_PER_SECOND", 1.0),
        check_every_n_seconds=env_float("CKAN_LLM_RATE_CHECK_SECONDS", 0.1),
        max_bucket_size=env_float("CKAN_LLM_MAX_BURST", 1.0),
    )


@lru_cache
def build_model(model_name: str | None = None) -> ChatOpenAI:
    # model_name overrides CKAN_LLM_MODEL for a single call site (used by the
    # evaluation harness to sweep models). Defaults preserve production behavior.
    resolved_model = model_name or env("CKAN_LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    return ChatOpenAI(
        model=resolved_model,
        api_key=required_env("OPENAI_API_KEY"),
        base_url=env("OPENAI_BASE_URL") or None,
        temperature=0,
        rate_limiter=build_rate_limiter(),
        max_retries=env_int("CKAN_LLM_MAX_RETRIES", 6),
        timeout=env_float("CKAN_LLM_TIMEOUT_SECONDS", 60.0),
    )

