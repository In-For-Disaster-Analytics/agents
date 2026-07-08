"""Experiment configuration: models, dataset name, and run metadata tags.

Every experiment is tagged with the prompt name/version, model, graph version
(git commit), OpenAPI spec source, and run date so results stay comparable over
time in LangSmith.
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime, timezone
from functools import lru_cache

from basic_ckan_agent.settings import DEFAULT_MODEL, ckan_openapi_url, env

# Name of the LangSmith dataset (created by dataset.py).
DATASET_NAME = env("CKAN_EVAL_DATASET", "ckan-agent-regression-suite")

# Models to sweep. Defaults to the deployed model; override with a comma-separated
# CKAN_EVAL_MODELS env var (e.g. "gpt-4.1,gpt-4.1-mini") when the endpoint serves
# multiple models. Add a model by appending its id here or via the env var.
_DEFAULT_MODELS = [env("CKAN_LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL]


def get_models() -> list[str]:
    raw = env("CKAN_EVAL_MODELS")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(_DEFAULT_MODELS)


# Model used by the LLM-as-judge and pairwise evaluators. Keep this fixed across
# experiments so judging is consistent; vary only the agent's model.
def judge_model() -> str:
    return env("CKAN_EVAL_JUDGE_MODEL") or env("CKAN_LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


@lru_cache
def graph_version() -> str:
    """Best-effort git commit of the current tree, for experiment tagging."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = out.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    return "unknown"


def spec_version() -> str:
    """Identifier for the OpenAPI tool schema in use."""
    return ckan_openapi_url()


def run_metadata(prompt_name: str, model_name: str) -> dict[str, str]:
    """Standard metadata block attached to every experiment run."""
    return {
        "prompt": prompt_name,
        "model": model_name,
        "graph_version": graph_version(),
        "spec_version": spec_version(),
        "run_date": date.today().isoformat(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
