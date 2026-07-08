from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import SchemaError, SchemaRegistry
from app.settings import PROJECT_ROOT

SEED_SCHEMAS_DIR = PROJECT_ROOT / "app" / "schemas"

VALID_PROFILE = """\
name: demo
description: Demo profile.
when_to_use: Use for tests.
dataset_type: dataset
defaults:
  collection_method: Model Output
controlled_vocab:
  categories: [Groundwater]
fields:
  - {key: title, required: true, guidance: "A title."}
"""


def _write(dir_: Path, name: str, text: str) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_profile_and_renders_tokens(tmp_path: Path) -> None:
    _write(tmp_path, "demo.yaml", VALID_PROFILE)
    profile = SchemaRegistry(tmp_path).get("demo")
    assert profile.dataset_type == "dataset"
    assert profile.defaults["collection_method"] == "Model Output"
    tokens = profile.render_tokens()
    assert set(tokens) == {"schema_fields", "controlled_vocab", "defaults"}
    assert json.loads(tokens["controlled_vocab"]) == {"categories": ["Groundwater"]}


def test_missing_required_key_raises(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "name: x\ndescription: y\n")  # no when_to_use
    with pytest.raises(SchemaError, match="missing required key"):
        SchemaRegistry(tmp_path).load_all()


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "name: x\n  : : bad")
    with pytest.raises(SchemaError):
        SchemaRegistry(tmp_path).load_all()


def test_unknown_profile_raises(tmp_path: Path) -> None:
    _write(tmp_path, "demo.yaml", VALID_PROFILE)
    with pytest.raises(SchemaError, match="not found"):
        SchemaRegistry(tmp_path).get("nope")


def test_duplicate_profile_name_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", VALID_PROFILE)
    _write(tmp_path, "b.yaml", VALID_PROFILE)
    with pytest.raises(SchemaError, match="Duplicate schema profile name"):
        SchemaRegistry(tmp_path).load_all()


def test_seed_subside_is_canonical_with_gam_defaults() -> None:
    registry = SchemaRegistry(SEED_SCHEMAS_DIR)
    subside = registry.get("subside")
    assert subside.dataset_type == "subside_dataset"
    assert subside.defaults["collection_method"] == "Model Output"
    assert "Groundwater" in subside.defaults["categories"]
    assert "Groundwater" in subside.controlled_vocab["categories"]
    names = {p["name"] for p in registry.list_profiles()}
    assert {"subside", "generic_ckan"} <= names
