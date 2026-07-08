"""User-extensible schema-profile registry.

A schema profile is a YAML file describing the target CKAN schema for a class of
datasets: the field list + guidance, controlled vocabularies, and hard defaults
applied at registration time. Each profile carries a ``when_to_use`` description so
the agent (and the user) can pick the right one. The default profile is configured
via ``settings.default_schema_profile`` (``subside``).

``app/schemas/subside.yaml`` is the canonical SUBSIDE schema (spec R5); the
ckanext-scheming deploy artifact in ``ckan-registration/schema/`` is derived from /
validated against it, not hand-maintained in parallel.

Security (spec R6): ``yaml.safe_load`` only, and discovered files are path-confined
to the schemas directory.

Profile YAML shape:

    name: subside
    description: ...
    when_to_use: ...
    dataset_type: subside_dataset
    defaults: {collection_method: Model Output, categories: [Groundwater]}
    controlled_vocab: {categories: [...], collection_method: [...]}
    fields:
      - {key: temporal_coverage_start, label: ..., required: false, guidance: "..."}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.settings import get_settings

REQUIRED_KEYS = {"name", "description", "when_to_use"}


class SchemaError(RuntimeError):
    """Raised when a schema profile is missing, malformed, or invalid."""


@dataclass(frozen=True)
class SchemaProfile:
    name: str
    description: str
    when_to_use: str
    dataset_type: str
    defaults: dict[str, Any] = field(default_factory=dict)
    controlled_vocab: dict[str, Any] = field(default_factory=dict)
    fields: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    def render_tokens(self) -> dict[str, str]:
        """JSON renderings for the ``{{schema_fields}}`` / ``{{controlled_vocab}}`` /
        ``{{defaults}}`` placeholders used in persona prompt bodies."""
        return {
            "schema_fields": json.dumps(self.fields, indent=2, ensure_ascii=False),
            "controlled_vocab": json.dumps(self.controlled_vocab, indent=2, ensure_ascii=False),
            "defaults": json.dumps(self.defaults, indent=2, ensure_ascii=False),
        }


class SchemaRegistry:
    def __init__(self, schemas_dir: Path | None = None) -> None:
        settings = get_settings()
        self.schemas_dir = (schemas_dir or settings.schemas_dir).resolve()
        self.default_profile = settings.default_schema_profile

    def _confined(self, path: Path) -> Path:
        resolved = path.resolve()
        if self.schemas_dir != resolved and self.schemas_dir not in resolved.parents:
            raise SchemaError(
                f"Schema file resolves outside the schemas directory and is refused: {path}"
            )
        return resolved

    def _parse_file(self, path: Path) -> SchemaProfile:
        resolved = self._confined(path)
        try:
            data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise SchemaError(f"Schema profile is not valid YAML ({path}): {exc}") from exc
        if not isinstance(data, dict):
            raise SchemaError(f"Schema profile must be a mapping: {path}")

        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise SchemaError(
                f"Schema profile {path.name} is missing required key(s): {sorted(missing)}"
            )
        return SchemaProfile(
            name=str(data["name"]).strip(),
            description=str(data["description"]).strip(),
            when_to_use=str(data["when_to_use"]).strip(),
            dataset_type=str(data.get("dataset_type") or "dataset").strip(),
            defaults=dict(data.get("defaults") or {}),
            controlled_vocab=dict(data.get("controlled_vocab") or {}),
            fields=list(data.get("fields") or []),
            path=resolved,
        )

    def load_all(self) -> list[SchemaProfile]:
        if not self.schemas_dir.is_dir():
            raise SchemaError(f"Schemas directory does not exist: {self.schemas_dir}")
        profiles = [self._parse_file(p) for p in sorted(self.schemas_dir.glob("*.yaml"))]
        names = [p.name for p in profiles]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise SchemaError(f"Duplicate schema profile name(s): {sorted(dupes)}")
        return profiles

    def get(self, name: str) -> SchemaProfile:
        for profile in self.load_all():
            if profile.name == name:
                return profile
        raise SchemaError(f"Schema profile not found: {name!r}")

    def default(self) -> SchemaProfile:
        return self.get(self.default_profile)

    def list_profiles(self) -> list[dict[str, str]]:
        """Lightweight listing (name + when_to_use) for selection UIs / the agent."""
        return [
            {"name": p.name, "description": p.description, "when_to_use": p.when_to_use}
            for p in self.load_all()
        ]
