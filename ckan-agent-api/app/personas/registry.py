"""User-extensible persona registry.

Personas are markdown files with YAML frontmatter, discovered from a directory
(``settings.personas_dir`` by default). A persona's markdown body becomes an LLM
system prompt, so this loader is a privilege/injection-sensitive surface — see the
hardening in ``_parse_file`` and ``load_all`` (spec R6):

- path-confinement: a discovered file must resolve to a real path *inside* the
  personas directory (guards against symlink / ``..`` escapes);
- ``yaml.safe_load`` only (never ``yaml.load``);
- loud validation: a malformed file, a missing required key, an unknown ``role``,
  or more than one enabled ``author`` raises at load time rather than silently
  degrading;
- approval-token guard: a persona body containing the literal CKAN approval token
  is flagged (logged) because it could socially-engineer a user into registering.

Frontmatter schema (required unless noted):

    ---
    name: data-curator
    description: FAIR-principles reviewer.
    role: evaluator                 # author | evaluator
    when_to_use: Run as a reviewer after the author drafts metadata.
    enabled: true                   # optional, default true
    ---
    <system-prompt body; may use {{schema_fields}} {{controlled_vocab}} {{defaults}} tokens>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.settings import get_settings

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"name", "description", "role", "when_to_use"}
VALID_ROLES = {"author", "evaluator"}
# Literal approval tokens that must never originate from a persona body.
APPROVAL_TOKENS = ("REGISTER", "DELETE_STALE_RESOURCES")


class PersonaError(RuntimeError):
    """Raised when a persona file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    role: str
    when_to_use: str
    enabled: bool
    body: str
    path: Path
    tools: tuple[str, ...] = ()  # optional allow-list of tool names this persona may call

    def render(self, **values: object) -> str:
        """Substitute ``{{token}}`` placeholders in the persona body."""
        rendered = self.body
        for key, value in values.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
        return rendered


def _split_frontmatter(text: str, path: Path) -> tuple[dict, str]:
    stripped = text.lstrip("﻿")
    if not stripped.startswith("---"):
        raise PersonaError(f"Persona file is missing a YAML frontmatter block: {path}")
    # Drop the leading '---' line, then split on the closing fence.
    after_open = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if "\n---" not in ("\n" + after_open):
        raise PersonaError(f"Persona frontmatter is not closed with '---': {path}")
    fm_text, _, body = after_open.partition("\n---")
    # body begins after the closing fence's own line.
    body = body.split("\n", 1)[1] if "\n" in body else ""
    try:
        meta = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise PersonaError(f"Persona frontmatter is not valid YAML ({path}): {exc}") from exc
    if not isinstance(meta, dict):
        raise PersonaError(f"Persona frontmatter must be a mapping: {path}")
    return meta, body.strip("\n")


class PersonaRegistry:
    def __init__(self, personas_dir: Path | None = None) -> None:
        settings = get_settings()
        self.personas_dir = (personas_dir or settings.personas_dir).resolve()

    def _confined(self, path: Path) -> Path:
        """Reject any path that resolves outside the personas directory."""
        resolved = path.resolve()
        if self.personas_dir != resolved and self.personas_dir not in resolved.parents:
            raise PersonaError(
                f"Persona file resolves outside the personas directory and is refused: {path}"
            )
        return resolved

    def _parse_file(self, path: Path) -> Persona:
        resolved = self._confined(path)
        meta, body = _split_frontmatter(resolved.read_text(encoding="utf-8"), resolved)

        missing = REQUIRED_KEYS - set(meta)
        if missing:
            raise PersonaError(
                f"Persona {path.name} is missing required frontmatter key(s): {sorted(missing)}"
            )
        role = str(meta["role"]).strip().lower()
        if role not in VALID_ROLES:
            raise PersonaError(
                f"Persona {path.name} has invalid role {meta['role']!r}; expected one of {sorted(VALID_ROLES)}"
            )
        if not body.strip():
            raise PersonaError(f"Persona {path.name} has an empty body (system prompt).")

        for token in APPROVAL_TOKENS:
            if token in body:
                logger.warning(
                    "Persona %s contains the approval token %r in its body; review for "
                    "prompt-injection before enabling.",
                    path.name,
                    token,
                )

        enabled = meta.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}

        tools_raw = meta.get("tools") or []
        if not isinstance(tools_raw, list):
            raise PersonaError(f"Persona {path.name} 'tools' must be a list of tool names.")

        return Persona(
            name=str(meta["name"]).strip(),
            description=str(meta["description"]).strip(),
            role=role,
            when_to_use=str(meta["when_to_use"]).strip(),
            enabled=bool(enabled),
            body=body,
            path=resolved,
            tools=tuple(str(t).strip() for t in tools_raw if str(t).strip()),
        )

    def load_all(self, *, include_disabled: bool = False) -> list[Persona]:
        """Discover and validate every persona file. Raises loudly on any bad file."""
        if not self.personas_dir.is_dir():
            raise PersonaError(f"Personas directory does not exist: {self.personas_dir}")

        personas = [self._parse_file(p) for p in sorted(self.personas_dir.glob("*.md"))]

        enabled = [p for p in personas if p.enabled]
        authors = [p for p in enabled if p.role == "author"]
        if len(authors) > 1:
            raise PersonaError(
                f"Exactly one enabled author persona is allowed; found {len(authors)}: "
                f"{[p.name for p in authors]}"
            )
        return personas if include_disabled else enabled

    def author(self) -> Persona:
        authors = [p for p in self.load_all() if p.role == "author"]
        if not authors:
            raise PersonaError("No enabled author persona found.")
        return authors[0]

    def evaluators(self) -> list[Persona]:
        return [p for p in self.load_all() if p.role == "evaluator"]
