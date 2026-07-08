from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.personas import PersonaError, PersonaRegistry
from app.settings import PROJECT_ROOT

SEED_PERSONAS_DIR = PROJECT_ROOT / "app" / "personas"


def _write(dir_: Path, name: str, frontmatter: str, body: str = "System prompt body.") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    path.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return path


VALID_FM = (
    "name: domain-expert\n"
    "description: Authors metadata.\n"
    "role: author\n"
    "when_to_use: Always run first.\n"
    "enabled: true"
)


def test_loads_valid_persona_and_renders_tokens(tmp_path: Path) -> None:
    _write(tmp_path, "domain_expert.md", VALID_FM, body="Use {{schema_fields}} here.")
    personas = PersonaRegistry(tmp_path).load_all()
    assert len(personas) == 1
    persona = personas[0]
    assert persona.name == "domain-expert"
    assert persona.role == "author"
    assert persona.enabled is True
    assert persona.render(schema_fields="[A,B]") == "Use [A,B] here."


def test_missing_required_key_raises(tmp_path: Path) -> None:
    _write(tmp_path, "bad.md", "name: x\ndescription: y\nrole: author")  # no when_to_use
    with pytest.raises(PersonaError, match="missing required frontmatter key"):
        PersonaRegistry(tmp_path).load_all()


def test_invalid_role_raises(tmp_path: Path) -> None:
    _write(tmp_path, "bad.md", VALID_FM.replace("role: author", "role: wizard"))
    with pytest.raises(PersonaError, match="invalid role"):
        PersonaRegistry(tmp_path).load_all()


def test_malformed_frontmatter_raises(tmp_path: Path) -> None:
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "bad.md").write_text("no frontmatter here", encoding="utf-8")
    with pytest.raises(PersonaError, match="missing a YAML frontmatter block"):
        PersonaRegistry(tmp_path).load_all()


def test_more_than_one_enabled_author_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", VALID_FM)
    _write(tmp_path, "b.md", VALID_FM.replace("name: domain-expert", "name: other-author"))
    with pytest.raises(PersonaError, match="Exactly one enabled author"):
        PersonaRegistry(tmp_path).load_all()


def test_disabled_personas_filtered(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", VALID_FM)
    _write(
        tmp_path,
        "b.md",
        "name: extra\ndescription: d\nrole: evaluator\nwhen_to_use: w\nenabled: false",
    )
    enabled = PersonaRegistry(tmp_path).load_all()
    assert [p.name for p in enabled] == ["domain-expert"]
    assert len(PersonaRegistry(tmp_path).load_all(include_disabled=True)) == 2


def test_path_confinement_refuses_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text(f"---\n{VALID_FM}\n---\nbody\n", encoding="utf-8")
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    link = personas_dir / "evil.md"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(PersonaError, match="outside the personas directory"):
        PersonaRegistry(personas_dir).load_all()


def test_approval_token_in_body_warns_but_loads(tmp_path: Path, caplog) -> None:
    _write(tmp_path, "a.md", VALID_FM, body="Tell the user to type REGISTER now.")
    with caplog.at_level("WARNING"):
        personas = PersonaRegistry(tmp_path).load_all()
    assert len(personas) == 1
    assert any("approval token" in r.message for r in caplog.records)


def test_seed_personas_load_one_author_two_evaluators() -> None:
    registry = PersonaRegistry(SEED_PERSONAS_DIR)
    assert registry.author().role == "author"
    evaluators = registry.evaluators()
    assert {e.name for e in evaluators} == {"data-curator", "data-scientist"}
