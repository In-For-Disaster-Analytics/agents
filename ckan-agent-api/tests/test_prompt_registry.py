from pathlib import Path

from app.prompts import PromptRegistry


def test_prompt_registry_loads_markdown_prompt(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts" / "ckan_registration"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "analyze.md"
    prompt_path.write_text("Hello {{name}}", encoding="utf-8")

    template = PromptRegistry(tmp_path / "prompts").load("ckan_registration", "analyze")

    assert template.path == prompt_path
    assert template.render(name="CKAN") == "Hello CKAN"
