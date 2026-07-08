from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.settings import get_settings


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    path: Path

    def render(self, **values: object) -> str:
        rendered = self.text
        for key, value in values.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
        return rendered


class PromptRegistry:
    def __init__(self, prompt_dir: Path | None = None) -> None:
        settings = get_settings()
        self.prompt_dir = prompt_dir or settings.prompt_dir

    def load(self, agent_name: str, prompt_name: str) -> PromptTemplate:
        safe_agent = agent_name.replace("/", "_").replace("..", "_")
        safe_prompt = prompt_name.replace("/", "_").replace("..", "_")
        path = self.prompt_dir / safe_agent / f"{safe_prompt}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return PromptTemplate(name=f"{safe_agent}.{safe_prompt}", text=path.read_text(encoding="utf-8"), path=path)


def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
