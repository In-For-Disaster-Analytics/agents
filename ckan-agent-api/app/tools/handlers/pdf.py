"""LLM-powered PDF summarization tool (map-reduce over the whole report)."""

from __future__ import annotations

from typing import Any

from app import llm
from app.files.pdf_summarize import map_reduce_pdf_summary
from app.files.safety import validate_readable_file
from app.settings import get_settings


def pdf_summarize(args: dict[str, Any]) -> dict[str, Any]:
    """Summarize a PDF section-by-section then combine — for reports longer than a few pages."""
    settings = get_settings()
    path = validate_readable_file(str(args["path"])).path

    def chat(system_prompt: str, user_text: str) -> str:
        return llm.invoke_chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            model=settings.ckan_llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "",
            temperature=0.1,
            max_tokens=500,
        )

    return map_reduce_pdf_summary(path, chat=chat)
