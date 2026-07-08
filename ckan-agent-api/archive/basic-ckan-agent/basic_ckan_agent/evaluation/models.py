"""DeepEval custom-model wrapper around the agent's own LLM endpoint.

DeepEval never supplies a model: the LLM judges (GEval, faithfulness) run on
*your* configured endpoint (OPENAI_BASE_URL / CKAN_LLM_MODEL — e.g. the TACC
Llama deployment). This wraps the project's ``build_model`` so every judge call
reuses the same client, rate limiter, and retry policy as the agent.
"""

from __future__ import annotations

from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM

from basic_ckan_agent.evaluation.config import judge_model
from basic_ckan_agent.evaluation.extraction import _last_json_object
from basic_ckan_agent.llm.model import build_model


class LocalChatModel(DeepEvalBaseLLM):
    """Adapter exposing the agent's ChatOpenAI client to DeepEval metrics."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or judge_model()
        self._client = build_model(self.model_name)

    def load_model(self) -> Any:
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any) -> Any:
        text = str(self._client.invoke(prompt).content)
        if schema is None:
            return text
        return self._coerce_to_schema(text, schema)

    async def a_generate(self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any) -> Any:
        result = await self._client.ainvoke(prompt)
        text = str(result.content)
        if schema is None:
            return text
        return self._coerce_to_schema(text, schema)

    # GEval and other metrics request structured JSON output. Non-native models
    # answer in text, so parse the JSON and build the requested pydantic schema,
    # tolerating extra keys the model may add.
    @staticmethod
    def _coerce_to_schema(text: str, schema: Any) -> Any:
        data = _last_json_object(text)
        if not isinstance(data, dict):
            raise ValueError(f"Judge did not return JSON for schema {getattr(schema, '__name__', schema)}: {text[:200]}")
        fields = getattr(schema, "model_fields", None)
        if fields:
            data = {key: value for key, value in data.items() if key in fields}
        return schema(**data)

    # Conservative capability flags: we emulate structured output via prompting.
    def supports_json_mode(self) -> bool:  # pragma: no cover - simple flags
        return False

    def supports_structured_outputs(self) -> bool:  # pragma: no cover
        return False
