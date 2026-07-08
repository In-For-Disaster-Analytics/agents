from __future__ import annotations

import json
from typing import Any


def parse_router_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return json.loads(text)


def safe_json_dumps(data: Any, *, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)

