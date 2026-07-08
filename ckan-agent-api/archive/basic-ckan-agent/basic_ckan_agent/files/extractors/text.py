from __future__ import annotations

from pathlib import Path
from typing import Any


def read_text_sample(path: Path, *, max_chars: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        head = handle.read(4096)
        if b"\x00" in head:
            return {
                "text": "",
                "encoding": None,
                "truncated": False,
                "binary_like": True,
                "message": "The file appears to contain binary data, so no text sample was returned.",
            }
        handle.seek(0)
        raw = handle.read(max_chars * 4 + 4096)

    text = raw.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    return {
        "text": text,
        "encoding": "utf-8",
        "truncated": truncated,
        "binary_like": False,
        "characters_returned": len(text),
    }
