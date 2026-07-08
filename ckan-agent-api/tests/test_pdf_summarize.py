from __future__ import annotations

from app.files.pdf_summarize import map_reduce_text_summary
from app.settings import PROJECT_ROOT
from app.tools import ToolRegistry


def _chat(system: str, user: str) -> str:
    # Distinguish the reduce step (combine) from the per-section map step.
    return "COMBINED SUMMARY" if system.startswith("Combine") else f"section[{user[:3]}]"


def test_map_reduce_combines_multiple_windows():
    text = "X" * 20_000  # 3 windows at 8000 chars
    out = map_reduce_text_summary(text, chat=_chat, window_chars=8000, max_windows=12)
    assert out["windows"] == 3
    assert out["summary"] == "COMBINED SUMMARY"
    assert len(out["section_summaries"]) == 3


def test_single_window_skips_reduce():
    out = map_reduce_text_summary("short text", chat=_chat, window_chars=8000)
    assert out["windows"] == 1
    assert out["summary"].startswith("section[")


def test_empty_text():
    out = map_reduce_text_summary("   ", chat=_chat)
    assert out["windows"] == 0
    assert out["summary"] == ""


def test_truncation_flag():
    text = "Y" * 40_000  # 5 windows, cap to 2
    out = map_reduce_text_summary(text, chat=_chat, window_chars=8000, max_windows=2)
    assert out["windows"] == 2
    assert out["truncated"] is True


def test_pdf_summarize_tool_registered():
    names = {s.name for s in ToolRegistry(PROJECT_ROOT / "app" / "tools" / "catalog").load_all()}
    assert "pdf_summarize" in names
