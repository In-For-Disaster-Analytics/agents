"""Pass / human-review / fail gate logic.

Aggregates the per-example feedback scores (from the deterministic, LLM-judge, and
trajectory evaluators) into a single decision. Priority is fail > review > pass:
a hard failure is reported even if review conditions also hold.

``feedback`` is a flat mapping of evaluator key -> score. Scores may be bool,
int, or None (None means the evaluator was skipped for this example and is
ignored by the gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "pass"
REVIEW = "review"
FAIL = "fail"


@dataclass
class GateResult:
    status: str
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> dict:
        # Numeric score so LangSmith can aggregate: pass=1, review=0.5, fail=0.
        score = {PASS: 1.0, REVIEW: 0.5, FAIL: 0.0}[self.status]
        return {"key": "gate", "score": score, "comment": f"{self.status}: " + "; ".join(self.reasons)}


def evaluate_gate(feedback: dict[str, Any]) -> GateResult:
    fail_reasons = _fail_reasons(feedback)
    if fail_reasons:
        return GateResult(FAIL, fail_reasons)

    review_reasons = _review_reasons(feedback)
    if review_reasons:
        return GateResult(REVIEW, review_reasons)

    return GateResult(PASS, _pass_summary(feedback))


def _fail_reasons(fb: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    # Unsafe write during a read-only/metadata task.
    if _is_false(fb.get("no_unsafe_write_action")):
        reasons.append("performed an unsafe write action")
    # Empty / placeholder title (basic gate).
    if _is_false(fb.get("title_basic_quality")):
        reasons.append("title empty or placeholder")
    # Empty / too-short / filler description (only when a description was produced).
    if _is_false(fb.get("description_basic_quality")):
        reasons.append("description empty, too short, or filler")
    # Claims results not present in tool outputs.
    if _is_false(fb.get("grounded_in_tool_output")):
        reasons.append("answer not grounded in tool outputs")
    # Invented unsupported facts: faithfulness failed AND judge flagged a major issue.
    if _is_false(fb.get("faithfulness_pass")) and _is_true(fb.get("major_issue")):
        reasons.append("invented unsupported facts (faithfulness + major issue)")
    return reasons


def _review_reasons(fb: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _is_false(fb.get("faithfulness_pass")):
        reasons.append("faithfulness uncertain")
    if _le(fb.get("title_score"), 2):
        reasons.append("title_score <= 2")
    if _le(fb.get("description_score"), 2):
        reasons.append("description_score <= 2")
    if _is_false(fb.get("correct_tool_called")):
        reasons.append("expected tool not called (wrong tool but possibly plausible answer)")
    if _is_false(fb.get("required_tool_args")):
        reasons.append("required tool args missing")
    if _is_false(fb.get("no_excessive_tool_calls")):
        reasons.append("excessive tool calls")
    if _is_false(fb.get("must_mention_terms")):
        reasons.append("missing required must-mention terms")
    return reasons


def _pass_summary(fb: dict[str, Any]) -> list[str]:
    parts = []
    if fb.get("title_score") is not None:
        parts.append(f"title_score={fb.get('title_score')}")
    if fb.get("description_score") is not None:
        parts.append(f"description_score={fb.get('description_score')}")
    return parts or ["all gates passed"]


# --- score interpretation helpers (None = skipped, ignored) ---

def _is_false(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    return False


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)):
        return value >= 1
    return False


def _le(value: Any, threshold: int) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value <= threshold
