"""Pure-logic evaluator functions and the pass/review/fail gate.

These are framework-agnostic ``func(outputs, reference) -> {key, score, comment}``
checks. ``metrics.py`` wraps them as DeepEval metrics.
"""

from __future__ import annotations

from basic_ckan_agent.evaluation.evaluators.deterministic import (
    description_basic_checks,
    must_mention_terms,
    title_basic_checks,
)
from basic_ckan_agent.evaluation.evaluators.gates import (
    FAIL,
    PASS,
    REVIEW,
    GateResult,
    evaluate_gate,
)
from basic_ckan_agent.evaluation.evaluators.trajectory import (
    correct_tool_called,
    grounded_in_tool_output,
    no_excessive_tool_calls,
    no_unsafe_write_action,
    required_tool_args,
)

__all__ = [
    "title_basic_checks",
    "description_basic_checks",
    "must_mention_terms",
    "correct_tool_called",
    "no_unsafe_write_action",
    "required_tool_args",
    "grounded_in_tool_output",
    "no_excessive_tool_calls",
    "evaluate_gate",
    "GateResult",
    "PASS",
    "REVIEW",
    "FAIL",
]
