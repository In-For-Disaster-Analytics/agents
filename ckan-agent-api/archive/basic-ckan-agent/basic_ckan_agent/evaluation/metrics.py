"""DeepEval metrics for the CKAN metadata agent.

Three families, all running locally against your own LLM:

* Deterministic metrics wrap the pure check functions (title/description basic
  quality, must-mention, and the tool-trajectory checks) as DeepEval metrics.
* GEval judges score title and description quality on a 1-5 rubric (normalized to
  0-1 by DeepEval).
* Built-in ``FaithfulnessMetric`` and ``ToolCorrectnessMetric`` cover the hard
  faithfulness gate and expected-tool behavior.

The runner reads structured fields from ``test_case.metadata`` (title,
description, trajectory, tools_called, tool_outputs) and the example's expected
behavior from ``metadata["reference"]``.
"""

from __future__ import annotations

from typing import Any, Callable

from deepeval.metrics import BaseMetric, GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase

try:  # SingleTurnParams is the current name; LLMTestCaseParams is the deprecated alias.
    from deepeval.test_case import SingleTurnParams as _Params
except ImportError:  # pragma: no cover
    from deepeval.test_case import LLMTestCaseParams as _Params

from basic_ckan_agent.evaluation.evaluators.deterministic import (
    description_basic_checks,
    must_mention_terms,
    title_basic_checks,
)
from basic_ckan_agent.evaluation.evaluators.trajectory import (
    grounded_in_tool_output,
    no_excessive_tool_calls,
    no_unsafe_write_action,
    required_tool_args,
)


def split_test_case(test_case: LLMTestCase) -> tuple[dict, dict]:
    """Reconstruct (outputs, reference) dicts from a test case's metadata."""
    meta = test_case.metadata or {}
    outputs = {
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "answer": meta.get("answer", ""),
        "trajectory": meta.get("trajectory", []),
        "tools_called": meta.get("tools_called", []),
        "tool_outputs": meta.get("tool_outputs", []),
    }
    reference = meta.get("reference", {}) or {}
    return outputs, reference


class FunctionMetric(BaseMetric):
    """Adapt a pure ``func(outputs, reference) -> {score, comment}`` to DeepEval.

    A function score of ``None`` means "not applicable" for this example; the
    metric marks itself skipped so it neither passes nor fails.
    """

    def __init__(self, func: Callable[[dict, dict], dict], name: str, threshold: float = 0.5) -> None:
        self._func = func
        self._name = name
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True
        self.strict_mode = False
        self.verbose_mode = False
        self.error: str | None = None
        self.skipped = False
        self.evaluation_cost = None
        self.raw_score: Any = None

    @property
    def __name__(self) -> str:  # shown in DeepEval reports
        return self._name

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outputs, reference = split_test_case(test_case)
        result = self._func(outputs, reference)
        self.raw_score = result.get("score")
        self.reason = result.get("comment", "")
        if self.raw_score is None:
            self.skipped = True
            self.score = 1.0
            self.success = True
            return self.score
        self.score = float(self.raw_score) if not isinstance(self.raw_score, bool) else float(self.raw_score)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


# --- Deterministic metric factories (gate keys match their names) ---

def title_basic_metric() -> FunctionMetric:
    return FunctionMetric(title_basic_checks, "title_basic_quality")


def description_basic_metric() -> FunctionMetric:
    return FunctionMetric(description_basic_checks, "description_basic_quality")


def must_mention_metric() -> FunctionMetric:
    return FunctionMetric(must_mention_terms, "must_mention_terms")


def no_unsafe_write_metric() -> FunctionMetric:
    return FunctionMetric(no_unsafe_write_action, "no_unsafe_write_action")


def required_tool_args_metric() -> FunctionMetric:
    return FunctionMetric(required_tool_args, "required_tool_args")


def grounded_metric() -> FunctionMetric:
    return FunctionMetric(grounded_in_tool_output, "grounded_in_tool_output")


def no_excessive_tool_calls_metric() -> FunctionMetric:
    return FunctionMetric(no_excessive_tool_calls, "no_excessive_tool_calls")


# --- LLM-as-judge (GEval) metrics ---

_TITLE_STEPS = [
    "Read the source metadata in Context and the generated title in Actual Output.",
    "Check that the title names the dataset's main subject and includes available geography, instrument, model, "
    "campaign, organization, or data type.",
    "Penalize vague labels (Dataset, Results, Metadata, File Upload) and bare filenames that are not descriptive.",
    "Heavily penalize any fact in the title that is not supported by the source metadata (invented agency, place, "
    "date, instrument, method, or conclusion).",
    "Reward concise, specific, faithful, searchable, publication-ready titles.",
]

_DESCRIPTION_STEPS = [
    "Read the source metadata in Context and the generated description in Actual Output.",
    "Check that the description explains what the dataset contains and adds available spatial, temporal, "
    "organizational, methodological, and topical context.",
    "Reward mention of key resources, variables, model outputs, or sensor outputs when present in the source.",
    "Heavily penalize unsupported claims, invented facts, marketing language, or claims of peer review/publication "
    "not present in the source.",
    "Reward faithful, clear, complete descriptions useful to a public data catalog user.",
]


_FAITHFULNESS_STEPS = [
    "Read the source metadata in Context and the generated title + description in Actual Output.",
    "Identify every concrete claim in the Actual Output: agencies, locations, dates, instruments, methods, "
    "variables, conclusions, and any claim of peer review or publication.",
    "Check each claim against the Context. A claim is unsupported if the Context neither states nor clearly implies it.",
    "Return a HIGH score only if every claim is supported by the Context.",
    "Return a LOW score if any claim is fabricated or unsupported, regardless of how polished the wording is.",
]

# async_mode=False keeps each judge's internal LLM calls sequential so they respect
# the agent's global rate limiter instead of bursting and triggering 429s.


def title_quality_geval(model: Any) -> GEval:
    return GEval(
        name="Title Quality",
        evaluation_steps=_TITLE_STEPS,
        evaluation_params=[_Params.INPUT, _Params.ACTUAL_OUTPUT, _Params.CONTEXT],
        model=model,
        threshold=0.6,  # ~3/5 on the rubric
        async_mode=False,
    )


def description_quality_geval(model: Any) -> GEval:
    return GEval(
        name="Description Quality",
        evaluation_steps=_DESCRIPTION_STEPS,
        evaluation_params=[_Params.INPUT, _Params.ACTUAL_OUTPUT, _Params.CONTEXT],
        model=model,
        threshold=0.6,
        async_mode=False,
    )


def faithfulness_metric(model: Any) -> GEval:
    # Single-call GEval faithfulness judge (cheaper than the built-in
    # FaithfulnessMetric, which makes one call per extracted claim). Judges whether
    # the title/description claims are supported by the source metadata in Context.
    return GEval(
        name="Faithfulness",
        evaluation_steps=_FAITHFULNESS_STEPS,
        evaluation_params=[_Params.ACTUAL_OUTPUT, _Params.CONTEXT],
        model=model,
        threshold=0.7,
        async_mode=False,
    )


def tool_correctness_metric() -> ToolCorrectnessMetric:
    # Compares tools_called vs expected_tools by name; subset match, order-agnostic.
    return ToolCorrectnessMetric(should_exact_match=False, should_consider_ordering=False)


def metrics_for(test_case: LLMTestCase, model: Any) -> list[BaseMetric]:
    """Pick the applicable metric set for one test case.

    Metadata-generation cases get the title/description judges + faithfulness;
    tool-driven cases get tool correctness + arg/grounding checks. The
    safety/trajectory checks run on everything.
    """
    meta = test_case.metadata or {}
    reference = meta.get("reference", {}) or {}
    has_metadata = bool(meta.get("title") or meta.get("description"))
    has_expected_tools = bool(reference.get("expected_tools"))

    metrics: list[BaseMetric] = [
        no_unsafe_write_metric(),
        no_excessive_tool_calls_metric(),
        grounded_metric(),
    ]
    if has_metadata:
        metrics += [
            title_basic_metric(),
            description_basic_metric(),
            must_mention_metric(),
            title_quality_geval(model),
            description_quality_geval(model),
        ]
        if test_case.context:
            metrics.append(faithfulness_metric(model))
    if has_expected_tools:
        metrics.append(tool_correctness_metric())
        metrics.append(required_tool_args_metric())
    return metrics
