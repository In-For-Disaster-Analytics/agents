"""Run the prompt x model evaluation matrix locally with DeepEval.

No external service: the agent and every judge run on your own LLM endpoint
(OPENAI_BASE_URL / CKAN_LLM_MODEL). Results are written to a JSON report and
summarized in the console with pass / review / fail gate decisions.

Examples:
    # Full prompt x model matrix:
    python -m basic_ckan_agent.evaluation.run_experiments

    # One configuration:
    python -m basic_ckan_agent.evaluation.run_experiments --prompt baseline --model Meta-Llama-3.3-70B-Instruct

    # Pairwise-compare two prompt variants on the same model:
    python -m basic_ckan_agent.evaluation.run_experiments --pairwise baseline schema_aware

    # Print the dataset summary:
    python -m basic_ckan_agent.evaluation.dataset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from basic_ckan_agent.evaluation.config import get_models, run_metadata
from basic_ckan_agent.evaluation.dataset import build_examples
from basic_ckan_agent.evaluation.evaluators import evaluate_gate
from basic_ckan_agent.evaluation.metrics import FunctionMetric, metrics_for
from basic_ckan_agent.evaluation.models import LocalChatModel
from basic_ckan_agent.evaluation.prompts import get_prompts
from basic_ckan_agent.evaluation.target import build_eval_target, build_request
from basic_ckan_agent.logging_config import LOG_DIR, logger

REPORT_DIR = LOG_DIR / "eval"


def build_test_case(inputs: dict, reference: dict, outputs: dict, example_id: str, task_type: str) -> LLMTestCase:
    """Assemble a DeepEval test case from one agent run."""
    title = outputs.get("title", "")
    description = outputs.get("description", "")
    has_metadata = bool(title or description)

    if has_metadata:
        actual_output = f"Title: {title}\n\nDescription: {description}"
    else:
        actual_output = outputs.get("answer", "") or "(no answer)"

    source_blocks: list[str] = []
    if inputs.get("metadata"):
        source_blocks.append(json.dumps(inputs["metadata"], indent=2, ensure_ascii=False, default=str))
    if inputs.get("source_context"):
        source_blocks.append(str(inputs["source_context"]))

    # Faithfulness/grounding reference = source metadata + any tool outputs.
    retrieval_context = list(source_blocks) + [str(t) for t in (outputs.get("tool_outputs") or [])]

    return LLMTestCase(
        input=build_request(inputs),
        actual_output=actual_output,
        expected_output=None,
        context=source_blocks or None,
        retrieval_context=retrieval_context or None,
        tools_called=[ToolCall(name=a) for a in (outputs.get("tools_called") or [])],
        expected_tools=[ToolCall(name=a) for a in (reference.get("expected_tools") or [])],
        metadata={
            "example_id": example_id,
            "task_type": task_type,
            "title": title,
            "description": description,
            "answer": outputs.get("answer", ""),
            "trajectory": outputs.get("trajectory", []),
            "tools_called": outputs.get("tools_called", []),
            "tool_outputs": outputs.get("tool_outputs", []),
            "reference": reference,
        },
    )


def _metric_to_feedback(metric: Any) -> dict[str, Any]:
    """Map one measured DeepEval metric to gate feedback keys."""
    if isinstance(metric, FunctionMetric):
        if metric.skipped:
            return {metric.__name__: None}
        return {metric.__name__: bool(metric.success)}
    if isinstance(metric, GEval):
        # DeepEval appends a " [GEval]" suffix to the configured name.
        name = metric.__name__
        score = float(metric.score or 0.0)
        if name.startswith("Title Quality"):
            return {"title_score": round(score * 5)}
        if name.startswith("Description Quality"):
            return {"description_score": round(score * 5)}
        if name.startswith("Faithfulness"):
            return {"faithfulness_pass": bool(metric.success), "major_issue": score < 0.3}
        return {name: round(score * 5)}
    if isinstance(metric, ToolCorrectnessMetric):
        return {"correct_tool_called": bool(metric.success)}
    return {}


def run_example(target: Any, judge: LocalChatModel, example: dict) -> dict:
    inputs = example["inputs"]
    reference = example["outputs"]
    meta = example["metadata"]
    example_id = meta["example_id"]

    outputs = target(inputs)
    test_case = build_test_case(inputs, reference, outputs, example_id, meta.get("task_type", ""))

    feedback: dict[str, Any] = {}
    metric_scores: dict[str, Any] = {}
    for metric in metrics_for(test_case, judge):
        try:
            metric.measure(test_case)
        except Exception as exc:  # never let one metric abort the example
            logger.exception("Metric %s failed for %s", getattr(metric, "__name__", metric), example_id)
            metric.error = str(exc)
            continue
        feedback.update(_metric_to_feedback(metric))
        metric_scores[getattr(metric, "__name__", str(metric))] = {
            "score": getattr(metric, "score", None),
            "success": getattr(metric, "success", None),
            "skipped": getattr(metric, "skipped", False),
            "reason": (getattr(metric, "reason", "") or "")[:300],
        }

    gate = evaluate_gate(feedback)
    return {
        "example_id": example_id,
        "task_type": meta.get("task_type", ""),
        "status": gate.status,
        "reasons": gate.reasons,
        "title": outputs.get("title", ""),
        "description": outputs.get("description", ""),
        "tools_called": outputs.get("tools_called", []),
        "feedback": feedback,
        "metrics": metric_scores,
    }


def run_experiment(prompt_name: str, prompt_text: str | None, model_name: str | None) -> dict:
    """Evaluate one (prompt, model) configuration over the whole dataset."""
    print(f"\n=== Experiment: prompt={prompt_name} model={model_name} ===")
    target = build_eval_target(prompt_text, model_name)
    judge = LocalChatModel(model_name)  # judge on the same endpoint
    examples = build_examples()

    rows = []
    for i, example in enumerate(examples, start=1):
        row = run_example(target, judge, example)
        rows.append(row)
        print(f"  [{i}/{len(examples)}] {row['example_id']:<34} {row['status'].upper()}")

    counts = Counter(r["status"] for r in rows)
    report = {
        "metadata": run_metadata(prompt_name, model_name or "default"),
        "summary": {"pass": counts.get("pass", 0), "review": counts.get("review", 0), "fail": counts.get("fail", 0)},
        "results": rows,
    }
    _write_report(prompt_name, model_name, report)
    print(f"  -> pass={report['summary']['pass']} review={report['summary']['review']} fail={report['summary']['fail']}")
    return report


def run_matrix(prompt_names: list[str] | None = None, model_names: list[str] | None = None) -> list[dict]:
    prompts = get_prompts()
    selected_prompts = prompt_names or list(prompts.keys())
    models = model_names or get_models()

    reports = []
    for prompt_name in selected_prompts:
        for model_name in models:
            reports.append(run_experiment(prompt_name, prompts[prompt_name], model_name))
    _print_matrix_summary(reports)
    _write_html(reports)
    return reports


def run_pairwise(prompt_a: str, prompt_b: str, model_name: str | None = None) -> None:
    from basic_ckan_agent.evaluation.compare import compare_configs

    prompts = get_prompts()
    model = model_name or get_models()[0]
    result = compare_configs(
        (prompt_a, prompts[prompt_a], model),
        (prompt_b, prompts[prompt_b], model),
        build_examples(),
    )
    print("\n" + result.summary())


def _write_report(prompt_name: str, model_name: str | None, report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = (model_name or "default").replace("/", "_")
    path = REPORT_DIR / f"eval-{prompt_name}-{safe_model}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  report: {path}")
    return path


def _write_html(reports: list[dict]) -> Path:
    from basic_ckan_agent.evaluation.report import write_html_report

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = write_html_report(reports, REPORT_DIR / f"report-{stamp}.html")
    print(f"\nHTML report: {path}")
    return path


def _print_matrix_summary(reports: list[dict]) -> None:
    print("\n=== Matrix summary ===")
    print(f"{'prompt':<16}{'model':<32}{'pass':>6}{'review':>8}{'fail':>6}")
    for report in reports:
        md = report["metadata"]
        s = report["summary"]
        print(f"{md['prompt']:<16}{md['model']:<32}{s['pass']:>6}{s['review']:>8}{s['fail']:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CKAN agent eval experiments locally with DeepEval.")
    parser.add_argument("--prompt", help="Run only this prompt variant.")
    parser.add_argument("--model", help="Run only this model.")
    parser.add_argument(
        "--pairwise",
        nargs=2,
        metavar=("PROMPT_A", "PROMPT_B"),
        help="Pairwise-compare two prompt variants on one model.",
    )
    args = parser.parse_args()

    if args.pairwise:
        run_pairwise(args.pairwise[0], args.pairwise[1], args.model)
        return

    prompt_names = [args.prompt] if args.prompt else None
    model_names = [args.model] if args.model else None
    run_matrix(prompt_names, model_names)


if __name__ == "__main__":
    main()
