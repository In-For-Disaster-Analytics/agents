"""Local pairwise comparison between two prompt/model configurations.

Runs the agent for both configs on each example and asks the judge LLM which
generated title + description is better for a CKAN catalog page. Output order is
randomized per example to reduce position bias. No external service required.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from basic_ckan_agent.evaluation.extraction import _last_json_object
from basic_ckan_agent.evaluation.models import LocalChatModel
from basic_ckan_agent.evaluation.target import build_eval_target
from basic_ckan_agent.logging_config import logger

PAIRWISE_PROMPT = """You are choosing which generated CKAN dataset metadata is better for a public catalog page.

Source metadata / context:
{source_context}

Option A:
title: {a_title}
description: {a_description}

Option B:
title: {b_title}
description: {b_description}

Prefer the option that is, in priority order:
1. More faithful to the source metadata (no invented agencies, locations, dates, instruments, methods, variables, conclusions, or publication claims).
2. More specific and searchable.
3. Clearer to a public data user.
4. More complete in the description.
5. Less verbose if both are equally complete.

Critical: if exactly one option invents unsupported facts, prefer the other regardless of polish.

Return JSON only: {{"preferred": "A" | "B" | "tie", "reason": "<short explanation>"}}."""


@dataclass
class PairwiseResult:
    a_label: str
    b_label: str
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    per_example: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        total = self.a_wins + self.b_wins + self.ties
        return (
            f"Pairwise: {self.a_label} vs {self.b_label} over {total} example(s)\n"
            f"  {self.a_label}: {self.a_wins} win(s)\n"
            f"  {self.b_label}: {self.b_wins} win(s)\n"
            f"  tie: {self.ties}"
        )


def compare_configs(
    config_a: tuple[str, str | None, str | None],
    config_b: tuple[str, str | None, str | None],
    examples: list[dict],
    *,
    judge: LocalChatModel | None = None,
    seed: int = 0,
) -> PairwiseResult:
    """Compare two ``(label, prompt_text, model_name)`` configs over examples.

    Only metadata-generation examples (those producing a title/description) are
    compared; others are skipped.
    """
    judge = judge or LocalChatModel()
    rng = random.Random(seed)
    label_a, prompt_a, model_a = config_a
    label_b, prompt_b, model_b = config_b
    target_a = build_eval_target(prompt_a, model_a)
    target_b = build_eval_target(prompt_b, model_b)

    result = PairwiseResult(a_label=label_a, b_label=label_b)
    for example in examples:
        inputs = example["inputs"]
        out_a = target_a(inputs)
        out_b = target_b(inputs)
        if not _has_metadata(out_a) and not _has_metadata(out_b):
            continue

        preferred = _judge_pair(judge, inputs, out_a, out_b, rng)
        if preferred == "A":
            result.a_wins += 1
        elif preferred == "B":
            result.b_wins += 1
        else:
            result.ties += 1
        result.per_example.append(
            {
                "example_id": example.get("metadata", {}).get("example_id"),
                "preferred": preferred,
                "a_title": out_a.get("title"),
                "b_title": out_b.get("title"),
            }
        )
    return result


def _judge_pair(judge: LocalChatModel, inputs: dict, out_a: dict, out_b: dict, rng: random.Random) -> str:
    flip = rng.random() < 0.5
    shown_a, shown_b = (out_b, out_a) if flip else (out_a, out_b)
    prompt = PAIRWISE_PROMPT.format(
        source_context=_source_context(inputs),
        a_title=shown_a.get("title") or "(none)",
        a_description=shown_a.get("description") or "(none)",
        b_title=shown_b.get("title") or "(none)",
        b_description=shown_b.get("description") or "(none)",
    )
    try:
        raw = judge.generate(prompt)
        verdict = _last_json_object(raw) or {}
    except Exception:
        logger.exception("Pairwise judge failed")
        return "tie"

    shown_pref = str(verdict.get("preferred", "tie")).strip().upper()
    if shown_pref == "A":
        return "B" if flip else "A"
    if shown_pref == "B":
        return "A" if flip else "B"
    return "tie"


def _has_metadata(outputs: dict) -> bool:
    return bool(outputs.get("title") or outputs.get("description"))


def _source_context(inputs: dict) -> str:
    blocks = []
    if inputs.get("metadata"):
        blocks.append(json.dumps(inputs["metadata"], indent=2, ensure_ascii=False, default=str))
    if inputs.get("source_context"):
        blocks.append(str(inputs["source_context"]))
    return "\n\n".join(blocks) if blocks else "(none provided)"
