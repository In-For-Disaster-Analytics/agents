"""Persona metadata authoring + evaluation engine (moved into ckan-agent-api per R4).

Pure, synchronous engine: an *author* persona drafts candidate metadata, then one or
more *evaluator* personas review it. It is interrupt-free — when an evaluator raises a
blocking question that no source can answer (``requires_human == true``, R1), the engine
stops immediately (R2 eager escalation) and returns ``stop_reason="needs_clarification"``
with the blocking questions. The graph node owns the human interrupt/resume; on resume it
folds the answers into ``organizational_metadata`` and calls the engine again.

Design commitments realised here:
- R1: evaluators emit structured questions ``{field, question, requires_human,
  reason_not_derivable}``; bare strings are coerced to ``requires_human=false``.
- R2: escalate on the *first* ``requires_human`` question rather than exhausting rounds.
- R4: lives in the agent package; the ``LLM_CALL_DELAY_SECONDS`` sleep defaults to 0 so it
  never blocks the FastAPI event loop (a graph node may still run this in a thread).
- Personas/schema are injected (from ``PersonaRegistry`` / ``SchemaProfile``); the prompt
  bodies are rendered with the schema profile's ``{{schema_fields}}`` / ``{{controlled_vocab}}``
  / ``{{defaults}}`` tokens.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from app import llm
from app.personas.registry import Persona
from app.schemas.registry import SchemaProfile

logger = logging.getLogger(__name__)

# A chat function takes (system_prompt, user_payload) and returns raw model text.
ChatFn = Callable[[str, dict[str, Any]], str]

STOP_CONVERGED = "converged"
STOP_NEEDS_CLARIFICATION = "needs_clarification"
STOP_MAX_ROUNDS = "max_rounds"
STOP_LLM_ERROR = "llm_error"


@dataclass
class EvaluatorVerdict:
    verdict: str  # "pass" | "revise"
    questions: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    persona_name: str = ""

    def human_questions(self) -> list[dict[str, Any]]:
        return [q for q in self.questions if q.get("requires_human")]


@dataclass
class LoopRound:
    round_number: int
    candidate_metadata: dict[str, Any]
    evaluator_verdicts: list[EvaluatorVerdict]
    converged: bool


@dataclass
class PersonaLoopResult:
    converged: bool
    rounds: int
    proposed_metadata: dict[str, Any]
    transcript: list[LoopRound]
    stop_reason: str
    clarification_questions: list[dict[str, Any]] = field(default_factory=list)
    model_id: str = ""
    timestamp: str = ""


def _coerce_questions(raw: Any) -> list[dict[str, Any]]:
    """Normalise evaluator questions to the R1 structured shape; drop empties."""
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            requires_human = item.get("requires_human", False)
            if isinstance(requires_human, str):
                requires_human = requires_human.strip().lower() in {"1", "true", "yes", "on"}
            out.append(
                {
                    "field": item.get("field"),
                    "question": question,
                    "requires_human": bool(requires_human),
                    "reason_not_derivable": item.get("reason_not_derivable"),
                }
            )
        else:
            out.append(
                {"field": None, "question": str(item).strip(), "requires_human": False, "reason_not_derivable": None}
            )
    return [q for q in out if q["question"]]


def _dedupe_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate clarification questions (same field, or same text) raised by
    multiple evaluators — the user should see each one once."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for q in questions:
        key = str(q.get("field") or q.get("question") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


# Fields the author writes/derives itself — never escalate these to the user.
NEVER_ASK_FIELDS = {
    "title", "name", "notes", "tag_string", "tags",
    "categories", "primary_tags", "secondary_tags",
    "collection_method", "collection_method_description",
}


def _is_actionable(question: dict[str, Any], candidate: dict[str, Any], org: dict[str, Any] | None) -> bool:
    """A question is worth acting on only if it targets a specific field that is genuinely
    missing — not author-derivable, not already supplied, not already valued. This enforces
    in code what the evaluator prompts request (LLMs don't reliably obey it)."""
    field = str(question.get("field") or "").strip()
    if not field or field in NEVER_ASK_FIELDS:
        return False
    if field in (org or {}):
        return False  # already supplied by the user/config this conversation
    value = candidate.get(field)
    if value not in (None, "", []):
        return False  # the author already produced a value
    return True


def _actionable_human_questions(
    questions: list[dict[str, Any]], candidate: dict[str, Any], org: dict[str, Any] | None
) -> list[dict[str, Any]]:
    return [q for q in questions if q.get("requires_human") and _is_actionable(q, candidate, org)]


def _parse_verdict(content: str, persona_name: str) -> EvaluatorVerdict:
    parsed = llm.parse_json_response(content)
    raw_verdict = str(parsed.get("verdict", "")).strip().lower()
    verdict = "pass" if raw_verdict == "pass" else "revise"
    recommendations = parsed.get("recommendations") or []
    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)]
    return EvaluatorVerdict(
        verdict=verdict,
        questions=_coerce_questions(parsed.get("questions")),
        recommendations=[str(r) for r in recommendations],
        persona_name=persona_name,
    )


def _author_payload(
    *,
    consolidated_inputs: dict[str, Any],
    schema_profile: SchemaProfile,
    resource_plan: list[dict[str, Any]],
    file_inventory: dict[str, Any] | None,
    bbox_geojson: str | None,
    organizational_metadata: dict[str, Any] | None,
    prior_evaluator_feedback: list[dict[str, Any]] | None,
    resolved_gaps: dict[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "consolidated_inputs": consolidated_inputs,
        "schema_profile": {"name": schema_profile.name, "dataset_type": schema_profile.dataset_type},
        "resource_plan_summary": [
            {
                "resource_name": r.get("resource_name", ""),
                "format": r.get("format", ""),
                "relative_path": r.get("relative_path", ""),
            }
            for r in (resource_plan or [])[:30]
        ],
        "resource_count": len(resource_plan or []),
        "bbox_geojson": bbox_geojson,
    }
    if file_inventory:
        payload["file_inventory"] = file_inventory
    if organizational_metadata:
        payload["organizational_metadata"] = organizational_metadata
    if prior_evaluator_feedback:
        payload["evaluator_feedback_from_prior_round"] = prior_evaluator_feedback
    if resolved_gaps:
        payload["resolved_gaps_do_not_reraise"] = resolved_gaps
    return payload


def _extract_gaps(candidate: dict[str, Any]) -> dict[str, str]:
    return {k[len("_gap_"):]: str(v) for k, v in candidate.items() if k.startswith("_gap_")}


def _author_tool_loop(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    tool_chat: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]],
    executor: Any,
    tools: list[dict[str, Any]],
    max_tool_calls: int,
) -> str:
    """Run the author as a tool-calling loop; returns the final assistant content (JSON).

    The model may call read-only tools (executed via ``executor``); results are fed back as
    tool messages. Capped at ``max_tool_calls`` total calls, after which one final tool-free
    turn is requested so the author always returns metadata JSON.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    calls_made = 0
    while True:
        offer_tools = tools if calls_made < max_tool_calls else None
        resp = tool_chat(messages, offer_tools)
        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            return resp.get("content") or ""
        messages.append(resp.get("raw_message") or {"role": "assistant", "content": resp.get("content") or ""})
        for call in tool_calls:
            result = executor.invoke(call.get("name", ""), call.get("arguments") or {})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result, default=str)[:8000],
                }
            )
            calls_made += 1
        if calls_made >= max_tool_calls:
            messages.append(
                {"role": "user", "content": "Tool budget reached. Return ONLY the final metadata JSON now."}
            )
            return tool_chat(messages, None).get("content") or ""


def _default_chat_fn(llm_model: str, llm_api_key: str, llm_base_url: str | None) -> ChatFn:
    def _call(system_prompt: str, user_payload: dict[str, Any]) -> str:
        return llm.invoke_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url or "",
            temperature=0.1,
        )

    return _call


def run_persona_metadata_loop(
    consolidated_inputs: dict[str, Any],
    *,
    author_persona: Persona,
    evaluator_personas: list[Persona],
    schema_profile: SchemaProfile,
    resource_plan: list[dict[str, Any]] | None = None,
    file_inventory: dict[str, Any] | None = None,
    bbox_geojson: str | None = None,
    organizational_metadata: dict[str, Any] | None = None,
    max_rounds: int = 3,
    chat_fn: ChatFn | None = None,
    tool_executor: Any = None,
    author_tool_specs: list[dict[str, Any]] | None = None,
    tool_chat_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]] | None = None,
    max_tool_calls: int = 6,
    llm_model: str = "",
    llm_api_key: str = "",
    llm_base_url: str | None = None,
    model_id: str = "",
    run_timestamp: str = "",
    runs_dir: Path | None = None,
    delay_seconds: float = 0.0,
) -> PersonaLoopResult:
    """Run the author + evaluator loop. Always returns; never raises.

    Stop reasons: ``converged`` (all evaluators pass), ``needs_clarification`` (a
    ``requires_human`` question appeared — eager), ``max_rounds``, ``llm_error``.
    """
    chat = chat_fn or _default_chat_fn(llm_model, llm_api_key, llm_base_url)
    tokens = schema_profile.render_tokens()
    author_system = author_persona.render(**tokens)
    evaluator_systems = [(p.name, p.render(**tokens)) for p in evaluator_personas]

    resource_plan = resource_plan or []
    transcript: list[LoopRound] = []
    prior_evaluator_feedback: list[dict[str, Any]] | None = None
    resolved_gaps: dict[str, str] = {}
    last_candidate: dict[str, Any] = {}

    def _sleep() -> None:
        if delay_seconds and delay_seconds > 0:
            time.sleep(delay_seconds)

    for round_num in range(1, max_rounds + 1):
        payload = _author_payload(
            consolidated_inputs=consolidated_inputs,
            schema_profile=schema_profile,
            resource_plan=resource_plan,
            file_inventory=file_inventory,
            bbox_geojson=bbox_geojson,
            organizational_metadata=organizational_metadata,
            prior_evaluator_feedback=prior_evaluator_feedback,
            resolved_gaps=resolved_gaps,
        )
        try:
            if tool_executor is not None and author_tool_specs and tool_chat_fn is not None:
                author_content = _author_tool_loop(
                    author_system,
                    payload,
                    tool_chat=tool_chat_fn,
                    executor=tool_executor,
                    tools=author_tool_specs,
                    max_tool_calls=max_tool_calls,
                )
            else:
                author_content = chat(author_system, payload)
            candidate = llm.parse_json_response(author_content)
        except Exception as exc:  # noqa: BLE001 - engine never raises
            logger.error("[persona_engine] author LLM failure round=%d: %s", round_num, exc)
            return _finish(
                PersonaLoopResult(
                    converged=False,
                    rounds=round_num - 1,
                    proposed_metadata=last_candidate or consolidated_inputs,
                    transcript=transcript,
                    stop_reason=STOP_LLM_ERROR,
                    model_id=model_id,
                    timestamp=run_timestamp,
                ),
                runs_dir,
            )
        _sleep()
        last_candidate = candidate

        verdicts: list[EvaluatorVerdict] = []
        try:
            for name, system in evaluator_systems:
                content = chat(system, {"candidate_metadata": candidate, "resolved_gaps_do_not_reraise": resolved_gaps})
                verdicts.append(_parse_verdict(content, name))
                _sleep()
        except Exception as exc:  # noqa: BLE001
            logger.error("[persona_engine] evaluator LLM failure round=%d: %s", round_num, exc)
            return _finish(
                PersonaLoopResult(
                    converged=False,
                    rounds=round_num,
                    proposed_metadata=candidate,
                    transcript=transcript,
                    stop_reason=STOP_LLM_ERROR,
                    model_id=model_id,
                    timestamp=run_timestamp,
                ),
                runs_dir,
            )

        all_human = _dedupe_questions([q for v in verdicts for q in v.human_questions()])
        human_questions = _actionable_human_questions(all_human, candidate, organizational_metadata)
        # Convergence is the evaluators' call. A "revise" (incl. recommendations to improve a
        # thin `notes`) loops the AUTHOR for another improvement pass (bounded by max_rounds) —
        # the user is only interrupted for genuinely-external `human_questions` (filtered above),
        # never for narrative/quality fixes the author can make from the sources itself.
        converged = all(v.verdict == "pass" for v in verdicts)
        transcript.append(LoopRound(round_num, candidate, verdicts, converged))

        # R2: eager escalation — the moment a non-derivable blocking question appears.
        if human_questions:
            logger.info("[persona_engine] needs_clarification round=%d (%d question(s))", round_num, len(human_questions))
            return _finish(
                PersonaLoopResult(
                    converged=False,
                    rounds=round_num,
                    proposed_metadata=candidate,
                    transcript=transcript,
                    stop_reason=STOP_NEEDS_CLARIFICATION,
                    clarification_questions=human_questions,
                    model_id=model_id,
                    timestamp=run_timestamp,
                ),
                runs_dir,
            )

        if converged:
            return _finish(
                PersonaLoopResult(
                    converged=True,
                    rounds=round_num,
                    proposed_metadata=candidate,
                    transcript=transcript,
                    stop_reason=STOP_CONVERGED,
                    model_id=model_id,
                    timestamp=run_timestamp,
                ),
                runs_dir,
            )

        # Source-derivable revises: feed feedback back to the author for another round.
        prior_evaluator_feedback = [
            {"persona": v.persona_name, "questions": v.questions, "recommendations": v.recommendations}
            for v in verdicts
            if v.verdict == "revise"
        ]
        resolved_gaps.update(_extract_gaps(candidate))

    return _finish(
        PersonaLoopResult(
            converged=False,
            rounds=max_rounds,
            proposed_metadata=last_candidate,
            transcript=transcript,
            stop_reason=STOP_MAX_ROUNDS,
            model_id=model_id,
            timestamp=run_timestamp,
        ),
        runs_dir,
    )


def _finish(result: PersonaLoopResult, runs_dir: Path | None) -> PersonaLoopResult:
    if runs_dir is not None:
        _write_audit_trail(result, runs_dir)
    return result


def _write_audit_trail(result: PersonaLoopResult, runs_dir: Path) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = (result.timestamp or "run").replace(" ", "_").replace(":", "")
        out = runs_dir / f"{result.model_id or 'persona'}_{stamp}.json"
        payload = {
            "model_id": result.model_id,
            "timestamp": result.timestamp,
            "converged": result.converged,
            "rounds": result.rounds,
            "stop_reason": result.stop_reason,
            "clarification_questions": result.clarification_questions,
            "proposed_metadata": result.proposed_metadata,
            "transcript": [
                {
                    "round_number": r.round_number,
                    "candidate_metadata": r.candidate_metadata,
                    "evaluator_verdicts": [asdict(v) for v in r.evaluator_verdicts],
                    "converged": r.converged,
                }
                for r in result.transcript
            ],
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - audit best-effort
        logger.warning("[persona_engine] failed to write audit trail: %s", exc)
