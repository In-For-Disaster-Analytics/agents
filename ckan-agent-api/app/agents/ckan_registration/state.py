from __future__ import annotations

from typing import Any, TypedDict


class CkanRegistrationState(TypedDict, total=False):
    thread_id: str
    action: str
    request: dict[str, Any]
    result: dict[str, Any]
    status: str
    error: str
    requires_action: dict[str, Any]
    messages: list[dict[str, Any]]

    # Persona-chat path (spec increments 2-3).
    schema_profile: str
    candidate_metadata: dict[str, Any]
    evaluator_verdicts: list[dict[str, Any]]
    persona_stop_reason: str
    clarification_questions: list[dict[str, Any]]
    clarification_round: int
    reviewed_files: list[str]
    # R3: org-level values are thread-sticky; dataset-specific answers reset per dataset.
    org_metadata: dict[str, Any]
    dataset_clarifications: dict[str, Any]
    # Fields locked after first LLM derivation to prevent drift across rounds (e.g. title).
    # Distinct from dataset_clarifications so they are labeled llm-derived, not user-supplied.
    llm_locked_fields: dict[str, Any]
    # Per-field validation errors from the last clarify round (e.g. invalid email format).
    # Shown at the top of the next question for that field so the user knows why they're re-asked.
    clarification_errors: dict[str, str]
    # Fields the user explicitly declined to provide ("no", "n/a", "skip").
    # The clarify node skips any question for a field in this set to prevent infinite re-asking.
    declined_fields: list[str]

    # Gated geo transform path (spec 2026-06-30). A persona may *propose* a transform; the
    # human approves; geo-apply executes it. The token is never stored here (injected server-side
    # at the node and scrubbed). `transforms_submitted` enforces the per-session cap.
    transform_request: dict[str, Any]
    transform_execution_id: str
    transforms_submitted: int

    # LLM-routed targeted field edit. Set by intake when the router picks revise_field;
    # consumed by the revise-field node. Carries {"field": str, "instruction": str}.
    revise_field_target: dict[str, Any]

    # LLM-routed focused show. Set by intake when the router picks show with a specific
    # field question. Carries {"field": str, "question": str}. Optional — omit for full dump.
    show_target: dict[str, Any]
