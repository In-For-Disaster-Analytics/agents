from __future__ import annotations

import base64
import json
import re
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agents.ckan_registration.graph import CkanRegistrationRunner, get_runner
from app.agents.ckan_registration.logging_config import logger
from app.agents.ckan_registration.schemas import (
    AgentRunResponse,
    ChatCompletionRequest,
    CkanResumeRequest,
    CkanRunRequest,
    ModelCard,
    ModelListResponse,
    ToolInvokeRequest,
)
from app.agents.ckan_registration.ckan_client import CkanClient
from app.agents.ckan_registration.tools import TOOL_SPECS, ToolExecutor
from app.api.security import merge_secret_headers
from app.auth_context import bind_request_ckan_auth, get_request_ckan_auth
from app.settings import Settings, get_settings

# bind_request_ckan_auth captures the request's Authorization: Bearer <jwt> into a contextvar
# for the duration of every agent call, so CKAN reads/writes authenticate as the chat user.
router = APIRouter(tags=["ckan-registration"], dependencies=[Depends(bind_request_ckan_auth)])


def require_ckan_org_access(settings: Settings = Depends(get_settings)) -> None:
    """FastAPI dependency: reject callers who are not members of any CKAN organization.

    Reads the per-request JWT already bound by bind_request_ckan_auth and calls
    organization_list_for_user on the CKAN portal. HTTP 401 from CKAN (invalid/expired
    token) is treated as no access and raises 403. Network failures or unreachable CKAN
    are treated as uncertain and allow the request through — the agent itself will surface
    a clear error if CKAN is actually down.
    """
    auth = get_request_ckan_auth()
    if not auth:
        raise HTTPException(
            status_code=401,
            detail="Authentication required — supply your Tapis JWT as 'Authorization: Bearer <token>'.",
        )
    client = CkanClient(base_url=settings.ckan_url, authorization_header=auth, timeout=15)
    orgs = client.organization_list_for_user()
    if orgs is not None and not orgs:
        raise HTTPException(
            status_code=403,
            detail=(
                "You must belong to at least one CKAN organization to use this agent. "
                "Contact your CKAN administrator to request access."
            ),
        )

WORKFLOW_HINT_RE = re.compile(
    r"\b("
    r"analy[sz]e|apply|ckan|dataset|dry[- ]?run|file|metadata|package|"
    r"publish|register|resource|revise|source|upload|url|upstream"
    r")\b|https?://|/\S+|\.(csv|json|pdf|zip)\b",
    re.IGNORECASE,
)
THREAD_ID_PATTERNS = (
    re.compile(r"\bThread ID:\s*`?([A-Za-z0-9_.:-]+)`?", re.IGNORECASE),
    re.compile(r'"(?:thread_id|session_id)"\s*:\s*"([^"]+)"', re.IGNORECASE),
    re.compile(r"/ckan-registration/([^/\s\"']+)\.json", re.IGNORECASE),
)
MISSING_STATE_RE = re.compile(r"No saved CKAN agent state found", re.IGNORECASE)
STATEFUL_ACTION_RE = re.compile(r"\b(dry[- ]?run|revise|change|edit|exclude|remove|show|status|register|apply)\b", re.I)


def _with_request_headers(payload: CkanRunRequest, request: Request) -> CkanRunRequest:
    headers = merge_secret_headers(payload.request_headers, request)
    return payload.model_copy(update={"request_headers": headers})


def _with_resume_headers(payload: CkanResumeRequest, request: Request) -> CkanResumeRequest:
    headers = merge_secret_headers(payload.request_headers, request)
    return payload.model_copy(update={"request_headers": headers})


@router.post(
    "/v1/ckan-registration/runs",
    response_model=AgentRunResponse,
    operation_id="createCkanRegistrationRun",
)
def create_ckan_registration_run(
    payload: CkanRunRequest,
    request: Request,
    runner: CkanRegistrationRunner = Depends(get_runner),
    _org_check: None = Depends(require_ckan_org_access),
) -> AgentRunResponse:
    return runner.invoke(_with_request_headers(payload, request))


@router.post(
    "/v1/ckan-registration/runs/{thread_id}/resume",
    response_model=AgentRunResponse,
    operation_id="resumeCkanRegistrationRun",
)
def resume_ckan_registration_run(
    thread_id: str,
    payload: CkanResumeRequest,
    request: Request,
    runner: CkanRegistrationRunner = Depends(get_runner),
    _org_check: None = Depends(require_ckan_org_access),
) -> AgentRunResponse:
    return runner.resume(thread_id, _with_resume_headers(payload, request))


@router.get(
    "/v1/ckan-registration/runs/{thread_id}",
    response_model=AgentRunResponse,
    operation_id="getCkanRegistrationRun",
)
def get_ckan_registration_run(
    thread_id: str,
    request: Request,
    runner: CkanRegistrationRunner = Depends(get_runner),
) -> AgentRunResponse:
    headers = merge_secret_headers(None, request)
    return runner.show(thread_id, request_headers=headers)


@router.post(
    "/v1/ckan-registration/tools/{tool_name}",
    operation_id="invokeCkanRegistrationTool",
)
def invoke_ckan_registration_tool(
    tool_name: str,
    payload: ToolInvokeRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    _org_check: None = Depends(require_ckan_org_access),
) -> dict[str, Any]:
    if tool_name not in TOOL_SPECS:
        raise HTTPException(status_code=404, detail=f"Unknown CKAN registration tool: {tool_name}")
    arguments = dict(payload.arguments)
    headers = merge_secret_headers(arguments.get("request_headers"), request)
    if headers:
        arguments["request_headers"] = headers
    return ToolExecutor(settings).invoke(tool_name, arguments)


@router.get("/models", response_model=ModelListResponse, include_in_schema=False)
@router.get("/v1/models", response_model=ModelListResponse, operation_id="listOpenAICompatibleModels")
def list_models() -> ModelListResponse:
    return ModelListResponse(data=[ModelCard(id="ckan-registration-agent")])


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
        return "\n".join(parts)
    return ""


def _message_content_types(content: object) -> list[str]:
    if isinstance(content, str):
        return ["text"]
    if isinstance(content, list):
        types = []
        for item in content:
            if isinstance(item, dict):
                types.append(str(item.get("type") or "object"))
            else:
                types.append(type(item).__name__)
        return types
    return [type(content).__name__]


def _nested_file_dict(item: dict[str, Any]) -> dict[str, Any]:
    nested = item.get("file")
    return nested if isinstance(nested, dict) else {}


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _decode_inline_file_data(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    if not text.startswith("data:") or "," not in text:
        return text
    header, payload = text.split(",", 1)
    if ";base64" not in header:
        return payload
    try:
        return base64.b64decode(payload).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _message_file_payloads(content: object) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    file_refs: list[dict[str, str]] = []
    inline_files: list[dict[str, str]] = []
    attachment_names: list[str] = []

    if not isinstance(content, list):
        return file_refs, inline_files, attachment_names

    for item in content:
        if not isinstance(item, dict):
            continue
        nested = _nested_file_dict(item)
        item_type = str(item.get("type") or "").lower()
        fileish = item_type in {"file", "input_file", "uploaded_file", "attachment"}
        name = str(
            _first_present(item, "filename", "file_name", "name", "title")
            or _first_present(nested, "filename", "file_name", "name", "title")
            or ""
        ).strip()
        path = str(
            _first_present(item, "path", "local_path", "file_path", "filepath", "upload_path", "tmp_path")
            or _first_present(nested, "path", "local_path", "file_path", "filepath", "upload_path", "tmp_path")
            or ""
        ).strip()
        content_text = _decode_inline_file_data(
            _first_present(item, "content", "text", "data", "file_data")
            or _first_present(nested, "content", "text", "data", "file_data")
        )
        mime_type = str(
            _first_present(item, "mime_type", "content_type")
            or _first_present(nested, "mime_type", "content_type")
            or ""
        ).strip()

        if path:
            file_refs.append({"path": path, "name": name or Path(path).name})
        elif content_text and (fileish or name):
            inline_files.append(
                {
                    "name": name or "uploaded_text.txt",
                    "content": content_text,
                    "mime_type": mime_type or "text/plain",
                }
            )
        elif name and fileish:
            attachment_names.append(name)

    return file_refs, inline_files, attachment_names


def _last_user_file_payloads(
    payload: ChatCompletionRequest,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    for message in reversed(payload.messages):
        if message.role == "user":
            return _message_file_payloads(message.content)
    return [], [], []


def _last_user_message(payload: ChatCompletionRequest) -> str:
    for message in reversed(payload.messages):
        if message.role == "user":
            return _message_content_to_text(message.content)
    return ""


def _conversation_thread_id(payload: ChatCompletionRequest) -> str:
    for message in reversed(payload.messages):
        text = _message_content_to_text(message.content)
        if MISSING_STATE_RE.search(text):
            continue
        for pattern in THREAD_ID_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    return ""


def _conversation_metadata_context(payload: ChatCompletionRequest, max_chars: int = 12_000) -> dict[str, Any]:
    turns: list[dict[str, str]] = []
    previous_assistant = ""
    has_metadata_report = False

    for message in payload.messages[-8:]:
        text = _message_content_to_text(message.content).strip()
        if not text:
            continue
        text = text[:4000]
        turns.append({"role": message.role, "content": text})
        if message.role == "assistant":
            previous_assistant = text
            if "Best-Guess CKAN Starter Metadata" in text or "Thread ID:" in text:
                has_metadata_report = True

    context_text = "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    if len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    return {
        "has_prior_metadata_report": has_metadata_report,
        "previous_assistant_response": previous_assistant,
        "recent_turns": turns,
        "context_text": context_text,
    }


def _chat_intro_content() -> str:
    return (
        "## Next Options\n"
        "- Send a dataset file, pasted file content, readable local path, or source URL.\n"
        "- Tell me whether this will be a new CKAN dataset or an update to an existing one.\n\n"
        "I will read what I can and return a file metadata report plus a starter CKAN metadata guess."
    )


def _has_workflow_metadata(metadata: dict[str, Any]) -> bool:
    ignored_keys = {"message", "request_headers"}
    return any(value not in (None, "", [], {}) for key, value in metadata.items() if key not in ignored_keys)


def _should_show_chat_intro(metadata: dict[str, Any], message: str) -> bool:
    if _has_workflow_metadata(metadata):
        return False
    return not WORKFLOW_HINT_RE.search(str(metadata.get("message") or message))


def _needs_existing_thread(message: str) -> bool:
    match = STATEFUL_ACTION_RE.search(message)
    if match:
        logger.debug(f"   STATEFUL_ACTION_RE matched: '{match.group()}' - looks like a stateful action request")
    return bool(match)


def _missing_thread_content(message: str) -> str:
    return (
        "I need an existing analyzed CKAN thread before I can do that. "
        "That means a saved local registration analysis session, not an existing CKAN dataset. "
        "Ask me to analyze the dataset first with files, an upload_dir, a source URL, or dataset details."
    )


def _chat_response(model: str, content: str) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chat_stream_response(model: str, content: str) -> StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict[str, str], finish_reason: str | None = None) -> str:
        return "data: " + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
        ) + "\n\n"

    def events() -> Iterator[str]:
        yield chunk({"role": "assistant"})
        if content:
            yield chunk({"content": content})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _next_options_for_result(result: AgentRunResponse) -> list[str]:
    status = str(result.status or result.result.get("status") or "")
    command = str(result.command or result.result.get("command") or "")

    if status == "metadata_report":
        return [
            "Reply `new dataset` if this should create a new CKAN package.",
            "Reply `update <dataset-name>` if this should update an existing CKAN package.",
            "Provide any missing fields or corrections.",
            "Reply `validate` or `dry run` after the target choice is clear.",
        ]
    if status == "needs_dataset_intent":
        return [
            "Reply `new dataset` to create a new CKAN package.",
            "Reply `update <dataset-name>` to update an existing CKAN package.",
            "If one of the listed matches is correct, use its dataset name exactly.",
        ]
    if status == "needs_existing_dataset_choice":
        return [
            "Choose one listed dataset with `update <dataset-name>`.",
            "Reply `new dataset` if none of the listed CKAN packages should be updated.",
        ]
    if status == "needs_dry_run":
        return [
            "Reply `validate` or `dry run` first.",
            "Review the CKAN diff before sending `REGISTER`.",
        ]
    if command == "dry-run" or status == "dry_run":
        return [
            "Review the CKAN changes and resource plan below.",
            "Send exact `REGISTER` to create/update the dataset and upload resources.",
            "Send corrections if any metadata or resource detail is wrong.",
        ]
    if command == "apply" or status == "applied":
        return [
            "Open the CKAN dataset URL and verify the package.",
            "Check the uploaded resources count and resource metadata.",
            "Send another dataset file/path to start a new registration.",
        ]
    if status == "needs_input":
        return [
            "Provide the requested missing information.",
            "Attach or reference readable files if the dataset resources were not available.",
        ]
    return [
        "Review the response below.",
        "Send corrections, add data, or ask to validate when ready.",
    ]


def _prepend_next_options(content: str, result: AgentRunResponse | None = None) -> str:
    if content.lstrip().startswith("## Next Options"):
        return content
    options = (
        _next_options_for_result(result)
        if result
        else ["Review the response below and send the next instruction."]
    )
    lines = ["## Next Options", *[f"- {option}" for option in options], "", content.strip()]
    return "\n".join(lines).strip()


def _chat_completion_content(
    payload: ChatCompletionRequest,
    request: Request,
    runner: CkanRegistrationRunner,
) -> str:
    logger.info(f"📝 /chat/completions endpoint called | Model: {payload.model}")
    metadata = dict(payload.metadata or {})
    message = _last_user_message(payload)
    logger.debug(f"   Message: {message[:100] if message else '(empty)'}")
    
    incoming_metadata_keys = sorted(str(key) for key in metadata if str(key).lower() != "request_headers")
    last_user_content_types: list[str] = []
    for item in reversed(payload.messages):
        if item.role == "user":
            last_user_content_types = _message_content_types(item.content)
            break
    file_refs, inline_files, attachment_names = _last_user_file_payloads(payload)
    conversation_context = _conversation_metadata_context(payload)
    previous_thread_id = ""
    thread_id_from_history = ""

    if not metadata.get("session_id") and not metadata.get("thread_id"):
        previous_thread_id = _conversation_thread_id(payload)
        if previous_thread_id:
            thread_id_from_history = previous_thread_id
            metadata["session_id"] = previous_thread_id
            logger.debug(f"   Found thread from history: {previous_thread_id}")
    
    if (
        _should_show_chat_intro(metadata, message)
        and not (file_refs or inline_files or attachment_names)
        and not conversation_context["has_prior_metadata_report"]
        and not previous_thread_id
    ):
        logger.debug("   → Showing chat intro")
        return _chat_intro_content()
    
    logger.debug("   Checking for session_id...")
    if not metadata.get("session_id") and not metadata.get("thread_id"):
        logger.debug("   No session_id, checking conversation history...")
        logger.debug("   Proceeding without thread - metadata report does not require an existing session")
    metadata.setdefault("message", message)
    metadata["conversation_context"] = conversation_context
    if file_refs:
        metadata["files"] = list(metadata.get("files") or []) + file_refs
    if inline_files:
        metadata["inline_files"] = list(metadata.get("inline_files") or []) + inline_files
    if attachment_names:
        metadata["attachment_filenames"] = list(metadata.get("attachment_filenames") or []) + attachment_names
    context = dict(metadata.get("agent_context") or {})
    context.update(
        {
            "interface": "openai_chat_compat",
            "incoming_metadata_keys": incoming_metadata_keys,
            "last_user_message_chars": len(message),
            "last_user_content_types": last_user_content_types,
            "thread_id_from_history": thread_id_from_history,
            "has_prior_metadata_report": conversation_context["has_prior_metadata_report"],
            "file_ref_count": len(file_refs),
            "inline_file_count": len(inline_files),
            "attachment_name_count": len(attachment_names),
        }
    )
    metadata["agent_context"] = context
    metadata["request_headers"] = merge_secret_headers(metadata.get("request_headers"), request)
    # If this conversation's thread is paused awaiting a clarification answer, deliver the
    # user's reply as a resume (so the persona clarify→propose loop continues) rather than
    # starting a fresh analyze.
    active_thread = str(metadata.get("session_id") or metadata.get("thread_id") or "").strip()
    pending_check = getattr(runner, "pending_interrupt", None)
    if active_thread and callable(pending_check) and pending_check(active_thread):
        logger.info(f"   Thread {active_thread} awaiting clarification → resume()")
        resume_request = CkanResumeRequest.model_validate(metadata)
        result = runner.resume(active_thread, resume_request)
    else:
        run_request = CkanRunRequest.model_validate(metadata)
        logger.info(f"   Calling runner.invoke() with action={run_request.action or 'none'}")
        result = runner.invoke(run_request)
    logger.info(f"   Runner returned with status={result.status}")
    if result.requires_action:
        # Format interrupt as human-readable message
        interrupt_msg = result.requires_action
        if isinstance(interrupt_msg, dict):
            message = interrupt_msg.get("message", "I need more information to proceed.")
            thread_id = interrupt_msg.get("thread_id") or result.thread_id
            return _prepend_next_options(f"{message}\n\nThread ID: `{thread_id}`", result)
        return _prepend_next_options(str(interrupt_msg), result)
    review_markdown = result.result.get("review_markdown")
    if result.status == "needs_input" or result.result.get("status") == "needs_input":
        if isinstance(review_markdown, str) and review_markdown.strip():
            return _prepend_next_options(review_markdown.strip(), result)
        message_text = result.result.get("message") or result.result.get("error") or result.error
        if message_text:
            return _prepend_next_options(str(message_text), result)
    if isinstance(review_markdown, str) and review_markdown.strip():
        status = result.status or result.command
        content = f"{review_markdown.strip()}\n\nThread ID: `{result.thread_id}`\nStatus: `{status}`"
        return _prepend_next_options(content, result)
    return _prepend_next_options(json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2), result)


@router.post("/chat/completions", response_model=None, include_in_schema=False)
@router.post("/v1/chat/completions", response_model=None, operation_id="createChatCompletion")
def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    runner: CkanRegistrationRunner = Depends(get_runner),
    _org_check: None = Depends(require_ckan_org_access),
) -> dict[str, Any] | StreamingResponse:
    print("🔴 ENDPOINT HANDLER CALLED", flush=True)
    logger.info("🔴 ENDPOINT HANDLER CALLED")
    content = _chat_completion_content(payload, request, runner)
    if payload.stream:
        return _chat_stream_response(payload.model, content)
    return _chat_response(payload.model, content)
