from __future__ import annotations

import json
from typing import Any


def select_relevant_actions(
    *,
    user_question: str,
    tool_catalog: list[dict[str, Any]],
    max_actions: int = 3,
    model_name: str | None = None,
) -> list[str]:
    from basic_ckan_agent.llm.model import build_model
    from basic_ckan_agent.logging_config import logger
    from basic_ckan_agent.prompts import get_prompt_registry
    from basic_ckan_agent.utils import parse_router_json

    prompt = get_prompt_registry().load("basic_ckan", "action_router").render(
        user_question=user_question,
        tool_catalog=json.dumps(tool_catalog, indent=2, ensure_ascii=False),
        max_actions=max_actions,
    )

    response = build_model(model_name).invoke([("user", prompt)])
    text = str(response.content)

    try:
        parsed = parse_router_json(text)
    except json.JSONDecodeError:
        logger.warning("Router returned non-JSON: %s", text)
        return ["package_search"]

    selected = parsed.get("selected_actions", [])
    if not isinstance(selected, list):
        return ["package_search"]

    selected = [action for action in selected if isinstance(action, str)]
    if not selected:
        selected = ["package_search"]

    logger.info("ROUTER selected_actions=%s reason=%s", selected, parsed.get("reason"))
    logger.debug("ROUTER raw=%s", text)
    return selected[:max_actions]


def expand_actions_for_task(user_question: str, selected: list[str]) -> list[str]:
    q = user_question.lower()
    actions = set(selected)

    if any(word in q for word in ["edit", "update", "modify", "change", "patch", "remove", "rename"]):
        actions.update(
            {
                "package_search",
                "package_show",
                "package_patch",
                "package_update",
            }
        )

    if "approve write" in q:
        actions.update({"package_show", "package_patch", "package_update"})

    if any(word in q for word in ["resource", "file", "download"]):
        actions.update(
            {
                "resource_search",
                "resource_show",
            }
        )

    return list(actions)
