from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from openai import BadRequestError

from basic_ckan_agent.ckan.constants import READ_ONLY_ACTIONS, WRITE_ACTIONS
from basic_ckan_agent.ckan.tools import build_tools_from_openapi
from basic_ckan_agent.files import build_file_tools, extract_file_paths
from basic_ckan_agent.llm.model import build_model
from basic_ckan_agent.llm.recovery import model_bad_request_recovery_message
from basic_ckan_agent.llm.router import expand_actions_for_task, select_relevant_actions
from basic_ckan_agent.logging_config import debug_print, logger
from basic_ckan_agent.openapi.catalog import build_operation_catalog
from basic_ckan_agent.openapi.spec import load_openapi_schema
from basic_ckan_agent.prompts import get_prompt_registry
from basic_ckan_agent.session.memory import memory_prompt
from basic_ckan_agent.session.task_planning import (
    file_metadata_plan_prompt,
    package_show_404_recovery_call,
    task_plan_prompt,
)
from basic_ckan_agent.settings import ckan_openapi_url, env_int


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass(frozen=True)
class ChatResult:
    answer: str
    messages: list[AnyMessage]


class ChatSession:
    def __init__(self, *, system_prompt: str | None = None, model_name: str | None = None) -> None:
        self.messages: list[AnyMessage] = []
        self.system_prompt = system_prompt
        self.model_name = model_name

    def ask(self, message: str) -> str:
        return self.ask_with_trace(message).answer

    def ask_with_trace(self, message: str) -> ChatResult:
        result = chat_turn(
            message,
            prior_messages=self.messages,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
        )
        self.messages = result.messages
        return result


@lru_cache
def get_spec() -> dict[str, Any]:
    return load_openapi_schema(ckan_openapi_url())


def build_graph_for_tools(selected_tools: list[StructuredTool], model_name: str | None = None):
    model_with_selected_tools = build_model(model_name).bind_tools(selected_tools)
    selected_tool_names = [tool.name for tool in selected_tools]

    def selected_chatbot(state: State) -> dict[str, list[AnyMessage]]:
        debug_print(
            "Chatbot input message count",
            {
                "count": len(state["messages"]),
                "last_message_type": state["messages"][-1].__class__.__name__
                if state["messages"]
                else None,
                "last_message_content": getattr(state["messages"][-1], "content", None)
                if state["messages"]
                else None,
                "selected_tools": selected_tool_names,
            },
        )

        recovery = package_show_404_recovery_call(state["messages"], selected_tool_names)
        if recovery:
            debug_print(
                "Recovering from package_show 404 with package_search",
                {
                    "content": recovery.content,
                    "tool_calls": recovery.tool_calls,
                },
            )
            return {"messages": [recovery]}

        try:
            response = model_with_selected_tools.invoke(state["messages"])
        except BadRequestError as exc:
            logger.debug("Model provider rejected function-calling response; using recovery text", exc_info=True)
            response = AIMessage(content=model_bad_request_recovery_message(exc, state["messages"]))

        debug_print(
            "Chatbot model response",
            {
                "response_type": response.__class__.__name__,
                "content": getattr(response, "content", None),
                "tool_calls": getattr(response, "tool_calls", None),
            },
        )
        return {"messages": [response]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", selected_chatbot)
    graph_builder.add_node("tools", ToolNode(selected_tools))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    return graph_builder.compile()


def chat_once(message: str) -> str:
    return chat_turn(message).answer


def chat_turn(
    message: str,
    prior_messages: list[AnyMessage] | None = None,
    *,
    system_prompt: str | None = None,
    model_name: str | None = None,
) -> ChatResult:
    logger.info("USER: %s", message)
    prior_messages = prior_messages or []

    spec = get_spec()
    tool_catalog = build_operation_catalog(
        spec,
        read_only_actions=READ_ONLY_ACTIONS,
        write_actions=WRITE_ACTIONS,
    )

    memory = memory_prompt(prior_messages)
    routing_question = message
    if memory:
        routing_question = f"{memory.content}\n\nCurrent user request:\n{message}"

    selected_actions = select_relevant_actions(
        user_question=routing_question,
        tool_catalog=tool_catalog,
        max_actions=3,
        model_name=model_name,
    )
    selected_actions = expand_actions_for_task(message, selected_actions)
    write_approved = "APPROVE WRITE" in message.upper()

    selected_tools = build_tools_from_openapi(
        spec,
        allowed_actions=set(selected_actions),
        write_approved=write_approved,
    )
    if not selected_tools:
        logger.warning(
            "No selected tools built for actions=%s; falling back to package_search",
            selected_actions,
        )
        selected_tools = build_tools_from_openapi(spec, allowed_actions={"package_search"})

    file_paths = extract_file_paths(message)
    if file_paths:
        selected_tools.extend(build_file_tools(allowed_paths=file_paths))

    logger.info(
        "TURN selected_tools=%s file_paths=%s",
        [tool.name for tool in selected_tools],
        file_paths,
    )
    turn_graph = build_graph_for_tools(selected_tools, model_name=model_name)

    system_text = system_prompt or get_prompt_registry().load("basic_ckan", "system").text
    input_messages: list[Any] = [SystemMessage(content=system_text)]
    if memory:
        input_messages.append(memory)
    file_prompt = file_metadata_plan_prompt(file_paths)
    if file_prompt:
        input_messages.append(file_prompt)
    plan_prompt = task_plan_prompt(message, selected_actions)
    if plan_prompt:
        input_messages.append(plan_prompt)
    input_messages.append(("user", message))

    inputs = {"messages": input_messages}

    # Bound the chatbot<->tools loop so a misbehaving turn cannot spin forever
    # (each step is one LLM call). LangGraph raises GraphRecursionError at the cap.
    recursion_limit = env_int("CKAN_AGENT_RECURSION_LIMIT", 12)

    final_result = None
    try:
        for step in turn_graph.stream(
            inputs, stream_mode="values", config={"recursion_limit": recursion_limit}
        ):
            final_result = step
            messages = step.get("messages", [])
            if not messages:
                continue

            last = messages[-1]
            debug_print(
                "Graph step last message",
                {
                    "message_type": last.__class__.__name__,
                    "content": getattr(last, "content", None),
                    "tool_calls": getattr(last, "tool_calls", None),
                    "name": getattr(last, "name", None),
                    "tool_call_id": getattr(last, "tool_call_id", None),
                },
            )
    except GraphRecursionError:
        logger.warning("Turn hit recursion_limit=%s; stopping the tool loop.", recursion_limit)
        if final_result and final_result.get("messages"):
            answer = (
                f"Stopped after reaching the {recursion_limit}-step tool limit without a final answer."
            )
            messages = _conversation_messages(prior_messages, final_result["messages"])
            return ChatResult(answer=answer, messages=messages)
        return ChatResult(
            answer=f"Stopped after reaching the {recursion_limit}-step tool limit.",
            messages=prior_messages,
        )

    if not final_result:
        logger.info("ASSISTANT: No result returned.")
        return ChatResult(answer="No result returned.", messages=prior_messages)

    answer = str(final_result["messages"][-1].content)
    logger.info("ASSISTANT: %s", answer)
    messages = _conversation_messages(prior_messages, final_result["messages"])
    return ChatResult(answer=answer, messages=messages)


def _conversation_messages(
    prior_messages: list[AnyMessage],
    turn_messages: list[AnyMessage],
) -> list[AnyMessage]:
    current = [
        message
        for message in turn_messages
        if message.__class__.__name__ != "SystemMessage"
    ]
    return [*prior_messages, *current]
