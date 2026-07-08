from app.agents.ckan_registration.tools import openai_tool_schema_bundle


def test_openai_tool_schema_bundle_contains_safe_and_apply_tools() -> None:
    bundle = openai_tool_schema_bundle()
    names = {tool["function"]["name"] for tool in bundle["chat_completions_tools"]}

    assert "ckan_analyze" in names
    assert "ckan_dry_run" in names
    assert "ckan_apply" in names


def test_apply_tool_documents_register_guardrail() -> None:
    bundle = openai_tool_schema_bundle()
    apply_tool = next(tool for tool in bundle["chat_completions_tools"] if tool["function"]["name"] == "ckan_apply")

    assert "REGISTER" in apply_tool["function"]["description"]


def test_analyze_tool_documents_debug_trace() -> None:
    bundle = openai_tool_schema_bundle()
    analyze_tool = next(tool for tool in bundle["chat_completions_tools"] if tool["function"]["name"] == "ckan_analyze")

    assert "debug_trace" in analyze_tool["function"]["parameters"]["properties"]
