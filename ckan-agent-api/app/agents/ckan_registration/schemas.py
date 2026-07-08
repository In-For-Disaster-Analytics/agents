from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ActionName = Literal["analyze", "revise", "dry-run", "apply", "show"]


class FileReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str = Field(..., description="Path to a file readable by the API service.")
    name: str | None = Field(default=None, description="Optional display name for the CKAN resource.")
    description: str | None = Field(default=None, description="Optional description or notes for the resource.")


class CkanConnectionOverride(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str | None = Field(default=None, description="CKAN base URL.")
    owner_org: str | None = Field(default=None, description="CKAN organization name or id that should own the dataset.")


class CkanDatasetOverride(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, description="CKAN dataset slug.")
    title: str | None = Field(default=None, description="Human-readable CKAN dataset title.")
    notes: str | None = Field(default=None, description="CKAN dataset description.")
    url: str | None = Field(default=None, description="Canonical source or project URL.")
    owner_org: str | None = Field(default=None, description="CKAN owner organization name or id.")
    private: bool | None = Field(default=None, description="Whether the CKAN dataset should be private.")
    author: str | None = None
    author_email: str | None = None
    maintainer: str | None = None
    maintainer_email: str | None = None
    license_id: str | None = None
    version: str | None = None
    type: str | None = None
    isopen: bool | None = None
    spatial: str | None = None
    temporal_coverage_start: str | None = None
    temporal_coverage_end: str | None = None
    tags: list[str] | None = Field(default=None, description="CKAN tag names.")


class CkanRegistrationBaseInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str | None = Field(default=None, description="Stable id for a resumable registration thread.")
    message: str | None = Field(default=None, description="User request or free-form registration instructions.")
    upload_dir: str | None = Field(default=None, description="Directory of files readable by the API service.")
    upload_dirs: list[str] | None = Field(default=None, description="Directories of files readable by the API service.")
    files: list[FileReference | str] | None = Field(default=None, description="Specific files to register.")
    source_url: str | None = Field(default=None, description="Primary source metadata URL.")
    source_urls: list[str] | None = Field(default=None, description="Source metadata URLs.")
    existing_ckan_entry: str | None = Field(default=None, description="Existing CKAN dataset name, id, or URL to update.")
    dataset: CkanDatasetOverride | None = Field(default=None, description="Dataset field overrides.")
    ckan: CkanConnectionOverride | None = Field(default=None, description="CKAN connection overrides.")
    use_llm: bool | None = Field(default=True, description="Whether analyze may call the configured LLM.")
    no_llm: bool | None = Field(default=None, description="Compatibility flag. If true, skips LLM metadata proposal.")
    allow_metadata_only: bool | None = Field(
        default=None,
        description="If true, allow analyze to create a metadata-only proposal without files, source URLs, or an existing CKAN entry.",
    )
    debug_trace: bool | None = Field(
        default=None,
        description=(
            "If true, include the structured audit/debug trace in the action response. "
            "This is not raw private model reasoning; it records observable inputs, branches, and rationale summaries."
        ),
    )
    upload_resources: bool | None = Field(default=None, description="Whether apply should upload or update resource files.")
    remove_stale_resources: bool | None = Field(default=None, description="Whether apply should delete CKAN resources absent locally.")
    resource_extra_fields: list[str] | None = Field(default=None, description="Extra resource fields to pass through to CKAN.")
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="Per-request secret headers forwarded by an API gateway. These are never saved to agent state.",
    )


class CkanAnalyzeInput(CkanRegistrationBaseInput):
    action: Literal["analyze"] = "analyze"


class CkanReviseInput(CkanRegistrationBaseInput):
    action: Literal["revise"] = "revise"
    exclude_resources: list[str] | None = Field(
        default=None,
        description="Resource names, relative paths, or local paths to remove from the saved plan.",
    )


class CkanDryRunInput(CkanRegistrationBaseInput):
    action: Literal["dry-run"] = "dry-run"


class CkanApplyInput(CkanRegistrationBaseInput):
    action: Literal["apply"] = "apply"
    approval: str | None = Field(default=None, description="Must be exactly REGISTER before CKAN writes are allowed.")
    delete_approval: str | None = Field(
        default=None,
        description="Must be exactly DELETE_STALE_RESOURCES before stale CKAN resources are deleted.",
    )


class CkanShowInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: Literal["show"] = "show"
    session_id: str | None = Field(default=None, description="Saved session id to inspect.")
    state_path: str | None = Field(default=None, description="Explicit saved state path to inspect.")
    request_headers: dict[str, str] | None = Field(default=None, description="Per-request secret headers; never saved.")


class CkanRunRequest(CkanRegistrationBaseInput):
    action: ActionName | None = Field(default=None, description="Requested action. If omitted, the service infers it.")
    approval: str | None = None
    delete_approval: str | None = None
    exclude_resources: list[str] | None = None


class CkanResumeRequest(CkanRunRequest):
    pass


class ToolInvokeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments.")


class AgentRunResponse(BaseModel):
    ok: bool
    thread_id: str
    command: str | None = None
    status: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    requires_action: dict[str, Any] | None = None
    error: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="ckan-registration-agent")
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional CKAN registration request fields such as upload_dir, source_urls, dataset, or session_id.",
    )


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "ckan-agent-api"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
