# CKAN Agent Tool Diagrams

These diagrams cover the tools registered in `app/agents/ckan_registration/tools.py`.
All tools are exposed through `POST /v1/ckan-registration/tools/{tool_name}` and
execute through `ToolExecutor` into the legacy CKAN registration worker.

## Shared Tool Invocation

```mermaid
flowchart TD
    caller["OpenAI tool call or API client"]
    endpoint["POST /v1/ckan-registration/tools/{tool_name}"]
    lookup{"tool_name in TOOL_SPECS?"}
    mergeHeaders["Merge secret request headers"]
    validate["Validate arguments with tool Pydantic model"]
    executor["ToolExecutor.invoke"]
    command["Map tool to legacy command"]
    worker["LegacyCkanWorker.run"]
    result["Return worker result"]
    notFound["404 unknown CKAN registration tool"]

    caller --> endpoint --> lookup
    lookup -- "no" --> notFound
    lookup -- "yes" --> mergeHeaders --> validate --> executor --> command --> worker --> result
```

## `ckan_analyze`

Builds proposed CKAN dataset metadata and a resource plan from staged files,
source URLs, or explicit dataset details. This is a safe-read tool.

```mermaid
flowchart TD
    start["ckan_analyze"]
    input["CkanAnalyzeInput"]
    sources{"Data source supplied?"}
    metadataOnly{"allow_metadata_only?"}
    needsInput["Return needs_input guidance"]
    llm{"LLM enabled and OPENAI_API_KEY set?"}
    deterministic["Create deterministic metadata guess"]
    guided["Request LLM metadata proposal"]
    merge["Merge grounded metadata and resource plan"]
    save["Save registration state"]
    report["Return metadata report, thread_id, next steps"]

    start --> input --> sources
    sources -- "files, upload_dir, source_url, or existing entry" --> llm
    sources -- "no" --> metadataOnly
    metadataOnly -- "no" --> needsInput
    metadataOnly -- "yes" --> llm
    llm -- "yes" --> guided --> merge
    llm -- "no" --> deterministic --> merge
    merge --> save --> report
```

## `ckan_revise`

Revises a saved CKAN registration proposal without writing to CKAN. This is a
safe-read tool.

```mermaid
flowchart TD
    start["ckan_revise"]
    input["CkanReviseInput"]
    state{"Saved session state found?"}
    load["Load existing proposal"]
    changes["Apply dataset updates and exclude_resources"]
    recompute["Recompute proposed package and resource plan"]
    save["Save revised state"]
    report["Return revised metadata report"]
    missing["Return missing state error"]

    start --> input --> state
    state -- "no" --> missing
    state -- "yes" --> load --> changes --> recompute --> save --> report
```

## `ckan_dry_run`

Compares the saved CKAN registration proposal against the target CKAN dataset
without writing changes. This is a safe-read tool.

```mermaid
flowchart TD
    start["ckan_dry_run"]
    input["CkanDryRunInput"]
    state{"Saved proposal found?"}
    target{"Target CKAN dataset known?"}
    resolve["Resolve CKAN URL, owner org, and dataset name"]
    fetch["Fetch current CKAN package/resources"]
    diff["Compare proposed package and resources"]
    save["Save dry-run state"]
    report["Return CKAN dry-run report"]
    missing["Return missing state or target guidance"]

    start --> input --> state
    state -- "no" --> missing
    state -- "yes" --> target
    target -- "no" --> missing
    target -- "yes" --> resolve --> fetch --> diff --> save --> report
```

## `ckan_apply`

Creates or patches the CKAN dataset and optionally uploads resources. This is
the mutating tool.

```mermaid
flowchart TD
    start["ckan_apply"]
    input["CkanApplyInput"]
    approval{"approval == REGISTER?"}
    state{"Saved dry-run/proposal state found?"}
    auth{"CKAN credentials available?"}
    stale{"remove_stale_resources?"}
    staleApproval{"delete_approval == DELETE_STALE_RESOURCES?"}
    packageWrite["Create or patch CKAN package"]
    resources["Upload or update resources when enabled"]
    deleteStale["Delete stale CKAN resources"]
    save["Save apply result state"]
    report["Return registration result"]
    blocked["Return guardrail error"]

    start --> input --> approval
    approval -- "no" --> blocked
    approval -- "yes" --> state
    state -- "no" --> blocked
    state -- "yes" --> auth
    auth -- "no" --> blocked
    auth -- "yes" --> stale
    stale -- "no" --> packageWrite
    stale -- "yes" --> staleApproval
    staleApproval -- "no" --> packageWrite
    staleApproval -- "yes" --> deleteStale --> packageWrite
    packageWrite --> resources --> save --> report
```

## `ckan_show`

Returns saved CKAN registration session state for debugging or review. This is a
safe-read tool.

```mermaid
flowchart TD
    start["ckan_show"]
    input["CkanShowInput"]
    locator{"session_id or state_path supplied?"}
    load["Load saved state JSON"]
    sanitize["Keep request headers and secrets out of persisted state"]
    report["Return saved session state"]
    missing["Return missing state guidance"]

    start --> input --> locator
    locator -- "no" --> missing
    locator -- "yes" --> load
    load --> sanitize --> report
```
