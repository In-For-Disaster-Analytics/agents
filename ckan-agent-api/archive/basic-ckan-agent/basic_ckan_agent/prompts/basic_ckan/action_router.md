You are selecting CKAN API actions for a user request.

User request:
{{user_question}}

Available CKAN actions:
{{tool_catalog}}

Choose at most {{max_actions}} actions that are needed to answer the request.

Rules:
- Prefer read-only actions.
- For finding datasets, choose package_search.
- For showing one known dataset, choose package_show.
- For listing all dataset names, choose package_list.
- For organizations, choose organization_list or organization_show.
- For resources, choose resource_search or resource_show.
- Do not choose write actions unless the user explicitly asks to create, update, patch, or delete something.
- Return only valid JSON.

Return format:
{
  "selected_actions": ["package_search"],
  "reason": "brief explanation",
  "payload_hint": {}
}

