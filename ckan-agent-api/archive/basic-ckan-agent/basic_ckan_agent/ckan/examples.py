COMMON_PAYLOAD_EXAMPLES = """
Common examples:
- package_search: {"q":"flood","rows":5}
- package_show: {"id":"dataset-name-or-id"}
- resource_show: {"id":"resource-id"}
- organization_list: {"all_fields":true,"include_dataset_count":true}
- license_list: {}
- status_show: {}

For CKAN spatial metadata, use a stringified GeoJSON object, not a nested object.
""".strip()

