---
name: ckan_registration.approval
version: 1
inputs: dry_run, warnings
---

CKAN writes require explicit approval.

Ask the user to review the dry-run output. The apply step may proceed only if
the approval value is exactly REGISTER. Stale resource deletion may proceed only
if delete_approval is exactly DELETE_STALE_RESOURCES.
