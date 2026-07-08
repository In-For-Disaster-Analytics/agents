---
name: ckan_registration.dry_run_summary
version: 1
inputs: changes, resource_changes, warnings
---

Summarize the CKAN dry-run result for review.

Clearly separate dataset metadata changes, resource creates, resource updates,
resource deletion candidates, and warnings. End by saying that CKAN will not be
modified unless the user approves with REGISTER.
