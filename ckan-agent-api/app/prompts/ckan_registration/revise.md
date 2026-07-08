---
name: ckan_registration.revise
version: 1
inputs: existing_state, requested_edits
---

Revise the saved CKAN registration proposal according to the requested edits.

Preserve existing proposal fields unless the user explicitly asks to change
them. Resource exclusions should remove only exact resource names, relative
paths, or local paths supplied by the user.
