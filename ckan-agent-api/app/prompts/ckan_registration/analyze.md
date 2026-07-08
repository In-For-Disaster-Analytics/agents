---
name: ckan_registration.analyze
version: 1
inputs: message, source_urls, file_inventory, dataset_overrides
---

Analyze the supplied message, source URLs, file inventory, and dataset
overrides. Produce CKAN-ready dataset metadata and a resource plan.

Use only evidence from the user request, source metadata, file names, file
previews, and explicit overrides. When a field is unknown, leave it empty rather
than inventing values.

Do not use follow-up clarification text as metadata. In particular, phrases like
"I do not have an existing dataset" mean the user wants a new CKAN dataset; they
are not a dataset title or notes.

If the user mentions a notebook, script, CSV, attachment, or base data but no
readable file inventory or upload directory is present, stop and ask for those
inputs. Do not create a metadata-only proposal unless the user provides explicit
dataset details or opts into metadata-only registration.

Expected output is structured JSON matching the CKAN registration response
schema.
