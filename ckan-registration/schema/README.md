# schema/

This directory holds the **proposed** SUBSIDE dataset scheming schema for
Capability C of the GAM discovery / TWDB enrichment feature.

## Files

| File | Purpose |
|---|---|
| `subside_dataset.proposed.yaml` | Proposed extended schema — **not yet deployed** |

## What this is

`subside_dataset.proposed.yaml` is the proposed extended version of the
`subside_dataset.yaml` that is **currently live** in the separate
`ckanext-dso_scheming` extension at
`modflow-suite/ckan-docker/src/ckanext-dso_scheming/ckanext/dso_scheming/subside_dataset.yaml`
and deployed to `ckan.tacc.utexas.edu`.

The proposed schema adds the missing fields from the 28-field SUBSIDE
metadata specification, adds controlled-vocabulary `choices` arrays to
structured fields, and resolves the field-placement decisions approved
in the design spec (2026-06-25):

- Moves `program_area`, `data_contact_email`, `caveats_usage`,
  `categories`, `primary_tags`, `secondary_tags`, `collection_method`,
  `quality_control_level`, and `spatial` from **resource level to dataset
  level** (OQ-1 resolved).
- Moves `mint_standard_variables` from **dataset level to resource level**
  (OQ-2 resolved).

## This file is NOT auto-applied

This YAML is for review and version control only.  It has no effect on the
live CKAN instance until the following gated steps are completed in order:

1. **Data migration approved and completed.** All existing `subside_dataset`
   entries on `ckan.tacc.utexas.edu` must have their classification fields
   migrated from resource level to dataset level, and all controlled-vocab
   values verified or corrected, **before** the schema is deployed.  This
   step is a hard gate — the PR must not merge until migration is verified
   complete.

2. **Explicit user approval for the external write.** Applying this schema
   requires editing `subside_dataset.yaml` in the `modflow-suite/ckan-docker`
   repo, committing, opening a PR, and redeploying the Docker stack.  All of
   these are external writes and require separate explicit approval at
   execution time per the project approval gates.

3. **Local schema validation.** Before the PR is opened, run
   `ckanext-scheming` schema validation locally (or against a staging
   instance) to confirm the YAML loads without errors.

4. **Redeploy `ckanext-dso_scheming`.** After the PR merges, the Docker
   stack must be rebuilt and redeployed to `ckan.tacc.utexas.edu`.

## How to apply (when approved)

Copy `subside_dataset.proposed.yaml` to replace
`subside_dataset.yaml` in the `modflow-suite/ckan-docker` repo, then follow
the rollout sequence in the design spec
(`docs/design/2026-06-25-tapis-gam-discovery-and-twdb-enrichment.md`,
Capability C4 section).

## Design spec reference

Full rationale, field descriptions, placement decisions, controlled-vocab
sources, and migration requirements are documented in:

`docs/design/2026-06-25-tapis-gam-discovery-and-twdb-enrichment.md`
— Capability C (sections C1 through C4).
