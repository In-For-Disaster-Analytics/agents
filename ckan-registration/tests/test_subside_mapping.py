"""Tests for subside_mapping.py — SUBSIDE dataset field mapping.

Built to the LIVE 15-column subside_dataset schema on ckan.tacc.utexas.edu.
Extra SUBSIDE fields (categories, collection_method, spatial, program_area,
caveats_usage, primary_tags, secondary_tags, quality_control_level,
data_contact_email, and others) now appear in package["extras"] as
{"key": ..., "value": ...} entries — NOT as top-level package keys.

mint_standard_variables is a DATASET COLUMN (live schema, multiple_text),
set from the mint_vars parameter — NOT a resource-level field.

Tests cover:
  - map_to_subside_dataset: type field, GAM defaults (in extras), field key
    mapping, spatial in extras as JSON string, mint_vars as dataset column,
    owner_org, tags, temporal coverage, null/empty field omission, extras
    structure, no unexpected top-level custom columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure src/ is on path so gam_registration package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.subside_mapping as subside_mapping  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _minimal_proposed(overrides: dict | None = None) -> dict:
    """Return a minimal proposed metadata dict."""
    base = {
        "dataset_title": "Blossom Aquifer Groundwater Availability Model",
        "dataset_notes": "Model files for the Blossom Aquifer GAM published by TWDB.",
        "dataset_name": "blossom-aquifer-gam",
        "dataset_url": "https://www.twdb.texas.gov/groundwater/models/gam/blsm/blsm.asp",
        "dataset_author": "Texas Water Development Board",
        "dataset_license_id": "notspecified",
        "temporal_coverage_start": "1980",
        "temporal_coverage_end": "2005",
        "dataset_tags": ["groundwater", "gam", "aquifer", "modflow"],
    }
    if overrides:
        base.update(overrides)
    return base


def _extras_dict(pkg: dict) -> dict[str, str]:
    """Return a key→value mapping of the package's extras list."""
    return {e["key"]: e["value"] for e in pkg.get("extras", [])}


# ===========================================================================
# type field
# ===========================================================================

class TestPackageType:
    def test_type_is_subside_dataset(self):
        """map_to_subside_dataset always sets type = 'subside_dataset'."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg["type"] == "subside_dataset"

    def test_type_cannot_be_overridden_by_proposed(self):
        """Proposed metadata cannot override the type field."""
        proposed = _minimal_proposed({"type": "dataset", "dataset_type": "dataset"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        assert pkg["type"] == "subside_dataset"


# ===========================================================================
# GAM defaults land in extras
# ===========================================================================

class TestGamDefaults:
    def test_collection_method_default_in_extras(self):
        """collection_method defaults to 'Model Output' and appears in extras."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        extras = _extras_dict(pkg)
        assert extras.get("collection_method") == "Model Output"
        # NOT a top-level key.
        assert "collection_method" not in pkg

    def test_categories_default_in_extras(self):
        """categories defaults to ['Groundwater'] and appears in extras as JSON string."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        extras = _extras_dict(pkg)
        cats_raw = extras.get("categories")
        assert cats_raw is not None
        cats = json.loads(cats_raw)
        assert "Groundwater" in cats
        # NOT a top-level key.
        assert "categories" not in pkg

    def test_proposed_collection_method_overrides_default(self):
        """If proposed contains collection_method, it is used instead of the default."""
        proposed = _minimal_proposed({"collection_method": "Survey"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("collection_method") == "Survey"

    def test_proposed_categories_override_default(self):
        """If proposed contains categories, they override the default."""
        proposed = _minimal_proposed({"categories": ["Water Quality", "Groundwater"]})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        cats = json.loads(extras.get("categories", "[]"))
        assert "Water Quality" in cats

    def test_custom_gam_defaults_parameter(self):
        """gam_defaults parameter overrides module-level defaults."""
        pkg = subside_mapping.map_to_subside_dataset(
            _minimal_proposed(),
            gam_defaults={"collection_method": "Digitization", "categories": ["Boundaries"]},
        )
        extras = _extras_dict(pkg)
        assert extras.get("collection_method") == "Digitization"
        cats = json.loads(extras.get("categories", "[]"))
        assert "Boundaries" in cats


# ===========================================================================
# Core CKAN column field mapping (top-level package keys)
# ===========================================================================

class TestFieldMapping:
    def test_title_mapped(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg["title"] == "Blossom Aquifer Groundwater Availability Model"

    def test_notes_mapped(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert "Blossom" in pkg["notes"]

    def test_name_mapped(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg["name"] == "blossom-aquifer-gam"

    def test_url_mapped(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg["url"] == "https://www.twdb.texas.gov/groundwater/models/gam/blsm/blsm.asp"

    def test_author_mapped(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg["author"] == "Texas Water Development Board"

    def test_license_id_mapped(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg["license_id"] == "notspecified"

    def test_temporal_coverage_mapped(self):
        # Bare years are normalized to full ISO dates for CKAN's date preset.
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        assert pkg.get("temporal_coverage_start") == "1980-01-01"
        assert pkg.get("temporal_coverage_end") == "2005-12-31"

    def test_tags_mapped_as_list_of_dicts(self):
        """dataset_tags list-of-strings is converted to list-of-dicts."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        tags = pkg.get("tags", [])
        assert isinstance(tags, list)
        assert all(isinstance(t, dict) for t in tags)
        tag_names = {t["name"] for t in tags}
        assert "groundwater" in tag_names

    def test_tags_list_of_dicts_passthrough(self):
        """Tags already in list-of-dicts format are preserved."""
        proposed = _minimal_proposed({"dataset_tags": [{"name": "gam"}, {"name": "modflow"}]})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        tag_names = {t["name"] for t in pkg.get("tags", [])}
        assert "gam" in tag_names
        assert "modflow" in tag_names

    def test_bare_field_names_also_accepted(self):
        """Bare field names (without 'dataset_' prefix) are also accepted."""
        proposed = {
            "title": "Test GAM",
            "notes": "Notes.",
            "name": "test-gam",
            "url": "https://example.com",
            "author": "TWDB",
        }
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        assert pkg["title"] == "Test GAM"
        assert pkg["notes"] == "Notes."


# ===========================================================================
# No unexpected top-level custom columns
# ===========================================================================

class TestNoUnexpectedTopLevelColumns:
    """Only the 15 live schema columns (+ type, extras, tags) may appear at the top level."""

    _ALLOWED_TOP_LEVEL = {
        "type", "name", "title", "notes", "tags", "license_id", "owner_org",
        "url", "version", "author", "author_email", "maintainer",
        "maintainer_email", "temporal_coverage_start", "temporal_coverage_end",
        "mint_standard_variables", "extras",
    }

    def test_no_custom_columns_at_top_level_minimal(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        unexpected = set(pkg.keys()) - self._ALLOWED_TOP_LEVEL
        assert not unexpected, f"Unexpected top-level keys: {unexpected}"

    def test_no_custom_columns_at_top_level_with_classification_fields(self):
        """Classification fields must NOT appear as top-level keys."""
        proposed = _minimal_proposed({
            "program_area": "Groundwater Management",
            "data_contact_email": "gam@twdb.texas.gov",
            "caveats_usage": "For research purposes only.",
            "primary_tags": "groundwater; aquifer",
            "secondary_tags": "modflow; texas",
            "quality_control_level": "Processed",
            "categories": ["Groundwater", "Surface Water"],
            "collection_method": "Model Output",
            "spatial": '{"type":"Point","coordinates":[0,0]}',
        })
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        unexpected = set(pkg.keys()) - self._ALLOWED_TOP_LEVEL
        assert not unexpected, f"Unexpected top-level keys: {unexpected}"


# ===========================================================================
# SUBSIDE classification fields appear in extras (not top-level)
# ===========================================================================

class TestExtrasClassificationFields:
    def test_program_area_in_extras(self):
        proposed = _minimal_proposed({"program_area": "Groundwater Management"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("program_area") == "Groundwater Management"
        assert "program_area" not in pkg

    def test_data_contact_email_in_extras(self):
        proposed = _minimal_proposed({"data_contact_email": "gam@twdb.texas.gov"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("data_contact_email") == "gam@twdb.texas.gov"
        assert "data_contact_email" not in pkg

    def test_caveats_usage_in_extras(self):
        proposed = _minimal_proposed({"caveats_usage": "For research purposes only."})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("caveats_usage") == "For research purposes only."
        assert "caveats_usage" not in pkg

    def test_quality_control_level_in_extras(self):
        proposed = _minimal_proposed({"quality_control_level": "Processed"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("quality_control_level") == "Processed"
        assert "quality_control_level" not in pkg

    def test_primary_tags_in_extras(self):
        proposed = _minimal_proposed({"primary_tags": "groundwater; aquifer"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("primary_tags") == "groundwater; aquifer"
        assert "primary_tags" not in pkg

    def test_secondary_tags_in_extras(self):
        proposed = _minimal_proposed({"secondary_tags": "modflow; texas"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("secondary_tags") == "modflow; texas"
        assert "secondary_tags" not in pkg

    def test_extras_is_list_of_dicts(self):
        """extras must be a list of {"key": ..., "value": ...} dicts."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        extras = pkg.get("extras", [])
        assert isinstance(extras, list)
        for entry in extras:
            assert isinstance(entry, dict)
            assert "key" in entry
            assert "value" in entry


# ===========================================================================
# Spatial field — in extras as JSON string
# ===========================================================================

class TestSpatialField:
    _GEOJSON_STR = '{"type":"Polygon","coordinates":[[[-94,31],[-94,32],[-95,32],[-95,31],[-94,31]]]}'
    _GEOJSON_DICT = {"type": "Polygon", "coordinates": [[[-94, 31], [-94, 32], [-95, 32], [-95, 31], [-94, 31]]]}

    def test_spatial_parameter_in_extras_as_json_string(self):
        """Explicit spatial parameter goes into extras as a JSON string."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed(), spatial=self._GEOJSON_STR)
        extras = _extras_dict(pkg)
        assert extras.get("spatial") == self._GEOJSON_STR
        # NOT a top-level key.
        assert "spatial" not in pkg

    def test_spatial_dict_encoded_as_json_string(self):
        """A dict spatial value is JSON-encoded in extras."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed(), spatial=self._GEOJSON_DICT)
        extras = _extras_dict(pkg)
        parsed = json.loads(extras["spatial"])
        assert parsed["type"] == "Polygon"

    def test_spatial_parameter_takes_precedence_over_proposed(self):
        """Explicit spatial parameter overrides any 'spatial' value in proposed."""
        proposed = _minimal_proposed({"spatial": "wrong geojson"})
        pkg = subside_mapping.map_to_subside_dataset(proposed, spatial=self._GEOJSON_STR)
        extras = _extras_dict(pkg)
        assert extras.get("spatial") == self._GEOJSON_STR

    def test_spatial_from_proposed_in_extras(self):
        """Uses 'spatial' from proposed when no explicit spatial parameter."""
        proposed = _minimal_proposed({"spatial": self._GEOJSON_STR})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("spatial") == self._GEOJSON_STR

    def test_dataset_spatial_key_also_accepted(self):
        """'dataset_spatial' key in proposed is also accepted."""
        proposed = _minimal_proposed({"dataset_spatial": "Texas"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert extras.get("spatial") == "Texas"

    def test_spatial_omitted_when_none_everywhere(self):
        """spatial is omitted from extras when not provided anywhere."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed())
        extras = _extras_dict(pkg)
        assert "spatial" not in extras
        assert "spatial" not in pkg


# ===========================================================================
# mint_standard_variables — DATASET COLUMN (live schema), NOT resource-level
# ===========================================================================

class TestMintVarsDatasetColumn:
    def test_mint_vars_placed_at_dataset_level(self):
        """mint_standard_variables IS in the package dict as a top-level list (live column)."""
        pkg = subside_mapping.map_to_subside_dataset(
            _minimal_proposed(),
            mint_vars=["groundwater__hydraulic_head", "aquifer_system__package_input_set"],
        )
        assert "mint_standard_variables" in pkg
        assert pkg["mint_standard_variables"] == [
            "groundwater__hydraulic_head",
            "aquifer_system__package_input_set",
        ]

    def test_mint_vars_omitted_when_none(self):
        """mint_standard_variables is omitted from the package when mint_vars is None."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed(), mint_vars=None)
        assert "mint_standard_variables" not in pkg

    def test_mint_vars_omitted_when_empty_list(self):
        """mint_standard_variables is omitted when mint_vars is an empty list."""
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed(), mint_vars=[])
        assert "mint_standard_variables" not in pkg

    def test_mint_vars_arg_accepted_without_error(self):
        """Passing mint_vars does not raise an error."""
        subside_mapping.map_to_subside_dataset(
            _minimal_proposed(),
            mint_vars=["some_variable"],
        )

    def test_mint_vars_not_in_extras(self):
        """mint_standard_variables must NOT appear in extras (it is a schema column)."""
        pkg = subside_mapping.map_to_subside_dataset(
            _minimal_proposed(),
            mint_vars=["groundwater__hydraulic_head"],
        )
        extras = _extras_dict(pkg)
        assert "mint_standard_variables" not in extras


# ===========================================================================
# owner_org
# ===========================================================================

class TestOwnerOrg:
    def test_owner_org_included_when_provided(self):
        pkg = subside_mapping.map_to_subside_dataset(
            _minimal_proposed(), owner_org="twdb-groundwater"
        )
        assert pkg.get("owner_org") == "twdb-groundwater"

    def test_owner_org_omitted_when_none(self):
        pkg = subside_mapping.map_to_subside_dataset(_minimal_proposed(), owner_org=None)
        assert "owner_org" not in pkg


# ===========================================================================
# Null / empty field omission
# ===========================================================================

class TestNullFieldOmission:
    def test_empty_proposed_still_has_type_and_defaults_in_extras(self):
        """Even an empty proposed dict produces type + GAM defaults in extras."""
        pkg = subside_mapping.map_to_subside_dataset({})
        assert pkg["type"] == "subside_dataset"
        extras = _extras_dict(pkg)
        assert extras.get("collection_method") == "Model Output"

    def test_null_values_in_proposed_are_omitted(self):
        """Fields explicitly set to None in proposed are omitted from the package."""
        proposed = _minimal_proposed({
            "dataset_author": None,
            "dataset_license_id": None,
            "temporal_coverage_start": None,
        })
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        assert "author" not in pkg
        assert "license_id" not in pkg
        assert "temporal_coverage_start" not in pkg

    def test_whitespace_only_values_omitted(self):
        """Fields with only whitespace are omitted from the package."""
        proposed = _minimal_proposed({"dataset_author": "   ", "dataset_notes": "\t\n"})
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        assert "author" not in pkg
        assert "notes" not in pkg

    def test_null_extra_fields_omitted_from_extras(self):
        """Extra fields set to None are omitted from the extras list."""
        proposed = _minimal_proposed({
            "program_area": None,
            "caveats_usage": None,
        })
        pkg = subside_mapping.map_to_subside_dataset(proposed)
        extras = _extras_dict(pkg)
        assert "program_area" not in extras
        assert "caveats_usage" not in extras


# ===========================================================================
# Deterministic output — calling twice with same input gives same output
# ===========================================================================

def test_deterministic_output():
    """map_to_subside_dataset is pure for the same inputs."""
    proposed = _minimal_proposed()
    pkg1 = subside_mapping.map_to_subside_dataset(proposed, spatial='{"type":"Point"}')
    pkg2 = subside_mapping.map_to_subside_dataset(proposed, spatial='{"type":"Point"}')
    assert pkg1 == pkg2


# ===========================================================================
# Temporal date normalization (CKAN date preset requires YYYY-MM-DD)
# ===========================================================================

class TestTemporalIsoNormalization:
    def test_bare_year_expands_start_and_end(self):
        import gam_registration.subside_mapping as sm
        assert sm._normalize_iso_date("1980", is_end=False) == "1980-01-01"
        assert sm._normalize_iso_date("1980", is_end=True) == "1980-12-31"

    def test_year_month_expands_to_month_bounds(self):
        import gam_registration.subside_mapping as sm
        assert sm._normalize_iso_date("1980-02", is_end=False) == "1980-02-01"
        assert sm._normalize_iso_date("1980-02", is_end=True) == "1980-02-29"  # 1980 is a leap year

    def test_full_iso_passthrough_and_datetime_trim(self):
        import gam_registration.subside_mapping as sm
        assert sm._normalize_iso_date("1995-06-15", is_end=False) == "1995-06-15"
        assert sm._normalize_iso_date("1995-06-15T00:00:00Z", is_end=True) == "1995-06-15"

    def test_range_in_one_field_uses_min_for_start_max_for_end(self):
        import gam_registration.subside_mapping as sm
        assert sm._normalize_iso_date("1980-1999", is_end=False) == "1980-01-01"
        assert sm._normalize_iso_date("1980-1999", is_end=True) == "1999-12-31"

    def test_unparseable_returns_none(self):
        import gam_registration.subside_mapping as sm
        assert sm._normalize_iso_date("steady-state", is_end=False) is None
        assert sm._normalize_iso_date("", is_end=True) is None
        assert sm._normalize_iso_date(None, is_end=False) is None

    def test_invalid_iso_month_falls_back_to_year(self):
        import gam_registration.subside_mapping as sm
        # 1980-13-40 is not a real date; falls back to the 4-digit year token.
        assert sm._normalize_iso_date("1980-13-40", is_end=False) == "1980-01-01"

    def test_package_temporal_columns_are_iso(self):
        import gam_registration.subside_mapping as sm
        pkg = sm.map_to_subside_dataset(
            {"dataset_name": "x", "dataset_title": "X", "dataset_notes": "n",
             "temporal_coverage_start": "1980", "temporal_coverage_end": "1999"},
        )
        assert pkg["temporal_coverage_start"] == "1980-01-01"
        assert pkg["temporal_coverage_end"] == "1999-12-31"

    def test_package_omits_unparseable_temporal(self):
        import gam_registration.subside_mapping as sm
        pkg = sm.map_to_subside_dataset(
            {"dataset_name": "x", "dataset_title": "X", "dataset_notes": "n",
             "temporal_coverage_start": "steady-state"},
        )
        assert "temporal_coverage_start" not in pkg
