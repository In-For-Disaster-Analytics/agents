"""gam_registration — TWDB GAM CKAN registration pipeline."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gam-registration")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "aquifer",
    "discovery",
    "pdf_extract",
    "resource_review",
    "twdb_enrich",
    "subside_mapping",
    "persona_loop",
    "orchestrate",
    "utils",
    "ckan_agent",
]
