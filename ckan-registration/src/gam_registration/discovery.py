"""Local recursive GAM model discovery and manifest generation.

Given a root directory (``GAM_ROOT_DIR`` env var or explicit ``root_path``
argument), walk the filesystem tree, identify MODFLOW model directories by
the presence of ``.nam`` / ``.NAM`` namefiles, attempt bounding-box derivation
via a four-step fallback chain, and write a fresh JSON manifest.

Fallback chain per model:

1. **DIS via flopy** — parse the MODFLOW discretisation file; extract xoff,
   yoff, NROW, NCOL, DELR, DELC, EPSG.  Apply Texas-bounds + zero-origin
   guard.  Status: ``ok_from_dis``.  If DIS is absent, unreadable, or
   flopy is unavailable, fall through.  If the guard rejects the bbox,
   status is ``suspicious_dis`` and Step 2 is tried.

2. **Geodatabase (.gdb)** — find a ``.gdb`` directory under the package
   folder, enumerate its layers, prefer a boundary/grid/extent layer, reproject
   to WGS84 and compute total_bounds.  Apply Texas-bounds sanity check.
   Status: ``ok_from_gdb``.  Requires geopandas (lazy import); if unavailable
   or all gdb reads fail, fall through.

3. **Aquifer-boundary lookup** via :mod:`aquifer` — look up the model's
   aquifer(s) in ``gam_aquifer_map.json``; fetch the TWDB ArcGIS polygon
   and compute bbox as min/max over all coordinate pairs.
   Status: ``ok_from_aquifer``.

4. **No spatial** — all steps failed; spatial fields are ``null``.
   Status: ``failed_no_spatial``.

Single-package mode:

When a user points discovery at a **single GAM folder** (e.g.
``Yegua-Jackson_Aquifer_GAM/``) whose immediate children are generic component
names (``Geodatabase/``, ``Model File/``, etc.), auto-detect fires and treats
the entire root as ONE package rather than creating bogus per-component
packages.  The ``single_package`` parameter to
:func:`discover_gam_models_from_local` controls this:

- ``None`` (default) — auto-detect.
- ``True`` — force single-package mode.
- ``False`` — force collection mode.

Full set of ``bbox_derivation_status`` values:
``ok_from_dis`` | ``suspicious_dis`` | ``ok_from_gdb`` | ``ok_from_aquifer``
| ``failed_no_spatial``.

After auto-discovery, ``gam_manifest_overrides.json`` (if present alongside
this file) is merged: any field in the overrides for a ``package_id``
overwrites the auto-discovered value.

Usage::

    from pathlib import Path
    from discovery import discover_gam_models_from_local

    manifest = discover_gam_models_from_local(Path("/corral-repl/.../twdb_gam_collection"))
    # manifest is also written to <root_path>/twdb_gam_manifest_generated.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Texas bounding box guard constants
# ---------------------------------------------------------------------------

TEXAS_LON_MIN = -107.0
TEXAS_LON_MAX = -93.0
TEXAS_LAT_MIN = 25.0
TEXAS_LAT_MAX = 37.0

# Default output manifest filename written inside root_path
MANIFEST_FILENAME = "twdb_gam_manifest_generated.json"

# Curated aquifer map and manifest overrides live in data/ at the repo root.
# Path(__file__).parent = src/gam_registration/
# Path(__file__).parent.parents[1] = repo root (ckan-registration/)
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parents[1]
GAM_AQUIFER_MAP_PATH = _REPO_ROOT / "data" / "gam_aquifer_map.json"
GAM_MANIFEST_OVERRIDES_PATH = _REPO_ROOT / "data" / "gam_manifest_overrides.json"
CURATED_MODFLOW_MANIFEST_PATH = _REPO_ROOT / "data" / "twdb_gam_modflow_locations_with_bbox_strings.json"


# ---------------------------------------------------------------------------
# Slug / title helpers
# ---------------------------------------------------------------------------


def _slugify(value: str) -> str:
    """Convert a directory name to a CKAN-style slug (lowercase, hyphens)."""
    value = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _human_title(directory_name: str) -> str:
    """Convert a snake/Pascal/camel directory name to a human-readable title."""
    title = directory_name.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


# ---------------------------------------------------------------------------
# Model discovery helpers
# ---------------------------------------------------------------------------


def _find_namefile_dirs(root: Path) -> dict[Path, list[Path]]:
    """Walk *root* and return a mapping of directory → list of .nam files.

    A directory qualifies as a MODFLOW model directory if it contains at
    least one file whose suffix matches ``.nam`` (case-insensitive).

    Args:
        root: Root directory to walk recursively.

    Returns:
        Dict mapping each model directory (Path) to its list of namefile
        Paths.  Directories that contain no .nam files are excluded.
    """
    dir_to_namefiles: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".nam":
            parent = path.parent
            dir_to_namefiles.setdefault(parent, []).append(path)
    return dir_to_namefiles


def _top_level_package_dir(path: Path, root: Path) -> Path:
    """Return the immediate child of *root* that is an ancestor of *path*.

    Args:
        path: An absolute Path inside *root*.
        root: The root directory.

    Returns:
        The top-level subdirectory of *root* that contains *path*.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path
    parts = rel.parts
    if not parts:
        return root
    return root / parts[0]


# ---------------------------------------------------------------------------
# Single-package auto-detection helpers
# ---------------------------------------------------------------------------

# Generic component folder names that, when present as the ONLY top-level
# children of the root, indicate the root itself is a single GAM package
# rather than a collection of GAM packages.
_GENERIC_COMPONENT_NAMES: frozenset[str] = frozenset(
    {
        "geodatabase",
        "model file",
        "model files",
        "modflow",
        "modflow model",
        "modflow models",
        "gwvistas",
        "input",
        "inputs",
        "output",
        "outputs",
        "documentation",
        "docs",
        "report",
        "reports",
        "gis",
        "shapefiles",
        "shapefile",
        "calibration",
        "model",
        "models",
        "data",
        "gms",
        "mt3d",
        "grid",
    }
)


def _normalize_component_name(name: str) -> str:
    """Normalize a directory name for generic-component matching.

    Lowercases, replaces underscores and hyphens with spaces, and collapses
    runs of whitespace.
    """
    return re.sub(r"\s+", " ", name.replace("_", " ").replace("-", " ").lower()).strip()


def _is_single_package(
    pkg_dirs: list[Path],
    root: Path,
) -> bool:
    """Return True when every top-level package dir has a generic component name.

    Args:
        pkg_dirs: List of immediate children of *root* that contain .nam files.
        root: The root directory.

    Returns:
        True if ALL pkg_dirs are generic component names (and there is at least
        one), implying the root is itself a single GAM package.
    """
    if not pkg_dirs:
        return False
    for pkg_dir in pkg_dirs:
        normalized = _normalize_component_name(pkg_dir.name)
        if normalized not in _GENERIC_COMPONENT_NAMES:
            return False
    return True


# ---------------------------------------------------------------------------
# DIS bbox derivation via flopy (lazy import)
# ---------------------------------------------------------------------------


def _derive_bbox_from_dis(
    model_dir: Path,
    default_epsg: int | None = None,
) -> tuple[dict[str, float], dict[str, Any], str] | None:
    """Attempt to derive a WGS84 bbox from a MODFLOW DIS file via flopy.

    Imports flopy lazily so the module can be imported even when flopy is
    not installed (the caller falls through to the aquifer fallback).

    Args:
        model_dir: Directory containing the MODFLOW model files.
        default_epsg: EPSG code to assume when the model DIS has no CRS.
            Sourced from ``MODFLOW_DEFAULT_EPSG`` env var if None.

    Returns:
        ``(bbox_dict, geojson_polygon, status)`` on success where
        ``status`` is ``"ok_from_dis"``, or ``None`` if derivation fails
        or should fall through.  When the guard rejects the bbox, returns
        ``(None, None, "suspicious_dis")`` — caller checks for this sentinel.
    """
    try:
        import flopy  # noqa: PLC0415 — lazy import intentional
    except ImportError:
        logger.debug("flopy not installed; skipping DIS bbox derivation for %s", model_dir)
        return None

    # Locate the first .dis file in the model directory
    dis_files = list(model_dir.glob("*.dis")) + list(model_dir.glob("*.DIS"))
    if not dis_files:
        logger.debug("No .dis file found in %s; skipping DIS derivation.", model_dir)
        return None

    dis_path = dis_files[0]
    logger.debug("Attempting DIS bbox derivation from %s", dis_path)

    try:
        # Try flopy.modflow.Modflow.load — best-effort, ignore load errors
        mf = flopy.modflow.Modflow.load(
            str(dis_path.parent / (dis_path.stem + ".nam"))
            if (dis_path.parent / (dis_path.stem + ".nam")).exists()
            else str(dis_path),
            model_ws=str(model_dir),
            load_only=["dis"],
            forgive=True,
            verbose=False,
            check=False,
        )
        dis = mf.dis
    except Exception as exc:
        logger.debug("flopy load failed for %s: %s", model_dir, exc)
        return None

    if dis is None:
        logger.debug("flopy returned no DIS object for %s", model_dir)
        return None

    try:
        nrow = int(dis.nrow)
        ncol = int(dis.ncol)
        # flopy stores DELR/DELC as util2d arrays; get the raw array
        delr = dis.delr.array.flatten()
        delc = dis.delc.array.flatten()
        xoff = float(getattr(dis, "xoffset", 0.0) or 0.0)
        yoff = float(getattr(dis, "yoffset", 0.0) or 0.0)
        rotation = float(getattr(dis, "rotation", 0.0) or 0.0)

        # Model width / height in model units
        width = float(sum(delr[:ncol]))
        height = float(sum(delc[:nrow]))
    except Exception as exc:
        logger.debug("Error reading DIS attributes for %s: %s", model_dir, exc)
        return None

    # ------------------------------------------------------------------
    # Texas-bounds + zero-origin guard
    # ------------------------------------------------------------------

    # Guard 1: zero origin — almost certainly missing georeferencing data
    if xoff == 0.0 and yoff == 0.0:
        logger.info(
            "DIS for %s has xoff=0, yoff=0 — rejecting as suspicious.", model_dir
        )
        return (None, None, "suspicious_dis")  # type: ignore[return-value]

    # Attempt CRS conversion to WGS84
    epsg = default_epsg or _get_default_epsg()
    try:
        min_lon, min_lat, max_lon, max_lat = _project_to_wgs84(
            xoff, yoff, width, height, rotation, epsg
        )
    except Exception as exc:
        logger.info("CRS projection failed for %s: %s — rejecting DIS bbox.", model_dir, exc)
        return (None, None, "suspicious_dis")  # type: ignore[return-value]

    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    # Guard 2: center outside Texas bounding box
    if not (
        TEXAS_LON_MIN <= center_lon <= TEXAS_LON_MAX
        and TEXAS_LAT_MIN <= center_lat <= TEXAS_LAT_MAX
    ):
        logger.info(
            "DIS bbox center (%.4f, %.4f) for %s outside Texas bounds — "
            "rejecting as suspicious.",
            center_lon,
            center_lat,
            model_dir,
        )
        return (None, None, "suspicious_dis")  # type: ignore[return-value]

    bbox = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    geojson = _bbox_to_geojson_polygon(bbox)
    logger.info(
        "DIS bbox OK for %s: lon [%.4f, %.4f] lat [%.4f, %.4f]",
        model_dir,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
    )
    return bbox, geojson, "ok_from_dis"


def _get_default_epsg() -> int | None:
    """Return the EPSG from the ``MODFLOW_DEFAULT_EPSG`` env var, or None."""
    val = os.environ.get("MODFLOW_DEFAULT_EPSG", "").strip()
    if val.isdigit():
        return int(val)
    return None


def _project_to_wgs84(
    xoff: float,
    yoff: float,
    width: float,
    height: float,
    rotation: float,
    epsg: int | None,
) -> tuple[float, float, float, float]:
    """Project a model grid extent from native CRS to WGS84.

    If *epsg* is None or not a projected CRS (EPSG < 32000 heuristic),
    assume the coordinates are already in decimal degrees (a common but
    unreliable fallback).

    Args:
        xoff: Grid origin X (easting or longitude).
        yoff: Grid origin Y (northing or latitude).
        width: Grid width in CRS units.
        height: Grid height in CRS units.
        rotation: Grid rotation in degrees (counter-clockwise from east).
        epsg: EPSG code, or None.

    Returns:
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 decimal degrees.

    Raises:
        RuntimeError: If projection requires pyproj and it is not installed.
    """
    if epsg is not None and epsg >= 32000:
        # Projected CRS — attempt pyproj conversion
        try:
            from pyproj import Transformer  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"pyproj required for EPSG:{epsg} → WGS84 projection but is not installed."
            ) from exc

        transformer = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )

        # Compute four corners of the grid (rotation handled via rotation matrix)
        rad = math.radians(rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)

        corners_local = [
            (0.0, 0.0),
            (width, 0.0),
            (width, height),
            (0.0, height),
        ]
        lons: list[float] = []
        lats: list[float] = []
        for dx, dy in corners_local:
            x_rot = xoff + dx * cos_r - dy * sin_r
            y_rot = yoff + dx * sin_r + dy * cos_r
            lon, lat = transformer.transform(x_rot, y_rot)
            lons.append(lon)
            lats.append(lat)

        return min(lons), min(lats), max(lons), max(lats)

    # Assume already in WGS84 decimal degrees (xoff = lon, yoff = lat)
    min_lon = xoff
    min_lat = yoff
    max_lon = xoff + width
    max_lat = yoff + height
    return min_lon, min_lat, max_lon, max_lat


# ---------------------------------------------------------------------------
# Geodatabase (.gdb) bbox derivation via geopandas (lazy import)
# ---------------------------------------------------------------------------

# Layer name fragments that indicate a preferred boundary/extent layer.
_GDB_PREFERRED_LAYER_FRAGMENTS: tuple[str, ...] = (
    "grid",
    "boundary",
    "active",
    "domain",
    "model",
    "extent",
    "outline",
)


def _derive_bbox_from_geodatabase(
    search_root: Path,
) -> tuple[dict[str, float], dict[str, Any], str, str | None] | None:
    """Attempt to derive a WGS84 bbox from a ``.gdb`` geodatabase under *search_root*.

    Searches *search_root* recursively for directories whose name ends in
    ``.gdb``.  For the first usable geodatabase found, enumerates its layers
    and prefers one whose name matches (case-insensitively) any of: grid,
    boundary, active, domain, model, extent, outline.  If no preferred layer
    is found, unions the extents of all layers.  Reprojects to EPSG:4326 and
    applies the same Texas-bounds sanity check used elsewhere.

    Also captures the ORIGINAL layer CRS (before reprojection) and returns it
    as a coordinate-system authority string (e.g. ``"EPSG:32614"``).

    geopandas is imported lazily; if unavailable (or any other error occurs)
    the function logs and returns ``None`` so the caller falls through to the
    next bbox source.

    Args:
        search_root: Directory under which to search for ``.gdb`` directories.

    Returns:
        ``(bbox_dict, geojson_polygon, "ok_from_gdb", crs_str)`` on success,
        where *crs_str* is the original CRS authority code (e.g.
        ``"EPSG:32614"``) or ``None`` if the CRS could not be determined.
        Returns ``None`` if derivation is impossible or should be skipped.
    """
    try:
        import geopandas as gpd  # noqa: PLC0415 — lazy import intentional
    except ImportError:
        logger.debug(
            "geopandas not installed; skipping geodatabase bbox derivation under %s",
            search_root,
        )
        return None

    # Find all .gdb directories under search_root
    gdb_paths: list[Path] = []
    for candidate in search_root.rglob("*.gdb"):
        if candidate.is_dir():
            gdb_paths.append(candidate)
    # Also check direct children (some layouts have <root>/<name>.gdb)
    if not gdb_paths:
        for candidate in search_root.glob("**/*.gdb"):
            if candidate.is_dir():
                gdb_paths.append(candidate)

    if not gdb_paths:
        logger.debug("No .gdb directories found under %s; skipping geodatabase bbox.", search_root)
        return None

    gdb_path = gdb_paths[0]
    logger.info("Attempting geodatabase bbox derivation from %s", gdb_path)

    try:
        import fiona  # noqa: PLC0415
        layers = fiona.listlayers(str(gdb_path))
    except Exception as exc:
        logger.debug("Could not list layers in %s: %s", gdb_path, exc)
        return None

    if not layers:
        logger.debug("No layers found in %s", gdb_path)
        return None

    # Choose preferred layer
    preferred_layer: str | None = None
    for layer in layers:
        for fragment in _GDB_PREFERRED_LAYER_FRAGMENTS:
            if fragment in layer.lower():
                preferred_layer = layer
                break
        if preferred_layer:
            break

    if preferred_layer:
        layers_to_read = [preferred_layer]
        logger.debug("Using preferred layer '%s' from %s", preferred_layer, gdb_path)
    else:
        layers_to_read = list(layers)
        logger.debug(
            "No preferred layer found in %s; unioning extents of all %d layers",
            gdb_path,
            len(layers_to_read),
        )

    # Read and union bounds; capture CRS from the first layer with usable geometry.
    all_min_lons: list[float] = []
    all_min_lats: list[float] = []
    all_max_lons: list[float] = []
    all_max_lats: list[float] = []
    original_crs_str: str | None = None  # CRS of the first layer (before reprojection)

    for layer_name in layers_to_read:
        try:
            gdf = gpd.read_file(str(gdb_path), layer=layer_name)
            if gdf.empty or gdf.geometry is None or gdf.geometry.isna().all():
                logger.debug("Layer '%s' in %s is empty or has no geometry; skipping.", layer_name, gdb_path)
                continue
            # Capture original CRS from the first layer that has usable geometry.
            if original_crs_str is None and gdf.crs is not None:
                try:
                    epsg_code = gdf.crs.to_epsg()
                    if epsg_code is not None:
                        original_crs_str = f"EPSG:{epsg_code}"
                    else:
                        # Fall back to the authority string from the CRS object.
                        original_crs_str = gdf.crs.to_string() or None
                except Exception:
                    original_crs_str = None
                logger.debug(
                    "Captured original CRS '%s' from layer '%s' in %s",
                    original_crs_str,
                    layer_name,
                    gdb_path,
                )
            # Reproject to WGS84
            if gdf.crs is None:
                logger.debug("Layer '%s' has no CRS; assuming EPSG:4326.", layer_name)
            else:
                gdf = gdf.to_crs("EPSG:4326")
            minx, miny, maxx, maxy = gdf.total_bounds
            all_min_lons.append(float(minx))
            all_min_lats.append(float(miny))
            all_max_lons.append(float(maxx))
            all_max_lats.append(float(maxy))
        except Exception as exc:
            logger.debug("Error reading layer '%s' from %s: %s", layer_name, gdb_path, exc)
            continue

    if not all_min_lons:
        logger.info("No usable geometry found in %s; skipping geodatabase bbox.", gdb_path)
        return None

    min_lon = min(all_min_lons)
    min_lat = min(all_min_lats)
    max_lon = max(all_max_lons)
    max_lat = max(all_max_lats)

    # Texas-bounds sanity check
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    if not (
        TEXAS_LON_MIN <= center_lon <= TEXAS_LON_MAX
        and TEXAS_LAT_MIN <= center_lat <= TEXAS_LAT_MAX
    ):
        logger.info(
            "Geodatabase bbox center (%.4f, %.4f) from %s is outside Texas bounds; skipping.",
            center_lon,
            center_lat,
            gdb_path,
        )
        return None

    bbox = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    geojson = _bbox_to_geojson_polygon(bbox)
    logger.info(
        "Geodatabase bbox OK from %s: lon [%.4f, %.4f] lat [%.4f, %.4f] crs=%s",
        gdb_path,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        original_crs_str,
    )
    return bbox, geojson, "ok_from_gdb", original_crs_str


# ---------------------------------------------------------------------------
# Aquifer fallback
# ---------------------------------------------------------------------------


def _derive_bbox_from_aquifer(
    package_id: str,
    aquifer_map: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], str] | None:
    """Look up the aquifer bbox for *package_id* via the curated aquifer map.

    Args:
        package_id: The model's package_id string.
        aquifer_map: Loaded contents of ``gam_aquifer_map.json``.

    Returns:
        ``(bbox_dict, geojson_polygon, "ok_from_aquifer")`` on success,
        or ``None`` on failure (no map entry, null name, or network error).
    """
    from .aquifer import get_aquifer_bbox  # noqa: PLC0415 — avoid circular at module level

    entry = aquifer_map.get(package_id)
    if not entry:
        logger.info("No aquifer map entry for '%s'; cannot use aquifer fallback.", package_id)
        return None

    aquifer_name = entry.get("aquifer_name")
    kind = entry.get("aquifer_kind", "minor")

    if aquifer_name is None:
        logger.info(
            "Aquifer map entry for '%s' has null aquifer_name; skipping fallback.",
            package_id,
        )
        return None

    if kind not in ("major", "minor"):
        logger.info(
            "Aquifer map entry for '%s' has unrecognised kind '%s'; skipping fallback.",
            package_id,
            kind,
        )
        return None

    result = get_aquifer_bbox(aquifer_name, kind=kind)
    if result is None:
        return None

    bbox, geojson_polygon = result
    return bbox, geojson_polygon, "ok_from_aquifer"


# ---------------------------------------------------------------------------
# GeoJSON polygon helper
# ---------------------------------------------------------------------------


def _bbox_to_geojson_polygon(bbox: dict[str, float]) -> dict[str, Any]:
    """Build a GeoJSON Polygon dict from a ``{min_lon, min_lat, max_lon, max_lat}`` bbox."""
    min_lon = bbox["min_lon"]
    min_lat = bbox["min_lat"]
    max_lon = bbox["max_lon"]
    max_lat = bbox["max_lat"]
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]
        ],
    }


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------


def discover_gam_models_from_local(
    root_path: Path,
    *,
    single_package: bool | None = None,
    max_files: int = 50000,
    output_path: Path | None = None,
    aquifer_map_path: Path | None = None,
    overrides_path: Path | None = None,
    curated_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Recursively walk *root_path*, identify MODFLOW model directories, derive
    spatial metadata, and write a fresh manifest JSON file.

    This function always writes a **fresh** manifest; it does NOT read or merge
    any previously generated manifest.  Manually edited values
    (``twdb_page_url``, ``report_url``, corrected bboxes) must be supplied via
    *overrides_path* (``gam_manifest_overrides.json``) so they survive
    regeneration.

    The bbox derivation chain per model:

    1. Parse DIS via flopy (lazy import) + Texas-bounds guard →
       ``"ok_from_dis"`` or ``"suspicious_dis"`` (falls through).
    2. Geodatabase (.gdb) — ``_derive_bbox_from_geodatabase`` (lazy geopandas)
       → ``"ok_from_gdb"``.  Falls through if geopandas unavailable or no
       usable geometry found.
    3. Aquifer-boundary lookup via ``gam_aquifer_map.json`` →
       ``"ok_from_aquifer"``.
    4. Null spatial → ``"failed_no_spatial"``.

    **Single-package mode** (``single_package`` parameter):

    When *root_path* points at a single GAM folder (e.g.
    ``Yegua-Jackson_Aquifer_GAM/``) rather than a collection, its immediate
    children (``Geodatabase/``, ``Model File/``) are generic component names,
    not individual GAM packages.  Set ``single_package=True`` (or rely on
    auto-detection) to treat the entire root as ONE package; *root_path.name*
    becomes the ``package_id`` slug and all ``.nam`` directories are grouped
    under that single entry.

    - ``None`` (default) — auto-detect: if every top-level child containing
      ``.nam`` files has a generic component name (see
      :data:`_GENERIC_COMPONENT_NAMES`), treat root as one package.
    - ``True`` — force single-package mode regardless of child names.
    - ``False`` — force collection mode (original behaviour).

    Args:
        root_path: Absolute path to the GAM collection root directory.
            Defaults to the ``GAM_ROOT_DIR`` env var if not supplied.
        single_package: Override auto-detection of single-package mode.
            ``None`` = auto-detect, ``True`` = force single, ``False`` = force
            collection.
        max_files: Maximum number of files to enumerate (guards against
            extremely large trees).  Only used for the file-count sanity
            check; does not truncate discovery results.
        output_path: Where to write the generated manifest JSON.  Defaults to
            ``<root_path>/<MANIFEST_FILENAME>``.
        aquifer_map_path: Path to ``gam_aquifer_map.json``.  Defaults to
            ``<module_dir>/gam_aquifer_map.json``.
        overrides_path: Path to ``gam_manifest_overrides.json``.  Defaults to
            ``<module_dir>/gam_manifest_overrides.json``.
        curated_manifest_path: Path to the curated TWDB MODFLOW locations
            manifest (``twdb_gam_modflow_locations_with_bbox_strings.json``).
            Used to backfill ``twdb_page_url`` and ``report_url`` when discovery
            writes empty values.  Precedence: discovered value → curated-manifest
            backfill → overrides (overrides always win, applied last).
            Defaults to ``data/twdb_gam_modflow_locations_with_bbox_strings.json``
            at the repo root.  If the file is missing or unreadable the backfill
            step is silently skipped (a warning is logged).

    Returns:
        The complete manifest dict (same structure written to disk).

    Raises:
        FileNotFoundError: If *root_path* does not exist.
        ValueError: If *root_path* is not a directory.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"GAM root directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"GAM root path is not a directory: {root_path}")

    if output_path is None:
        output_path = root_path / MANIFEST_FILENAME

    # ------------------------------------------------------------------
    # Load curated aquifer map
    # ------------------------------------------------------------------
    _aquifer_map_path = aquifer_map_path or GAM_AQUIFER_MAP_PATH
    aquifer_map: dict[str, Any] = {}
    if _aquifer_map_path.exists():
        try:
            with _aquifer_map_path.open(encoding="utf-8") as fh:
                aquifer_map = json.load(fh)
            logger.info("Loaded aquifer map from %s (%d entries)", _aquifer_map_path, len(aquifer_map))
        except Exception as exc:
            logger.warning("Failed to load aquifer map from %s: %s", _aquifer_map_path, exc)
    else:
        logger.warning("Aquifer map not found at %s; aquifer fallback disabled.", _aquifer_map_path)

    default_epsg = _get_default_epsg()

    # ------------------------------------------------------------------
    # Walk the directory tree and find model directories
    # ------------------------------------------------------------------
    logger.info("Scanning %s for MODFLOW model directories ...", root_path)
    dir_to_namefiles = _find_namefile_dirs(root_path)
    logger.info("Found %d candidate model directories.", len(dir_to_namefiles))

    # Group by top-level package folder
    pkg_to_model_dirs: dict[Path, dict[Path, list[Path]]] = {}
    for model_dir, namefiles in dir_to_namefiles.items():
        pkg_dir = _top_level_package_dir(model_dir, root_path)
        pkg_to_model_dirs.setdefault(pkg_dir, {})[model_dir] = namefiles

    # ------------------------------------------------------------------
    # Single-package vs collection mode detection
    # ------------------------------------------------------------------
    top_level_pkg_dirs = list(pkg_to_model_dirs.keys())

    if single_package is True:
        _use_single_package = True
        logger.info(
            "Single-package mode: forced ON by caller (single_package=True)."
        )
    elif single_package is False:
        _use_single_package = False
        logger.info(
            "Collection mode: forced ON by caller (single_package=False)."
        )
    else:
        # Auto-detect: if every top-level pkg_dir has a generic component name,
        # treat root_path itself as the one package.
        _use_single_package = _is_single_package(top_level_pkg_dirs, root_path)
        if _use_single_package:
            normalized_names = [
                _normalize_component_name(d.name) for d in top_level_pkg_dirs
            ]
            logger.info(
                "Single-package mode AUTO-DETECTED: all top-level dirs are generic "
                "component names %s — treating root '%s' as one package.",
                normalized_names,
                root_path.name,
            )
        else:
            logger.info(
                "Collection mode AUTO-DETECTED: root '%s' has %d non-generic top-level "
                "package dirs.",
                root_path.name,
                len(top_level_pkg_dirs),
            )

    # In single-package mode, re-group ALL model dirs under the root itself.
    if _use_single_package:
        all_model_dirs: dict[Path, list[Path]] = {}
        for _pkg_model_dirs in pkg_to_model_dirs.values():
            all_model_dirs.update(_pkg_model_dirs)
        pkg_to_model_dirs = {root_path: all_model_dirs}

    models: list[dict[str, Any]] = []

    for pkg_dir in sorted(pkg_to_model_dirs.keys()):
        model_dirs = pkg_to_model_dirs[pkg_dir]

        if _use_single_package and pkg_dir == root_path:
            # The package IS the root directory itself.
            package_folder_name = root_path.name
            package_id = _slugify(root_path.name)
            title = _human_title(root_path.name)
            gdb_search_root = root_path
        else:
            package_folder_name = pkg_dir.name
            package_id = _slugify(package_folder_name)
            title = _human_title(package_folder_name)
            gdb_search_root = pkg_dir

        logger.info("Processing package: %s (%s)", package_id, pkg_dir)

        # Build modflow_model_directories list
        modflow_model_directories: list[dict[str, Any]] = []
        for model_dir in sorted(model_dirs.keys()):
            namefiles_in_dir = sorted(model_dirs[model_dir])
            try:
                rel_dir = str(model_dir.relative_to(root_path))
            except ValueError:
                rel_dir = str(model_dir)

            namefile_entries = []
            for nf in namefiles_in_dir:
                try:
                    rel_nf = str(nf.relative_to(root_path))
                except ValueError:
                    rel_nf = str(nf)
                namefile_entries.append(
                    {
                        "filename": nf.name,
                        "relative_path": rel_nf,
                        "absolute_path": str(nf),
                    }
                )

            modflow_model_directories.append(
                {
                    "relative_directory": rel_dir,
                    "absolute_directory": str(model_dir),
                    "namefiles": namefile_entries,
                }
            )

        # ------------------------------------------------------------------
        # Bbox derivation chain — four-step fallback
        # ------------------------------------------------------------------
        bbox_wgs84: dict[str, float] | None = None
        bbox_geojson: dict[str, Any] | None = None
        bbox_derivation_status = "failed_no_spatial"
        coordinate_system: str | None = None  # set when gdb step captures CRS

        # Step 1: DIS via flopy
        first_model_dir = sorted(model_dirs.keys())[0]
        dis_result = _derive_bbox_from_dis(first_model_dir, default_epsg=default_epsg)

        if dis_result is not None:
            maybe_bbox, maybe_geojson, dis_status = dis_result
            if dis_status == "ok_from_dis":
                bbox_wgs84 = maybe_bbox
                bbox_geojson = maybe_geojson
                bbox_derivation_status = "ok_from_dis"
            elif dis_status == "suspicious_dis":
                bbox_derivation_status = "suspicious_dis"
                logger.info(
                    "DIS bbox suspicious for '%s'; trying geodatabase fallback.", package_id
                )
                # Fall through to Step 2
        else:
            logger.debug(
                "DIS derivation returned None for '%s'; trying geodatabase fallback.", package_id
            )

        # Step 2: Geodatabase (.gdb) fallback
        # _derive_bbox_from_geodatabase returns a 4-tuple:
        #   (bbox, geojson, status, crs_str)
        if bbox_wgs84 is None:
            gdb_result = _derive_bbox_from_geodatabase(gdb_search_root)
            if gdb_result is not None:
                bbox_wgs84, bbox_geojson, bbox_derivation_status, coordinate_system = gdb_result
                logger.info(
                    "Geodatabase fallback succeeded for '%s': status=%s crs=%s",
                    package_id,
                    bbox_derivation_status,
                    coordinate_system,
                )
            else:
                logger.debug(
                    "Geodatabase derivation returned None for '%s'; trying aquifer fallback.",
                    package_id,
                )

        # Step 3: Aquifer-boundary fallback
        if bbox_wgs84 is None:
            aquifer_result = _derive_bbox_from_aquifer(package_id, aquifer_map)
            if aquifer_result is not None:
                bbox_wgs84, bbox_geojson, bbox_derivation_status = aquifer_result
                logger.info(
                    "Aquifer fallback succeeded for '%s': status=%s", package_id, bbox_derivation_status
                )
            else:
                # Step 4: No spatial
                bbox_derivation_status = "failed_no_spatial"
                logger.info(
                    "No spatial data available for '%s'; status=failed_no_spatial.", package_id
                )

        # Convert geojson dict to string for dataset_spatial (matches existing manifest schema)
        dataset_spatial: str | None = (
            json.dumps(bbox_geojson) if bbox_geojson is not None else None
        )
        boundary_bbox_geojson: str | None = (
            json.dumps(bbox_geojson) if bbox_geojson is not None else None
        )

        model_entry: dict[str, Any] = {
            "package_id": package_id,
            "package_folder": str(pkg_dir),
            "title": title,
            "modflow_model_directories": modflow_model_directories,
            "twdb_page_url": "",
            "report_url": "",
            "boundary_bbox_wgs84": bbox_wgs84,
            "boundary_bbox_geojson": boundary_bbox_geojson,
            "dataset_spatial": dataset_spatial,
            "bbox_derivation_status": bbox_derivation_status,
        }
        if coordinate_system is not None:
            model_entry["coordinate_system"] = coordinate_system
        models.append(model_entry)

    # ------------------------------------------------------------------
    # Curated-manifest backfill (twdb_page_url, report_url)
    # Precedence: discovered (empty) → curated-manifest backfill → overrides (next block)
    # ------------------------------------------------------------------
    _curated_manifest_path = curated_manifest_path or CURATED_MODFLOW_MANIFEST_PATH
    curated_lookup: dict[str, dict[str, str]] = {}
    if _curated_manifest_path.exists():
        try:
            with _curated_manifest_path.open(encoding="utf-8") as fh:
                curated_data = json.load(fh)
            # The curated manifest is structured as {"models": [...], ...}
            curated_models = curated_data.get("models", []) if isinstance(curated_data, dict) else []
            for entry in curated_models:
                pid = entry.get("package_id")
                if pid:
                    curated_lookup[pid] = {
                        "twdb_page_url": entry.get("twdb_page_url") or "",
                        "report_url": entry.get("report_url") or "",
                    }
            logger.info(
                "Loaded curated manifest from %s (%d entries) for URL backfill.",
                _curated_manifest_path,
                len(curated_lookup),
            )
        except Exception as exc:
            logger.warning(
                "Failed to load curated manifest from %s for URL backfill: %s",
                _curated_manifest_path,
                exc,
            )
    else:
        logger.warning(
            "Curated manifest not found at %s; twdb_page_url/report_url backfill disabled.",
            _curated_manifest_path,
        )

    if curated_lookup:
        for model in models:
            pid = model["package_id"]
            curated_entry = curated_lookup.get(pid)
            if curated_entry is None:
                continue
            for field in ("twdb_page_url", "report_url"):
                if not model.get(field) and curated_entry.get(field):
                    model[field] = curated_entry[field]
                    logger.info(
                        "Curated-manifest backfill for '%s': %s = %r",
                        pid,
                        field,
                        curated_entry[field],
                    )

    # ------------------------------------------------------------------
    # Merge overrides (final step — overrides take precedence)
    # ------------------------------------------------------------------
    _overrides_path = overrides_path or GAM_MANIFEST_OVERRIDES_PATH
    overrides: dict[str, Any] = {}
    if _overrides_path.exists():
        try:
            with _overrides_path.open(encoding="utf-8") as fh:
                overrides = json.load(fh)
            logger.info(
                "Loaded manifest overrides from %s (%d entries)", _overrides_path, len(overrides)
            )
        except Exception as exc:
            logger.warning(
                "Failed to load manifest overrides from %s: %s", _overrides_path, exc
            )

    for model in models:
        pid = model["package_id"]
        if pid in overrides:
            override_fields = overrides[pid]
            for key, value in override_fields.items():
                model[key] = value
                logger.debug("Override applied for '%s': %s = %r", pid, key, value)

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest: dict[str, Any] = {
        "generated_by": "discovery.discover_gam_models_from_local",
        "source_tree_root": str(root_path),
        "record_count": len(models),
        "models": models,
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info(
            "Manifest written to %s (%d models).", output_path, len(models)
        )
    except Exception as exc:
        logger.warning("Failed to write manifest to %s: %s", output_path, exc)

    return manifest
