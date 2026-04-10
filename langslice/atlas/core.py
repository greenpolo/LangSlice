import importlib
import logging
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Protocol, cast

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from langslice.atlas.space import atlas_space_context, require_coronal_layout

logger = logging.getLogger(__name__)

DEFAULT_ATLAS_NAME = "allen_mouse_25um"

_ATLAS_ALIASES: dict[str, str] = {
    "whs_sd_rat": "whs_sd_rat_39um",
}


def species_from_atlas_name(atlas_name: str) -> str:
    """Extract a readable species name from a BrainGlobe atlas identifier."""
    lower = atlas_name.lower()
    for species in ("mouse", "rat", "human", "zebrafish", "fish"):
        if species in lower:
            return species
    return "animal"


class _AtlasLike(Protocol):
    atlas_name: str
    orientation: str
    reference: np.ndarray
    annotation: np.ndarray
    resolution: Sequence[float]
    metadata: dict[str, object]


def _shape3d(volume: np.ndarray) -> tuple[int, int, int]:
    shape = cast(tuple[int, ...], volume.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D atlas volume, got shape={shape}")
    return shape[0], shape[1], shape[2]


def _safe_index(axis_name: str, index: int, upper_bound: int) -> int:
    if index < 0 or index >= upper_bound:
        raise ValueError(f"{axis_name} index {index} out of range [0, {upper_bound - 1}]")
    return index


def _as_scalar_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if np.isscalar(value):
        try:
            return int(np.asarray(value).item())
        except (TypeError, ValueError):
            return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_version(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, tuple):
        tuple_value = cast(tuple[object, ...], raw)
        return ".".join(str(part) for part in tuple_value)
    return str(raw)


def _lookup_structure_record(atlas: _AtlasLike, structure_id: int) -> dict[str, str]:
    structures = getattr(atlas, "structures", None)
    if structures is not None:
        try:
            data = structures[structure_id]
            if isinstance(data, dict):
                return {
                    "acronym": str(data.get("acronym", "")),
                    "name": str(data.get("name", "")),
                }
        except Exception:
            pass

    lookup_df = getattr(atlas, "lookup_df", None)
    if lookup_df is not None:
        try:
            matches = lookup_df.loc[lookup_df["id"] == structure_id]
            if not getattr(matches, "empty", True):
                row = matches.iloc[0]
                return {
                    "acronym": str(row["acronym"]),
                    "name": str(row["name"]),
                }
        except Exception:
            pass

    return {"acronym": "", "name": ""}


class BrainGlobeAtlas(_AtlasLike, Protocol):
    pass


def canonicalize_atlas_name(name: str) -> str:
    """Normalize atlas identifiers and resolve legacy aliases."""
    cleaned = name.strip()
    if not cleaned:
        return DEFAULT_ATLAS_NAME
    return _ATLAS_ALIASES.get(cleaned, cleaned)


@lru_cache(maxsize=4)
def load_atlas(name: str) -> BrainGlobeAtlas:
    """Load and cache a BrainGlobe atlas. First call downloads if needed."""
    atlas_name = canonicalize_atlas_name(name)
    try:
        module = importlib.import_module("brainglobe_atlasapi")
        brain_globe_atlas = cast(Callable[[str], BrainGlobeAtlas], module.BrainGlobeAtlas)
        atlas = brain_globe_atlas(atlas_name)
    except Exception as exc:  # pragma: no cover - passthrough from external library
        raise ValueError(f"Atlas '{atlas_name}' not found or failed to load: {exc}") from exc
    return atlas


def position_mm_to_index(atlas: _AtlasLike, position_mm: float) -> int:
    """Convert a physical position (mm from anterior edge) to an array index."""
    context = require_coronal_layout(atlas_space_context(atlas))
    res_mm = context.ap_resolution_mm
    n_slices = context.shape[context.ap_axis_index]

    idx = int(round(position_mm / res_mm))
    if idx < 0 or idx >= n_slices:
        _, max_pos = get_position_range_mm(atlas)
        raise ValueError(
            f"Position {position_mm:.3f}mm maps to index {idx}, out of range [0, {n_slices - 1}]. "
            + f"Valid range for '{atlas.atlas_name}': 0.0mm to {max_pos:.3f}mm"
        )
    return idx


def index_to_position_mm(atlas: _AtlasLike, idx: int) -> float:
    """Convert an array index along axis 0 to a physical position in mm."""
    context = require_coronal_layout(atlas_space_context(atlas))
    return idx * context.ap_resolution_mm


def get_position_range_mm(atlas: _AtlasLike) -> tuple[float, float]:
    """Return physical position range as (0.0, max_mm)."""
    context = require_coronal_layout(atlas_space_context(atlas))
    n_slices = context.shape[context.ap_axis_index]
    return 0.0, (n_slices - 1) * context.ap_resolution_mm


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 [0, 255]."""
    if arr.size == 0:
        return arr.astype(np.uint8)

    arr_float = arr.astype(np.float32, copy=False)
    max_val = float(np.max(arr_float))
    if max_val <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr_float / max_val) * 255.0, 0, 255).astype(np.uint8)


def get_reference_slice(atlas: _AtlasLike, position_mm: float) -> Image.Image:
    """Get coronal reference slice as grayscale PIL image."""
    idx = position_mm_to_index(atlas, position_mm)
    reference_slice = np.asarray(atlas.reference[idx, :, :])
    normalized = _normalize_to_uint8(reference_slice)
    return Image.fromarray(normalized, mode="L")


def _annotation_to_boundaries(annotation_slice: np.ndarray) -> np.ndarray:
    """Extract binary region boundaries from an annotation slice."""
    h, w = cast(tuple[int, int], annotation_slice.shape)
    edges = np.zeros((h, w), dtype=np.uint8)

    dx = cast(np.ndarray, annotation_slice[:, 1:] != annotation_slice[:, :-1])
    dy = cast(np.ndarray, annotation_slice[1:, :] != annotation_slice[:-1, :])
    dx_u8 = np.zeros(cast(tuple[int, int], dx.shape), dtype=np.uint8)
    dy_u8 = np.zeros(cast(tuple[int, int], dy.shape), dtype=np.uint8)
    dx_u8[dx] = 255
    dy_u8[dy] = 255

    edges[:, 1:] |= dx_u8
    edges[:, :-1] |= dx_u8
    edges[1:, :] |= dy_u8
    edges[:-1, :] |= dy_u8
    return edges


def get_boundary_slice(atlas: _AtlasLike, position_mm: float) -> Image.Image:
    """Get coronal annotation boundaries as grayscale PIL image."""
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])
    edges = _annotation_to_boundaries(annotation_slice)
    return Image.fromarray(edges, mode="L")


def get_composite_slice(atlas: _AtlasLike, position_mm: float, opacity: float = 0.4) -> Image.Image:
    """Overlay annotation boundaries on reference image and return RGB PIL image."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}")

    idx = position_mm_to_index(atlas, position_mm)
    ref_slice = np.asarray(atlas.reference[idx, :, :])
    ref_norm = _normalize_to_uint8(ref_slice)
    ref_rgb = np.stack([ref_norm, ref_norm, ref_norm], axis=-1).astype(np.float32)

    ann_slice = np.asarray(atlas.annotation[idx, :, :])
    edges = _annotation_to_boundaries(ann_slice)

    result = ref_rgb.copy()
    edge_mask = edges > 0
    boundary_color = np.array([0.0, 255.0, 100.0], dtype=np.float32)
    result[edge_mask] = result[edge_mask] * (1.0 - opacity) + boundary_color * opacity

    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def get_colored_region_slice(atlas: _AtlasLike, position_mm: float) -> Image.Image:
    """Get coronal annotation slice as RGB PIL image with official atlas region colors."""
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])
    h, w = cast(tuple[int, int], annotation_slice.shape)

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    unique_ids = np.unique(annotation_slice)
    structures = getattr(atlas, "structures", None)

    for uid in unique_ids:
        uid_int = int(uid)
        if uid_int == 0:
            continue
        try:
            color = structures[uid_int]["rgb_triplet"]  # type: ignore[index]
            rgb[annotation_slice == uid] = color
        except Exception:
            pass

    return Image.fromarray(rgb, mode="RGB")


def get_smoothed_boundary_slice(
    atlas: _AtlasLike, position_mm: float, *, target_size: tuple[int, int] | None = None
) -> Image.Image:
    """Get coronal annotation boundaries with smoothed vector contours as grayscale PIL image."""
    idx = position_mm_to_index(atlas, position_mm)
    annotation_slice = np.asarray(atlas.annotation[idx, :, :])

    if target_size is not None:
        tw, th = target_size
        annotation_slice = cv2.resize(
            annotation_slice, (tw, th), interpolation=cv2.INTER_NEAREST
        )

    h, w = cast(tuple[int, int], annotation_slice.shape)
    canvas = np.zeros((h, w), dtype=np.uint8)

    unique_ids = np.unique(annotation_slice)
    for uid in unique_ids:
        uid_int = int(uid)
        if uid_int == 0:
            continue

        mask = np.where(annotation_slice == uid, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if len(contour) < 10:
                continue
            pts = contour.squeeze(axis=1).astype(np.float64)
            pts[:, 0] = gaussian_filter1d(pts[:, 0], sigma=3.0, mode="wrap")
            pts[:, 1] = gaussian_filter1d(pts[:, 1], sigma=3.0, mode="wrap")
            smoothed = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(canvas, [smoothed], -1, 255, 1, cv2.LINE_AA)

    return Image.fromarray(canvas, mode="L")


def list_additional_references(atlas: _AtlasLike) -> list[str]:
    """Return available secondary reference names for an atlas."""
    refs = getattr(atlas, "additional_references", None)
    if refs is None:
        return []
    if hasattr(refs, "keys"):
        return sorted(str(key) for key in refs.keys())
    return []


def get_additional_reference_slice(
    atlas: _AtlasLike, reference_name: str, position_mm: float
) -> Image.Image:
    """Get a coronal slice from an atlas additional reference volume."""
    refs = getattr(atlas, "additional_references", None)
    if refs is None:
        raise ValueError(f"Atlas '{atlas.atlas_name}' has no additional references")

    try:
        ref_volume = np.asarray(refs[reference_name])
    except Exception as exc:
        available = list_additional_references(atlas)
        raise ValueError(
            f"Unknown additional reference '{reference_name}'. Available: {available}"
        ) from exc

    _, _, _ = _shape3d(ref_volume)
    idx = position_mm_to_index(atlas, position_mm)
    if idx >= ref_volume.shape[0]:
        raise ValueError(
            f"Reference '{reference_name}' only has {ref_volume.shape[0]} AP slices, "
            + f"requested index {idx}"
        )

    slice_2d = np.asarray(ref_volume[idx, :, :])
    normalized = _normalize_to_uint8(slice_2d)
    return Image.fromarray(normalized, mode="L")


def get_structure_hierarchy(atlas: _AtlasLike, structure: int | str) -> dict[str, object]:
    """Return ancestor and descendant acronyms for a structure."""
    ancestors_fn = getattr(atlas, "get_structure_ancestors", None)
    descendants_fn = getattr(atlas, "get_structure_descendants", None)

    ancestors: list[str] = []
    descendants: list[str] = []

    if callable(ancestors_fn):
        try:
            raw = ancestors_fn(structure)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                ancestors = [str(item) for item in raw]
        except Exception:
            ancestors = []

    if callable(descendants_fn):
        try:
            raw = descendants_fn(structure)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                descendants = [str(item) for item in raw]
        except Exception:
            descendants = []

    return {
        "structure": structure,
        "ancestors": ancestors,
        "descendants": descendants,
        "n_ancestors": len(ancestors),
        "n_descendants": len(descendants),
    }


def get_structure_mask_slice(
    atlas: _AtlasLike, structure: int | str, position_mm: float
) -> Image.Image:
    """Get a binary coronal mask slice for one structure (including descendants)."""
    get_mask_fn = getattr(atlas, "get_structure_mask", None)
    if not callable(get_mask_fn):
        raise ValueError(
            "Atlas backend does not expose get_structure_mask(). "
            + "Requires brainglobe-atlasapi structure utilities."
        )

    mask_volume = np.asarray(get_mask_fn(structure))
    _, _, _ = _shape3d(mask_volume)

    idx = position_mm_to_index(atlas, position_mm)
    if idx >= mask_volume.shape[0]:
        raise ValueError(
            f"Structure mask has {mask_volume.shape[0]} AP slices, requested index {idx}"
        )

    slice_mask = np.asarray(mask_volume[idx, :, :])
    binary = np.where(slice_mask > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def get_slice_region_metadata(atlas: _AtlasLike, position_mm: float) -> list[dict[str, object]]:
    """Enumerate regions visible on a coronal slice with metadata for VLM prompts.

    Returns a list of region dicts sorted by area (largest first), each containing:
    ``id``, ``acronym``, ``name``, ``centroid`` (row, col pixels),
    ``area_fraction``, and ``centroid_normalized`` ([y, x] in 0–1000 range).
    Background (structure ID 0) is excluded.
    """
    idx = position_mm_to_index(atlas, position_mm)
    ann = np.asarray(atlas.annotation[idx, :, :])
    h, w = cast(tuple[int, int], ann.shape)

    unique_ids = np.unique(ann)
    unique_ids = unique_ids[unique_ids > 0]  # filter background

    regions: list[dict[str, object]] = []
    for uid in unique_ids:
        uid_int = int(uid)
        record = _lookup_structure_record(atlas, uid_int)
        mask = ann == uid
        ys, xs = np.where(mask)
        centroid_row = int(np.mean(ys))
        centroid_col = int(np.mean(xs))
        area_fraction = float(mask.sum()) / float(h * w)
        regions.append(
            {
                "id": uid_int,
                "acronym": record["acronym"],
                "name": record["name"],
                "centroid": (centroid_row, centroid_col),
                "area_fraction": area_fraction,
                "centroid_normalized": (
                    int(round(centroid_row * 1000 / max(h - 1, 1))),
                    int(round(centroid_col * 1000 / max(w - 1, 1))),
                ),
            }
        )

    regions.sort(key=lambda r: cast(float, r["area_fraction"]), reverse=True)
    return regions


def get_region_at_position(
    atlas: _AtlasLike,
    position_mm: float,
    *,
    dv_index: int | None = None,
    ml_index: int | None = None,
    include_hierarchy: bool = False,
) -> dict[str, Any]:
    """Resolve structure and hemisphere at a specific AP position and in-slice index."""
    ap_idx = position_mm_to_index(atlas, position_mm)
    _, n_dv, n_ml = _shape3d(atlas.reference)

    dv_idx = n_dv // 2 if dv_index is None else _safe_index("DV", dv_index, n_dv)
    ml_idx = n_ml // 2 if ml_index is None else _safe_index("ML", ml_index, n_ml)
    coords = (ap_idx, dv_idx, ml_idx)

    structure_id = _as_scalar_int(atlas.annotation[coords])
    if structure_id is None:
        structure_id = 0

    structure_fn = getattr(atlas, "structure_from_coords", None)
    acronym = ""
    if callable(structure_fn):
        try:
            sid = _as_scalar_int(structure_fn(coords, microns=False, as_acronym=False))
            if sid is not None:
                structure_id = sid
            acronym = str(
                structure_fn(
                    coords,
                    microns=False,
                    as_acronym=True,
                    key_error_string="Outside atlas",
                )
            )
        except Exception:
            pass

    record = _lookup_structure_record(atlas, structure_id)
    if not acronym:
        acronym = record["acronym"]

    hemisphere_fn = getattr(atlas, "hemisphere_from_coords", None)
    hemisphere = "unknown"
    if callable(hemisphere_fn):
        try:
            hemisphere = str(hemisphere_fn(coords, microns=False, as_string=True))
        except Exception:
            hemisphere = "unknown"

    hierarchy = None
    if include_hierarchy and structure_id > 0:
        hierarchy = get_structure_hierarchy(atlas, structure_id)

    return {
        "position_mm": float(position_mm),
        "indices": {
            "ap": ap_idx,
            "dv": dv_idx,
            "ml": ml_idx,
        },
        "hemisphere": hemisphere,
        "structure": {
            "id": structure_id,
            "acronym": acronym,
            "name": record["name"],
        },
        "hierarchy": hierarchy,
    }


def get_atlas_info(atlas: _AtlasLike) -> dict[str, object]:
    """Return atlas metadata and position range info."""
    min_pos, max_pos = get_position_range_mm(atlas)
    resolution_mm = [float(r) / 1000.0 for r in atlas.resolution]

    shape = cast(tuple[int, ...], atlas.reference.shape)

    check_latest_version = getattr(atlas, "check_latest_version", None)
    is_latest: bool | None = None
    if callable(check_latest_version):
        try:
            result = check_latest_version(print_warning=False)
            if isinstance(result, bool):
                is_latest = result
        except Exception:
            is_latest = None

    return {
        "name": atlas.atlas_name,
        "orientation": atlas.orientation,
        "shape": list(shape),
        "resolution_um": list(atlas.resolution),
        "resolution_mm": resolution_mm,
        "position_range_mm": {
            "min": min_pos,
            "max": max_pos,
        },
        "n_coronal_slices": shape[0],
        "species": atlas.metadata.get("species", "unknown"),
        "citation": atlas.metadata.get("citation", ""),
        "local_version": _normalize_version(getattr(atlas, "local_version", None)),
        "remote_version": _normalize_version(getattr(atlas, "remote_version", None)),
        "is_latest_version": is_latest,
        "additional_references": list_additional_references(atlas),
    }


def list_downloaded_atlases() -> list[str]:
    """Return names of locally available BrainGlobe atlases."""
    try:
        module = importlib.import_module("brainglobe_atlasapi.list_atlases")
    except Exception:
        module = importlib.import_module("brainglobe_atlasapi")

    try:
        get_downloaded_atlases = cast(
            Callable[[], Sequence[object]],
            module.get_downloaded_atlases,
        )
    except AttributeError:
        logger.warning("BrainGlobe API does not expose get_downloaded_atlases().")
        return []

    raw_items = get_downloaded_atlases()
    names: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            names.append(canonicalize_atlas_name(item))
            continue
        if isinstance(item, Sequence) and len(item) > 0 and isinstance(item[0], str):
            names.append(canonicalize_atlas_name(item[0]))

    return sorted(set(names))


def list_available_atlases() -> list[str]:
    """Return remote atlas names advertised by BrainGlobe."""
    try:
        module = importlib.import_module("brainglobe_atlasapi.list_atlases")
        get_atlases_lastversions = cast(
            Callable[[], object],
            module.get_atlases_lastversions,
        )
    except Exception as exc:
        logger.warning("Could not access BrainGlobe remote atlas listing: %s", exc)
        return []

    try:
        result = get_atlases_lastversions()
    except Exception as exc:
        logger.warning("Failed to fetch BrainGlobe remote atlas listing: %s", exc)
        return []

    if isinstance(result, dict):
        return sorted({canonicalize_atlas_name(str(name)) for name in result.keys()})
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return sorted({canonicalize_atlas_name(str(name)) for name in result})
    return []
