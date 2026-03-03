import importlib
import logging
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Protocol, cast

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_ATLAS_NAME = "allen_mouse_25um"

_ATLAS_ALIASES: dict[str, str] = {
    "whs_sd_rat": "whs_sd_rat_39um",
}


class _AtlasLike(Protocol):
    atlas_name: str
    orientation: str
    reference: np.ndarray
    annotation: np.ndarray
    resolution: Sequence[float]
    metadata: dict[str, object]


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
        brain_globe_atlas = cast(Callable[[str], BrainGlobeAtlas], getattr(module, "BrainGlobeAtlas"))
        atlas = brain_globe_atlas(atlas_name)
    except Exception as exc:  # pragma: no cover - passthrough from external library
        raise ValueError(f"Atlas '{atlas_name}' not found or failed to load: {exc}") from exc
    return atlas


def position_mm_to_index(atlas: _AtlasLike, position_mm: float) -> int:
    """Convert a physical position (mm from anterior edge) to an array index."""
    res_mm = float(atlas.resolution[0]) / 1000.0
    shape = cast(tuple[int, ...], atlas.reference.shape)
    n_slices = shape[0]

    idx = int(round(position_mm / res_mm))
    if idx < 0 or idx >= n_slices:
        _, max_pos = get_position_range_mm(atlas)
        raise ValueError(
            f"Position {position_mm:.3f}mm maps to index {idx}, out of range [0, {n_slices - 1}]. "
            f"Valid range for '{atlas.atlas_name}': 0.0mm to {max_pos:.3f}mm"
        )
    return idx


def index_to_position_mm(atlas: _AtlasLike, idx: int) -> float:
    """Convert an array index along axis 0 to a physical position in mm."""
    res_mm = float(atlas.resolution[0]) / 1000.0
    return idx * res_mm


def get_position_range_mm(atlas: _AtlasLike) -> tuple[float, float]:
    """Return physical position range as (0.0, max_mm)."""
    shape = cast(tuple[int, ...], atlas.reference.shape)
    n_slices = shape[0]
    res_mm = float(atlas.resolution[0]) / 1000.0
    return 0.0, (n_slices - 1) * res_mm


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


def get_atlas_info(atlas: _AtlasLike) -> dict[str, object]:
    """Return atlas metadata and position range info."""
    min_pos, max_pos = get_position_range_mm(atlas)
    resolution_mm = [float(r) / 1000.0 for r in atlas.resolution]

    shape = cast(tuple[int, ...], atlas.reference.shape)
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
    }


def list_downloaded_atlases() -> list[str]:
    """Return names of locally available BrainGlobe atlases."""
    try:
        module = importlib.import_module("brainglobe_atlasapi.list_atlases")
    except Exception:
        module = importlib.import_module("brainglobe_atlasapi")

    try:
        get_downloaded_atlases = cast(Callable[[], Sequence[object]], getattr(module, "get_downloaded_atlases"))
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
        get_atlases_lastversions = cast(Callable[[], object], getattr(module, "get_atlases_lastversions"))
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
