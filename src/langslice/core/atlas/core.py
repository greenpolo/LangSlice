import importlib
import logging
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Protocol, cast

import numpy as np
from PIL import Image

from langslice.core.space import (
    Plane,
    atlas_space_context,
    slice_axis_index,
)

logger = logging.getLogger(__name__)

DEFAULT_ATLAS_NAME = "allen_mouse_25um"

_ATLAS_ALIASES: dict[str, str] = {
    "whs_sd_rat": "whs_sd_rat_39um",
}


class _AtlasLike(Protocol):
    atlas_name: str
    orientation: str
    template: np.ndarray
    annotation: np.ndarray
    resolution: Sequence[float]
    metadata: dict[str, object]


BrainGlobeAtlas = _AtlasLike


def canonicalize_atlas_name(name: str) -> str:
    """Normalize atlas identifiers and resolve legacy aliases."""
    cleaned = name.strip()
    if not cleaned:
        return DEFAULT_ATLAS_NAME
    return _ATLAS_ALIASES.get(cleaned, cleaned)


@lru_cache(maxsize=64)
def load_atlas(name: str) -> BrainGlobeAtlas:
    """Load and cache a BrainGlobe atlas. First call downloads if needed.

    The cache is generous so a session that moves across many atlases (the
    per-age developmental ones) does not reload them.
    """
    atlas_name = canonicalize_atlas_name(name)
    # When the GIN version server is unreachable, BrainGlobe blocks for minutes
    # (no-timeout HTTP, 5 retries) on every load: the latest-version check for
    # cached atlases, and remote_version resolution for uncached ones. Setting
    # LANGSLICE_ATLAS_SKIP_LATEST_CHECK enables offline mode — cached atlases
    # load without phoning home, and uncached atlases fail fast instead of
    # hanging on a download attempt.
    skip_latest = os.environ.get("LANGSLICE_ATLAS_SKIP_LATEST_CHECK") not in (None, "", "0")
    try:
        module = importlib.import_module("brainglobe_atlasapi")
        if skip_latest:
            from brainglobe_atlasapi import config as bg_config

            bg_dir = bg_config.get_brainglobe_dir()
            if not list(bg_dir.glob(f"{atlas_name}_v*")):
                raise FileNotFoundError(
                    f"atlas not present in local cache {bg_dir} "
                    f"(offline mode; GIN download skipped)"
                )
        brain_globe_atlas = cast(Callable[..., BrainGlobeAtlas], module.BrainGlobeAtlas)
        atlas = (
            brain_globe_atlas(atlas_name, check_latest=False)
            if skip_latest
            else brain_globe_atlas(atlas_name)
        )
    except Exception as exc:  # pragma: no cover - passthrough from external library
        raise ValueError(f"Atlas '{atlas_name}' not found or failed to load: {exc}") from exc
    return atlas


def _resolution_mm_for_plane(atlas: _AtlasLike, plane: Plane) -> float:
    context = atlas_space_context(atlas)
    axis = slice_axis_index(context, plane)
    return context.resolution_um[axis] / 1000.0


def _n_slices_for_plane(atlas: _AtlasLike, plane: Plane) -> int:
    context = atlas_space_context(atlas)
    axis = slice_axis_index(context, plane)
    return context.shape[axis]


def position_mm_to_index(
    atlas: _AtlasLike, position_mm: float, *, plane: Plane = "coronal"
) -> int:
    """Convert a physical position (mm) to an array index along the slice-normal axis."""
    res_mm = _resolution_mm_for_plane(atlas, plane)
    n_slices = _n_slices_for_plane(atlas, plane)
    idx = int(round(position_mm / res_mm))
    if idx < 0 or idx >= n_slices:
        _, max_pos = get_position_range_mm(atlas, plane=plane)
        raise ValueError(
            f"Position {position_mm:.3f}mm (plane={plane}) maps to index {idx}, "
            f"out of range [0, {n_slices - 1}]. "
            f"Valid range for '{atlas.atlas_name}': 0.0mm to {max_pos:.3f}mm"
        )
    return idx


def index_to_position_mm(
    atlas: _AtlasLike, idx: int, *, plane: Plane = "coronal"
) -> float:
    """Convert an array index along the slice-normal axis to a physical position (mm)."""
    return idx * _resolution_mm_for_plane(atlas, plane)


def get_position_range_mm(
    atlas: _AtlasLike, *, plane: Plane = "coronal"
) -> tuple[float, float]:
    """Return (min_mm, max_mm) along the given slicing plane's normal axis."""
    res_mm = _resolution_mm_for_plane(atlas, plane)
    n_slices = _n_slices_for_plane(atlas, plane)
    return 0.0, (n_slices - 1) * res_mm


def _resolve_idx_axis(
    atlas: _AtlasLike, position_mm: float, plane: Plane
) -> tuple[int, int]:
    """Resolve the slice index and slice-normal axis for a position/plane pair."""
    idx = position_mm_to_index(atlas, position_mm, plane=plane)
    axis = slice_axis_index(atlas_space_context(atlas), plane)
    return idx, axis


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 [0, 255]."""
    if arr.size == 0:
        return arr.astype(np.uint8)

    arr_float = arr.astype(np.float32, copy=False)
    max_val = float(np.max(arr_float))
    if max_val <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr_float / max_val) * 255.0, 0, 255).astype(np.uint8)


def orient_slice_for_display(slice_2d: np.ndarray, plane: Plane) -> np.ndarray:
    """Orient atlas slices for visual comparison with histology images.

    For sagittal and horizontal slices the raw axis order is awkward for
    overlaying onto histology; swap the in-plane axes so the AP axis lies
    horizontally in the resulting image.
    """
    if plane in {"sagittal", "horizontal"}:
        return np.swapaxes(slice_2d, 0, 1)
    return slice_2d


def get_reference_slice(
    atlas: _AtlasLike,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> Image.Image:
    """Get a reference slice along the chosen plane as grayscale PIL image.

    Non-zero cutting angles reslice the template obliquely instead of taking
    it flat off the voxel grid.
    """
    if pitch_deg or yaw_deg:
        from langslice.core.oblique import sample_oblique_plane

        return Image.fromarray(
            _normalize_to_uint8(
                sample_oblique_plane(
                    atlas, position_mm, plane, pitch_deg, yaw_deg, volume="template", order=3
                )
            ),
            mode="L",
        )
    idx, axis = _resolve_idx_axis(atlas, position_mm, plane)
    reference_slice = np.take(np.asarray(atlas.template), idx, axis=axis)
    reference_slice = orient_slice_for_display(reference_slice, plane)
    normalized = _normalize_to_uint8(reference_slice)
    return Image.fromarray(normalized, mode="L")


def get_root_mask(
    atlas: _AtlasLike,
    position_mm: float,
    target_size: tuple[int, int],
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Binary tissue silhouette of the atlas at *position_mm*, at *target_size*.

    Pixels with a non-zero annotation id are opaque (255), the out-of-tissue
    background (annotation == 0) transparent (0). Resized NEAREST so the mask
    stays binary — bilinear interpolation would leave a halo around the brain
    silhouette.

    Args:
        atlas: BrainGlobe-style atlas exposing ``.annotation``.
        position_mm: AP/ML/DV position (per *plane*) at which to slice.
        target_size: ``(width, height)`` of the desired mask.
        plane: Slicing plane, resolved the same way as every other accessor
            here so the slab axis lines up across the pipeline.
        pitch_deg, yaw_deg: Cutting angles; non-zero samples the plane
            obliquely (:func:`langslice.core.oblique.sample_oblique_annotation`),
            the plane every picture of an angled stack shows.
    """
    if pitch_deg or yaw_deg:
        from langslice.core.oblique import sample_oblique_annotation

        labels = sample_oblique_annotation(atlas, position_mm, plane, pitch_deg, yaw_deg)
        mask = (labels != 0).astype(np.uint8) * 255
    else:
        idx, axis = _resolve_idx_axis(atlas, position_mm, plane)
        annotation_slice = np.asarray(np.take(atlas.annotation, idx, axis=axis))
        annotation_slice = orient_slice_for_display(annotation_slice, plane)
        mask = (annotation_slice != 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L").resize(
        target_size, resample=Image.Resampling.NEAREST
    )
    return np.asarray(mask_img, dtype=np.uint8)

