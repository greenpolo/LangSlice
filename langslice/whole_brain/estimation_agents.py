"""Async wrappers around existing AP estimators.

These functions run the existing synchronous estimator code on a thread
via ``asyncio.to_thread()`` so the pipeline can execute multiple slices
concurrently.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from PIL import Image

from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.estimation import APResult, estimate_position_image_gen
from langslice.image_prep import adaptive_preprocess, normalize_image, prepare_image_for_vlm
from langslice.whole_brain.window import compute_search_bounds

# Match the single-slice CLI: normalize → downscale → CLAHE.
_VLM_MAX_LONG_EDGE = 2048

logger = logging.getLogger(__name__)


def _prepare_slice(image_path: str) -> Image.Image:
    """Load and preprocess a slice image, matching the single-slice CLI path."""
    raw = Image.open(image_path).convert("RGB")
    canonical = normalize_image(raw)
    downscaled = prepare_image_for_vlm(canonical, max_long_edge=_VLM_MAX_LONG_EDGE).image
    return adaptive_preprocess(downscaled)


async def run_anchor_estimation(
    *,
    image_path: str,
    atlas_name: str,
    coarse_model: str | None = None,
    fine_model: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Two-stage anchor estimation: image-gen 2-pass coarse then fine pass.

    Stage A: Image-gen broad→neighborhood scan (2 passes) to find the
    general AP region.  More deterministic than tool-use for coarse
    estimation (validated in hyp 018: anchor 3 coarse 0.298mm off GT
    vs tool-use 0.6-1.6mm).
    Stage B: Nano-banana fine pass centered on the coarse result refines to
    ~0.05mm precision within ±0.5mm bounds.

    Falls back to atlas midpoint if both stages fail (3-tier fallback).
    """
    image = _prepare_slice(image_path)

    # Stage A: image-gen 2-pass coarse (broad + neighborhood only)
    coarse_mm: float
    try:
        coarse = await asyncio.to_thread(
            estimate_position_image_gen,
            image,
            atlas_name,
            on_progress=on_progress,
            debug_dir=debug_dir,
            model_name=coarse_model,
            send_individually=True,
            atlas_resolution=1024,
            max_passes=2,
        )
        coarse_mm = coarse.position_mm
        logger.info("Anchor coarse (image-gen 2-pass): %.3fmm (%s)", coarse_mm, image_path)
    except Exception:
        # Fallback: atlas midpoint (zero API cost)
        atlas = load_atlas(atlas_name)
        lo, hi = get_position_range_mm(atlas)
        coarse_mm = (lo + hi) / 2.0
        logger.warning(
            "Anchor Stage A failed, using atlas midpoint %.3fmm (%s)",
            coarse_mm, image_path,
        )

    # Stage B: nano-banana fine pass centered on coarse result.
    # ±0.5mm bounds keep the fine pass from drifting too far while still
    # being wide enough for all coarse estimates observed with image-gen
    # 2-pass (max coarse error ~0.4mm in hyp 018).
    _ANCHOR_FINE_HALF_MM = 0.5
    fine_lo = coarse_mm - _ANCHOR_FINE_HALF_MM
    fine_hi = coarse_mm + _ANCHOR_FINE_HALF_MM
    try:
        fine = await asyncio.to_thread(
            estimate_position_image_gen,
            image,
            atlas_name,
            on_progress=on_progress,
            debug_dir=debug_dir,
            model_name=fine_model,
            send_individually=True,
            atlas_resolution=1024,
            center_mm=coarse_mm,
            bounds=(fine_lo, fine_hi),
        )
        logger.info("Anchor fine: %.3fmm (drift %.3fmm) (%s)",
                    fine.position_mm,
                    abs(fine.position_mm - coarse_mm),
                    image_path)
        return fine
    except Exception:
        logger.warning(
            "Anchor Stage B failed, returning coarse %.3fmm (%s)",
            coarse_mm, image_path,
        )
        return APResult(position_mm=coarse_mm, reasoning="Stage B failed, coarse only")


async def run_slice_estimation(
    *,
    image_path: str,
    atlas_name: str,
    center_mm: float,
    atlas_range: tuple[float, float] | None = None,
    window_half_mm: float = 3.0,
    slices_per_pass: int = 17,
    edge_anchor_mm: float | None = None,
    edge_side: str | None = None,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Run 2-pass nano-banana estimation with confirmation pass for a non-anchor slice.

    The broad scan is skipped; the search starts in a neighborhood around
    *center_mm* (derived from anchor interpolation).  The window is wide
    enough (~±3 mm) for the model to correct a bad interpolation.

    Using 17 slices per pass (vs the default 13) gives finer neighborhood
    resolution and a wider fine-pass range (0.8 mm vs 0.6 mm), improving
    precision for near-miss estimates.

    After the main 2-pass estimation, a single confirmation pass (9 slices,
    ±0.25mm, ~0.06mm spacing) refines the result. This pushes near-threshold
    estimates below 0.1mm error (validated in hyp 015: 6/8 targeted cases
    improved).
    """
    image = _prepare_slice(image_path)

    if atlas_range is None:
        bounds = (center_mm - window_half_mm, center_mm + window_half_mm)
    else:
        bounds = compute_search_bounds(
            center_mm=center_mm,
            atlas_range=atlas_range,
            window_half_mm=window_half_mm,
            edge_anchor_mm=edge_anchor_mm,
            edge_side=edge_side,
        )

    result = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        model_name=model_name,
        send_individually=True,
        atlas_resolution=1024,
        center_mm=center_mm,
        bounds=bounds,
        slices_per_pass=slices_per_pass,
        fine_resolution_mm=0.10,
    )

    # Confirmation pass: single ultra-fine pass centered on the initial result.
    # 9 slices over ±0.25mm gives ~0.06mm spacing — finer than the main fine
    # pass (0.10mm).  This reliably pushes near-threshold estimates below
    # 0.1mm error without degrading already-accurate ones.
    _CONFIRM_HALF_MM = 0.25
    confirm_center = result.position_mm
    try:
        confirmed = await asyncio.to_thread(
            estimate_position_image_gen,
            image,
            atlas_name,
            on_progress=on_progress,
            debug_dir=debug_dir,
            model_name=model_name,
            send_individually=True,
            atlas_resolution=1024,
            center_mm=confirm_center,
            bounds=(confirm_center - _CONFIRM_HALF_MM, confirm_center + _CONFIRM_HALF_MM),
            slices_per_pass=9,
            max_passes=1,
        )
        logger.info(
            "Slice confirmation: %.3f → %.3fmm (shift %.3fmm) (%s)",
            confirm_center, confirmed.position_mm,
            abs(confirmed.position_mm - confirm_center),
            image_path,
        )
        return confirmed
    except Exception:
        logger.warning(
            "Confirmation pass failed, returning initial %.3fmm (%s)",
            confirm_center, image_path,
        )
        return result
