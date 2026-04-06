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

from langslice.ai.estimator import APResult, estimate_position
from langslice.ai.estimator_image_gen import estimate_position_image_gen
from langslice.image_prep import adaptive_preprocess, normalize_image, prepare_image_for_vlm

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
    """Two-stage anchor estimation: coarse tool-use then nano-banana fine pass.

    Stage A: Multi-turn tool-use agent explores the atlas adaptively to find
    the general AP region.
    Stage B: Nano-banana fine pass centered on the coarse result refines to
    ~0.05mm precision.

    Anchor results feed into interpolation and the final isotonic fit like
    every other slice — they are not locked as truth.
    """
    image = _prepare_slice(image_path)

    # Stage A: coarse tool-use exploration
    coarse = await asyncio.to_thread(
        estimate_position,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        model_name=coarse_model,
    )
    logger.info("Anchor coarse: %.3fmm (%s)", coarse.position_mm, image_path)

    # Stage B: nano-banana fine pass centered on coarse result.
    # Restrict the search window to ±0.5 mm around the coarse estimate so the
    # fine pass can only *refine*, not *relocate*.  The tighter window also
    # increases atlas reference density from ~0.25 mm to ~0.08 mm spacing,
    # giving the VLM more precise comparison images.  Without these bounds the
    # neighbourhood zoom covers ±1.5 mm and the model can drift >1 mm from a
    # good coarse estimate (observed in prior experiments).
    _ANCHOR_FINE_HALF_MM = 0.5
    fine_lo = coarse.position_mm - _ANCHOR_FINE_HALF_MM
    fine_hi = coarse.position_mm + _ANCHOR_FINE_HALF_MM
    fine = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        model_name=fine_model,
        send_individually=True,
        atlas_resolution=1024,
        center_mm=coarse.position_mm,
        bounds=(fine_lo, fine_hi),
    )
    logger.info("Anchor fine: %.3fmm (drift %.3fmm) (%s)",
                fine.position_mm,
                abs(fine.position_mm - coarse.position_mm),
                image_path)

    return fine


async def run_slice_estimation(
    *,
    image_path: str,
    atlas_name: str,
    center_mm: float,
    window_half_mm: float = 3.0,
    slices_per_pass: int = 17,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Run 2-pass nano-banana estimation for a non-anchor slice.

    The broad scan is skipped; the search starts in a neighborhood around
    *center_mm* (derived from anchor interpolation).  The window is wide
    enough (~±3 mm) for the model to correct a bad interpolation.

    Using 17 slices per pass (vs the default 13) gives finer neighborhood
    resolution and a wider fine-pass range (0.8 mm vs 0.6 mm), improving
    precision for near-miss estimates.
    """
    image = _prepare_slice(image_path)

    lo = center_mm - window_half_mm
    hi = center_mm + window_half_mm

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
        bounds=(lo, hi),
        slices_per_pass=slices_per_pass,
    )

    return result
