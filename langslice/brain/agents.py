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

    # Stage B: nano-banana fine pass centered on coarse result
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
    )
    logger.info("Anchor fine: %.3fmm (%s)", fine.position_mm, image_path)

    return fine


async def run_slice_estimation(
    *,
    image_path: str,
    atlas_name: str,
    center_mm: float,
    window_half_mm: float = 2.0,
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Run 2-pass nano-banana estimation for a non-anchor slice.

    The broad scan is skipped; the search starts in a neighborhood around
    *center_mm* (derived from anchor interpolation).  The window is wide
    enough (~±2 mm) for the model to correct a bad interpolation.
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
    )

    return result
