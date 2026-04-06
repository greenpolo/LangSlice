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
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Run full AP estimation + nano-banana refinement for an anchor slice.

    Stage A: multi-turn tool-use estimation (coarse).
    Stage B: nano-banana fine pass centered on Stage A result.
    """
    image = _prepare_slice(image_path)

    # Stage A: coarse estimation
    coarse = await asyncio.to_thread(
        estimate_position,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
    )
    logger.info("Anchor coarse: %.3fmm (%s)", coarse.position_mm, image_path)

    # Stage B: nano-banana fine pass centered on coarse result
    fine = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        send_individually=True,
        atlas_resolution=1024,
        center_mm=coarse.position_mm,
    )
    logger.info("Anchor fine: %.3fmm (%s)", fine.position_mm, image_path)

    return fine


async def run_refinement(
    *,
    image_path: str,
    atlas_name: str,
    window_lo: float,
    window_hi: float,
    window_center: float,
    n_images: int,
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult | None:
    """Run nano-banana refinement for a single slice within a bounded window.

    Returns ``None`` if *n_images* is 0 (window too narrow, skip).
    """
    if n_images == 0:
        return None

    image = _prepare_slice(image_path)

    result = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        send_individually=True,
        atlas_resolution=1024,
        center_mm=window_center,
        bounds=(window_lo, window_hi),
    )

    # Clamp result to window bounds
    clamped_mm = max(window_lo, min(window_hi, result.position_mm))
    return APResult(
        position_mm=clamped_mm,
        reasoning=result.reasoning,
        debug_dir=result.debug_dir,
    )
