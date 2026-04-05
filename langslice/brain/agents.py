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
from langslice.image_prep import normalize_image

logger = logging.getLogger(__name__)


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
    image = Image.open(image_path).convert("RGB")
    image = normalize_image(image)

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

    image = Image.open(image_path).convert("RGB")
    image = normalize_image(image)

    # TODO: pass window bounds to nano-banana to constrain atlas image range.
    # For now, use the standard nano-banana call.  The window parameters will
    # be wired into estimate_position_image_gen once the API is extended to
    # accept explicit position bounds.
    result = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        send_individually=True,
        atlas_resolution=1024,
    )

    # Clamp result to window bounds
    clamped_mm = max(window_lo, min(window_hi, result.position_mm))
    return APResult(
        position_mm=clamped_mm,
        reasoning=result.reasoning,
        debug_dir=result.debug_dir,
    )
