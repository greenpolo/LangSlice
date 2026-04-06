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

from langslice.ai.estimator import APResult
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
    model_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Run 3-pass nano-banana estimation for an anchor slice.

    Anchors use the full atlas range (no center_mm) so the broad scan
    establishes the general region.  Their results are NOT locked — they
    feed into interpolation and the final isotonic fit like every other slice.
    """
    image = _prepare_slice(image_path)

    result = await asyncio.to_thread(
        estimate_position_image_gen,
        image,
        atlas_name,
        on_progress=on_progress,
        debug_dir=debug_dir,
        model_name=model_name,
        send_individually=True,
        atlas_resolution=1024,
    )
    logger.info("Anchor estimate: %.3fmm (%s)", result.position_mm, image_path)

    return result


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
