"""Async wrappers around the single-slice tool-use AP estimator.

These run the synchronous :func:`estimate_position` agent on a thread via
``asyncio.to_thread()`` so the pipeline can estimate multiple slices
concurrently. Each call is an independent full-atlas tool-use estimate; the
estimator handles its own image normalization (normalize -> downscale ->
CLAHE) internally, so callers pass the raw slice image.
"""

from __future__ import annotations

import asyncio
import logging

from PIL import Image

from langslice_harness.atlas.core import get_position_range_mm, load_atlas
from langslice_harness.estimation import APResult, estimate_position

logger = logging.getLogger(__name__)


def _load_slice(image_path: str) -> Image.Image:
    """Load a slice image as RGB; the estimator handles preprocessing."""
    return Image.open(image_path).convert("RGB")


async def run_anchor_estimation(
    *,
    image_path: str,
    atlas_name: str,
    model_name: str | None = None,
) -> APResult:
    """Estimate an anchor slice's AP position with the tool-use agent.

    Falls back to the atlas midpoint if the agent errors, since interpolation
    and the isotonic fit lean on anchor positions being present.
    """
    image = _load_slice(image_path)
    try:
        result = await asyncio.to_thread(
            estimate_position,
            image,
            atlas_name,
            model_name=model_name,
        )
        logger.info("Anchor (tool-use): %.3fmm (%s)", result.position_mm, image_path)
        return result
    except Exception:
        atlas = load_atlas(atlas_name)
        lo, hi = get_position_range_mm(atlas)
        mid = (lo + hi) / 2.0
        logger.warning(
            "Anchor estimation failed, using atlas midpoint %.3fmm (%s)",
            mid,
            image_path,
        )
        return APResult(
            position_mm=mid,
            reasoning="Anchor estimation failed; fell back to atlas midpoint.",
        )


async def run_slice_estimation(
    *,
    image_path: str,
    atlas_name: str,
    model_name: str | None = None,
) -> APResult:
    """Estimate a non-anchor slice's AP position with the tool-use agent.

    Each slice is estimated independently over the full atlas. The pipeline's
    interpolated position is used only as an outlier reference afterwards
    (Phase 3.5), not as a search prior.
    """
    image = _load_slice(image_path)
    result = await asyncio.to_thread(
        estimate_position,
        image,
        atlas_name,
        model_name=model_name,
    )
    logger.info("Slice (tool-use): %.3fmm (%s)", result.position_mm, image_path)
    return result
