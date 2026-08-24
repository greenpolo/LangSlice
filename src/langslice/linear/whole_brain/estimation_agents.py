"""Single-slice estimation as an async worker for the positioning step.

Wraps the synchronous :func:`estimate_position` agent on a thread so the
engine can escalate several stubborn slices concurrently. Each call is an
independent full-atlas tool-use estimate; the estimator handles its own image
preprocessing (normalize -> downscale -> CLAHE) internally, so callers pass
the raw slice image.
"""

from __future__ import annotations

import asyncio
import logging

from PIL import Image

from langslice.linear import APResult, estimate_position

logger = logging.getLogger(__name__)


def _load_slice(image_path: str) -> Image.Image:
    """Load a slice image as RGB; the estimator handles preprocessing."""
    return Image.open(image_path).convert("RGB")


async def run_slice_estimation(
    *,
    image_path: str,
    atlas_name: str,
    model_name: str | None = None,
) -> APResult:
    """Estimate one slice's position with the single-slice tool-use agent.

    Errors propagate: the positioning step decides what a failed escalation
    means for that slice.
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
