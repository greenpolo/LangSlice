"""Single-slice estimation as an async worker for the positioning step.

Wraps the synchronous :func:`estimate_position` agent on a thread so the engine
can run estimates without blocking its event loop. Each call is an independent
full-atlas tool-use estimate; the estimator does its own image preprocessing
(normalize -> downscale -> CLAHE) internally, so callers pass the raw slice
image and switch the CLAHE step with *apply_clahe* rather than preprocessing it
themselves — preprocessing twice is worse than not at all.
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
    plane: str = "coronal",
    model_name: str | None = None,
    apply_clahe: bool = True,
) -> APResult:
    """Estimate one slice's position with the single-slice tool-use agent.

    Errors propagate: the positioning step decides what a failed estimate
    means for that slice.
    """
    image = _load_slice(image_path)
    result = await asyncio.to_thread(
        estimate_position,
        image,
        atlas_name,
        plane=plane,
        model_name=model_name,
        apply_clahe=apply_clahe,
    )
    logger.info("Slice (tool-use): %.3fmm (%s)", result.position_mm, image_path)
    return result
