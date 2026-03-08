"""Backend-neutral registration orchestration."""

from __future__ import annotations

import logging
from typing import Callable

from PIL import Image

from langslice.registration.ants_backend import (
    RegistrationFailure,
    estimate_affine_with_ants,
)
from langslice.registration.types import AffineResult

logger = logging.getLogger(__name__)

AffineFallback = Callable[
    [Image.Image, Callable[[str], None] | None, str | None, float | None, float | None],
    AffineResult,
]


def estimate_affine_registration(
    image: Image.Image,
    *,
    on_progress: Callable[[str], None] | None = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    pixel_size_um: float | None = None,
    fallback: AffineFallback | None = None,
) -> AffineResult:
    """Run ANTsPyX affine registration and fall back when needed."""

    def _progress(message: str) -> None:
        if on_progress:
            on_progress(message)
        logger.info(message)

    _ = pixel_size_um

    if atlas_name is None or position_mm is None:
        if fallback is None:
            raise RegistrationFailure("Atlas context is required for affine registration.")
        _progress("Atlas context missing; using Gemini affine fallback.")
        return fallback(image, on_progress, atlas_name, position_mm, pixel_size_um)

    try:
        result = estimate_affine_with_ants(
            image=image,
            atlas_name=atlas_name,
            position_mm=position_mm,
            on_progress=on_progress,
        )
        _progress(
            "ANTsPyX affine completed: "
            f"rot={result.rotation_deg:.2f} deg, "
            f"scale=({result.scale[0]:.3f}, {result.scale[1]:.3f}), "
            f"shear={result.shear:.3f}"
        )
        return result
    except Exception as exc:
        if fallback is None:
            raise

        _progress(f"ANTsPyX affine failed: {exc}")
        fallback_result = fallback(image, on_progress, atlas_name, position_mm, pixel_size_um)
        fallback_result.reasoning = (
            f"ANTsPyX backend failed: {exc}\n\n{fallback_result.reasoning}"
        )
        return fallback_result
