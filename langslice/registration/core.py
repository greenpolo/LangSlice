"""Registration orchestration and backward-compatible affine entry points."""

from __future__ import annotations

import logging
from typing import Callable

from PIL import Image

from langslice.registration.runtime import RegistrationFailure, estimate_registration
from langslice.registration.types import AffineResult, RegistrationResult

logger = logging.getLogger(__name__)


def estimate_registration_runtime(
    image: Image.Image,
    *,
    on_progress: Callable[[str], None] | None = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    pixel_size_um: float | None = None,
    fallback: object | None = None,
) -> RegistrationResult:
    """Run the separate registration runtime and return full results."""

    def _progress(message: str) -> None:
        if on_progress:
            on_progress(message)
        logger.info(message)

    _ = fallback
    if atlas_name is None or position_mm is None:
        raise RegistrationFailure("Atlas context is required for registration.")

    result = estimate_registration(
        image=image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        pixel_size_um=pixel_size_um,
        on_progress=on_progress,
    )
    _progress(
        "Registration affine derived: "
        f"rot={result.affine_result.rotation_deg:.2f} deg, "
        f"scale=({result.affine_result.scale[0]:.3f}, {result.affine_result.scale[1]:.3f}), "
        f"shear={result.affine_result.shear:.3f}, state={result.qc_state}"
    )
    return result


def estimate_affine_registration(
    image: Image.Image,
    *,
    on_progress: Callable[[str], None] | None = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    pixel_size_um: float | None = None,
    fallback: object | None = None,
) -> AffineResult:
    """Backward-compatible affine entry point built on the registration runtime."""
    return estimate_registration_runtime(
        image=image,
        on_progress=on_progress,
        atlas_name=atlas_name,
        position_mm=position_mm,
        pixel_size_um=pixel_size_um,
        fallback=fallback,
    ).affine_result
