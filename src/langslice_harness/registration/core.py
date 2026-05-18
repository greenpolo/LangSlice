"""Registration runtime orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PIL import Image

from langslice_harness.atlas.space import Plane
from langslice_harness.registration.runtime import RegistrationFailure, estimate_registration
from langslice_harness.registration.types import (
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    RegistrationResult,
)

logger = logging.getLogger(__name__)


def estimate_registration_runtime(
    image: Image.Image,
    *,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    plane: Plane = "coronal",
    registration_mode: str = "direct",
    on_correspondences: Callable[[list[RegistrationCorrespondence]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
    debug_dir: str | None = None,
    provider: str = "google",
    image_provider: str | None = None,
    image_model: str | None = None,
    openai_image_route: str = "images",
    review_model: str | object | None = None,
    max_candidates: int = 3,
) -> RegistrationResult:
    """Run the separate registration runtime and return full results."""

    def _progress(message: str) -> None:
        if on_progress:
            on_progress(message)
        logger.info(message)

    if atlas_name is None or position_mm is None:
        raise RegistrationFailure("Atlas context is required for registration.")

    result = estimate_registration(
        image=image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        plane=plane,
        registration_mode=registration_mode,
        on_correspondences=on_correspondences,
        on_annotation_session=on_annotation_session,
        on_progress=on_progress,
        on_trace=on_trace,
        debug_dir=debug_dir,
        provider=provider,
        image_provider=image_provider,
        image_model=image_model,
        openai_image_route=openai_image_route,
        review_model=review_model,
        max_candidates=max_candidates,
    )
    _progress(
        "Registration outputs derived: "
        f"rot={result.affine_result.rotation_deg:.2f} deg, "
        f"scale=({result.affine_result.scale[0]:.3f}, {result.affine_result.scale[1]:.3f}), "
        f"shear={result.affine_result.shear:.3f}"
    )
    return result
