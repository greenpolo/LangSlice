"""Registration runtime orchestration."""

from __future__ import annotations

import logging
from typing import Callable

from PIL import Image

from langslice.registration.runtime import RegistrationFailure, estimate_registration
from langslice.registration.types import (
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
    pixel_size_um: float | None = None,
    target_landmark_count: int = 12,
    workflow: str = "single_pass",
    show_atlas_borders: bool = True,
    on_correspondences: Callable[[list[RegistrationCorrespondence]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
    debug_dir: str | None = None,
    enable_code_execution: bool | None = None,
    tool_loop_max_steps: int | None = None,
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
        pixel_size_um=pixel_size_um,
        target_landmark_count=target_landmark_count,
        workflow=workflow,
        show_atlas_borders=show_atlas_borders,
        on_correspondences=on_correspondences,
        on_annotation_session=on_annotation_session,
        on_progress=on_progress,
        on_trace=on_trace,
        debug_dir=debug_dir,
        enable_code_execution=enable_code_execution,
        tool_loop_max_steps=tool_loop_max_steps,
    )
    _progress(
        "Registration outputs derived: "
        f"rot={result.affine_result.rotation_deg:.2f} deg, "
        f"scale=({result.affine_result.scale[0]:.3f}, {result.affine_result.scale[1]:.3f}), "
        f"shear={result.affine_result.shear:.3f}, state={result.qc_state}"
    )
    return result
