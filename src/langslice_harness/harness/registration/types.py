"""Harness registration candidate types for dense image-gen registration flows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from langslice_harness.registration.types import (
    AffineResult,
    NonlinearResult,
    RegistrationAnnotationSession,
    RegistrationResult,
    identity_affine_matrix,
)


@dataclass
class GeneratedSegmentation:
    """Image-gen output for a dense registration candidate."""

    image: Image.Image
    provider: str
    model: str
    route: str
    revised_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationCandidate:
    """Dense image-gen registration candidate and its annotation session."""

    candidate_id: str
    generated_segmentation: Image.Image
    warped_atlas: Image.Image
    warped_border_overlay: Image.Image
    markers: list[list[float]]
    annotation_session: RegistrationAnnotationSession
    metadata: dict[str, Any] = field(default_factory=dict)


def candidate_to_registration_result(
    candidate: RegistrationCandidate,
    image_size: tuple[int, int],
    debug_dir: str | None = None,
) -> RegistrationResult:
    """Convert a dense registration candidate into a runtime result."""

    session = candidate.annotation_session
    session_metadata = deepcopy(session.metadata)
    session_metadata.update(
        {
            "visualign_markers": deepcopy(candidate.markers),
            "n_markers": len(candidate.markers),
            "candidate_id": candidate.candidate_id,
            "candidate_metadata": deepcopy(candidate.metadata),
        }
    )
    session.metadata = session_metadata

    affine_result = AffineResult(
        matrix=identity_affine_matrix(),
        source_size=image_size,
        output_size=image_size,
        backend="image_gen_registration_dense",
        reasoning="Dense VisuAlign markers are stored in annotation metadata.",
    )
    nonlinear_result = NonlinearResult(
        atlas_points=np.zeros((0, 2), dtype=np.float64),
        slice_points=np.zeros((0, 2), dtype=np.float64),
        smoothing=0.0,
        backend="elastix_bspline_visualign",
        reasoning=(
            "Dense VisuAlign markers are the transform representation stored "
            "in annotation metadata."
        ),
        output_size=image_size,
    )

    return RegistrationResult(
        correspondences=[],
        accepted_correspondences=[],
        affine_result=affine_result,
        nonlinear_result=nonlinear_result,
        debug_dir=debug_dir,
        annotation_session=session,
    )
