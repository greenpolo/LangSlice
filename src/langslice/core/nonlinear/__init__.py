"""Nonlinear registration: matrix-first affine/nonlinear types plus the image-gen harness.

The image-model border route's core half (``CLAUDE.md`` in this package);
the model transport is :mod:`langslice.providers.images`.
"""

from __future__ import annotations

from langslice.core.nonlinear.image_gen_registration import generate_registration_candidate
from langslice.core.nonlinear.types import (
    AffineResult,
    GeneratedSegmentation,
    LandmarkAnnotation,
    NonlinearResult,
    RegistrationAnnotationSession,
    RegistrationCandidate,
    RegistrationCorrespondence,
    RegistrationResult,
    affine_matrix_from_legacy_params,
    annotation_session_to_dict,
    apply_affine_to_points,
    candidate_to_registration_result,
    coerce_affine_matrix,
    decompose_affine_matrix,
    identity_affine_matrix,
    is_valid_affine_matrix,
    render_landmark_annotations,
)

__all__ = [
    "AffineResult",
    "GeneratedSegmentation",
    "LandmarkAnnotation",
    "NonlinearResult",
    "RegistrationAnnotationSession",
    "RegistrationCandidate",
    "RegistrationCorrespondence",
    "RegistrationResult",
    "affine_matrix_from_legacy_params",
    "annotation_session_to_dict",
    "apply_affine_to_points",
    "candidate_to_registration_result",
    "coerce_affine_matrix",
    "decompose_affine_matrix",
    "generate_registration_candidate",
    "identity_affine_matrix",
    "is_valid_affine_matrix",
    "render_landmark_annotations",
]
