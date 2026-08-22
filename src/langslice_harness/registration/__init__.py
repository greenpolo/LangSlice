"""Registration subsystem for matrix-first affine and nonlinear alignment."""

from __future__ import annotations

from langslice_harness.registration.types import (
    AffineResult,
    LandmarkAnnotation,
    NonlinearResult,
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    RegistrationResult,
    affine_matrix_from_legacy_params,
    annotation_session_to_dict,
    apply_affine_to_points,
    coerce_affine_matrix,
    decompose_affine_matrix,
    identity_affine_matrix,
    is_valid_affine_matrix,
    render_landmark_annotations,
)

__all__ = [
    "AffineResult",
    "LandmarkAnnotation",
    "NonlinearResult",
    "RegistrationAnnotationSession",
    "RegistrationCorrespondence",
    "RegistrationResult",
    "annotation_session_to_dict",
    "affine_matrix_from_legacy_params",
    "apply_affine_to_points",
    "coerce_affine_matrix",
    "decompose_affine_matrix",
    "identity_affine_matrix",
    "is_valid_affine_matrix",
    "render_landmark_annotations",
]
