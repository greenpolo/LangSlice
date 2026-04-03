"""Registration subsystem for matrix-first affine and nonlinear alignment."""

from langslice.registration.core import estimate_registration_runtime
from langslice.registration.types import (
    AffineResult,
    LandmarkAnnotation,
    NonlinearResult,
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    RegistrationResult,
    affine_matrix_from_legacy_params,
    annotation_session_to_dict,
    apply_affine_to_points,
    build_annotation_session_from_correspondences,
    coerce_affine_matrix,
    decompose_affine_matrix,
    get_session_marker_points,
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
    "build_annotation_session_from_correspondences",
    "coerce_affine_matrix",
    "decompose_affine_matrix",
    "estimate_registration_runtime",
    "get_session_marker_points",
    "identity_affine_matrix",
    "is_valid_affine_matrix",
    "render_landmark_annotations",
]
