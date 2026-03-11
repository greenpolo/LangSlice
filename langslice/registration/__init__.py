"""Registration subsystem for matrix-first affine and nonlinear alignment."""

from langslice.registration.core import estimate_affine_registration, estimate_registration_runtime
from langslice.registration.types import (
    AffineResult,
    NonlinearResult,
    RegistrationCorrespondence,
    RegistrationResult,
    affine_matrix_from_legacy_params,
    apply_affine_to_points,
    coerce_affine_matrix,
    decompose_affine_matrix,
    identity_affine_matrix,
    is_valid_affine_matrix,
)

__all__ = [
    "AffineResult",
    "NonlinearResult",
    "RegistrationCorrespondence",
    "RegistrationResult",
    "affine_matrix_from_legacy_params",
    "apply_affine_to_points",
    "coerce_affine_matrix",
    "decompose_affine_matrix",
    "estimate_affine_registration",
    "estimate_registration_runtime",
    "identity_affine_matrix",
    "is_valid_affine_matrix",
]
