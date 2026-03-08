"""Registration subsystem for matrix-first affine alignment."""

from langslice.registration.core import estimate_affine_registration
from langslice.registration.types import (
    AffineResult,
    affine_matrix_from_legacy_params,
    apply_affine_to_points,
    coerce_affine_matrix,
    decompose_affine_matrix,
    identity_affine_matrix,
    is_valid_affine_matrix,
)

__all__ = [
    "AffineResult",
    "affine_matrix_from_legacy_params",
    "apply_affine_to_points",
    "coerce_affine_matrix",
    "decompose_affine_matrix",
    "estimate_affine_registration",
    "identity_affine_matrix",
    "is_valid_affine_matrix",
]
