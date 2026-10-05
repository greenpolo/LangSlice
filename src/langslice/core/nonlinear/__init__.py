"""The image-model border route's core half (``CLAUDE.md`` in this package).

The model transport is :mod:`langslice.providers.images`. The three matrix
helpers are re-exported from :mod:`langslice.core.affine` for
``langslice.job.quint``.
"""

from __future__ import annotations

from langslice.core.affine import (
    affine_matrix_from_legacy_params,
    apply_affine_to_points,
    coerce_affine_matrix,
)
from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest

__all__ = [
    "GeneratedSegmentation",
    "SegmentationGenerationRequest",
    "affine_matrix_from_legacy_params",
    "apply_affine_to_points",
    "coerce_affine_matrix",
]
