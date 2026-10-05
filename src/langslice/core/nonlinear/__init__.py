"""The image-model border route's core half (``CLAUDE.md`` in this package).

The model transport is :mod:`langslice.providers.images`.
"""

from __future__ import annotations

from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest

__all__ = [
    "GeneratedSegmentation",
    "SegmentationGenerationRequest",
]
