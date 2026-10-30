"""The image-model border route's core half (``CLAUDE.md`` in this package).

The model transport is :mod:`langslice.providers.image_calls`.
"""

from __future__ import annotations

from langslice.core.fits.trace.request import (
    BorderTraceRequest,
    GeneratedSegmentation,
)

__all__ = [
    "BorderTraceRequest",
    "GeneratedSegmentation",
]
