"""The flags a fit carries: what about its result a reader should check.

A flag is not a refusal: the fit is written, and its row lists each flag
as ``{"code": ..., ...}`` with the numbers behind it. The codes are strings
the agent reads, so a member's value never changes.
"""

from __future__ import annotations

from langslice.core.refusal import TextEnum


class FitFlag(TextEnum):
    """Every flag code of a fit, once, with its meaning."""

    DISPLACEMENT_OUTSIZED = "DISPLACEMENT_OUTSIZED"
    FOLDS = "FOLDS"
    REGION_COMPRESSED = "REGION_COMPRESSED"
    REGION_EXPANDED = "REGION_EXPANDED"
    REGION_FOLDED = "REGION_FOLDED"
    REGION_VANISHED = "REGION_VANISHED"
    SCALE_CHANGED = "SCALE_CHANGED"
    SHEAR = "SHEAR"


SECTION_FLAGS = (FitFlag.DISPLACEMENT_OUTSIZED, FitFlag.FOLDS)
