"""The flags a fit carries: what about its result a reader should check.

A flag is not a refusal: the fit is written, and its row lists each flag
as ``{"code": ..., ...}`` with the numbers behind it. The codes are strings
the agent reads, so a member's value never changes.
"""

from __future__ import annotations

from langslice.core.refusal import TextEnum


class FitFlag(TextEnum):
    """Every flag code of a fit, once, with its meaning."""

    #: The deformation moves the tissue further than its brain allows.
    DISPLACEMENT_OUTSIZED = "DISPLACEMENT_OUTSIZED"
    #: The deformation folds more of the tissue than its limit.
    FOLDS = "FOLDS"
    #: A region shrinks below its area limit.
    REGION_COMPRESSED = "REGION_COMPRESSED"
    #: A region grows past its area limit.
    REGION_EXPANDED = "REGION_EXPANDED"
    #: A region's warped area cannot be measured: it folds over itself.
    REGION_FOLDED = "REGION_FOLDED"
    #: A region drawn on the placement is gone after the warp.
    REGION_VANISHED = "REGION_VANISHED"
    #: An affine fit changes the section's scale past its limits.
    SCALE_CHANGED = "SCALE_CHANGED"
    #: An affine fit changes the section's shear past its limit.
    SHEAR = "SHEAR"


#: Flags about the whole section, listed before any per-region flag.
SECTION_FLAGS = (FitFlag.DISPLACEMENT_OUTSIZED, FitFlag.FOLDS)
