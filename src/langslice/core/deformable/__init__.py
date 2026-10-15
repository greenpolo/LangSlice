"""Deformable fit of a linearly placed atlas plane onto one section.

Shared by the linear agent's tooling and nonlinear, like ``core/affine.py``:
library engines (FireANTs SyN), atlas images, masks, and the
canonical per-section record. See ``CLAUDE.md`` in this package.
"""

from langslice.core.deformable.fit import (
    CandidateFailure,
    PreparedFit,
    finish_fit,
    fit_prepared,
    prepare_fit,
    prepare_warp,
)
from langslice.core.deformable.geometry import Placement, placement_from_handoff
from langslice.core.deformable.plane_images import (
    ventricle_ids,
)
from langslice.core.deformable.record import DeformableRecord, diagnose
from langslice.core.deformable.render import (
    draw_warped_borders,
    resampled_record,
    warp_section_image,
    warp_source_points,
    warped_border_layers,
)
from langslice.core.deformable.settings import FitSettings

__all__ = [
    "CandidateFailure",
    "DeformableRecord",
    "FitSettings",
    "Placement",
    "PreparedFit",
    "diagnose",
    "draw_warped_borders",
    "finish_fit",
    "fit_prepared",
    "placement_from_handoff",
    "prepare_fit",
    "prepare_warp",
    "resampled_record",
    "ventricle_ids",
    "warp_section_image",
    "warp_source_points",
    "warped_border_layers",
]
