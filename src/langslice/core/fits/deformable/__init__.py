"""Deformable fit of a linearly placed atlas plane onto one section.

Shared by the linear agent's tooling and nonlinear, like
``core/geometry/affine.py``: library engines (FireANTs SyN), atlas images,
masks, and the canonical per-section record. See ``CLAUDE.md`` in this package.
"""

from langslice.core.fits.deformable.fit import (
    CandidateFailure,
    PreparedFit,
    finish_fit,
    fit_prepared,
    prepare_fit,
    prepare_warp,
    warp_from_native_map,
)
from langslice.core.fits.deformable.geometry import (
    Placement,
    placement_from_handoff,
)
from langslice.core.fits.deformable.plane_images import ventricle_ids
from langslice.core.fits.deformable.record import DeformableRecord, diagnose
from langslice.core.fits.deformable.render import (
    draw_warped_borders,
    resampled_record,
    warp_section_image,
    warp_source_points,
    warped_border_layers,
)
from langslice.core.fits.deformable.settings import FitSettings

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
    "warp_from_native_map",
    "warp_section_image",
    "warp_source_points",
    "warped_border_layers",
]
