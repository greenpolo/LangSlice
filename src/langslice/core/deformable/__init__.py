"""Deformable fit of a linearly placed atlas plane onto one section.

Shared by the linear agent's tooling and nonlinear, like ``affine.py``:
library engines (ANTs SyN, Elastix B-spline), atlas images, masks, and the
canonical per-section record. See ``CLAUDE.md`` in this package.
"""

from langslice.core.deformable.abba_atlas import AbbaAtlas
from langslice.core.deformable.atlas_images import (
    atlas_images_for_host,
    excluded_ids,
    ventricle_ids,
)
from langslice.core.deformable.fit import (
    CandidateFailure,
    PreparedFit,
    finish_fit,
    fit_candidates,
    fit_prepared,
    fit_section,
    prepare_fit,
)
from langslice.core.deformable.geometry import Placement, placement_from_handoff
from langslice.core.deformable.record import DeformableRecord, diagnose
from langslice.core.deformable.render import (
    draw_warped_borders,
    resampled_record,
    warp_section_image,
    warped_border_coverage,
    warped_border_layers,
)
from langslice.core.deformable.settings import FitSettings, traced_settings

__all__ = [
    "AbbaAtlas",
    "CandidateFailure",
    "DeformableRecord",
    "FitSettings",
    "Placement",
    "PreparedFit",
    "atlas_images_for_host",
    "diagnose",
    "draw_warped_borders",
    "excluded_ids",
    "finish_fit",
    "fit_candidates",
    "fit_prepared",
    "fit_section",
    "placement_from_handoff",
    "prepare_fit",
    "resampled_record",
    "traced_settings",
    "ventricle_ids",
    "warp_section_image",
    "warped_border_coverage",
    "warped_border_layers",
]
