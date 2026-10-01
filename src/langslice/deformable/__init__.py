"""Deformable fit of a linearly placed atlas plane onto one section.

Shared by the linear agent's tooling and nonlinear, like ``affine.py``:
library engines (ANTs SyN, Elastix B-spline), atlas images, masks, and the
canonical per-section record. See ``CLAUDE.md`` in this package.
"""

from langslice.deformable.abba_atlas import AbbaAtlas
from langslice.deformable.atlas_images import atlas_images_for_host, excluded_ids, ventricle_ids
from langslice.deformable.fit import (
    CandidateFailure,
    PreparedFit,
    finish_fit,
    fit_candidates,
    fit_section,
    prepare_fit,
)
from langslice.deformable.geometry import Placement, placement_from_handoff
from langslice.deformable.record import DeformableRecord, diagnose
from langslice.deformable.render import draw_warped_borders, warped_border_coverage
from langslice.deformable.settings import FitSettings

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
    "fit_section",
    "placement_from_handoff",
    "prepare_fit",
    "ventricle_ids",
    "warped_border_coverage",
]
