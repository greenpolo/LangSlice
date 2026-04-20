"""Session-state builder and artifact-key conventions for the estimation harness."""

from __future__ import annotations

from typing import Any

from langslice.atlas.space import Plane

ARTIFACT_TARGET = "target"  # single-slice target image key
ARTIFACT_TARGET_PREFIX = "target:"  # multi-slice: target:1, target:2, ...
ARTIFACT_ATLAS_PREFIX = "atlas:"  # cached atlas slices: atlas:3.20
ARTIFACT_ZOOM_PREFIX = "zoom:"
ARTIFACT_SIDE_BY_SIDE_PREFIX = "side_by_side:"


_PLANE_TO_AXIS_LABEL: dict[str, str] = {
    "coronal": "AP",
    "sagittal": "ML",
    "horizontal": "DV",
}


def axis_label_for(plane: Plane) -> str:
    return _PLANE_TO_AXIS_LABEL[plane]


def build_initial_state(
    *,
    atlas_name: str,
    plane: Plane,
    pos_lo: float,
    pos_hi: float,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Return the initial `tool_context.state` dict for a run."""
    return {
        "atlas": atlas_name,
        "plane": plane,
        "axis_label": axis_label_for(plane),
        "pos_lo": pos_lo,
        "pos_hi": pos_hi,
        "n_slices": n_slices,
        "interval_mm": interval_mm,
        "thickness_um": thickness_um,
        "fetched_positions": [],
        "saw_broad_sweep": False,
        "saw_narrow_sweep": False,
        "images_fetched": 0,
        "submit_attempts": 0,
        "result": None,
        "max_iterations": max_iterations,
    }


def atlas_key(position_mm: float) -> str:
    """Canonical artifact key for an atlas slice at a given position."""
    return f"{ARTIFACT_ATLAS_PREFIX}{position_mm:.2f}"


def target_key(index: int | None = None) -> str:
    """Canonical artifact key for a target image ('target' or 'target:N')."""
    if index is None:
        return ARTIFACT_TARGET
    return f"{ARTIFACT_TARGET_PREFIX}{index}"
