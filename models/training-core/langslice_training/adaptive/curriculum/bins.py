"""Quintile binning for curriculum sampling.

Wraps slicebench's existing :func:`slicebench.score._bin_index` so the same
binning logic that drives slicebench's ``per_coord_bin`` aggregation also
drives RLVR / SFT curriculum weighting. The two views must agree, otherwise
weight updates computed from slicebench MAE land on the wrong rows.

Phase A — read-only: callers compute ``section_id → bin_idx`` once at
allocation-load time, log per-section reward / abs-error during rollouts,
and aggregate offline. No sampler change yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import slicebench's authoritative binning. The slicebench package lives at
# the repo root; pyproject's pythonpath puts the repo root on sys.path via
# the ``src`` entry implicitly during pytest, but slicebench itself needs
# the repo root explicitly added by callers (run from repo root, or via the
# ``langslice-rlvr`` shim which already does this).
from slicebench.score import _bin_index  # noqa: PLC2701 — stable internal API

if TYPE_CHECKING:
    from langslice_training.rl_core.dataset import Plane, SingleSliceExample


# Cache of (atlas_name, plane) → plane_extent_mm so we don't reload BrainGlobe
# atlases when the allocation pool revisits the same atlas+plane pair (it
# always does — there are ~20 unique pairs in a 14k allocation).
_PLANE_EXTENT_CACHE: dict[tuple[str, str], float] = {}


def _plane_extent_mm(atlas_name: str, plane: str) -> float:
    """Return ``pos_hi - pos_lo`` for ``(atlas_name, plane)``, cached.

    Loads the BrainGlobe atlas via ``langslice_harness.atlas.core.load_atlas``
    once per pair and computes extent from
    :func:`langslice_harness.atlas.core.get_position_range_mm`.
    """
    key = (atlas_name, plane)
    cached = _PLANE_EXTENT_CACHE.get(key)
    if cached is not None:
        return cached

    from langslice_harness.atlas.core import (
        get_position_range_mm,
        load_atlas,
    )

    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)  # type: ignore[arg-type]
    extent = float(pos_hi) - float(pos_lo)
    _PLANE_EXTENT_CACHE[key] = extent
    return extent


def compute_section_bins(
    allocation: list["SingleSliceExample"],
    n_bins: int = 5,
    *,
    plane_extent_overrides: dict[tuple[str, str], float] | None = None,
) -> dict[str, int]:
    """Compute the curriculum ``bin_idx`` for every section in ``allocation``.

    Each example's ``ap_mm`` (the slice-normal-axis position in mm) is bucketed
    against its ``(atlas_name, plane)`` extent using slicebench's
    :func:`_bin_index` — i.e. the same plane-relative quintile rule that
    slicebench's ``per_coord_bin`` block reports against. Returns
    ``{section_id: bin_idx}`` where ``bin_idx`` is in ``[0, n_bins)``.

    Parameters
    ----------
    allocation:
        List of :class:`rlvr.dataset.SingleSliceExample` rows (the canonical
        RLVR allocation pool — see :func:`rlvr.dataset.load_rlvr_allocation`).
    n_bins:
        Number of quintile buckets. Defaults to 5 to match slicebench.
    plane_extent_overrides:
        Optional ``{(atlas_name, plane): extent_mm}`` mapping. When provided,
        skips the BrainGlobe atlas load for the listed pairs. Mostly useful
        in unit tests so we can build deterministic synthetic allocations
        without dragging real atlases into the test environment.

    Notes
    -----
    Sagittal positions are NOT canonicalized here; the binning operates on
    the raw ``ap_mm`` value. ``_bin_index`` clamps the fractional position
    to ``[0, 1)`` so out-of-range values land in the boundary bins rather
    than overflowing.
    """
    overrides = plane_extent_overrides or {}
    out: dict[str, int] = {}
    for ex in allocation:
        key = (ex.atlas_name, str(ex.plane))
        if key in overrides:
            extent = float(overrides[key])
        else:
            extent = _plane_extent_mm(ex.atlas_name, str(ex.plane))
        bin_idx = _bin_index(float(ex.ap_mm), extent, n_bins)
        out[ex.section_id] = int(bin_idx)
    return out


def clear_plane_extent_cache() -> None:
    """Drop the in-process plane-extent cache (test helper)."""
    _PLANE_EXTENT_CACHE.clear()
