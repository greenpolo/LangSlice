"""Constraint enforcement for slice positions."""

from __future__ import annotations

from langslice.brain.types import SlicePosition


def enforce_constraints(
    slices: list[SlicePosition],
    *,
    ordering: str,
    thickness_mm: float,
    z_axis: str,
) -> list[SlicePosition]:
    """Validate and fix ordering/spacing violations.

    Returns a new list.  Locked positions may be nudged to satisfy hard
    constraints (minimum spacing).
    """
    if len(slices) <= 1:
        return list(slices)

    out = [
        SlicePosition(s.filename, s.index, s.position_mm, s.source, s.locked)
        for s in slices
    ]

    # For PA axis, we work in negated space so "increasing" logic applies,
    # then negate back.
    if z_axis == "PA":
        for s in out:
            s.position_mm = -s.position_mm

    if ordering == "loose":
        _apply_swaps(out)

    if ordering in ("strict", "loose"):
        _enforce_monotonic(out, thickness_mm)

    _enforce_min_spacing(out, thickness_mm)

    if z_axis == "PA":
        for s in out:
            s.position_mm = -s.position_mm

    return out


def _apply_swaps(slices: list[SlicePosition]) -> None:
    """Scan for adjacent pairs that are reversed and swap them (one pass)."""
    i = 0
    while i < len(slices) - 1:
        if slices[i].position_mm > slices[i + 1].position_mm:
            # Swap positions (keep filenames/indices in place)
            slices[i].position_mm, slices[i + 1].position_mm = (
                slices[i + 1].position_mm,
                slices[i].position_mm,
            )
            slices[i].source, slices[i + 1].source = (
                slices[i + 1].source,
                slices[i].source,
            )
            i += 2  # skip past swapped pair to avoid cascading
        else:
            i += 1


def _enforce_monotonic(slices: list[SlicePosition], thickness_mm: float) -> None:
    """Clamp any non-monotonic slice to midpoint between neighbors."""
    for i in range(1, len(slices)):
        if slices[i].position_mm <= slices[i - 1].position_mm:
            if i < len(slices) - 1:
                mid = (slices[i - 1].position_mm + slices[i + 1].position_mm) / 2
                slices[i].position_mm = max(
                    slices[i - 1].position_mm + thickness_mm, mid
                )
            else:
                slices[i].position_mm = slices[i - 1].position_mm + thickness_mm


def _enforce_min_spacing(slices: list[SlicePosition], thickness_mm: float) -> None:
    """Nudge slices that are closer than thickness_mm apart."""
    for i in range(1, len(slices)):
        gap = abs(slices[i].position_mm - slices[i - 1].position_mm)
        if gap < thickness_mm:
            if slices[i].position_mm >= slices[i - 1].position_mm:
                slices[i].position_mm = slices[i - 1].position_mm + thickness_mm
            else:
                slices[i].position_mm = slices[i - 1].position_mm - thickness_mm
