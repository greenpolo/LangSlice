"""Tests for ``langslice_training.adaptive.curriculum.bins.compute_section_bins``.

We use mock plane extents (passed via ``plane_extent_overrides``) so the
test environment doesn't have to load real BrainGlobe atlases. Concrete
extents below match the public Allen / DeMBA / Princeton atlases roughly,
but they're chosen for round numbers that make the expected bin
distribution easy to read.
"""

from __future__ import annotations

from pathlib import Path

from langslice_training.adaptive.curriculum.bins import compute_section_bins
from langslice_training.rl.common.dataset import SingleSliceExample


def _make(section_id: str, atlas: str, plane: str, ap_mm: float) -> SingleSliceExample:
    return SingleSliceExample(
        image_path=Path("/dev/null"),
        atlas_name=atlas,
        plane=plane,  # type: ignore[arg-type]
        ap_mm=ap_mm,
        subject_id=section_id.split(":", 1)[0],
        section_id=section_id,
    )


# Plane extents picked for clean quintile boundaries:
# coronal:    13.18mm  → quintiles at 2.636, 5.272, 7.908, 10.544, 13.18
# sagittal:   11.40mm  → quintiles at 2.28, 4.56, 6.84, 9.12, 11.40
# horizontal:  8.00mm  → quintiles at 1.6, 3.2, 4.8, 6.4, 8.0
PLANE_EXTENTS = {
    ("allen_mouse_25um", "coronal"): 13.18,
    ("allen_mouse_25um", "sagittal"): 11.40,
    ("allen_mouse_25um", "horizontal"): 8.00,
}


def test_quintile_distribution_coronal() -> None:
    """One section per coronal quintile — verify each lands in its bin."""
    # Pick AP values near the centre of each quintile so binning is
    # unambiguous: q0=1.0, q1=4.0, q2=6.5, q3=9.0, q4=12.0
    examples = [
        _make("c:q0", "allen_mouse_25um", "coronal", 1.0),
        _make("c:q1", "allen_mouse_25um", "coronal", 4.0),
        _make("c:q2", "allen_mouse_25um", "coronal", 6.5),
        _make("c:q3", "allen_mouse_25um", "coronal", 9.0),
        _make("c:q4", "allen_mouse_25um", "coronal", 12.0),
    ]
    bins = compute_section_bins(examples, plane_extent_overrides=PLANE_EXTENTS)
    assert bins == {"c:q0": 0, "c:q1": 1, "c:q2": 2, "c:q3": 3, "c:q4": 4}


def test_quintile_distribution_all_three_planes() -> None:
    """Mix all three planes at once and verify per-plane binning is independent."""
    examples = [
        # coronal: extent 13.18, mids ~ 1.318, 3.954, 6.59, 9.226, 11.862
        _make("co:q0", "allen_mouse_25um", "coronal", 1.318),
        _make("co:q4", "allen_mouse_25um", "coronal", 11.862),
        # sagittal: extent 11.40, mids ~ 1.14, 3.42, 5.7, 7.98, 10.26
        _make("sa:q0", "allen_mouse_25um", "sagittal", 1.14),
        _make("sa:q2", "allen_mouse_25um", "sagittal", 5.7),
        # horizontal: extent 8.00, mids ~ 0.8, 2.4, 4.0, 5.6, 7.2
        _make("ho:q1", "allen_mouse_25um", "horizontal", 2.4),
        _make("ho:q4", "allen_mouse_25um", "horizontal", 7.2),
    ]
    bins = compute_section_bins(examples, plane_extent_overrides=PLANE_EXTENTS)
    assert bins == {
        "co:q0": 0,
        "co:q4": 4,
        "sa:q0": 0,
        "sa:q2": 2,
        "ho:q1": 1,
        "ho:q4": 4,
    }


def test_uniform_allocation_produces_uniform_bin_counts() -> None:
    """An evenly-spaced ramp across the plane extent produces ~uniform bin counts.

    100 evenly-spaced positions on a 5-bin schedule should produce 20 in each
    bin. ``_bin_index`` clamps the topmost fraction to 0.999999 so even the
    final position lands in q4 instead of overflowing to a sixth bin.
    """
    extent = 13.18
    extents = {("allen_mouse_25um", "coronal"): extent}
    n = 100
    examples = [
        _make(f"c:s{i}", "allen_mouse_25um", "coronal", extent * (i + 0.5) / n)
        for i in range(n)
    ]
    bins = compute_section_bins(examples, plane_extent_overrides=extents)
    counts = [0] * 5
    for v in bins.values():
        counts[v] += 1
    assert counts == [20, 20, 20, 20, 20]


def test_custom_n_bins() -> None:
    """``n_bins`` parameter controls the partitioning."""
    extent = 10.0
    extents = {("allen_mouse_25um", "coronal"): extent}
    # 10 evenly-spaced examples on n_bins=10 → one per bin
    examples = [
        _make(f"c:s{i}", "allen_mouse_25um", "coronal", extent * (i + 0.5) / 10)
        for i in range(10)
    ]
    bins = compute_section_bins(examples, n_bins=10, plane_extent_overrides=extents)
    assert sorted(bins.values()) == list(range(10))


def test_zero_extent_falls_back_to_bin_zero() -> None:
    """Defensive: if a plane has zero extent (degenerate atlas) all rows land in bin 0.

    ``_bin_index`` returns 0 for ``plane_extent_mm <= 0`` — verify our
    wrapper inherits that behaviour.
    """
    extents = {("degenerate_atlas", "coronal"): 0.0}
    examples = [
        _make("c:s0", "degenerate_atlas", "coronal", 0.0),
        _make("c:s1", "degenerate_atlas", "coronal", 5.0),
    ]
    bins = compute_section_bins(examples, plane_extent_overrides=extents)
    assert bins == {"c:s0": 0, "c:s1": 0}
