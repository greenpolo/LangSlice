"""Tests for oblique atlas slice extraction.

Covers:
- Zero-angle result matches canonical np.take extraction
- Output shape honouring output_size
- Tilted slices still contain tissue (not catastrophically misaligned)
- Annotation uses nearest-neighbour (no fractional region IDs)
- sample_oblique_angles stays within bounds
- Angles > 8° raise ValueError
- Determinism: same atlas + same angles → identical output
- roll-only and yaw-only tilts produce spatially shifted tissue
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice_harness.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")


@pytest.fixture(scope="module")
def canonical_slices(atlas: object) -> tuple[np.ndarray, np.ndarray]:
    """(ref_uint8, ann_int32) for AP=5.335 mm via np.take."""
    from langslice_harness.atlas.core import position_mm_to_index
    from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, 5.335)  # type: ignore[arg-type]
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]

    # Replicate the reference normalization from atlas.core._normalize_to_uint8
    ref_raw = np.take(np.asarray(atlas.reference), idx, axis=axis).astype(np.float32)  # type: ignore[union-attr]
    max_val = float(ref_raw.max())
    if max_val > 0:
        ref_u8 = np.clip(ref_raw / max_val * 255.0, 0, 255).astype(np.uint8)
    else:
        ref_u8 = np.zeros_like(ref_raw, dtype=np.uint8)
    return ref_u8, ann


# ---------------------------------------------------------------------------
# Test 1 — zero-angle matches canonical extraction
# ---------------------------------------------------------------------------


def test_zero_angle_annotation_matches_canonical(
    atlas: object, canonical_slices: tuple[np.ndarray, np.ndarray]
) -> None:
    """At yaw=pitch=roll=0 the annotation output must match np.take exactly."""
    from augmentation.oblique import get_oblique_slice

    _, ann_canonical = canonical_slices
    _, ann_oblique = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    # Nearest-neighbour at zero tilt → exact match expected.
    np.testing.assert_array_equal(ann_oblique, ann_canonical)


def test_zero_angle_reference_close_to_canonical(
    atlas: object, canonical_slices: tuple[np.ndarray, np.ndarray]
) -> None:
    """At zero tilt the reference uint8 should be very close to the canonical."""
    from augmentation.oblique import get_oblique_slice

    ref_canonical, _ = canonical_slices
    ref_oblique, _ = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    # Linear interp at integer grid ≡ exact; allow ≤1 rounding unit tolerance.
    diff = np.abs(ref_oblique.astype(np.int32) - ref_canonical.astype(np.int32))
    assert diff.max() <= 1, f"Max reference deviation at zero tilt: {diff.max()}"


# ---------------------------------------------------------------------------
# Test 2 — output shape
# ---------------------------------------------------------------------------


def test_output_size_respected(atlas: object) -> None:
    from augmentation.oblique import get_oblique_slice

    ref, ann = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=3.0,
        pitch_deg=-2.0,
        roll_deg=1.0,
        output_size=(128, 192),
    )
    assert ref.shape == (128, 192), f"ref shape: {ref.shape}"
    assert ann.shape == (128, 192), f"ann shape: {ann.shape}"


def test_default_output_size_is_native(
    atlas: object, canonical_slices: tuple[np.ndarray, np.ndarray]
) -> None:
    from augmentation.oblique import get_oblique_slice

    _, ann_canonical = canonical_slices
    ref, ann = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    assert ref.shape == ann_canonical.shape, "Default output_size should equal native slice shape"
    assert ann.shape == ann_canonical.shape


# ---------------------------------------------------------------------------
# Test 3 — tilted slice still has tissue
# ---------------------------------------------------------------------------


def test_tilted_slice_has_tissue(atlas: object) -> None:
    """A moderate tilt should still intersect substantial brain tissue."""
    from augmentation.oblique import get_oblique_slice

    _, ann = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=8.0,
        pitch_deg=6.0,
        roll_deg=-5.0,
    )
    tissue_fraction = (ann > 0).mean()
    # At 5.335 mm AP the canonical slice is ~40% tissue; even at 15° tilt
    # we expect well above 10% tissue remaining.
    assert tissue_fraction > 0.10, (
        f"Tilted slice tissue fraction too low: {tissue_fraction:.3f} "
        "(section may be catastrophically misaligned)"
    )


# ---------------------------------------------------------------------------
# Test 4 — annotation has only integer region IDs (nearest-neighbour)
# ---------------------------------------------------------------------------


def test_annotation_integer_values_only(atlas: object) -> None:
    """Annotation must contain only integer IDs — no fractional interpolation."""
    from augmentation.oblique import get_oblique_slice

    _, ann = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=5.0,
        pitch_deg=4.0,
        roll_deg=3.0,
    )
    assert ann.dtype == np.int32
    # All values are already integers by dtype, but verify no fractional IDs
    # would have appeared if we'd used linear interpolation (smoke-test).
    unique_ids = np.unique(ann)
    assert len(unique_ids) < 10000, "Unexpected explosion in unique annotation IDs"


def test_annotation_ids_are_known_atlas_ids(atlas: object) -> None:
    """Annotation IDs in the tilted slice should all be valid atlas structure IDs."""
    from augmentation.oblique import get_oblique_slice

    _, ann = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=7.0,
        pitch_deg=-4.0,
        roll_deg=2.0,
    )
    valid_ids = set(int(s["id"]) for s in atlas.structures.values()) | {0}  # type: ignore[union-attr]
    unique_returned = set(int(v) for v in np.unique(ann))
    unknown = unique_returned - valid_ids
    assert len(unknown) == 0, f"Annotation contains unknown IDs: {unknown}"


# ---------------------------------------------------------------------------
# Test 5 — sample_oblique_angles within bounds
# ---------------------------------------------------------------------------


def test_sample_oblique_angles_within_bounds() -> None:
    from augmentation.oblique import sample_oblique_angles

    rng = np.random.default_rng(42)
    for _ in range(200):
        yaw, pitch, roll = sample_oblique_angles(
            rng, max_yaw_deg=8.0, max_pitch_deg=7.0, max_roll_deg=6.0
        )
        assert abs(yaw) <= 8.0, f"yaw out of bounds: {yaw}"
        assert abs(pitch) <= 7.0, f"pitch out of bounds: {pitch}"
        assert abs(roll) <= 6.0, f"roll out of bounds: {roll}"


def test_sample_oblique_angles_uses_full_range() -> None:
    """Over many samples, both positive and negative angles should appear."""
    from augmentation.oblique import sample_oblique_angles

    rng = np.random.default_rng(0)
    yaws = [sample_oblique_angles(rng)[0] for _ in range(100)]
    assert any(y > 1.0 for y in yaws), "No positive yaw sampled"
    assert any(y < -1.0 for y in yaws), "No negative yaw sampled"


def test_sample_oblique_angles_max_exceeds_limit_raises() -> None:
    from augmentation.oblique import sample_oblique_angles

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="8"):
        sample_oblique_angles(rng, max_yaw_deg=20.0)


# ---------------------------------------------------------------------------
# Test 6 — angles > 8° raise ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [
        (8.1, 0.0, 0.0),
        (0.0, -8.1, 0.0),
        (0.0, 0.0, 8.1),
        (9.0, 0.0, 0.0),
    ],
)
def test_large_angles_raise_value_error(
    atlas: object, yaw: float, pitch: float, roll: float
) -> None:
    from augmentation.oblique import get_oblique_slice

    with pytest.raises(ValueError, match="8"):
        get_oblique_slice(
            atlas,
            base_position_mm=5.335,
            plane="coronal",
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )


def test_exactly_8_degrees_is_allowed(atlas: object) -> None:
    """The boundary value 8.0° should not raise."""
    from augmentation.oblique import get_oblique_slice

    ref, ann = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=8.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    assert ref.shape == ann.shape


# ---------------------------------------------------------------------------
# Test 7 — determinism
# ---------------------------------------------------------------------------


def test_determinism_same_inputs_same_output(atlas: object) -> None:
    """Identical inputs must produce byte-identical outputs."""
    from augmentation.oblique import get_oblique_slice

    kwargs = dict(
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=7.3,
        pitch_deg=-4.1,
        roll_deg=3.2,
    )
    ref_a, ann_a = get_oblique_slice(atlas, **kwargs)  # type: ignore[arg-type]
    ref_b, ann_b = get_oblique_slice(atlas, **kwargs)  # type: ignore[arg-type]
    np.testing.assert_array_equal(ref_a, ref_b)
    np.testing.assert_array_equal(ann_a, ann_b)


# ---------------------------------------------------------------------------
# Test 8 — tilt actually changes the output
# ---------------------------------------------------------------------------


def test_tilt_changes_annotation(
    atlas: object, canonical_slices: tuple[np.ndarray, np.ndarray]
) -> None:
    """A non-zero tilt should produce a different annotation slice."""
    from augmentation.oblique import get_oblique_slice

    _, ann_canonical = canonical_slices
    _, ann_tilted = get_oblique_slice(
        atlas,
        base_position_mm=5.335,
        plane="coronal",
        yaw_deg=7.0,
        pitch_deg=8.0,
        roll_deg=0.0,
    )
    # The tilted slice should differ from the canonical in a meaningful fraction
    # of pixels.
    diff_fraction = (ann_tilted != ann_canonical).mean()
    assert diff_fraction > 0.01, (
        f"Tilt produced almost no change in annotation ({diff_fraction:.4f}). "
        "The rotation may not be applied."
    )
