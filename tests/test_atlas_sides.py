"""One side of a region ("CTX:left"), in the section's displayed frame; outsized warps."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from langslice.core.atlas.sides import (
    SideError,
    has_sides,
    ml_halves,
    native_left,
    split_side,
)
from langslice.core.deformable import FitSettings, draw_warped_borders, fit_section, prepare_fit
from langslice.core.deformable.atlas_images import regions_mask, whole_region_ids
from langslice.core.deformable.record import (
    OUTSIZED_MAX_FRACTION,
    OUTSIZED_MEDIAN_MM,
    diagnose,
    displacement_report,
)
from tests.deformable_synthetic import (
    CTX,
    HY,
    SMOOTH_FIELD,
    STR,
    SyntheticAtlas,
    placement,
    render_section,
)

#: The synthetic atlas's ML axis has 160 voxels: BrainGlobe's split is at 80.
SPLIT = 80


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


# --- the grammar ---------------------------------------------------------------


def test_entries_name_a_region_and_optionally_a_side():
    assert split_side("CTX") == ("CTX", None)
    assert split_side(" CTX:Left ") == ("CTX", "left")
    assert split_side(315) == ("315", None)
    assert split_side("315:right") == ("315", "right")
    with pytest.raises(ValueError, match="Unknown side"):
        split_side("CTX:dorsal")
    assert has_sides(["STR", "CTX:left"]) and not has_sides(["STR", "CTX"])


# --- left and right in the displayed frame ---------------------------------------


def test_left_follows_the_placement_not_the_atlas_labels(atlas):
    high, gradient = ml_halves(atlas, 0.1, "coronal")
    # Coronal asr: native columns are the ML axis; the second half starts at 80.
    assert gradient == pytest.approx([1.0, 0.0])
    assert not high[:, :SPLIT].any() and high[:, SPLIT:].all()
    upright = native_left(atlas, 0.1, "coronal", 0, 0, np.eye(2))
    assert upright[:, :SPLIT].all() and not upright[:, SPLIT:].any()
    # A placement that mirrors the atlas swaps the sides, as it swaps the tissue.
    mirrored = native_left(atlas, 0.1, "coronal", 0, 0, np.diag([-2.0, 2.0]))
    assert np.array_equal(mirrored, ~upright)
    # Upside down (180 degrees) is a mirror in x as well.
    turned = native_left(atlas, 0.1, "coronal", 0, 0, -np.eye(2))
    assert np.array_equal(turned, ~upright)
    # A small tilt keeps the sides.
    angle = np.deg2rad(20)
    tilt = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert np.array_equal(native_left(atlas, 0.1, "coronal", 0, 0, tilt), upright)


def test_sides_are_refused_where_they_mean_nothing(atlas):
    quarter = np.array([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(SideError) as turned:
        native_left(atlas, 0.1, "coronal", 0, 0, quarter)
    assert turned.value.code == "SIDES_AMBIGUOUS"
    with pytest.raises(SideError) as sagittal:
        native_left(atlas, 0.1, "sagittal", 0, 0, np.eye(2))
    assert sagittal.value.code == "NO_SIDES"


def test_an_asymmetric_atlas_is_split_by_its_own_hemispheres(atlas):
    lopsided = SyntheticAtlas()
    lopsided.metadata = {"species": "mouse", "symmetric": False}
    hemispheres = np.ones(lopsided.annotation.shape, dtype=np.uint8)
    hemispheres[..., 70:] = 2  # the midline is NOT the volume's centre here
    lopsided.hemispheres = hemispheres  # type: ignore[attr-defined]
    high, _ = ml_halves(lopsided, 0.1, "coronal")
    assert not high[:, :70].any() and high[:, 70:].all()


def test_region_masks_take_one_side(atlas):
    labels = atlas.annotation[0]
    left = native_left(atlas, 0.1, "coronal", 0, 0, np.eye(2))
    both = regions_mask(atlas, labels, ["CTX"])
    one = regions_mask(atlas, labels, ["CTX:left"], left)
    assert one.any() and not one[:, SPLIT:].any()
    assert np.array_equal(one, both & left)
    assert whole_region_ids(atlas, ["CTX:left", "HY"]) == frozenset({HY})
    with pytest.raises(ValueError, match="needs the placement's sides"):
        regions_mask(atlas, labels, ["CTX:left"])


# --- the deformable fit ----------------------------------------------------------


def test_a_one_sided_exclusion_keeps_the_other_side(atlas):
    image, _ = render_section(atlas, SMOOTH_FIELD())
    settings = FitSettings(engine="elastix", detail="coarse", exclude=("CTX:left",))
    prepared = prepare_fit(image, atlas, placement(), settings)
    mask = prepared.excluded_mask
    assert mask is not None
    cortex = atlas.annotation[0] == CTX
    assert mask[cortex & (np.arange(160) < SPLIT)].all()
    assert not mask[:, SPLIT:].any()
    # Whole-region ids (skipped by the diagnostics) do not include CTX.
    assert CTX not in prepared.excluded
    # The moving (atlas) image is blanked on the left only (intensities are
    # renormalized over what is kept, so compare where it is blank).
    both = prepare_fit(image, atlas, placement(), replace(settings, exclude=("CTX",)))
    third = prepared.inputs.moving.shape[1] // 3
    blank, blank_both = prepared.inputs.moving == 0, both.inputs.moving == 0
    assert np.array_equal(blank[:, :third], blank_both[:, :third])
    assert blank[:, -third:].sum() < blank_both[:, -third:].sum()
    assert prepared.inputs.moving_mask.sum() > both.inputs.moving_mask.sum()
    # A one-sided structure restricts the fit to that side's neighbourhood.
    restricted = prepare_fit(image, atlas, placement(),
                             FitSettings(engine="elastix", detail="coarse",
                                         structures=("STR:left",)))
    assert restricted.details["structures"] == [STR]
    with pytest.raises(ValueError, match="None of the chosen structures"):
        prepare_fit(image, atlas, placement(),
                    FitSettings(engine="elastix", detail="coarse", structures=("STR:right",)))


def test_one_sided_highlights_and_marks_are_drawn_on_that_side(atlas):
    pytest.importorskip("ants", reason="the deformable fit needs antspyx")
    image, _ = render_section(atlas, SMOOTH_FIELD())
    record = fit_section(image, atlas, placement(),
                         FitSettings(engine="ants", detail="coarse", exclude=("CTX:left",)))
    pink = np.asarray(draw_warped_borders(image, record, atlas, marked=["CTX:left"]))
    marked = (pink[..., 0] > 200) & (pink[..., 1] < 140)
    # Native column 80 sits at section x = 2 * 80 + 10 = 170.
    assert marked[:, :160].sum() > 50 and marked[:, 180:].sum() == 0


# --- outsized displacement -------------------------------------------------------


def test_outsized_displacement_is_flagged_by_median_or_by_max():
    tissue = np.zeros((200, 200), dtype=bool)
    tissue[20:180, 20:180] = True  # 160 px at 0.05 mm: an 8 mm section
    mm = 0.05
    labels = np.where(tissue, 7, 0)

    def flags(field: np.ndarray) -> list[str]:
        report = diagnose(field, mm, labels, labels, tissue, ventricles=frozenset(),
                          excluded=frozenset(), acronyms={})
        return [flag["code"] for flag in report["flags"]]

    small = np.zeros((200, 200, 2))
    small[..., 0] = 0.1
    assert "DISPLACEMENT_OUTSIZED" not in flags(small)
    drift = np.zeros((200, 200, 2))
    drift[..., 0] = OUTSIZED_MEDIAN_MM + 0.05  # a uniform shift: only the median sees it
    assert "DISPLACEMENT_OUTSIZED" in flags(drift)
    spike = np.zeros((200, 200, 2))
    yy, xx = np.indices((200, 200))
    spike[..., 1] = 1.2 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 15.0 ** 2))
    report = displacement_report(spike, mm, tissue)
    assert report["tissue_extent_mm"] == pytest.approx(8.0)
    assert report["max_limit_mm"] == pytest.approx(OUTSIZED_MAX_FRACTION * 8.0)
    assert report["median_mm"] < 0.1 and report["outsized"]
    assert "DISPLACEMENT_OUTSIZED" in flags(spike)
