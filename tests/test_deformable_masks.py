"""The masks a deformable fit is given (``core.deformable.masks``,
``prepare_fit(section_weights=)``) and the FireANTs engine's determinism.

No fit sorts the section by itself: a section mask (``set_mask``'s EM
sorting, :func:`~langslice.core.deformable.masks.em_weights`) reaches a fit as
weights thresholded at 0.5 and ANDed into the section (fixed) mask; the
atlas-side damage zone stays as it was. The Elastix affine and the fits of
traced lines keep the footprint masks, built exactly as before.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.core.deformable import FitSettings, prepare_fit
from langslice.core.deformable.fit import (
    RESTRICTED_TAPER_MM,
    run_fit,
    run_prepared,
    with_section_mask,
)
from langslice.core.deformable.masks import (
    EM_MATCH_THRESHOLD,
    SORTING_KEYS,
    Masks,
    dilate,
    em_weights,
    footprint_masks,
)
from tests.deformable_synthetic import (
    SyntheticAtlas,
    bump_field,
    input_fingerprint,
    placement,
    render_section,
    same_inputs,
)

needs_engine = pytest.mark.skipif(importlib.util.find_spec("fireants") is None,
                                  reason="FireANTs missing")
#: A 0.4 mm hole in the synthetic section's cortex. On these flat, textureless
#: regions the EM's 0.8 mm contrast blocks absorb a hole of 0.5 mm or more
#: (a block's offset fits it); textured tissue keeps larger holes apart.
HOLE = (150, 112, 8)


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _hole(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    yy, xx = np.indices((height, width))
    cx, cy, r = HOLE
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2


def _section(atlas: SyntheticAtlas, *, fluorescent: bool) -> tuple[Image.Image, np.ndarray]:
    """A section with a hole (slide background): brightfield (dark tissue on a
    light slide) or fluorescent (bright tissue on black)."""
    field = bump_field([(130, 120, 45, 0.05, 0.02)])
    hole = _hole((340, 260))
    image, labels = render_section(atlas, field, remove=hole)
    if fluorescent:
        image = Image.fromarray(255 - np.asarray(image))
    return image, labels


@pytest.mark.parametrize("fluorescent", [False, True])
def test_section_weights_leave_a_hole_out_of_the_fixed_mask(atlas, fluorescent):
    """The weights a mask holds (here the EM sorting of the section against
    the atlas image placed on it) drop the hole from the fit's section mask
    and keep the matching tissue; the atlas-side mask is untouched."""
    image, labels = _section(atlas, fluorescent=fluorescent)
    settings = FitSettings(detail="coarse")
    plain = prepare_fit(image, atlas, placement(), settings)
    hole = _hole((340, 260))
    # The sorting of this section: P(match) 0 in the hole and on the slide, 1 on tissue.
    weights = np.where(labels != 0, 1.0, 0.0).astype(np.float32)
    weights[hole] = 0.0
    masked = prepare_fit(image, atlas, placement(), settings, section_weights=weights)
    assert plain.inputs is not None and masked.inputs is not None
    assert masked.details["section_mask"] == "weights thresholded at 0.5"
    assert "section_mask" not in plain.details
    # No fit sorts by itself: the unmasked fit compares the whole footprint, hole included.
    in_hole = masked.grid.to_working(hole.astype(np.float32)) > 0.5
    assert plain.inputs.fixed_mask[in_hole].all() and not masked.inputs.fixed_mask[in_hole].any()
    assert np.array_equal(masked.inputs.fixed_mask & plain.inputs.fixed_mask,
                          masked.inputs.fixed_mask)
    assert np.array_equal(masked.inputs.moving_mask, plain.inputs.moving_mask)
    tissue = masked.grid.to_working(((labels != 0) & ~dilate(hole, 12)).astype(np.float32)) > 0.5
    assert masked.inputs.fixed_mask[tissue].mean() > 0.99


def test_weights_are_thresholded_at_one_half_and_checked(atlas):
    image, _ = _section(atlas, fluorescent=False)
    settings = FitSettings(detail="coarse")
    plain = prepare_fit(image, atlas, placement(), settings)
    assert plain.inputs is not None
    for level in (EM_MATCH_THRESHOLD, 1.0):  # at the threshold counts as matching
        weights = np.full((260, 340), level, dtype=np.float32)
        masked = prepare_fit(image, atlas, placement(), settings, section_weights=weights)
        assert masked.inputs is not None
        assert np.array_equal(masked.inputs.fixed_mask, plain.inputs.fixed_mask)
    with pytest.raises(ValueError, match="masks are empty"):  # just below it does not
        prepare_fit(image, atlas, placement(), settings,
                    section_weights=np.full((260, 340), 0.49, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        prepare_fit(image, atlas, placement(), settings,
                    section_weights=np.ones((10, 10), dtype=np.float32))


def test_a_script_s_boolean_mask_and_the_weights_both_narrow_the_fixed_mask(atlas):
    image, _ = _section(atlas, fluorescent=False)
    prepared = prepare_fit(image, atlas, placement(), FitSettings(detail="coarse"))
    assert prepared.inputs is not None
    before = int(prepared.inputs.fixed_mask.sum())
    keep = np.zeros((260, 340), dtype=bool)
    keep[:, :170] = True
    with_section_mask(prepared, keep)
    assert 0 < int(prepared.inputs.fixed_mask.sum()) < before
    assert not hasattr(prepared, "em")


def test_there_is_no_em_path_inside_a_fit():
    from langslice.core.deformable import fit, masks

    assert not hasattr(masks, "EMContext") and not hasattr(masks, "em_masks")
    assert not hasattr(fit.PreparedFit, "em")
    assert set(SORTING_KEYS) == {"match_sd", "background_sd", "artifact_sd",
                                 "background_mean", "artifact_mean"}


def test_em_weights_follow_the_contrast_map_on_both_polarities():
    """Textured tissue under a linear contrast (dark tissue on a light slide,
    or bright tissue on black) keeps P(match) high; a 0.6 mm hole drops."""
    rng = np.random.default_rng(1)
    yy, xx = np.indices((400, 520))  # a 20 um grid
    tissue = (xx - 260) ** 2 / 220 ** 2 + (yy - 200) ** 2 / 160 ** 2 < 1
    coarse = cv2.GaussianBlur(rng.random((20, 26)).astype(np.float32), (0, 0), 1)
    texture = cv2.resize(coarse, (520, 400), interpolation=cv2.INTER_CUBIC)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    atlas = np.where(tissue, 0.2 + 0.8 * texture, 0.0).astype(np.float32)
    hole = (xx - 300) ** 2 + (yy - 180) ** 2 < 15 ** 2
    for polarity, section, background in (("brightfield", 0.85 - 0.6 * atlas, 0.95),
                                          ("fluorescence", 0.05 + 0.7 * atlas, 0.02)):
        section = np.where(tissue & ~hole, section, background)
        section = np.clip(section + rng.normal(0, 0.03, section.shape), 0, 1)
        weights = em_weights(section.astype(np.float32), atlas, dilate(tissue, 10), 0.02)
        kept = weights["WM"] >= 0.5
        assert kept[tissue & ~dilate(hole, 12)].mean() > 0.99, polarity
        assert not kept[~dilate(~hole, 4)].any(), polarity


def test_em_weights_take_the_sorting_widths_and_means_and_are_deterministic():
    rng = np.random.default_rng(2)
    placed = np.zeros((120, 160), dtype=np.int32)
    placed[20:100, 30:130] = 5
    excluded = np.zeros(placed.shape, dtype=bool)
    excluded[50:60, 60:70] = True
    first, second = (footprint_masks(placed, excluded, 0.02) for _ in range(2))
    assert all(np.array_equal(getattr(first, k), getattr(second, k))
               for k in ("fixed", "moving", "damage_zone"))
    J = rng.random(placed.shape).astype(np.float32)
    warped = rng.random(placed.shape).astype(np.float32)
    a = em_weights(J, warped, first.fixed, 0.02)
    b = em_weights(J, warped, first.fixed, 0.02)
    assert np.array_equal(a["WM"], b["WM"]) and a["params"] == b["params"]
    given = em_weights(J, warped, first.fixed, 0.02, sorting={
        "match_sd": 0.3, "background_sd": 0.2, "artifact_sd": 0.4,
        "background_mean": 0.8, "artifact_mean": 0.1})
    assert given["params"]["sigmaM"] == 0.3 and given["params"]["sigmaB"] == 0.2
    assert given["params"]["sigmaA"] == 0.4
    assert given["params"]["muB"] == 0.8 and given["params"]["muA"] == 0.1
    assert not np.array_equal(given["WM"], a["WM"])
    with pytest.raises(ValueError, match="Unknown sorting key"):
        em_weights(J, warped, first.fixed, 0.02, sorting={"threshold": 0.5})
    with pytest.raises(ValueError, match="match_sd must be above 0"):
        em_weights(J, warped, first.fixed, 0.02, sorting={"match_sd": 0.0})


#: The engine inputs (masks, images, edge channels) prepare_fit built for
#: these settings before the EM mask and FireANTs (main at f1770df2), as fingerprints:
#: the Elastix affine, a restricted step's and a whole-section fit's first run.
UNCHANGED_INPUTS = {
    "elastix": ("85b2b163014fdea2", [
        23948.11, 19925.92, 9606.64, 5949.57, 3090.42, 1370.41, 3199.30, 1548.77]),
    "restricted": ("66b977e835f94980", [
        23948.11, 19925.92, 9606.64, 5949.57, 2993.03, 1291.40, 2502.89, 959.24]),
    "first_run": ("85b2b163014fdea2", [
        23948.11, 19925.92, 9606.64, 5949.57, 3090.42, 1370.41, 3199.30, 1548.77]),
}


def test_the_affine_and_restricted_steps_get_the_inputs_they_always_got(atlas):
    """The Elastix affine keeps the footprint masks and one run; a restricted
    step's and a whole-section fit's first run are given the same as before."""
    image, _ = render_section(atlas, bump_field([(130, 120, 45, 0.05, 0.02)]))
    for name, settings in (
        ("elastix", FitSettings(engine="elastix", detail="coarse", exclude=("TH",))),
        ("restricted", FitSettings(detail="coarse", structures=("STR",), exclude=("TH",))),
        ("first_run", FitSettings(detail="coarse", exclude=("TH",))),
    ):
        prepared = prepare_fit(image, atlas, placement(), settings)
        inputs = prepared.inputs
        assert inputs is not None
        fingerprint = input_fingerprint([
            inputs.fixed_mask, inputs.moving_mask, inputs.fixed, inputs.moving,
            inputs.fixed_edges, inputs.moving_edges])
        assert same_inputs(fingerprint, UNCHANGED_INPUTS[name]), name


def _fit_in_worker(args):  # a spawn worker's whole fit, as the CPU pool runs it
    return run_fit(*args).field_mm


@needs_engine
def test_identical_inputs_give_identical_fields_in_and_across_processes(atlas):
    """Bit for bit: twice in this process and once in a fresh one."""
    image, _ = _section(atlas, fluorescent=False)
    prepared = prepare_fit(image, atlas, placement(), FitSettings(detail="coarse"))
    assert prepared.inputs is not None and prepared.settings is not None
    args = (prepared.inputs, prepared.settings)
    first, second = run_fit(*args), run_fit(*args)
    assert np.array_equal(first.field_mm, second.field_mm)
    assert np.array_equal(first.inverse_field_mm, second.inverse_field_mm)
    assert first.left_out is None  # no fit estimates a sorting of its own
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        elsewhere = pool.submit(_fit_in_worker, args).result()
    assert np.array_equal(first.field_mm, elsewhere)


def test_masks_value_type():
    zone = np.zeros((4, 4), dtype=bool)
    masks = Masks(fixed=~zone, moving=~zone, damage_zone=zone)
    assert masks.fixed.all() and not masks.damage_zone.any()


# --- a section mask in a region-restricted fit -----------------------------------


@needs_engine
def test_a_restricted_stain_fit_keeps_the_mask_inside_its_neighbourhood_and_fades(atlas):
    image, labels = _section(atlas, fluorescent=False)
    settings = FitSettings(detail="coarse", stain_metric="mutual_information",
                           stain_edges=False)
    first = run_prepared(prepare_fit(image, atlas, placement(), settings))
    hole = _hole((340, 260))
    weights = np.where(labels != 0, 1.0, 0.0).astype(np.float32)
    weights[hole] = 0.0
    restricted = FitSettings(detail="coarse", structures=("CTX",), neighbourhood_um=250,
                             stain_metric="mutual_information", stain_edges=False)
    prepared = prepare_fit(image, atlas, placement(), restricted, previous=first,
                           section_weights=weights)
    plain = prepare_fit(image, atlas, placement(), restricted, previous=first)
    assert prepared.inputs is not None and plain.inputs is not None
    in_hole = prepared.grid.to_working(hole.astype(np.float32)) > 0.5
    assert plain.inputs.fixed_mask[in_hole].any() and not prepared.inputs.fixed_mask[in_hole].any()
    second = run_prepared(prepared)
    assert second.left_out is None and second.weights is None
    near = prepared.inputs.fixed_mask
    near = cv2.resize(near.astype(np.uint8), (340, 260), interpolation=cv2.INTER_NEAREST) > 0
    mm = second.placement.section_mm_per_px
    distance = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 5)
    change = np.linalg.norm(second.field_mm - first.field_mm, axis=-1)
    beyond = (distance * mm > RESTRICTED_TAPER_MM + 2 * mm + 0.25) & (labels != 0)
    assert beyond.any() and change[beyond].max() < 1e-4
