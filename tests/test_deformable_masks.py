"""The masks a deformable fit is given (``core.deformable.masks``) and the
FireANTs engine's determinism.

Stain fits, whole-section or restricted to regions, run twice, the second
time with the EM section mask: holes leave the metric, matching tissue stays
in, on brightfield and on fluorescence. A restricted fit keeps both masks
inside its neighbourhood and its field fades beyond. The Elastix affine keeps
the footprint masks, built exactly as before.
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
from langslice.core.deformable.fit import RESTRICTED_TAPER_MM, run_fit, run_prepared
from langslice.core.deformable.masks import (
    EMContext,
    Masks,
    dilate,
    em_masks,
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


@needs_engine
@pytest.mark.parametrize("fluorescent", [False, True])
def test_the_em_mask_drops_a_hole_and_keeps_matching_tissue(atlas, fluorescent):
    image, labels = _section(atlas, fluorescent=fluorescent)
    prepared = prepare_fit(image, atlas, placement(), FitSettings(detail="coarse"))
    assert prepared.em is not None and prepared.em.neighbourhood is None
    record = run_prepared(prepared)
    assert record.left_out is not None and record.left_out.shape == labels.shape[:2]
    hole = _hole((340, 260))
    core = ~dilate(~hole, 4)  # the hole's interior, clear of its rim
    tissue = (labels != 0) & ~dilate(hole, 12)
    assert record.left_out[core].mean() > 0.9
    assert record.left_out[tissue].mean() < 0.05
    mask = record.engine["native_parameters"]["section_mask"]
    assert mask["kind"] == "em" and mask["match_threshold"] == 0.5


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


def test_masks_are_deterministic():
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
    context = EMContext(first.fixed, first.damage_zone)
    a, facts_a = em_masks(J, warped, context, 0.02)
    b, facts_b = em_masks(J, warped, context, 0.02)
    assert np.array_equal(a.fixed, b.fixed) and np.array_equal(a.moving, b.moving)
    assert np.array_equal(facts_a["left_out"], facts_b["left_out"])
    assert facts_a["params"] == facts_b["params"]
    # The second fit's atlas mask is everything but the damage zone.
    assert np.array_equal(a.moving, ~first.damage_zone)


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
        assert (prepared.em is None) == (name == "elastix"), name


def _fit_in_worker(args):  # a spawn worker's whole fit, as the CPU pool runs it
    return run_fit(*args).field_mm


@needs_engine
def test_identical_inputs_give_identical_fields_in_and_across_processes(atlas):
    """Bit for bit: twice in this process and once in a fresh one, through both
    of the EM path's runs."""
    image, _ = _section(atlas, fluorescent=False)
    prepared = prepare_fit(image, atlas, placement(), FitSettings(detail="coarse"))
    assert prepared.inputs is not None and prepared.settings is not None
    args = (prepared.inputs, prepared.settings, prepared.em)
    first, second = run_fit(*args), run_fit(*args)
    assert np.array_equal(first.field_mm, second.field_mm)
    assert np.array_equal(first.inverse_field_mm, second.inverse_field_mm)
    assert first.left_out is not None and second.left_out is not None
    assert np.array_equal(first.left_out, second.left_out)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        elsewhere = pool.submit(_fit_in_worker, args).result()
    assert np.array_equal(first.field_mm, elsewhere)


def test_masks_value_type():
    zone = np.zeros((4, 4), dtype=bool)
    masks = Masks(fixed=~zone, moving=~zone, damage_zone=zone)
    assert masks.fixed.all() and not masks.damage_zone.any()


# --- the EM mask in a region-restricted fit -----------------------------------


def _textured_pair(rng: np.random.Generator):
    """A tissue ellipse with texture, its section under a linear contrast with
    a 0.4 mm hole, and a region (a box) whose neighbourhood holds the hole."""
    yy, xx = np.indices((400, 520))  # a 20 um grid
    tissue = (xx - 260) ** 2 / 220 ** 2 + (yy - 200) ** 2 / 160 ** 2 < 1
    coarse = cv2.GaussianBlur(rng.random((20, 26)).astype(np.float32), (0, 0), 1)
    texture = cv2.resize(coarse, (520, 400), interpolation=cv2.INTER_CUBIC)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    atlas = np.where(tissue, 0.2 + 0.8 * texture, 0.0).astype(np.float32)
    hole = (xx - 300) ** 2 + (yy - 180) ** 2 < 10 ** 2
    section = np.where(tissue & ~hole, 0.85 - 0.6 * atlas, 0.95)
    section = np.clip(section + rng.normal(0, 0.03, section.shape), 0, 1).astype(np.float32)
    region = (xx > 280) & (xx < 340) & (yy > 150) & (yy < 210)
    return tissue, atlas, section, hole, region


def test_a_restricted_em_mask_is_the_whole_sections_inside_the_neighbourhood():
    """The classes are estimated over the whole section, so the weights do not
    depend on the region; the masks are those inside its neighbourhood, and a
    hole there leaves the metric."""
    tissue, atlas, section, hole, region = _textured_pair(np.random.default_rng(3))
    footprint = dilate(tissue, 10)
    zone = np.zeros(tissue.shape, dtype=bool)
    zone[300:320, 200:230] = True
    neighbourhood = dilate(region, 15)  # 0.3 mm
    whole, whole_facts = em_masks(section, atlas, EMContext(footprint, zone), 0.02)
    near, facts = em_masks(section, atlas, EMContext(footprint, zone, neighbourhood), 0.02)
    assert np.array_equal(near.fixed, whole.fixed & neighbourhood)
    assert np.array_equal(near.moving, ~zone & neighbourhood)
    assert not near.fixed[~neighbourhood].any() and not facts["left_out"][~neighbourhood].any()
    inside = ~dilate(~hole, 3)
    assert inside.any() and not near.fixed[inside].any() and facts["left_out"][inside].all()
    kept = tissue & neighbourhood & ~dilate(hole, 8)
    assert near.fixed[kept].mean() > 0.99
    # The share left out is measured in the neighbourhood, not the section.
    assert facts["left_out_in_footprint"] > whole_facts["left_out_in_footprint"]
    again, _ = em_masks(section, atlas, EMContext(footprint, zone, neighbourhood), 0.02)
    assert np.array_equal(near.fixed, again.fixed)


@needs_engine
def test_a_restricted_stain_fit_runs_the_em_path_and_its_field_still_fades(atlas):
    image, labels = _section(atlas, fluorescent=False)
    settings = FitSettings(detail="coarse", stain_metric="mutual_information",
                           stain_edges=False)
    first = run_prepared(prepare_fit(image, atlas, placement(), settings))
    prepared = prepare_fit(image, atlas, placement(), FitSettings(
        detail="coarse", structures=("CTX",), neighbourhood_um=250,
        stain_metric="mutual_information", stain_edges=False), previous=first)
    assert prepared.em is not None and prepared.em.neighbourhood is not None
    assert prepared.details["section_mask"] == "em"
    second = run_prepared(prepared)
    assert second.left_out is not None
    mask = second.engine["native_parameters"]["section_mask"]
    assert mask["kind"] == "em"
    hole = _hole((340, 260))
    core = ~dilate(~hole, 4)
    assert second.left_out[core].mean() > 0.9
    mm = second.placement.section_mm_per_px
    near = cv2.resize(prepared.em.neighbourhood.astype(np.uint8), (340, 260),
                      interpolation=cv2.INTER_NEAREST) > 0
    assert not second.left_out[~near].any()
    distance = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 5)
    change = np.linalg.norm(second.field_mm - first.field_mm, axis=-1)
    beyond = (distance * mm > RESTRICTED_TAPER_MM + 2 * mm) & (labels != 0)
    assert beyond.any() and change[beyond].max() < 1e-4
