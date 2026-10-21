"""The atlas brain's size scales the deformable fit's distances
(``core.atlas.size``): the adult mouse brain is 1 and fits as it always did."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from langslice.core.atlas import size
from langslice.core.deformable import FitSettings, prepare_fit
from langslice.core.deformable.masks import em_parameters
from langslice.core.deformable.record import diagnose
from tests.deformable_synthetic import (
    SyntheticAtlas,
    bump_field,
    input_fingerprint,
    placement,
    render_section,
    same_inputs,
)


@pytest.fixture
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


@pytest.fixture
def scaled(monkeypatch):
    """Make every atlas a brain *factor* times the adult mouse's linear size."""
    def set_scale(factor: float) -> None:
        monkeypatch.setattr(size, "brain_volume_mm3",
                            lambda atlas: size.REFERENCE_VOLUME_MM3 * factor ** 3)
        monkeypatch.setattr(size, "_scales", {})
    return set_scale


def _cases(atlas: SyntheticAtlas) -> dict[str, tuple[Image.Image, FitSettings, np.ndarray | None]]:
    image, labels = render_section(atlas, bump_field([(130, 120, 45, 0.05, 0.02)]))
    lines = np.zeros(labels.shape, dtype=bool)
    lines[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    lines[1:, :] |= labels[1:, :] != labels[:-1, :]
    return {
        "stain": (image, FitSettings(exclude=("TH",)), None),
        "restricted": (image, FitSettings(structures=("STR",), exclude=("TH",)), None),
        "elastix": (image, FitSettings(engine="elastix", structures=("HY",)), None),
        "traced": (image, FitSettings(atlas_image="borders_merged", section_image="lines",
                                      labels="model"), lines),
    }


def _digest(prepared) -> tuple[str, list[float]]:
    """Every array and number a prepared fit hands the engines."""
    inputs = prepared.inputs
    arrays = [inputs.fixed_mask, inputs.moving_mask, inputs.fixed, inputs.moving,
              *inputs.fixed_labels, *inputs.moving_labels, prepared.footprint]
    arrays += [a for a in (inputs.fixed_edges, inputs.moving_edges, prepared.step_weight)
               if a is not None]
    numbers = json.dumps([list(inputs.spacing_mm), list(inputs.origin_mm),
                          inputs.correlation_radius_px, inputs.edge_weight,
                          inputs.label_weights, prepared.details], sort_keys=True, default=str)
    return input_fingerprint(arrays, numbers)


#: What prepare_fit built for these fits (the stain and restricted hashes since the EM left
#: prepare_fit; their arrays are unchanged) at standard detail before distances
#: followed the brain's size (main at 8ac2c91b), hashed: the adult mouse brain
#: (scale 1) must keep getting exactly these.
BEFORE_SCALING = {
    "stain": ("89e7d2c23e6af7b2", [
        61225.60, 50625.84, 24728.41, 15330.64, 7092.25, 3009.99, 7696.39, 3688.30]),
    "restricted": ("1e6ce51ba2094e48", [
        61225.60, 50625.84, 24728.41, 15330.64, 7564.62, 3362.36, 6534.27, 2685.52, 10189.07,
        9643.09]),
    "elastix": ("158bc238d8b45124", [
        61225.60, 50625.84, 26067.32, 15765.94, 14436.51, 8142.36, 14402.35, 8905.77, 8117.22,
        7601.52]),
    "traced": ("593e0c925340ef04", [
        10360.54, 7447.87, 11390.50, 8564.82, 47038.00, 45973.85, 34183.00, 32101.02, 4606.00,
        4271.28, 4767.00, 4433.09, 2955.00, 2648.08, 527.00, 407.74, 46511.00, 45427.19, 33581.00,
        31440.59, 4622.00, 4277.06, 4809.00, 4465.11, 2956.00, 2642.72, 543.00, 418.44]),
}


def test_the_adult_mouse_brain_prepares_the_fits_it_did_before_scaling(atlas):
    assert size.brain_scale(atlas) == 1.0
    for name, (image, settings, lines) in _cases(atlas).items():
        prepared = prepare_fit(image, atlas, placement(), settings, lines=lines)
        assert prepared.brain_scale == 1.0
        assert same_inputs(_digest(prepared), BEFORE_SCALING[name]), name


def test_the_adult_mouse_brain_keeps_its_diagnostics_and_em_settings():
    field = np.zeros((40, 40, 2), dtype=np.float32)
    field[..., 0] = 0.61
    labels = np.ones((40, 40), dtype=np.int32)
    report = diagnose(field, 0.5, labels, labels, labels > 0, ventricles=frozenset(),
                      excluded=frozenset(), acronyms={})
    assert report["thresholds"]["outsized_median_mm"] == 0.6
    assert report["thresholds"]["min_flag_area_mm2"] == 0.02
    assert [flag["code"] for flag in report["flags"]] == ["DISPLACEMENT_OUTSIZED"]
    assert em_parameters() == {
        "method": "three-class EM after emlddmm (Tward et al. 2025)", "grid_um": 50.0,
        "block_px": 16, "iterations": 20, "ridge": 1e-7, "sigma_scale": 0.5,
        "priors": [0.9, 0.05, 0.05], "background_distance_um": 500.0, "match_threshold": 0.5,
    }


def test_a_larger_brain_is_fitted_as_the_mouse_drawn_smaller(atlas, scaled):
    """At scale 2 the working pixels are twice as large and every distance
    twice as long: the fit sees the same number of pixels across the brain."""
    image, settings, _lines = _cases(atlas)["restricted"]
    reference = prepare_fit(image, atlas, placement(), settings)
    scaled(2.0)
    assert size.brain_scale(atlas) == 2.0
    prepared = prepare_fit(image, atlas, placement(), settings)
    assert prepared.brain_scale == 2.0
    assert prepared.inputs.brain_scale == 2.0
    # Standard detail's 20 um is finer than this section's 25 um pixels, so
    # the mouse works at 25 um; twice 20 um is 40 um.
    assert reference.grid.spacing_mm == (0.025, 0.025)
    assert np.allclose(prepared.grid.spacing_mm, 0.040, rtol=0.01)
    assert prepared.details["edge_channel"]["sigma_um"] == 60.0
    taper = prepared.details["taper_um"]
    assert taper[1] - taper[0] == pytest.approx(600.0)
    params = em_parameters(2.0)
    assert params["block_px"] == 32 and params["background_distance_um"] == 1000.0
    assert params["grid_um"] == 50.0  # the slide's holes and debris keep their size
    field = np.zeros((40, 40, 2), dtype=np.float32)
    field[..., 0] = 0.9
    labels = np.ones((40, 40), dtype=np.int32)
    report = diagnose(field, 0.5, labels, labels, labels > 0, ventricles=frozenset(),
                      excluded=frozenset(), acronyms={}, brain_scale=2.0)
    assert report["thresholds"]["outsized_median_mm"] == 1.2
    assert report["thresholds"]["min_flag_area_mm2"] == pytest.approx(0.08)
    assert not report["flags"]  # 0.9 mm is a mouse's 0.45 mm: within the limit


def test_the_factor_is_the_cube_root_of_the_volume_ratio_rounded():
    assert size.scale_for_volume(size.REFERENCE_VOLUME_MM3) == 1.0
    assert size.scale_for_volume(size.REFERENCE_VOLUME_MM3 * 1.02) == 1.0
    assert size.scale_for_volume(size.REFERENCE_VOLUME_MM3 * 1.69 ** 3) == 1.70
    assert size.scale_for_volume(size.REFERENCE_VOLUME_MM3 * 0.724 ** 3) == 0.70
    with pytest.raises(ValueError):
        size.scale_for_volume(0.0)
    annotation = np.zeros((40, 40, 40), dtype=np.int32)
    annotation[:20, :20, :20] = 7  # 20 voxels of 25 um: a 0.5 mm cube
    assert size.annotated_volume_mm3(annotation, (25.0, 25.0, 25.0)) == pytest.approx(0.125)


def test_the_ruler_margins_follow_the_brain():
    from langslice.core.positioning import ruler_range

    assert ruler_range([5.0], (0.0, 20.0)) == (4.5, 5.5)
    assert ruler_range([5.0], (0.0, 20.0), brain_scale=2.0) == (4.0, 6.0)


def test_a_description_states_the_neighbourhood_at_the_brain_scale():
    from langslice.doors.declarations import Variant, declaration

    assert "atlas within 300 um of them" in declaration("elastix_affine", Variant()).doc
    rat = declaration("elastix_affine", Variant(brain_scale=1.7)).doc
    assert "atlas within 510 um of them" in rat and "300 um" not in rat


@pytest.mark.slow
def test_real_atlases_measure_their_brains():
    from langslice.core.atlas.core import load_atlas

    expected = {"allen_mouse_25um": 1.0, "allen_mouse_50um": 1.0,
                "ccfv3augmented_mouse_25um": 1.0, "whs_sd_rat_39um": 1.7,
                "demba_allen_seg_dev_mouse_p4_25um": 0.7,
                "demba_allen_seg_dev_mouse_p14_25um": 0.95}
    for name, factor in expected.items():
        atlas = load_atlas(name)
        volume = size.annotated_volume_mm3(atlas.annotation, atlas.resolution)
        assert size.scale_for_volume(volume) == factor, name
