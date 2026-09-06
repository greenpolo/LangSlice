"""Leaf borders are a model-facing decoration and nothing more."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import langslice.atlas.render as atlas_render
import langslice.nonlinear.image_gen_helpers as helpers
from langslice.atlas.recolor import color_lut, use_palette
from langslice.nonlinear.render import BORDER_DARKEN

WHITE = [255, 255, 255]

#: An Allen-tree atlas whose hippocampal subfields share one family color.
_ROWS = {
    997: ("root", [997], "root"),
    8: ("grey", [997, 8], "Basic cell groups and regions"),
    315: ("Isocortex", [997, 8, 315], "Isocortex"),
    1089: ("HPF", [997, 8, 1089], "Hippocampal formation"),
    382: ("CA1", [997, 8, 1089, 382], "Field CA1"),
    463: ("CA3", [997, 8, 1089, 463], "Field CA3"),
    726: ("DG", [997, 8, 1089, 726], "Dentate gyrus"),
    512: ("CB", [997, 8, 512], "Cerebellum"),
}


@pytest.fixture
def atlas(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A 200x200 four-region slice on the Allen structure tree."""
    annotation = np.zeros((1, 200, 200), dtype=np.int32)
    annotation[0, 10:90, 10:90] = 382
    annotation[0, 10:90, 110:190] = 463
    annotation[0, 110:190, 10:90] = 726
    annotation[0, 110:190, 110:190] = 512
    fake = SimpleNamespace(
        annotation=annotation,
        structures={
            sid: {
                "id": sid,
                "acronym": acronym,
                "name": name,
                "structure_id_path": path,
                "rgb_triplet": list(WHITE),
            }
            for sid, (acronym, path, name) in _ROWS.items()
        },
    )
    # the annotation slice lives in atlas.render now; patch it where it is defined
    monkeypatch.setattr(
        atlas_render, "position_mm_to_index", lambda a, p, plane="coronal": 0
    )
    monkeypatch.setattr(atlas_render, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(atlas_render, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(atlas_render, "orient_slice_for_display", lambda a, plane: a)
    return fake


def _render(atlas: SimpleNamespace, smooth: bool, palette: str) -> np.ndarray:
    """The atlas the way the MODEL sees it."""
    with use_palette(palette):
        return np.asarray(
            helpers._generate_colored_region_slice(atlas, 0.0, (400, 400), smooth=smooth),
            dtype=np.uint8,
        )


def _classify(atlas: SimpleNamespace, rgb: np.ndarray, **kwargs) -> np.ndarray:
    return helpers._classify_pixels_to_region_ids(rgb, atlas, 0.0, **kwargs)


@pytest.mark.parametrize("smooth", [True, False])
def test_the_model_facing_render_classifies_back_to_its_own_regions(
    atlas: SimpleNamespace, smooth: bool
) -> None:
    """Smoothed or not, every painted pixel is an exact palette color."""
    classified = _classify(
        atlas, _render(atlas, smooth, "family"), off_palette_background=False
    )
    lut = color_lut(atlas)
    # one id per COLOR on the section: the hippocampal subfields share theirs
    assert {lut[int(uid)] for uid in np.unique(classified) if uid} == {
        lut[uid] for uid in (382, 463, 726, 512)
    }


def test_borders_change_only_the_model_facing_render(atlas: SimpleNamespace) -> None:
    """The Elastix-side render, which everything downstream is built on, is
    untouched by the style; the smooth one gains darker lines."""
    assert (
        _render(atlas, False, "leaf-borders") == _render(atlas, False, "family")
    ).all()
    flat = _render(atlas, True, "family")
    bordered = _render(atlas, True, "leaf-borders")
    assert not (flat == bordered).all()

    # every changed pixel is a darkened version of some region's own color
    changed = np.any(flat != bordered, axis=2)
    palette = np.array(sorted(set(color_lut(atlas).values())), dtype=float)
    lines = bordered[changed].astype(float)
    distance = np.linalg.norm(
        lines[:, None, :] - (palette * BORDER_DARKEN)[None, :, :], axis=2
    )
    assert float(np.median(distance.min(axis=1))) < 30.0  # anti-aliased edges aside


def test_a_bordered_render_still_classifies_to_its_own_families(
    atlas: SimpleNamespace,
) -> None:
    """The assumption the style rests on: if the model paints the hairlines
    back, the classifier absorbs them instead of inventing regions."""
    truth = _classify(atlas, _render(atlas, True, "family"), off_palette_background=False)
    with use_palette("leaf-borders"):  # the style is in force for the whole run
        as_model_output = _classify(atlas, _render(atlas, True, "leaf-borders"))

    inside = truth != 0
    assert float((as_model_output[inside] == truth[inside]).mean()) >= 0.99
    assert set(np.unique(as_model_output)) <= set(np.unique(truth))


def test_registration_render_uses_the_same_colors_the_classifier_expects(
    atlas: SimpleNamespace,
) -> None:
    classified = _classify(
        atlas, _render(atlas, False, "family"), off_palette_background=False
    )
    registration_rgb = helpers._registration_rgb(classified, atlas)
    reclassified = _classify(atlas, registration_rgb, off_palette_background=False)
    assert (reclassified == classified).all()


def test_the_palette_keeps_its_colors_apart_by_more_than_the_merge_radius(
    atlas: SimpleNamespace,
) -> None:
    colors = np.array(sorted(set(color_lut(atlas).values())), float)
    distances = np.linalg.norm(colors[:, None, :] - colors[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    from langslice.atlas.recolor import MERGE_EPS

    assert distances.min() > MERGE_EPS
