"""Render styles are model-facing decoration and nothing more."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import langslice.nonlinear.image_gen_helpers as helpers
from langslice.atlas.recolor import PALETTES, color_lut, use_palette
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
    monkeypatch.setattr(helpers, "position_mm_to_index", lambda a, p, plane="coronal": 0)
    monkeypatch.setattr(helpers, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(helpers, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(helpers, "orient_slice_for_display", lambda a, plane: a)
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


# --- family-flat: one flat color per registration family --------------------

#: Digests of the two ORIGINAL styles on the fixture above, pinned the day
#: "family-flat" was added. They are a promise, not a preference: a new style
#: may not move a pixel of the styles already in use. Update only alongside a
#: deliberate change to how "family"/"leaf-borders" draw.
_ORIGINAL_RENDERS = {
    "family": "da5afd1f53040693",
    "leaf-borders": "4cf0f38934df653a",
}


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def allen() -> object:
    """The real Allen atlas: the fixture above has no near-miss colors, and
    collapsing leaves onto families is only interesting where there are some."""
    from langslice.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")


def _merged(atlas: object, position_mm: float, palette: str) -> np.ndarray:
    """Render -> classify -> merge, the way one whole run sees the plane."""
    with use_palette(palette):
        rgb = np.asarray(
            helpers._generate_colored_region_slice(
                atlas, position_mm, (456, 320), smooth=True
            ),
            dtype=np.uint8,
        )
        classified = helpers._classify_pixels_to_region_ids(
            rgb, atlas, position_mm, off_palette_background=False
        )
    return helpers._merge_classified(classified, atlas)


@pytest.mark.parametrize("position_mm", [3.9, 9.0])
def test_family_flat_merges_to_the_same_map_as_family(
    allen: object, position_mm: float
) -> None:
    """The point of the style: the model is shown the granularity the fit is
    scored at, and nothing downstream can tell which style drew the map."""
    truth = _merged(allen, position_mm, "family")
    flat = _merged(allen, position_mm, "family-flat")

    foreground = (truth != 0) | (flat != 0)
    agreement = float((flat[foreground] == truth[foreground]).mean())
    assert agreement > 0.98, f"family-flat agrees with family on {agreement:.4f}"

    # whatever does differ is a delineation line, not displaced anatomy
    from scipy import ndimage

    edge = np.zeros(truth.shape, dtype=bool)
    edge[:, 1:] |= truth[:, 1:] != truth[:, :-1]
    edge[1:, :] |= truth[1:, :] != truth[:-1, :]
    band = np.asarray(ndimage.binary_dilation(edge, iterations=3))
    assert not ((flat != truth) & foreground & ~band).any()


def test_family_flat_classification_round_trips_through_the_merge(
    allen: object,
) -> None:
    """It classifies straight to representative ids, so merging them again is
    the identity — which is what keeps the two styles' maps comparable."""
    with use_palette("family-flat"):
        rgb = np.asarray(
            helpers._generate_colored_region_slice(allen, 9.0, (456, 320), smooth=True),
            dtype=np.uint8,
        )
        classified = helpers._classify_pixels_to_region_ids(
            rgb, allen, 9.0, off_palette_background=False
        )
    assert (helpers._merge_classified(classified, allen) == classified).all()
    families = helpers._plane_families(helpers._annotation_slice(allen, 9.0), allen)
    assert set(np.unique(classified)) - {0} <= set(families.values())


def test_family_flat_classifies_the_elastix_side_render_into_the_same_families(
    allen: object,
) -> None:
    """The Elastix pair is drawn per LEAF whatever the style, and a leaf's
    NEAREST family color is not always its own family (10 of the 64 leaf
    colors at 3.9mm) — so every leaf color stays in the palette, keyed to the
    representative of the family it actually belongs to."""
    with use_palette("family"):
        rgb = np.asarray(
            helpers._generate_colored_region_slice(allen, 3.9, None, smooth=False),
            dtype=np.uint8,
        )
        truth = helpers._merge_classified(
            helpers._classify_pixels_to_region_ids(
                rgb, allen, 3.9, off_palette_background=False
            ),
            allen,
        )
    with use_palette("family-flat"):
        flat = helpers._classify_pixels_to_region_ids(
            rgb, allen, 3.9, off_palette_background=False
        )
    assert (flat == truth).all()


def test_family_flat_paints_fewer_colors_than_family(allen: object) -> None:
    """Flat means flat: the leaf shade bands are gone from the model's copy."""

    def colors(palette: str) -> int:
        with use_palette(palette):
            rgb = np.asarray(
                helpers._generate_colored_region_slice(
                    allen, 9.0, (456, 320), smooth=True
                ),
                dtype=np.uint8,
            )
        return len(np.unique(rgb.reshape(-1, 3), axis=0))

    families = helpers._plane_families(helpers._annotation_slice(allen, 9.0), allen)
    # at most one fill and one delineation shade per family, plus background
    assert colors("family-flat") <= 2 * len(set(families.values())) + 1
    assert colors("family-flat") < colors("family") + len(set(families.values()))


def test_the_styles_already_in_use_are_untouched(atlas: SimpleNamespace) -> None:
    """The Elastix-side render ignores the style, and the two original
    model-facing renders are byte-for-byte what they were before family-flat."""
    for palette in PALETTES:
        assert (_render(atlas, False, palette) == _render(atlas, False, "family")).all()
    for palette, digest in _ORIGINAL_RENDERS.items():
        assert _digest(_render(atlas, True, palette)) == digest, palette


def test_every_style_name_is_accepted_everywhere_a_style_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langslice.atlas.recolor import PALETTE_ENV, active_palette
    from langslice.cli import _build_parser

    assert "family-flat" in PALETTES
    for palette in PALETTES:
        monkeypatch.setenv(PALETTE_ENV, palette)
        assert active_palette() == palette
        with use_palette(palette):
            assert active_palette() == palette
        args = _build_parser().parse_args(
            ["nonlinear", "register", "slice.png", "--position", "5.0",
             "--palette", palette]
        )
        assert args.palette == palette
