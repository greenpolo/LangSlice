"""The geometry rules of the model-facing lineup.

Two images in ONE frame for route "atlas": the outlined grayscale atlas
template, and the section. The atlas renders keep their own proportions and
are letterboxed into the section's aspect; whatever comes back is cropped to
that aspect, never stretched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.core.nonlinear.image_gen_registration import (
    MODEL_MAP_MIN_LONG_EDGE,
    crop_to_aspect,
    letterbox_to_aspect,
    outlined_atlas_template,
    upscale_to_min_long_edge,
)


def test_letterbox_widens_a_tall_render_without_scaling_it() -> None:
    render = Image.new("RGB", (300, 400), (10, 20, 30))
    boxed = letterbox_to_aspect(render, 2.0)

    assert boxed.size == (800, 400)
    # the render is centered, untouched, on black
    assert boxed.getpixel((400, 200)) == (10, 20, 30)
    assert boxed.getpixel((10, 200)) == (0, 0, 0)
    assert boxed.getpixel((790, 200)) == (0, 0, 0)


def test_letterbox_heightens_a_wide_render() -> None:
    boxed = letterbox_to_aspect(Image.new("RGB", (400, 100), (9, 9, 9)), 1.0)

    assert boxed.size == (400, 400)
    assert boxed.getpixel((200, 200)) == (9, 9, 9)
    assert boxed.getpixel((200, 5)) == (0, 0, 0)


def test_upscale_reaches_the_minimum_long_edge_and_never_shrinks() -> None:
    small = Image.new("RGB", (456, 320))
    grown = upscale_to_min_long_edge(small, Image.Resampling.NEAREST)
    assert max(grown.size) == MODEL_MAP_MIN_LONG_EDGE
    assert abs(grown.width / grown.height - small.width / small.height) < 0.01

    big = Image.new("RGB", (3000, 2000))
    assert upscale_to_min_long_edge(big, Image.Resampling.NEAREST) is big


def test_upscaling_a_flat_map_nearest_invents_no_colors() -> None:
    plate = Image.new("RGB", (4, 3), (255, 0, 0))
    plate.putpixel((2, 1), (0, 255, 0))
    grown = upscale_to_min_long_edge(plate, Image.Resampling.NEAREST)
    assert {tuple(c) for c in np.asarray(grown).reshape(-1, 3)} == {(255, 0, 0), (0, 255, 0)}


def test_an_output_in_a_wider_frame_is_cropped_back_to_the_section_aspect() -> None:
    # the section's frame letterboxed inside a 3:2 answer
    returned = Image.new("RGB", (1536, 1024))
    cropped = crop_to_aspect(returned, 1.0)

    assert cropped.size == (1024, 1024)


def test_an_output_in_a_taller_frame_is_cropped_back() -> None:
    cropped = crop_to_aspect(Image.new("RGB", (1024, 1536)), 1.0)
    assert cropped.size == (1024, 1024)


def test_an_output_already_in_the_frame_is_returned_untouched() -> None:
    returned = Image.new("RGB", (1024, 683))  # 1.4993 vs 1.5 — within tolerance
    assert crop_to_aspect(returned, 1.5) is returned


def test_cropping_keeps_the_center_of_the_painting() -> None:
    returned = Image.new("RGB", (300, 100), (0, 0, 0))
    for x in range(100, 200):
        for y in range(100):
            returned.putpixel((x, y), (255, 255, 255))

    cropped = crop_to_aspect(returned, 1.0)

    assert cropped.size == (100, 100)
    assert {tuple(c) for c in np.asarray(cropped).reshape(-1, 3)} == {(255, 255, 255)}


# -------------------------------------------------------- outlined atlas render


@pytest.fixture
def _toy_atlas_render(monkeypatch: pytest.MonkeyPatch):
    """A tiny two-region plane, wired so outlined_atlas_template runs with no atlas I/O."""
    import langslice.core.nonlinear.image_gen_helpers as helpers

    labels = np.zeros((20, 30), dtype=np.int32)
    labels[:, 15:] = 1
    gray = np.full((20, 30), 120.0, dtype=np.float32)

    atlas = SimpleNamespace(atlas_name="toy", structures={})
    monkeypatch.setattr(helpers, "annotation_slice", lambda *a, **k: labels.copy())
    monkeypatch.setattr("langslice.core.atlas.get_reference_slice", lambda *a, **k: gray.copy())
    # No real structure tree to merge families from; the render only needs a
    # boundary between two ids, so pass classified ids through unchanged.
    monkeypatch.setattr(helpers, "_merge_classified", lambda ids, atlas, merge_eps=40.0: ids)
    return atlas, labels


def test_outlined_atlas_template_draws_only_yellow_on_a_grayscale_plate(
    _toy_atlas_render,
) -> None:
    atlas, _labels = _toy_atlas_render

    outlined = outlined_atlas_template(atlas, 1.0, "coronal")

    assert max(outlined.size) >= MODEL_MAP_MIN_LONG_EDGE
    assert outlined.size[0] / outlined.size[1] == pytest.approx(30 / 20, rel=0.02)
    arr = np.asarray(outlined)
    # Achromatic (r == g == b) is the grayscale plate or the border's black
    # rim; the only chromatic pixels are the yellow line core.
    chromatic = arr[(arr[..., 0] != arr[..., 1]) | (arr[..., 1] != arr[..., 2])]
    assert chromatic.size, "no boundary line was drawn at all"
    assert {tuple(c) for c in chromatic.reshape(-1, 3)} == {(255, 255, 0)}


def test_outlined_atlas_template_letterboxes_to_the_section_aspect(
    _toy_atlas_render,
) -> None:
    atlas, _labels = _toy_atlas_render

    outlined = outlined_atlas_template(atlas, 1.0, "coronal", section_aspect=2.5)

    assert outlined.size[0] / outlined.size[1] == pytest.approx(2.5, rel=0.02)


def test_outlined_atlas_template_reflection_is_explicit_and_flips_the_render(
    _toy_atlas_render,
) -> None:
    atlas, _labels = _toy_atlas_render

    plain = np.asarray(outlined_atlas_template(atlas, 1.0, "coronal"))
    mirrored = np.asarray(
        outlined_atlas_template(atlas, 1.0, "coronal", atlas_mirror_lr=True)
    )

    assert not np.array_equal(plain, mirrored)
    assert np.array_equal(mirrored, plain[:, ::-1])
