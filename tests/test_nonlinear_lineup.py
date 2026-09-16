"""The geometry rules of the model-facing lineup.

Three images in ONE frame: the colored atlas map the model edits, the
grayscale template, and the section. The atlas renders keep their own
proportions and are letterboxed into the section's aspect; whatever comes
back is cropped to that aspect, never stretched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from langslice.nonlinear.image_gen_registration import (
    MODEL_MAP_MIN_LONG_EDGE,
    crop_to_aspect,
    letterbox_to_aspect,
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
