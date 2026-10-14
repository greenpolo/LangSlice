"""The geometry rule of the model-facing lineup: whatever comes back is
cropped to the section's aspect, never stretched."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from langslice.core.nonlinear.image_gen_registration import (
    crop_to_aspect,
)


@pytest.mark.parametrize("size", [(1536, 1024), (1024, 1536)])
def test_an_output_in_a_wider_or_taller_frame_is_cropped_back(size) -> None:
    # the section's frame letterboxed inside a 3:2 answer (either way up)
    assert crop_to_aspect(Image.new("RGB", size), 1.0).size == (1024, 1024)


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
