"""The atlas root mask and the border traces' working canvas."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests.fakes import SlabAtlas


def _make_slice(size: tuple[int, int] = (12, 8)) -> Image.Image:
    image = Image.new("RGB", size, color=(30, 40, 50))
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), (30 + x, 40 + y, 50))
    return image


def test_root_mask_is_binary_alpha_at_target_size():
    """Non-zero annotation is opaque (255), the background transparent (0), and
    resizing is NEAREST so the silhouette gets no halo."""
    from langslice.core.atlas.core import get_root_mask

    mask = get_root_mask(SlabAtlas(), 5.0, (32, 24), plane="coronal")
    assert mask.shape == (24, 32) and mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {0, 255}
    assert mask[12, 16] == 255 and mask[0, 0] == 0


@pytest.mark.parametrize(("size", "padded", "offset"), [
    # gpt-image-2.5-sunburst accepts 1:3..3:1: a 4:1 strip black-pads down to 3:1,
    # centred; an in-range section is left untouched.
    ((48, 12), (48, 16), (0.0, 2.0)),
    ((24, 9), (24, 9), (0.0, 0.0)),
])
def test_canvas_aspect_clamps_into_the_supported_range(size, padded, offset):
    from langslice.core.nonlinear.image_gen_registration import prepare_canvas

    canvas, render_to_canvas = prepare_canvas(
        _make_slice(size), provider="none", image_model="gpt-image-2.5-sunburst",
    )
    assert canvas.size == padded
    np.testing.assert_allclose(render_to_canvas, [[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])
