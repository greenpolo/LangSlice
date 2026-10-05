"""A border trace's yellow lines: drawn on the photograph, read back off a reply."""

import numpy as np
from PIL import Image

from langslice.core.nonlinear import border_refinement as borders


def test_lines_replace_only_their_own_pixels_of_the_photograph():
    photo = Image.fromarray(np.full((40, 60, 3), (70, 80, 90), dtype=np.uint8))
    mask = np.zeros((40, 60), dtype=bool)
    mask[6:34, 30] = True
    drawn = np.asarray(borders.border_overlay(photo, mask))
    assert (drawn[mask] == (255, 255, 0)).all()
    np.testing.assert_array_equal(drawn[~mask], np.asarray(photo)[~mask])


def test_a_letterboxed_reply_is_cropped_before_resampling():
    """A lane answering in a wider frame letterboxes ours inside it: the line
    lands where it was drawn on the canvas, not stretched."""
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[7:33, 40] = (255, 255, 0)  # x=30 after the symmetric 10 px crop
    mask = borders.extract_thinned_lines(Image.fromarray(pixels), (60, 40))
    assert mask.shape == (40, 60)
    assert np.all(np.nonzero(mask)[1] == 30)


def test_a_reply_with_no_yellow_has_no_lines():
    photo = Image.fromarray(np.full((40, 60, 3), (70, 80, 90), dtype=np.uint8))
    assert not borders.extract_thinned_lines(photo, photo.size).any()


def test_thinning_keeps_connected_line_and_yellow_tolerance():
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    rgb[5:25, 12:17] = (240, 210, 20)
    mask = borders.thin(borders.yellow_mask(rgb))
    assert mask.sum() > 10
    assert np.unique(np.nonzero(mask)[1]).size == 1
    assert not borders.yellow_mask(np.full((3, 3, 3), 150, dtype=np.uint8)).any()
