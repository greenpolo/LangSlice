"""Zoom (core/zoom.py): a box of an earlier picture, drawn again.

The box is read on the picture's content and composed onto the unzoomed
picture's window, so a zoom of a zoom maps back; it lands where it was
asked (a white square on the marked section); a changed stack is drawn as it
was and flagged; a picture without a recipe is cropped, not redrawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.core.layers import collecting, note_for
from langslice.core.look import LookRequest, look
from langslice.core.zoom import (
    CROP_RENDERER,
    STALE_NOTE,
    ZoomError,
    box_fractions,
    compose,
    redraw,
)
from tests.look_synthetic import MARKED, record, stack


def _content(picture) -> np.ndarray:
    """The zoom's content (above its caption band), gray."""
    width, height = picture.recipe["shown"]
    return np.asarray(picture.image.convert("L"))[:height, :width].astype(np.float32)


def test_box_fractions_and_compose():
    assert box_fractions([50, 25, 0, 75], (100, 100)) == (0.0, 0.25, 0.5, 0.75)
    assert box_fractions([-10, -10, 500, 500], (100, 50)) == (0.0, 0.0, 1.0, 1.0)
    for box, code in (([1, 2, 3], "BAD_BOX"), (["a", 0, 1, 1], "BAD_BOX"),
                      ([10, 10, 10.5, 40], "EMPTY_BOX"), ([200, 0, 300, 50], "EMPTY_BOX")):
        with pytest.raises(ZoomError) as caught:
            box_fractions(box, (100, 100))
        assert caught.value.code == code
    assert compose((), (0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4)
    assert compose((0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 0.5, 0.5)) == (0.5, 0.5, 0.75, 0.75)


def test_section_zoom_lands_on_the_box_and_zooms_again(tmp_path: Path):
    ws, state = stack(tmp_path, marked=True)
    first = look(ws, state, LookRequest("section", sections=(MARKED,)))[0]
    content = np.asarray(first.image.convert("L"))[:first.recipe["shown"][1]]
    ys, xs = np.nonzero(content > 250)  # the white square
    box = [xs.min() + 2, ys.min() + 2, xs.max() - 2, ys.max() - 2]
    with collecting() as notes:
        zoomed = redraw(first.recipe, box, ws, state=state)
    assert zoomed.redrawn and not zoomed.stale
    assert zoomed.sections == (MARKED,) and zoomed.mode == "section"
    assert _content(zoomed).mean() > 200  # mostly the square
    assert (zoomed.um_per_px or 0.0) < first.um_per_px
    window = zoomed.recipe["args"]["zoom"]
    assert window == pytest.approx(list(box_fractions(box, first.recipe["shown"])))
    held = note_for(zoomed.image, notes)
    assert held is not None and held.recipe == zoomed.recipe
    json.dumps(zoomed.recipe)
    # Zoom of a zoom: the right half of the zoom is the right half of the box.
    width, height = zoomed.recipe["shown"]
    again = redraw(zoomed.recipe, [width / 2, 0, width, height], ws, state=state)
    expected = compose(window, (0.5, 0.0, 1.0, 1.0))
    assert again.recipe["args"]["zoom"] == pytest.approx(list(expected))
    assert again.recipe["base"] == first.recipe["base"]
    # Off the square: the slab's gray, not white.
    away = redraw(first.recipe, [box[2] + 30, box[3] + 30, box[2] + 90, box[3] + 70], ws,
                  state=state)
    assert _content(away).mean() < 200


def test_positioning_zoom_redraws_larger_at_more_detail(tmp_path: Path):
    ws, state = stack(tmp_path)
    first = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.2)))[0]
    width, height = first.recipe["shown"]
    zoomed = redraw(first.recipe, [0, height * 0.5, width * 0.5, height], ws, state=state)
    assert zoomed.redrawn and zoomed.mode == "positioning"
    assert (zoomed.um_per_px or 0.0) < first.um_per_px
    assert zoomed.recipe["args"]["zoom"] == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert max(zoomed.recipe["shown"]) > 0.5 * width  # redrawn larger, not cropped
    assert "zoomed" in zoomed.caption


def test_overlay_and_atlas_zoom(tmp_path: Path):
    ws, state = stack(tmp_path)
    layers = ("template", "borders")
    for request in (LookRequest("overlay", sections=("s0.png",)),
                    LookRequest("atlas", positions_mm=(0.1,), atlas_layers=layers)):
        first = look(ws, state, request)[0]
        width, height = first.recipe["shown"]
        zoomed = redraw(first.recipe, [width * 0.25, height * 0.25, width * 0.75, height * 0.75],
                        ws, state=state)
        assert zoomed.redrawn and zoomed.mode == request.mode
        # The synthetic overlay is drawn at its source pixels: no finer.
        assert (zoomed.um_per_px or 0.0) <= first.um_per_px
        assert zoomed.recipe["args"]["zoom"] == pytest.approx([0.25, 0.25, 0.75, 0.75], abs=0.01)


def test_changed_stack_is_drawn_as_it_was_and_flagged(tmp_path: Path):
    ws, state = stack(tmp_path)
    first = look(ws, state, LookRequest("overlay", sections=("s1.png",)))[0]
    record(state, "s1.png").position_mm = 0.22
    width, height = first.recipe["shown"]
    zoomed = redraw(first.recipe, [0, 0, width / 2, height / 2], ws, state=state)
    assert zoomed.redrawn and zoomed.stale
    assert "at 0.15 mm" in zoomed.caption and STALE_NOTE in zoomed.caption
    assert zoomed.recipe["state"] == first.recipe["state"]  # still the picture as it was
    assert record(state, "s1.png").position_mm == 0.22  # the stack is untouched
    fresh = redraw(first.recipe | {"state": None}, [0, 0, width / 2, height / 2], ws, state=state)
    assert not fresh.stale and "at 0.22 mm" in fresh.caption


def test_no_recipe_is_cropped_and_a_crop_of_a_crop_maps_back(tmp_path: Path):
    ws, state = stack(tmp_path)
    source = Image.new("RGB", (400, 200), (0, 0, 0))
    source.paste((255, 255, 255), (300, 100, 400, 200))  # bottom-right block
    crop = redraw(None, [200, 0, 400, 200], ws, state=state, picture=source)
    assert not crop.redrawn and crop.recipe["renderer"] == CROP_RENDERER
    assert crop.recipe["args"]["box"] == [200.0, 0.0, 400.0, 200.0]
    width, height = crop.recipe["shown"]
    again = redraw(crop.recipe, [width / 2, height / 2, width, height], ws, state=state,
                   picture=source)
    assert again.recipe["args"]["box"] == [300.0, 100.0, 400.0, 200.0]
    assert _content(again).mean() > 250
    with pytest.raises(ZoomError) as caught:
        redraw(None, [0, 0, 10, 10], ws, state=state)
    assert caught.value.code == "NO_PICTURE"


def test_section_gone_falls_back_to_the_saved_picture(tmp_path: Path):
    ws, state = stack(tmp_path)
    first = look(ws, state, LookRequest("section", sections=("s2.png",)))[0]
    state.slices = [s for s in state.slices if s.id != "s2.png"]
    width, height = first.recipe["shown"]
    crop = redraw(first.recipe, [0, 0, width / 2, height / 2], ws, state=state,
                  picture=first.image)
    assert not crop.redrawn and crop.stale
