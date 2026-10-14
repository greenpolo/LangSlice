"""Zoom (core/zoom.py): a box of an earlier picture, drawn again.

Box maths and composition, positioning zoom, a section gone falls back to the
saved picture, and region zoom (selected borders only, displayed side, refusals).
The verb-level zoom (landing on the box, stale redraw, crops of pictures
without a recipe) is in test_ops_look.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from langslice.core.layers import collecting, note_for
from langslice.core.look import LookRequest, look
from langslice.core.zoom import (
    ZoomError,
    box_fractions,
    compose,
    redraw,
)
from tests.look_synthetic import record, stack


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


def test_section_gone_falls_back_to_the_saved_picture(tmp_path: Path):
    ws, state = stack(tmp_path)
    first = look(ws, state, LookRequest("section", sections=("s2.png",)))[0]
    state.slices = [s for s in state.slices if s.id != "s2.png"]
    width, height = first.recipe["shown"]
    crop = redraw(first.recipe, [0, 0, width / 2, height / 2], ws, state=state,
                  picture=first.image)
    assert not crop.redrawn and crop.stale


def test_region_zoom_crops_only_selected_border_and_preserves_saved_state(tmp_path: Path):
    from langslice.core.look import LookError

    ws, state = stack(tmp_path)
    first = look(ws, state, LookRequest("overlay", sections=("s1.png",)))[0]
    with collecting() as notes:
        zoomed = redraw(first.recipe, [], ws, state=state, region="STR")
    assert zoomed.redrawn and zoomed.mode == "overlay"
    held = note_for(zoomed.image, notes)
    assert held is not None and held.panel is not None
    assert not held.panel.lines and held.panel.highlighted
    window = zoomed.recipe["args"]["zoom"]
    assert window[2] - window[0] < 0.6
    assert window[3] - window[1] < 0.6
    assert zoomed.um_per_px < first.um_per_px
    # An explicit box on this result keeps the selected-only borders.
    width, height = zoomed.recipe["shown"]
    again = redraw(zoomed.recipe, [0, 0, width / 2, height], ws, state=state)
    assert again.recipe["args"]["region_only"]
    assert again.recipe["args"]["highlight"] == ["STR"]
    # Moving the live section does not move a zoom of the saved overlay.
    record(state, "s1.png").position_mm = 0.22
    stale = redraw(first.recipe, [], ws, state=state, region="STR")
    assert stale.stale and "at 0.15 mm" in stale.caption
    assert stale.recipe["args"]["zoom"] == window
    assert np.array_equal(np.asarray(stale.image)[:height], np.asarray(zoomed.image)[:height])
    with pytest.raises(LookError) as caught:
        redraw(first.recipe, [], ws, state=state, region="not-a-region")
    assert caught.value.code == "UNKNOWN_REGIONS"


def test_region_zoom_refusals(tmp_path: Path):
    ws, state = stack(tmp_path)
    for mode in ("section", "atlas", "positioning"):
        first = look(ws, state, LookRequest(mode, sections=("s1.png",), positions_mm=(0.15,)))[0]
        with pytest.raises(ZoomError) as caught:
            redraw(first.recipe, [], ws, state=state, region="STR")
        assert caught.value.code == "BAD_REGION_PICTURE"
    first = look(ws, state, LookRequest("overlay", sections=("s1.png",)))[0]
    for box, region in (([0, 0, 20, 20], "STR"), ([], ["STR", "TH"])):
        with pytest.raises(ZoomError) as caught:
            redraw(first.recipe, box, ws, state=state, region=region)
        assert caught.value.code == "BAD_ARGS"


def test_region_zoom_resolves_displayed_side_and_reports_absent_region(tmp_path: Path):
    from langslice.core.look import LookError

    ws, state = stack(tmp_path)
    first = look(ws, state, LookRequest("overlay", sections=("s1.png",)))[0]
    left = redraw(first.recipe, [], ws, state=state, region="CTX:left")
    right = redraw(first.recipe, [], ws, state=state, region="CTX:right")
    a, b = left.recipe["args"]["zoom"], right.recipe["args"]["zoom"]
    assert a[0] < b[0] and a[2] < b[2]
    with pytest.raises(LookError) as caught:
        redraw(first.recipe, [], ws, state=state, region="TH:left")
    assert caught.value.code == "REGION_NOT_IN_PLANE"
