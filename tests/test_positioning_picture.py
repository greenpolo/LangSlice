"""The positioning picture's layout (core/positioning.py).

The geometry is pure (:func:`plan`): sections in position order (ties in
stack order, no position last), so no two lines cross; a long stack split
into pictures of consecutive positions at one scale, each with its own
ruler segment; atlas thumbnails over their millimetres, in the picture
whose positions surround them. :func:`paint` and the stack's pictures are
checked on the synthetic atlas.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from langslice.core import positioning
from langslice.core.layers import collecting, note_for
from langslice.core.look import LookError, LookRequest, look
from langslice.core.positioning import (
    GAP,
    MARGIN,
    PER_PICTURE,
    POSITIONING_MAX_WIDTH,
    RULER_MIN_SPAN_MM,
    SECTION_LINK_PX,
    SectionEntry,
    paint,
    plan,
    position_order,
    ruler_range,
    split,
)
from langslice.core.zoom import redraw
from tests.look_synthetic import SECTIONS, stack

#: A coronal-ish section and atlas plane, micrometres.
SECTION_UM = (9000.0, 6500.0)
ATLAS_UM = (10000.0, 7000.0)
RANGE = (0.0, 13.2)


def _entries(positions: Sequence[float | None]) -> list[SectionEntry]:
    return [SectionEntry(id=f"s{k}", labels=(f"{k}: s{k}.tif", str(k)), position_mm=p,
                         extent_um=SECTION_UM) for k, p in enumerate(positions)]


def _plan(positions: Sequence[float | None], atlas_mm: tuple[float, ...] = (3.0, 6.5, 10.0), **kw):
    return plan(_entries(positions), [(mm, ATLAS_UM) for mm in atlas_mm],
                range_mm=RANGE, tile_edge=kw.pop("tile_edge", 512), **kw)


def _one(positions: Sequence[float | None], atlas_mm: tuple[float, ...] = (3.0, 6.5, 10.0), **kw):
    layouts = _plan(positions, atlas_mm, **kw)
    assert len(layouts) == 1
    return layouts[0]


def _segments_cross(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Whether segments a and b ((x0, y0, x1, y1)) cross strictly inside."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    p1, p2, p3, p4 = (a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])
    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


# --- order, parts and rulers -------------------------------------------------------


def test_position_order_keeps_stack_order_for_ties_and_puts_unplaced_last():
    entries = _entries([5.0, None, 2.0, 5.0, 1.0, None, 2.0])
    assert [e.id for e in position_order(entries)] == ["s4", "s2", "s6", "s0", "s3", "s1", "s5"]


def test_split_cuts_even_consecutive_runs():
    positions = [float(k) * 0.25 for k in range(38)]
    pieces = split(positions, [], per_picture=6)
    assert [len(s) for s, _a in pieces] == [6, 6, 6, 5, 5, 5, 5]
    assert [k for s, _a in pieces for k in s] == list(range(38))
    assert split([1.0] * 6, [], per_picture=6) == [(list(range(6)), [])]
    assert split([], [1.0, 2.0], per_picture=6) == [([], [0, 1])]


def test_split_gives_each_atlas_position_to_the_run_around_it():
    positions = [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
    # Runs [1, 2, 3] and [7, 8, 9]: the boundary is halfway, at 5 mm.
    pieces = split(positions, [0.5, 2.5, 4.9, 5.1, 12.0], per_picture=3)
    assert pieces == [([0, 1, 2], [0, 1, 2]), ([3, 4, 5], [3, 4])]
    # Unplaced sections go last, and no atlas position follows them there.
    pieces = split([1.0, 2.0, None, None], [9.0], per_picture=2)
    assert pieces == [([0, 1], [0]), ([2, 3], [])]


def test_split_splits_a_run_holding_too_many_atlas_positions():
    pieces = split([5.0], [4.0, 4.5, 5.0, 5.5, 6.0], per_picture=2)
    assert all(len(s) <= 2 and len(a) <= 2 for s, a in pieces)
    assert [k for _s, a in pieces for k in a] == [0, 1, 2, 3, 4]
    assert [k for s, _a in pieces for k in s] == [0]


def test_ruler_range_spans_the_positions_with_a_margin():
    lo, hi = ruler_range([3.0, 5.0], RANGE)
    assert lo < 3.0 and hi > 5.0 and hi - lo < 2.6
    lo, hi = ruler_range([6.1], RANGE)
    assert hi - lo == pytest.approx(RULER_MIN_SPAN_MM) and lo < 6.1 < hi
    lo, hi = ruler_range([0.05], RANGE)  # kept inside the atlas
    assert lo == pytest.approx(0.0) and hi == pytest.approx(RULER_MIN_SPAN_MM)
    lo, hi = ruler_range([-0.5, 1.0], RANGE)  # unless a position lies outside it
    assert lo < -0.5
    assert ruler_range([], RANGE) == RANGE


# --- the layout ------------------------------------------------------------------


@pytest.mark.parametrize("positions", [
    [2.0, 4.0, 6.0, 8.0], [2.0, 6.0, 4.0, 8.0], [9.0, 2.0, 3.0, 4.0, 5.0],
    [8.0, 6.0, 7.0, 2.0], [3.0, 3.0, 2.0, 5.0, 1.0, 9.0],
])
def test_sections_in_position_order_and_lines_never_cross(positions):
    layout = _one(positions)
    xs = [slot.centre_x for slot in layout.sections]
    assert xs == sorted(xs)
    shown = [slot.position_mm for slot in layout.sections]
    assert shown == sorted(positions)
    lines = {slot.key: (slot.centre_x, slot.top, layout.x_at(slot.position_mm or 0.0),
                        layout.ruler[2]) for slot in layout.sections}
    for a, b in itertools.combinations(lines, 2):
        assert not _segments_cross(lines[a], lines[b]), (a, b)


def test_ties_keep_stack_order_and_labels_keep_the_original_index():
    layout = _one([3.0, 3.0, 2.0])
    assert [slot.key for slot in layout.sections] == ["s2", "s0", "s1"]
    assert layout.sections[0].label == ("2: s2.tif", "2.00 mm")


def test_atlas_thumbnails_over_their_positions_above_the_ruler():
    layout = _one([2.0, 10.0], atlas_mm=(8.0, 3.0, 6.0), tile_edge=120)
    assert [slot.position_mm for slot in layout.atlas] == [3.0, 6.0, 8.0]
    for slot in layout.atlas:  # room for each over its own millimetre
        assert slot.centre_x == pytest.approx(layout.x_at(slot.position_mm or 0.0))
    ruler_y = layout.ruler[2]
    assert all(slot.bottom < ruler_y for slot in layout.atlas)
    assert all(slot.top > ruler_y for slot in layout.sections)
    assert layout.atlas[0].label == ("atlas 3.00 mm",)


def test_crowded_atlas_thumbnails_move_aside_without_overlapping():
    layout = _one([6.1], atlas_mm=(5.9, 6.0, 6.1, 6.2))
    boxes = [slot.box for slot in layout.atlas]
    for left, right in itertools.pairwise(boxes):
        assert right[0] - left[2] >= GAP - 1e-6
    assert boxes[0][0] >= MARGIN - 1e-6 and boxes[-1][2] <= layout.width - MARGIN + 1e-6


def test_one_scale_for_sections_and_atlas():
    layout = _one([2.0, 4.0, 6.0])
    section, atlas = layout.sections[0], layout.atlas[0]
    assert section.width / atlas.width == pytest.approx(SECTION_UM[0] / ATLAS_UM[0], rel=1e-6)
    assert atlas.width * layout.um_per_px == pytest.approx(ATLAS_UM[0])
    # The largest item fills the tile.
    assert max(atlas.width, atlas.height) == pytest.approx(layout.tile)


def test_tile_grows_as_the_item_count_falls_and_a_long_stack_is_split_not_shrunk():
    one = _one([6.1], atlas_mm=(5.9, 6.1, 6.3))
    six = _one([3.0, 5.1, 6.1, 8.7, 8.9, 9.1])
    many = _plan([float(k) * 0.3 for k in range(38)])
    assert one.tile > six.tile
    assert len(many) == 7
    floor = (POSITIONING_MAX_WIDTH - 2 * MARGIN - (PER_PICTURE - 1) * GAP) // PER_PICTURE
    assert six.tile == floor
    for layout in many:  # every picture at one scale, the six-section tile
        assert layout.tile == floor and layout.um_per_px == pytest.approx(six.um_per_px)
        assert len(layout.sections) <= PER_PICTURE
        assert layout.width <= POSITIONING_MAX_WIDTH
    assert one.width <= POSITIONING_MAX_WIDTH
    # A small request stops at the picture size, not the width.
    assert _one([6.1], atlas_mm=(6.0,), tile_edge=300).tile == 300


def test_each_part_has_its_own_ruler_and_atlas_and_numbers():
    positions = [float(k) * 0.3 for k in range(14)]
    layouts = _plan(positions, atlas_mm=(0.5, 2.0, 3.6))
    assert [(lay.part, lay.parts) for lay in layouts] == [(0, 3), (1, 3), (2, 3)]
    assert [(lay.first, lay.last, lay.total) for lay in layouts] == [
        (1, 5, 14), (6, 10, 14), (11, 14, 14)]
    assert [[slot.position_mm for slot in lay.atlas] for lay in layouts] == [
        [0.5], [2.0], [3.6]]
    for layout in layouts:
        held = [slot.position_mm or 0.0 for slot in layout.sections + layout.atlas]
        lo, hi = layout.range_mm
        assert lo < min(held) and max(held) < hi
        assert hi - lo < 2.0  # a segment, not the whole atlas
    assert layouts[0].range_mm[1] <= layouts[1].range_mm[0] + 0.5


def test_finest_scale_is_never_passed():
    layout = _one([6.1], atlas_mm=(6.0,), finest_um=40.0)
    assert layout.um_per_px == pytest.approx(40.0)


def test_labels_shorten_to_fit_small_tiles():
    entries = [SectionEntry(id=f"M04_A_{k:02d}_Overlay.tif",
                            labels=(f"{k}: M04_A_{k:02d}_Overlay.tif",
                                    f"{k}: M04_A_{k:02d}_Overlay", str(k)),
                            position_mm=k * 0.3, extent_um=SECTION_UM) for k in range(38)]
    many = plan(entries, [], range_mm=RANGE, tile_edge=512, per_picture=38)[0]
    assert [slot.label[0] for slot in many.sections] == [str(k) for k in range(38)]
    assert many.sections[0].label[1] == "0.00"  # the unit dropped too
    few = plan(entries[:3], [], range_mm=RANGE, tile_edge=512)[0]
    assert few.sections[0].label == ("0: M04_A_00_Overlay.tif", "0.00 mm")


def test_ruler_ticks_and_section_band():
    layout = _one([2.0, 9.0], atlas_mm=())
    assert layout.label_step_mm == 1.0
    lo, hi = layout.range_mm
    assert layout.x_at(lo) == layout.ruler[0] and layout.x_at(hi) == layout.ruler[1]
    assert layout.x_at(-5.0) == layout.ruler[0] and layout.x_at(99.0) == layout.ruler[1]
    band = layout.sections[0].top - layout.ruler[2]
    assert band >= SECTION_LINK_PX[0]
    tiny = plan(_entries([0.05, 0.15]), [], range_mm=(0.0, 0.25), tile_edge=512)[0]
    assert tiny.label_step_mm < 0.25


# --- drawing ---------------------------------------------------------------------


def _paint(layout, **kw):
    from PIL import Image

    def tile(width: float, height: float) -> Image.Image:
        return Image.new("RGB", (max(1, round(width)), max(1, round(height))), (90, 90, 90))

    return paint(layout,
                 draw_section=lambda _id, um: tile(*(v / um for v in SECTION_UM)),
                 draw_atlas=lambda _mm, um: tile(*(v / um for v in ATLAS_UM)), **kw)


def test_lines_drawn_in_one_colour():
    image = np.asarray(_paint(_one([2.0, 6.0, 4.0, 8.0])), dtype=np.int16)
    reddish = (image[..., 0] - np.maximum(image[..., 1], image[..., 2])) > 60
    assert not reddish.any()


def test_paint_crop_is_the_box_at_its_magnification():
    layout = _one([2.0, 6.0, 4.0])
    whole = _paint(layout)
    assert whole.size == (layout.width, layout.height)
    window = (0.25, 0.5, 0.75, 1.0)
    zoomed = _paint(layout, scale=3.0, window=window)
    assert zoomed.width == pytest.approx(0.5 * layout.width * 3.0, abs=2)
    assert zoomed.height == pytest.approx(0.5 * layout.height * 3.0, abs=2)


# --- a stack's pictures (synthetic atlas) ---------------------------------------------


def test_positioning_look_one_picture_with_recipe_and_caption(tmp_path: Path):
    ws, state = stack(tmp_path)
    with collecting() as notes:
        pictures = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.2)))
    assert len(pictures) == 1
    picture = pictures[0]
    # Position order: s0 (0.05), s2 (0.10), s1 (0.15).
    assert picture.sections == ("s0.png", "s2.png", "s1.png")
    assert "in position order" in picture.caption and "red" not in picture.caption
    assert picture.extra["part"] == 0 and picture.extra["parts"] == 1
    recipe = json.loads(json.dumps(picture.recipe))
    assert recipe["renderer"] == "look" and recipe["args"]["mode"] == "positioning"
    assert recipe["args"]["sections"] == list(SECTIONS) and recipe["args"]["part"] == 0
    assert set(recipe["state"]["sections"]) == set(SECTIONS)
    assert "view_angles" in recipe["state"]
    held = note_for(picture.image, notes)
    assert held is not None and held.recipe == picture.recipe and held.caption == picture.caption
    assert held.mode == "positioning"


def test_positioning_split_into_parts_each_redrawn_by_its_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(positioning, "PER_PICTURE", 2)
    ws, state = stack(tmp_path)
    pictures = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.14)))
    assert [p.sections for p in pictures] == [("s0.png", "s2.png"), ("s1.png",)]
    assert [p.recipe["args"]["part"] for p in pictures] == [0, 1]
    assert [p.extra["positions_mm"] for p in pictures] == [[0.05], [0.14]]
    assert "part 2 of 2: section 3 of 3" in pictures[1].caption
    assert pictures[0].um_per_px == pytest.approx(pictures[1].um_per_px)
    second = pictures[1]
    width, height = second.recipe["shown"]
    zoomed = redraw(second.recipe, [0, 0, width / 2, height], ws, state=state)
    assert zoomed.sections == ("s1.png",) and zoomed.recipe["args"]["part"] == 1
    assert "part 2 of 2 zoomed" in zoomed.caption
    only = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.14), part=1))
    assert [p.sections for p in only] == [("s1.png",)]
    with pytest.raises(LookError) as caught:
        look(ws, state, LookRequest("positioning", part=5))
    assert caught.value.code == "UNKNOWN_PART"


def test_positioning_sections_ordered_by_position_not_index(tmp_path: Path):
    ws, state = stack(tmp_path)
    for record in state.slices:  # corrected order reversed: the order is the positions'
        record.index_corrected = len(state.slices) - 1 - record.index_original
    picture = look(ws, state, LookRequest("positioning", sections=("s1.png", "s0.png")))[0]
    assert picture.sections == ("s0.png", "s1.png")
