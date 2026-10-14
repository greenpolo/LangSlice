"""The positioning picture's layout (core/positioning.py).

The geometry is pure (:func:`plan`): sections in position order (ties in
stack order, no position last), so no two lines cross; a long stack split
into pictures of consecutive positions at one scale, each with its own
ruler segment; atlas thumbnails over their millimetres, in the picture
whose positions surround them. The stack's pictures (order, parts, writes
nothing) are checked on the synthetic atlas.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import pytest

from langslice.core import positioning
from langslice.core.look import LookError, LookRequest, look
from langslice.core.positioning import (
    GAP,
    MARGIN,
    PER_PICTURE,
    POSITIONING_MAX_WIDTH,
    SectionEntry,
    plan,
    position_order,
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


# --- a stack's pictures (synthetic atlas) ---------------------------------------------


def test_positioning_split_into_parts_each_redrawn_by_its_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(positioning, "PER_PICTURE", 2)
    ws, state = stack(tmp_path)
    pictures = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.14)))
    assert [p.sections for p in pictures] == [("s0.png", "s2.png"), ("s1.png",)]
    assert [p.recipe["args"]["part"] for p in pictures] == [0, 1]
    assert [p.extra["positions_mm"] for p in pictures] == [[0.05], [0.14]]
    assert "part 2 of 2: 1 of the 3 sections (s1.png)" in pictures[1].caption
    assert "part 1 of 2: 2 of the 3 sections (s0.png to s2.png)" in pictures[0].caption
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


@pytest.mark.parametrize("placed", [False, True])
def test_unplaced_sections_can_compare_candidates_without_writing(tmp_path: Path, placed: bool):
    ws, state = stack(tmp_path)
    for record in state.slices:
        record.position_mm = None
    if placed:
        state.slices[1].position_mm = 0.15
    before = state.to_dict()
    pictures = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.2)))
    assert state.to_dict() == before
    assert len(pictures) == 1
    picture = pictures[0]
    assert picture.extra["positions_mm"] == [0.05, 0.2]
    assert f"{2 if placed else 3} without a position" in picture.caption
    expected = ("s1.png", "s0.png", "s2.png") if placed else SECTIONS
    assert picture.sections == expected
    width, height = picture.recipe["shown"]
    zoomed = redraw(picture.recipe, [0, 0, width, height], ws, state=state)
    assert zoomed.sections == expected
    assert state.to_dict() == before
