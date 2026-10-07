"""The positioning picture's layout (core/positioning.py).

The geometry is pure (:func:`plan`): stack order, the direction the row runs,
which lines cross, one scale for every thumbnail, and the tile growing as the
item count falls. :func:`paint` and the stack's picture are checked on the
synthetic atlas.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from langslice.core.layers import collecting, note_for
from langslice.core.look import LookRequest, look
from langslice.core.positioning import (
    CROSSING_COLOR,
    MIN_TILE,
    SECTION_LINK_PX,
    SectionEntry,
    crossing_pairs,
    paint,
    plan,
    stack_direction,
)
from tests.look_synthetic import SECTIONS, stack

#: A coronal-ish section and atlas plane, micrometres.
SECTION_UM = (9000.0, 6500.0)
ATLAS_UM = (10000.0, 7000.0)


def _entries(positions: list[float | None]) -> list[SectionEntry]:
    return [SectionEntry(id=f"s{k}", labels=(f"{k}: s{k}.tif", str(k)), position_mm=p,
                         extent_um=SECTION_UM) for k, p in enumerate(positions)]


def _plan(positions: list[float | None], atlas_mm: tuple[float, ...] = (3.0, 6.5, 10.0), **kw):
    return plan(_entries(positions), [(mm, ATLAS_UM) for mm in atlas_mm],
                range_mm=(0.0, 13.2), tile_edge=kw.pop("tile_edge", 512), **kw)


def _segments_cross(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Whether segments a and b ((x0, y0, x1, y1)) cross strictly inside."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    p1, p2, p3, p4 = (a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3])
    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


# --- order and crossings ---------------------------------------------------------


def test_stack_direction_follows_the_majority_of_pairs():
    assert stack_direction([1.0, 2.0, 3.0]) == 1
    assert stack_direction([3.0, 2.0, 1.0]) == -1
    assert stack_direction([1.0, 3.0, 2.0, 4.0]) == 1
    assert stack_direction([2.0, 1.0]) == -1
    assert stack_direction([1.0, 1.0]) == 1  # a tie runs left to right
    assert stack_direction([None, 2.0, None]) == 1


def test_crossing_pairs_are_the_pairs_against_the_direction():
    assert crossing_pairs([1.0, 2.0, 4.0, 3.0, 5.0]) == [(2, 3)]
    # A stack numbered posterior to anterior: no pair crosses...
    assert crossing_pairs([5.0, 4.0, 3.0, 2.0, 1.0]) == []
    # ...except the one swapped pair.
    assert crossing_pairs([5.0, 4.0, 2.0, 3.0, 1.0]) == [(2, 3)]
    # One section far out of order crosses every section it passes.
    assert crossing_pairs([1.0, 9.0, 2.0, 3.0]) == [(1, 2), (1, 3)]
    # Equal positions touch, they do not cross; no position never crosses.
    assert crossing_pairs([1.0, 1.0, None, 0.5, 2.0, 3.0]) == [(0, 3), (1, 3)]


# --- the layout ------------------------------------------------------------------


def test_sections_run_in_stack_order_and_flip_for_a_reversed_stack():
    forward = _plan([2.0, 4.0, 6.0, 8.0])
    xs = [slot.centre_x for slot in forward.sections]
    assert xs == sorted(xs) and forward.direction == 1
    assert [slot.key for slot in forward.sections] == ["s0", "s1", "s2", "s3"]
    backward = _plan([8.0, 6.0, 4.0, 2.0])
    xs = [slot.centre_x for slot in backward.sections]
    assert xs == sorted(xs, reverse=True) and backward.direction == -1
    assert not backward.crossings


@pytest.mark.parametrize("positions", [
    [2.0, 4.0, 6.0, 8.0], [2.0, 6.0, 4.0, 8.0], [9.0, 2.0, 3.0, 4.0, 5.0],
    [8.0, 6.0, 7.0, 2.0], [3.0, 3.0, 2.0, 5.0, 1.0, 9.0],
])
def test_drawn_lines_cross_exactly_for_the_flagged_pairs(positions):
    layout = _plan(positions)
    lines = {slot.key: (slot.centre_x, slot.top, layout.x_at(slot.position_mm or 0.0),
                        layout.ruler[2]) for slot in layout.sections}
    flagged = {frozenset(pair) for pair in layout.crossings}
    for a, b in itertools.combinations(lines, 2):
        assert _segments_cross(lines[a], lines[b]) == (frozenset((a, b)) in flagged), (a, b)
    crossing = {key for pair in layout.crossings for key in pair}
    assert {slot.key for slot in layout.sections if slot.crossing} == crossing


def test_atlas_thumbnails_in_position_order_above_the_ruler():
    layout = _plan([2.0, 4.0], atlas_mm=(10.0, 3.0, 6.5))
    assert [slot.position_mm for slot in layout.atlas] == [3.0, 6.5, 10.0]
    xs = [slot.centre_x for slot in layout.atlas]
    assert xs == sorted(xs)
    ruler_y = layout.ruler[2]
    assert all(slot.bottom < ruler_y for slot in layout.atlas)
    assert all(slot.top > ruler_y for slot in layout.sections)
    assert layout.atlas[0].label == ("atlas 3.00 mm",)
    assert layout.sections[0].label == ("0: s0.tif", "2.00 mm")


def test_one_scale_for_sections_and_atlas():
    layout = _plan([2.0, 4.0, 6.0])
    section, atlas = layout.sections[0], layout.atlas[0]
    assert section.width / atlas.width == pytest.approx(SECTION_UM[0] / ATLAS_UM[0], rel=1e-6)
    assert atlas.width * layout.um_per_px == pytest.approx(ATLAS_UM[0])
    # The largest item fills the tile.
    assert max(atlas.width, atlas.height) == pytest.approx(layout.tile)


def test_tile_grows_as_the_item_count_falls():
    one = _plan([6.1], atlas_mm=(5.9, 6.1, 6.3))
    six = _plan([3.0, 5.1, 6.1, 8.7, 8.9, 9.1])
    many = _plan([float(k) * 0.3 for k in range(38)])
    assert one.tile > six.tile > many.tile >= MIN_TILE
    assert one.um_per_px < six.um_per_px < many.um_per_px
    for layout in (one, six, many):
        assert layout.width <= 1568
    # A small request stops at the picture size, not the width.
    assert _plan([6.1], atlas_mm=(6.0,), tile_edge=300).tile == 300


def test_finest_scale_is_never_passed():
    layout = _plan([6.1], atlas_mm=(6.0,), finest_um=40.0)
    assert layout.um_per_px == pytest.approx(40.0)


def test_labels_shorten_to_fit_small_tiles():
    entries = [SectionEntry(id=f"M04_A_{k:02d}_Overlay.tif",
                            labels=(f"{k}: M04_A_{k:02d}_Overlay.tif",
                                    f"{k}: M04_A_{k:02d}_Overlay", str(k)),
                            position_mm=k * 0.3, extent_um=SECTION_UM) for k in range(38)]
    many = plan(entries, [], range_mm=(0.0, 13.2), tile_edge=512)
    assert [slot.label[0] for slot in many.sections] == [str(k) for k in range(38)]
    assert many.sections[0].label[1] == "0.00"  # the unit dropped too
    few = plan(entries[:3], [], range_mm=(0.0, 13.2), tile_edge=512)
    assert few.sections[0].label == ("0: M04_A_00_Overlay.tif", "0.00 mm")


def test_ruler_ticks_and_section_band():
    layout = _plan([2.0, 4.0])
    assert layout.label_step_mm == 1.0
    assert layout.x_at(0.0) == layout.ruler[0] and layout.x_at(13.2) == layout.ruler[1]
    assert layout.x_at(-5.0) == layout.ruler[0] and layout.x_at(99.0) == layout.ruler[1]
    band = layout.sections[0].top - layout.ruler[2]
    assert band >= SECTION_LINK_PX[0]
    tiny = plan(_entries([0.05, 0.15]), [], range_mm=(0.0, 0.25), tile_edge=512)
    assert tiny.label_step_mm < 0.25


# --- drawing ---------------------------------------------------------------------


def _paint(layout, **kw):
    from PIL import Image

    def tile(width: float, height: float) -> Image.Image:
        return Image.new("RGB", (max(1, round(width)), max(1, round(height))), (90, 90, 90))

    return paint(layout,
                 draw_section=lambda _id, um: tile(*(v / um for v in SECTION_UM)),
                 draw_atlas=lambda _mm, um: tile(*(v / um for v in ATLAS_UM)), **kw)


def _has_colour(image, rgb, tolerance=12) -> bool:
    pixels = np.asarray(image, dtype=np.int16)
    return bool((np.abs(pixels - np.asarray(rgb)).max(axis=-1) <= tolerance).any())


def test_crossing_lines_drawn_in_their_colour_only_when_they_cross():
    assert _has_colour(_paint(_plan([2.0, 6.0, 4.0, 8.0])), CROSSING_COLOR)
    assert not _has_colour(_paint(_plan([2.0, 4.0, 6.0, 8.0])), CROSSING_COLOR)


def test_paint_crop_is_the_box_at_its_magnification():
    layout = _plan([2.0, 6.0, 4.0])
    whole = _paint(layout)
    assert whole.size == (layout.width, layout.height)
    window = (0.25, 0.5, 0.75, 1.0)
    zoomed = _paint(layout, scale=3.0, window=window)
    assert zoomed.width == pytest.approx(0.5 * layout.width * 3.0, abs=2)
    assert zoomed.height == pytest.approx(0.5 * layout.height * 3.0, abs=2)


# --- a stack's picture (synthetic atlas) ----------------------------------------------


def test_positioning_look_one_picture_with_recipe_and_caption(tmp_path: Path):
    ws, state = stack(tmp_path)
    with collecting() as notes:
        pictures = look(ws, state, LookRequest("positioning", positions_mm=(0.05, 0.2)))
    assert len(pictures) == 1
    picture = pictures[0]
    assert picture.sections == SECTIONS
    # s1 (0.15) and s2 (0.10) are out of order.
    assert picture.extra["crossings"] == [["s1.png", "s2.png"]]
    assert "1 pair whose order and positions disagree" in picture.caption
    assert "s1.png / s2.png" in picture.caption
    recipe = json.loads(json.dumps(picture.recipe))
    assert recipe["renderer"] == "look" and recipe["args"]["mode"] == "positioning"
    assert recipe["args"]["sections"] == list(SECTIONS)
    assert set(recipe["state"]["sections"]) == set(SECTIONS)
    assert "view_angles" in recipe["state"]
    held = note_for(picture.image, notes)
    assert held is not None and held.recipe == picture.recipe and held.caption == picture.caption
    assert held.mode == "positioning"


def test_positioning_sections_in_index_original_order(tmp_path: Path):
    ws, state = stack(tmp_path)
    for record in state.slices:  # corrected order reversed: stack order stays the files'
        record.index_corrected = len(state.slices) - 1 - record.index_original
    picture = look(ws, state, LookRequest("positioning", sections=("s2.png", "s0.png")))[0]
    assert picture.sections == ("s0.png", "s2.png")
