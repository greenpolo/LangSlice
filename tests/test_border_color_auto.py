"""Atlas borders are cased lines: a black or white core edged on both sides by
the other, the arrangement chosen per picture from the lightness under the
lines (Nash, 2026-10-14: "black with white outlines or white with black
outlines"). The image model's trace stays plain yellow."""

from __future__ import annotations

import numpy as np
import pytest

from langslice.core.canvas import (
    CASING_PX,
    _draw_polys,
    border_words,
    pick_border_core,
    resolve_border_color,
)

LINE = [np.array([[10.0, 30.0], [50.0, 30.0], [50.0, 31.0]])]


def _flat(value: int, size: int = 60) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _band(size: int = 60) -> np.ndarray:
    coverage = np.zeros((size, size), dtype=np.float32)
    coverage[size // 2 - 3 : size // 2 + 3, :] = 1.0
    return coverage


def test_a_light_picture_gets_a_black_core_and_a_dark_one_a_white_core():
    assert pick_border_core(_flat(235), _band()) == "black"
    assert pick_border_core(_flat(40), _band()) == "white"
    assert resolve_border_color("auto", 1.0, rgb=_flat(40), coverage=_band()) == ("white", 1.0)
    assert resolve_border_color("auto", 1.0) == ("black", 1.0)  # nothing drawn: unused
    assert border_words("black") == "black, white-edged"
    assert border_words("white") == "white, black-edged"


def test_the_lightness_is_read_under_the_lines_only():
    rgb = _flat(30)
    rgb[:20] = 240  # a pale band away from the lines
    assert pick_border_core(rgb, _band()) == "white"


def test_an_explicit_core_is_kept_and_anything_else_is_refused():
    assert resolve_border_color("white", 0.75) == ("white", 0.75)
    with pytest.raises(ValueError, match="border_color"):
        resolve_border_color("yellow", 0.75)


@pytest.mark.parametrize(("core", "ground"), [("black", 200), ("white", 60)])
def test_a_line_has_its_core_and_a_casing_on_both_sides(core, ground):
    canvas = _flat(ground, 80)
    _draw_polys(canvas, LINE, core, thickness=1.0)
    column = canvas[:, 30, 0].astype(int)
    ink = 0 if core == "black" else 255
    assert column[30] == pytest.approx(ink, abs=40)  # the core
    other = 255 - ink
    for side in (28, 32):  # the casing, one pixel either side of the core
        assert abs(column[side] - other) < abs(column[side] - ink)
    assert column[30 - 4] == ground and column[30 + 4] == ground  # nothing beyond it
    assert CASING_PX >= 1.0


def test_a_faint_line_fades_core_and_casing_together():
    canvas = _flat(128, 80)
    _draw_polys(canvas, LINE, "black", thickness=1.0, alpha=0.35)
    column = canvas[:, 30, 0].astype(int)
    assert 60 < column[30] < 128 < column[32] < 190
