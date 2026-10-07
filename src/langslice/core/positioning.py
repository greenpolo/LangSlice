"""The positioning picture: the stack laid out along its slicing axis, as in ABBA.

A horizontal millimetre ruler; above it, atlas thumbnails at the requested
positions, each over its own millimetre (moved aside only as far as its
neighbours need) and joined by a short line to it; below it, the sections
in POSITION order (their current ``position_mm``, ties in stack order,
``index_original``; sections without a position last), evenly spaced,
each joined by a thin line to its position. Both rows are sorted along
the ruler, so no two lines cross; each section's label keeps
its original index and filename, so a filename order that disagrees with
the positions shows in the labels.

A stack is never drawn smaller to fit: at most :data:`PER_PICTURE` sections
(and as many atlas thumbnails) go in one picture, and a longer stack is
split into several pictures, each a consecutive run of positions
(:func:`split`) with its own ruler segment over the positions it holds
(:func:`ruler_range`). Every thumbnail of every picture is drawn at ONE
micrometres per pixel (:func:`langslice.core.scale.pair_um_per_px`): the
largest item of the whole call fills a tile; the tile is the width shared
by the fullest picture's row, so it grows as the item count falls (one
section beside three atlas positions is drawn large), and is never smaller
than :data:`POSITIONING_MAX_WIDTH` divided among :data:`PER_PICTURE` tiles.

:func:`plan` is the pure geometry (sizes in micrometres in, one
:class:`Layout` per picture out); :func:`paint` draws a layout at a
magnification and crop, which is how :mod:`langslice.core.zoom` redraws a
box of it at more detail (the furniture scales with the tiles, so a box of
the picture is the same box at any magnification);
:func:`positioning_pictures` does both for a stack.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.core.appearance import Look
from langslice.core.atlas.render import atlas_um_per_px
from langslice.core.atlas_fetch import atlas_section
from langslice.core.canvas import zoom_box
from langslice.core.captions import _font, angles_label, caption
from langslice.core.display import DisplayOptions, framed_atlas
from langslice.core.opening import CLAUDE_MAX_IMAGE_EDGE
from langslice.core.scale import (
    finest_um_per_px,
    framed_um_per_px,
    pair_um_per_px,
    section_at,
    section_um_per_px,
)
from langslice.core.sections import PREVIEW_LONG_EDGE, render_slice
from langslice.core.state import Angles, SliceState, StackState
from langslice.core.workspace import Workspace

#: Widest positioning picture: the largest image every model lane takes in
#: without shrinking it.
POSITIONING_MAX_WIDTH = CLAUDE_MAX_IMAGE_EDGE
#: Most sections, and most atlas thumbnails, in one picture. Six tiles share
#: :data:`POSITIONING_MAX_WIDTH` at about 240 px each, a coronal mouse
#: section about 190 px wide; a longer stack is split, never shrunk.
PER_PICTURE = 6
#: Narrowest picture: the ruler stays readable with one or two tiles.
MIN_WIDTH = 720
#: Space at the left and right ends, and between two tiles of a row.
MARGIN = 28
GAP = 10
PAD = 14
#: Height of the band between the atlas row and the ruler.
ATLAS_LINK_PX = 30
#: Height of the band between the ruler and the section row: the share of
#: the picture's width, between these bounds.
SECTION_LINK_SHARE = 0.06
SECTION_LINK_PX = (56, 96)
#: A picture's ruler spans the positions it holds plus this share of their
#: span at each end (at least :data:`RULER_MIN_MARGIN_MM`), and at least
#: :data:`RULER_MIN_SPAN_MM` in all.
RULER_MARGIN_SHARE = 0.08
RULER_MIN_MARGIN_MM = 0.15
RULER_MIN_SPAN_MM = 1.0
#: How far a zoom may draw the atlas past its voxels (it holds no finer detail,
#: but its lines stay sharp and its pixels legible).
ATLAS_UPSAMPLE = 4.0

BACKGROUND = (0, 0, 0)
RULER_COLOR = (215, 215, 215)
TICK_TEXT_COLOR = (185, 185, 185)
TEXT_COLOR = (235, 235, 235)
#: A section's line to its position, and its dot on the ruler.
SECTION_LINE_COLOR = (165, 165, 165)
SECTION_DOT_COLOR = (240, 240, 240)
#: An atlas thumbnail's line to its position, and its marker (the atlas
#: borders' yellow, dimmed).
ATLAS_LINE_COLOR = (200, 185, 70)


# --- order and parts ----------------------------------------------------------------


@dataclass(frozen=True)
class SectionEntry:
    """One section as :func:`plan` takes it."""

    id: str
    #: First-line label candidates, longest first; the first that fits its
    #: slot is drawn (the last is always drawn).
    labels: tuple[str, ...]
    position_mm: float | None
    #: ``(width, height)`` of its tissue-framed picture, micrometres.
    extent_um: tuple[float, float]


def position_order(entries: Sequence[SectionEntry]) -> list[SectionEntry]:
    """*entries* (given in stack order) sorted by position; ties keep the
    given order, sections without a position go last."""
    return sorted(entries, key=lambda entry: (entry.position_mm is None,
                                              entry.position_mm or 0.0))


def _even(count: int, parts: int) -> list[int]:
    """*count* split into *parts* sizes differing by at most one, larger first."""
    base, extra = divmod(count, parts)
    return [base + (1 if k < extra else 0) for k in range(parts)]


def split(
    positions: Sequence[float | None], atlas_mm: Sequence[float], per_picture: int = PER_PICTURE,
) -> list[tuple[list[int], list[int]]]:
    """The pictures of a call: ``(section indices, atlas indices)`` each.

    *positions* are the sections' in position order (None last),
    *atlas_mm* sorted. The sections are cut into the fewest consecutive runs
    of at most *per_picture*, their sizes as even as possible; each atlas
    position goes to the run whose positions surround it (the boundary
    between two runs is halfway between them). A run given more than
    *per_picture* atlas positions is split again, both rows evenly.
    """
    per = max(1, int(per_picture))
    count = len(positions)
    runs: list[list[int]] = []
    start = 0
    for size in _even(count, max(1, math.ceil(count / per))):
        runs.append(list(range(start, start + size)))
        start += size
    bounds: list[float] = []
    for left, right in zip(runs, runs[1:], strict=False):
        a, b = positions[left[-1]], positions[right[0]]
        bounds.append(math.inf if a is None or b is None else (a + b) / 2.0)
    held: list[list[int]] = [[] for _ in runs]
    for index, mm in enumerate(atlas_mm):
        held[sum(1 for bound in bounds if mm > bound)].append(index)
    out: list[tuple[list[int], list[int]]] = []
    for sections, atlas in zip(runs, held, strict=True):
        pieces = max(1, math.ceil(len(atlas) / per))
        s_sizes, a_sizes = _even(len(sections), pieces), _even(len(atlas), pieces)
        s0 = a0 = 0
        for s_size, a_size in zip(s_sizes, a_sizes, strict=True):
            out.append((sections[s0:s0 + s_size], atlas[a0:a0 + a_size]))
            s0, a0 = s0 + s_size, a0 + a_size
    return out


def ruler_range(values: Sequence[float], bounds: tuple[float, float]) -> tuple[float, float]:
    """The ruler segment for the positions *values*: their span plus a
    margin (:data:`RULER_MARGIN_SHARE`, at least :data:`RULER_MIN_MARGIN_MM`),
    at least :data:`RULER_MIN_SPAN_MM` long, kept inside the atlas's
    *bounds* where the positions allow. No values: *bounds*."""
    b0, b1 = float(bounds[0]), float(bounds[1])
    if not values:
        return b0, b1
    vmin, vmax = float(min(values)), float(max(values))
    margin = max((vmax - vmin) * RULER_MARGIN_SHARE, RULER_MIN_MARGIN_MM)
    lo, hi = vmin - margin, vmax + margin
    if hi - lo < RULER_MIN_SPAN_MM:
        centre = (vmin + vmax) / 2.0
        lo, hi = centre - RULER_MIN_SPAN_MM / 2.0, centre + RULER_MIN_SPAN_MM / 2.0
    floor, ceiling = min(b0, vmin - 0.02), max(b1, vmax + 0.02)
    if lo < floor:
        lo, hi = floor, hi + (floor - lo)
    if hi > ceiling:
        lo, hi = lo - (hi - ceiling), ceiling
    return max(lo, floor), min(hi, ceiling)


# --- the geometry ------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One thumbnail's place in a :class:`Layout` (picture pixels at
    magnification 1): centred on ``centre_x``, ``top`` its upper edge."""

    key: str
    centre_x: float
    top: float
    width: float
    height: float
    position_mm: float | None
    label: tuple[str, ...]

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def box(self) -> tuple[float, float, float, float]:
        half = self.width / 2.0
        return (self.centre_x - half, self.top, self.centre_x + half, self.bottom)


@dataclass(frozen=True)
class Layout:
    """Where everything of one positioning picture goes, at magnification 1."""

    width: int
    height: int
    #: Micrometres per pixel of every thumbnail (of every picture of the call).
    um_per_px: float
    tile: int
    font_px: int
    small_px: int
    ruler: tuple[float, float, float]
    #: The millimetres the ruler spans, left to right.
    range_mm: tuple[float, float]
    atlas: tuple[Slot, ...]
    #: In position order, left to right.
    sections: tuple[Slot, ...]
    label_step_mm: float
    tick_step_mm: float
    #: Top of the atlas labels' line and of the section labels' first line:
    #: one line per row, whatever each thumbnail's height.
    atlas_label_y: float = 0.0
    section_label_y: float = 0.0
    #: This picture's number among the call's (0-based) and their count.
    part: int = 0
    parts: int = 1
    #: The 1-based ranks, in position order, of this picture's first and last
    #: section among the call's ``total`` (0, 0 without sections).
    first: int = 0
    last: int = 0
    total: int = 0

    def x_at(self, position_mm: float) -> float:
        """The ruler's x at *position_mm* (clamped to the ruler)."""
        x0, x1, _y = self.ruler
        lo, hi = self.range_mm
        fraction = (float(position_mm) - lo) / max(hi - lo, 1e-9)
        return x0 + (x1 - x0) * min(max(fraction, 0.0), 1.0)


def _fits(text: str, px: int, width: float) -> bool:
    return float(_font(px).getlength(text)) <= width


def _first_fitting(candidates: Sequence[str], px: int, width: float) -> str:
    for text in candidates:
        if _fits(text, px, width):
            return text
    return candidates[-1]


#: Ruler label steps (mm), the smallest whose labels stay apart is used.
_LABEL_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _steps(px_per_mm: float, small_px: int) -> tuple[float, float]:
    """``(label step, tick step)`` in mm: labels at least seven digits apart,
    ticks halfway between labels when there is room."""
    label = next((step for step in _LABEL_STEPS if step * px_per_mm >= 7 * small_px),
                 _LABEL_STEPS[-1])
    tick = label / 2.0 if label / 2.0 * px_per_mm >= 10 else label
    return label, tick


def _spread(wanted: Sequence[float], widths: Sequence[float], lo: float, hi: float
            ) -> list[float]:
    """Centres as near *wanted* (ascending) as boxes of *widths*, *GAP*
    apart and inside ``[lo, hi]``, allow, order kept."""
    xs = [float(x) for x in wanted]
    for k, width in enumerate(widths):  # pushed right, off the left end and each other
        floor = lo + width / 2.0 if k == 0 else xs[k - 1] + (widths[k - 1] + width) / 2.0 + GAP
        xs[k] = max(xs[k], floor)
    for k in range(len(xs) - 1, -1, -1):  # then left, off the right end
        ceiling = (hi - widths[k] / 2.0 if k == len(xs) - 1
                   else xs[k + 1] - (widths[k + 1] + widths[k]) / 2.0 - GAP)
        xs[k] = min(xs[k], ceiling)
    return xs


def plan(
    sections: Sequence[SectionEntry],
    atlas: Sequence[tuple[float, tuple[float, float]]],
    *,
    range_mm: tuple[float, float],
    tile_edge: int,
    max_width: int = POSITIONING_MAX_WIDTH,
    finest_um: float = 0.0,
    per_picture: int = PER_PICTURE,
) -> list[Layout]:
    """The layouts of *sections* (given in stack order; drawn in position
    order) and *atlas* (``(mm, (width, height) um)`` per thumbnail, any
    order), one per picture (:func:`split`); *range_mm* is the atlas's
    extent along the axis (a ruler's bounds, :func:`ruler_range`).

    The tile is *tile_edge* while the fullest picture's row fits
    *max_width*, else that row's share of it; one micrometres per pixel
    puts the largest item's long edge at the tile, never finer than
    *finest_um* (a section's own pixels).
    """
    ordered = position_order(sections)
    marks = sorted(atlas, key=lambda item: item[0])
    pieces = split([entry.position_mm for entry in ordered], [mm for mm, _e in marks],
                   per_picture)
    count = max([max(len(s), len(a), 1) for s, a in pieces] or [1])
    room = max_width - 2 * MARGIN - (count - 1) * GAP
    tile = int(max(1, min(int(tile_edge), room // count)))
    extents = [max(entry.extent_um) for entry in ordered] + [max(e) for _mm, e in marks]
    um_per_px = pair_um_per_px(extents or [tile], tile, finest=finest_um)
    inner = max(count * tile + (count - 1) * GAP, MIN_WIDTH - 2 * MARGIN)
    out: list[Layout] = []
    for part, (indices, atlas_indices) in enumerate(pieces):
        out.append(_layout(
            [ordered[k] for k in indices], [marks[k] for k in atlas_indices],
            bounds=range_mm, tile=tile, inner=inner, um_per_px=um_per_px, part=part,
            parts=len(pieces), first=indices[0] + 1 if indices else 0,
            last=indices[-1] + 1 if indices else 0, total=len(ordered)))
    return out


def _layout(
    sections: Sequence[SectionEntry], atlas: Sequence[tuple[float, tuple[float, float]]], *,
    bounds: tuple[float, float], tile: int, inner: int, um_per_px: float, part: int, parts: int,
    first: int, last: int, total: int,
) -> Layout:
    """One picture: *sections* in position order, *atlas* sorted."""
    width = inner + 2 * MARGIN
    font_px = 14 if tile >= 110 else 12 if tile >= 56 else 11
    small_px = 12 if tile >= 56 else 11
    label_h = font_px + 5

    def centres(n: int) -> list[float]:
        pitch = inner / max(n, 1)
        return [MARGIN + pitch * (k + 0.5) for k in range(n)]

    def room(n: int) -> float:
        return inner / max(n, 1) - 4

    values = [float(e.position_mm) for e in sections if e.position_mm is not None]
    lo, hi = ruler_range(values + [float(mm) for mm, _e in atlas], bounds)
    x0, x1 = float(MARGIN), float(width - MARGIN)

    def x_at(mm: float) -> float:
        return x0 + (x1 - x0) * min(max((mm - lo) / max(hi - lo, 1e-9), 0.0), 1.0)

    y = float(PAD)
    atlas_label_y = y
    atlas_slots: list[Slot] = []
    if atlas:
        # Each atlas thumbnail above its own millimetre, moved aside only as
        # far as its neighbours need.
        sizes = [(e[0] / um_per_px, e[1] / um_per_px) for _mm, e in atlas]
        texts = [_first_fitting((f"atlas {mm:.2f} mm", f"{mm:.2f} mm", f"{mm:.1f}"),
                                font_px, max(float(tile), w)) for (mm, _e), (w, _h)
                 in zip(atlas, sizes, strict=True)]
        boxes = [max(w, float(_font(font_px).getlength(t))) for (w, _h), t
                 in zip(sizes, texts, strict=True)]
        xs = _spread([x_at(mm) for mm, _e in atlas], boxes, x0, x1)
        bottom = y + label_h + max(h for _w, h in sizes)
        for (mm, _e), (w, h), x, text in zip(atlas, sizes, xs, texts, strict=True):
            atlas_slots.append(Slot(key=f"{mm:g}", centre_x=x, top=bottom - h, width=w,
                                    height=h, position_mm=float(mm), label=(text,)))
        y = bottom + ATLAS_LINK_PX
    else:
        y += 6
    ruler_y = y + 4
    label_step, tick_step = _steps((x1 - x0) / max(hi - lo, 1e-9), small_px)
    link = min(max(round(width * SECTION_LINK_SHARE), SECTION_LINK_PX[0]), SECTION_LINK_PX[1])
    top = ruler_y + 8 + small_px + 4 + link

    section_slots: list[Slot] = []
    for entry, x in zip(sections, centres(len(sections)), strict=True):
        w, h = entry.extent_um[0] / um_per_px, entry.extent_um[1] / um_per_px
        where = ((f"{entry.position_mm:.2f} mm", f"{entry.position_mm:.2f}")
                 if entry.position_mm is not None else ("no position", "none"))
        label = (_first_fitting(entry.labels, font_px, room(len(sections))),
                 _first_fitting(where, font_px, room(len(sections))))
        section_slots.append(Slot(key=entry.id, centre_x=x, top=top, width=w, height=h,
                                  position_mm=entry.position_mm, label=label))
    tallest = max((slot.height for slot in section_slots), default=0.0)
    section_label_y = top + tallest + 4
    height = section_label_y + (2 * (font_px + 4) if section_slots else 0) + PAD
    return Layout(
        width=int(width), height=int(math.ceil(height)), um_per_px=float(um_per_px),
        tile=tile, font_px=font_px, small_px=small_px,
        ruler=(x0, x1, float(ruler_y)), range_mm=(lo, hi),
        atlas=tuple(atlas_slots), sections=tuple(section_slots),
        label_step_mm=label_step, tick_step_mm=tick_step,
        atlas_label_y=atlas_label_y, section_label_y=section_label_y,
        part=part, parts=parts, first=first, last=last, total=total,
    )


def visible(layout: Layout, window: Sequence[float] | None) -> tuple[list[Slot], list[Slot]]:
    """``(sections, atlas)`` slots that meet *window* (fractions of the
    picture; empty: all)."""
    if not window:
        return list(layout.sections), list(layout.atlas)
    x0, y0, x1, y1 = (window[0] * layout.width, window[1] * layout.height,
                      window[2] * layout.width, window[3] * layout.height)

    def meets(slot: Slot) -> bool:
        a, b, c, d = slot.box
        return a < x1 and c > x0 and b < y1 and d > y0

    return ([s for s in layout.sections if meets(s)], [s for s in layout.atlas if meets(s)])


# --- drawing -------------------------------------------------------------------------

#: Sub-pixel bits for OpenCV's anti-aliased lines.
_SHIFT = 4


def _pt(x: float, y: float) -> tuple[int, int]:
    return (round(x * (1 << _SHIFT)), round(y * (1 << _SHIFT)))


def paint(
    layout: Layout,
    *,
    draw_section: Callable[[str, float], Image.Image],
    draw_atlas: Callable[[float, float], Image.Image],
    scale: float = 1.0,
    window: Sequence[float] | None = None,
) -> Image.Image:
    """*layout* drawn at magnification *scale*, cropped to *window*
    (``[x0, y0, x1, y1]`` fractions of the picture; empty: all).

    ``draw_section(id, um_per_px)`` and ``draw_atlas(mm, um_per_px)`` return
    each thumbnail at that scale; only the thumbnails inside the crop are
    drawn. Every length (tiles, gaps, lines, fonts) scales together, so the
    crop shows exactly the box of the unmagnified picture.
    """
    s = float(scale)
    full = (max(1, round(layout.width * s)), max(1, round(layout.height * s)))
    cx0, cy0, cx1, cy1 = zoom_box(list(window) if window else [], full)
    out = Image.new("RGB", (cx1 - cx0, cy1 - cy0), BACKGROUND)

    def X(v: float) -> float:  # noqa: N802 - picture x of a layout x
        return v * s - cx0

    def Y(v: float) -> float:  # noqa: N802
        return v * s - cy0

    shown_sections, shown_atlas = visible(layout, window)
    um = layout.um_per_px / s
    for slot in shown_atlas:
        tile = draw_atlas(float(slot.position_mm or 0.0), um).convert("RGB")
        out.paste(tile, (round(X(slot.centre_x) - tile.width / 2.0),
                         round(Y(slot.bottom) - tile.height)))
    for slot in shown_sections:
        tile = draw_section(slot.key, um).convert("RGB")
        out.paste(tile, (round(X(slot.centre_x) - tile.width / 2.0), round(Y(slot.top))))

    canvas = np.asarray(out, dtype=np.uint8).copy()
    thin = max(1, round(s))
    x0, x1, ry = layout.ruler
    lo, hi = layout.range_mm
    cv2.line(canvas, _pt(X(x0), Y(ry)), _pt(X(x1), Y(ry)), RULER_COLOR, thin, cv2.LINE_AA,
             _SHIFT)
    ticks: list[tuple[float, bool]] = []
    step = layout.tick_step_mm
    k = math.ceil(lo / step - 1e-9)
    while k * step <= hi + 1e-9:
        mm = k * step
        major = abs(mm / layout.label_step_mm - round(mm / layout.label_step_mm)) < 1e-6
        ticks.append((mm, major))
        length = (6 if major else 3) * s
        x = X(layout.x_at(mm))
        cv2.line(canvas, _pt(x, Y(ry) - length), _pt(x, Y(ry) + length), RULER_COLOR, thin,
                 cv2.LINE_AA, _SHIFT)
        k += 1
    for slot in layout.atlas:
        if slot.position_mm is None:
            continue
        x = X(layout.x_at(slot.position_mm))
        cv2.line(canvas, _pt(X(slot.centre_x), Y(slot.bottom) + 2 * s), _pt(x, Y(ry) - 5 * s),
                 ATLAS_LINE_COLOR, thin, cv2.LINE_AA, _SHIFT)
        size = 4.5 * s
        triangle = np.array([_pt(x - size, Y(ry) - 1.6 * size), _pt(x + size, Y(ry) - 1.6 * size),
                             _pt(x, Y(ry) - 1)], dtype=np.int32)
        cv2.fillPoly(canvas, [triangle], ATLAS_LINE_COLOR, cv2.LINE_AA, _SHIFT)
    for slot in layout.sections:
        if slot.position_mm is None:
            continue
        x = X(layout.x_at(slot.position_mm))
        cv2.line(canvas, _pt(X(slot.centre_x), Y(slot.top) - 3 * s), _pt(x, Y(ry)),
                 SECTION_LINE_COLOR, thin, cv2.LINE_AA, _SHIFT)
        cv2.circle(canvas, _pt(x, Y(ry)), round(3.0 * s * (1 << _SHIFT)), SECTION_DOT_COLOR,
                   -1, cv2.LINE_AA, _SHIFT)
    out = Image.fromarray(canvas, mode="RGB")

    draw = ImageDraw.Draw(out)
    small = _font(max(6, round(layout.small_px * s)))
    font = _font(max(6, round(layout.font_px * s)))
    majors = [mm for mm, major in ticks if major]
    for mm in majors:
        number = f"{mm:g}"
        # The number is centred on its tick; the last one is followed by the
        # unit, kept inside the picture.
        text = number + (" mm" if mm == majors[-1] else "")
        left = min(X(layout.x_at(mm)) - float(small.getlength(number)) / 2.0,
                   X(layout.width - 6) - float(small.getlength(text)))
        draw.text((left, Y(ry) + 8 * s), text, fill=TICK_TEXT_COLOR, font=small,
                  stroke_width=max(1, round(2 * s)), stroke_fill=BACKGROUND)
    for slot in layout.atlas:
        _centred(draw, slot.label[0], X(slot.centre_x), Y(layout.atlas_label_y), font,
                 TEXT_COLOR)
    line_h = (layout.font_px + 4) * s
    for slot in layout.sections:
        for row, text in enumerate(slot.label):
            _centred(draw, text, X(slot.centre_x), Y(layout.section_label_y) + row * line_h,
                     font, TEXT_COLOR)
    return out


def _centred(draw: ImageDraw.ImageDraw, text: str, x: float, y: float, font: Any,
             color: tuple[int, int, int]) -> None:
    draw.text((x - float(font.getlength(text)) / 2.0, y), text, fill=color, font=font)


# --- a stack's pictures ---------------------------------------------------------------


@dataclass(frozen=True)
class Positioned:
    """One positioning picture: the captioned image, its burned caption, the
    layout it was drawn from, the magnification and window it was drawn at,
    and the micrometres per pixel of its thumbnails as shown."""

    image: Image.Image
    caption: str
    layout: Layout
    scale: float
    window: tuple[float, ...]
    um_per_px: float
    #: The picture's size above its caption band.
    content: tuple[int, int]


def section_extent(
    ws: Workspace, state: StackState, record: SliceState,
) -> tuple[tuple[float, float], float, float]:
    """``((width, height) um, working um/px, finest um/px)`` of *record*'s
    tissue-framed picture as its placement draws it
    (:func:`langslice.core.scale.section_um_per_px`)."""
    working_um = section_um_per_px(ws, state, record)
    framed = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE, frame=True)
    per_px = framed_um_per_px(ws, record, working_um, long_edge=PREVIEW_LONG_EDGE)
    return ((framed.width * per_px, framed.height * per_px), working_um,
            finest_um_per_px(ws, record, working_um))


def atlas_extent(
    ws: Workspace, state: StackState, position_mm: float, angles: Angles,
) -> tuple[float, float]:
    """``(width, height)`` micrometres of the anatomy-framed atlas plane."""
    picture = atlas_section(ws, state, position_mm, frame=True, angles=angles)
    voxel = atlas_um_per_px(ws.atlas)
    return (picture.width * voxel, picture.height * voxel)


def label_candidates(record: SliceState) -> tuple[str, ...]:
    """A section's first-line labels, longest first: its original index
    (``index_original``, the filename order) and filename with short
    correction flags, then without the file extension, then the index alone."""
    flags = []
    if record.rotation_deg:
        flags.append(f"rot {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append("damaged")
    full = f"{record.index_original}: {record.id}" + (f" [{', '.join(flags)}]" if flags else "")
    stem = record.id.rsplit(".", 1)[0] if "." in record.id else record.id
    return (full, f"{record.index_original}: {stem}", str(record.index_original))


def positioning_pictures(
    ws: Workspace,
    state: StackState,
    records: Sequence[SliceState],
    positions_mm: Sequence[float],
    *,
    look_of: Callable[[SliceState], Look],
    atlas_options: DisplayOptions,
    angles: Angles,
    tile_edge: int,
    max_width: int = POSITIONING_MAX_WIDTH,
    per_picture: int | None = None,
    part: int | None = None,
    window: Sequence[float] = (),
    zoom_edge: int | None = None,
    shown: str = "",
) -> list[Positioned]:
    """The positioning pictures of *records* and the atlas at
    *positions_mm*, captioned: every picture of the call, or with *part*
    only that one (0-based; :class:`IndexError` past the last), at most
    *per_picture* (None: :data:`PER_PICTURE`) sections in a picture.

    Sections are drawn in *look_of*'s look (*shown* words it for the
    caption), the atlas in *atlas_options*' layers at *angles*. With
    *window* (fractions of the unzoomed picture; *part* None: the first),
    the box is redrawn at the magnification that brings its long side to
    *zoom_edge* (None: the unzoomed picture's long side). Past the detail of
    the sections in it (their working copies; the atlas: its voxels) the
    thumbnails are enlarged, at most :data:`ATLAS_UPSAMPLE` times, and the
    caption says so.
    """
    ordered = sorted(records, key=lambda record: record.index_original)
    by_id = {record.id: record for record in ordered}
    entries: dict[str, SectionEntry] = {}
    working: dict[str, float] = {}
    finest: dict[str, float] = {}
    for record in ordered:
        extent, working[record.id], finest[record.id] = section_extent(ws, state, record)
        entries[record.id] = SectionEntry(
            id=record.id, labels=label_candidates(record),
            position_mm=None if record.position_mm is None else float(record.position_mm),
            extent_um=extent)
    atlas = [(float(mm), atlas_extent(ws, state, float(mm), angles)) for mm in positions_mm]
    layouts = plan(list(entries.values()), atlas, range_mm=ws.position_range,
                   tile_edge=tile_edge, max_width=max_width,
                   finest_um=max(finest.values(), default=0.0),
                   per_picture=per_picture or PER_PICTURE)
    frame = tuple(float(v) for v in window) if window else ()
    if part is not None:
        if not 0 <= int(part) < len(layouts):
            raise IndexError(f"part {part} of a call with {len(layouts)} pictures")
        chosen = [layouts[int(part)]]
    else:
        chosen = layouts[:1] if frame else layouts

    def draw_section(section_id: str, um_per_px: float) -> Image.Image:
        record = by_id[section_id]
        # Rendered at least as large as it is shown (render_slice stops at
        # the working copy), then brought to the exact scale.
        edge = max(64, math.ceil(max(entries[section_id].extent_um) / um_per_px) + 2)
        return section_at(ws, record, um_per_px, working_um=working[section_id],
                          long_edge=edge, look=look_of(record))

    def draw_atlas(position_mm: float, um_per_px: float) -> Image.Image:
        return framed_atlas(ws, state, position_mm, atlas_options, um_per_px=um_per_px,
                            angles=angles)

    out: list[Positioned] = []
    for layout in chosen:
        scale = enlarged = 1.0
        if frame:
            box_w = (frame[2] - frame[0]) * layout.width
            box_h = (frame[3] - frame[1]) * layout.height
            target = float(zoom_edge or max(layout.width, layout.height))
            wanted = target / max(box_w, box_h, 1.0)
            seen, _atlas_seen = visible(layout, frame)
            detail = (layout.um_per_px / max(finest[slot.key] for slot in seen) if seen
                      else layout.um_per_px / atlas_um_per_px(ws.atlas))
            scale = max(1.0, min(wanted, max(detail, 1.0) * ATLAS_UPSAMPLE))
            enlarged = scale / max(detail, 1.0)
        picture = paint(layout, draw_section=draw_section, draw_atlas=draw_atlas, scale=scale,
                        window=frame)
        text = positioning_caption(ws, layout, angles=angles, shown=shown,
                                   atlas_name=atlas_options.atlas_name(),
                                   um_per_px=layout.um_per_px / scale,
                                   zoom=scale if frame else None, enlarged=enlarged)
        out.append(Positioned(image=caption(picture, text), caption=text, layout=layout,
                              scale=scale, window=frame, um_per_px=layout.um_per_px / scale,
                              content=picture.size))
    return out


def positioning_caption(
    ws: Workspace, layout: Layout, *, angles: Angles, shown: str, atlas_name: str,
    um_per_px: float, zoom: float | None = None, enlarged: float = 1.0,
) -> str:
    """The caption burned under a positioning picture; *enlarged*: how far
    a zoom shows its thumbnails past their source pixels."""
    low, high = ws.axis_ends
    parts: list[str] = []
    head = "positioning"
    if layout.parts > 1:
        head += f" part {layout.part + 1} of {layout.parts}"
    if zoom:
        head += f" zoomed x{zoom:.1f}"
        if enlarged > 1.05:
            head += f" (source pixels enlarged x{enlarged:.1f})"
    count = len(layout.sections)
    if count:
        if layout.parts > 1:
            which = (f"section {layout.first} of {layout.total}" if count == 1 else
                     f"sections {layout.first}-{layout.last} of {layout.total}")
        else:
            which = f"{count} section{'s' if count != 1 else ''}"
        placed = [slot.position_mm for slot in layout.sections if slot.position_mm is not None]
        span = ""
        if placed:
            span = (f" at {placed[0]:.2f} mm" if min(placed) == max(placed) else
                    f", {min(placed):.2f}-{max(placed):.2f} mm")
        unplaced = count - len(placed)
        if unplaced:
            span += f", {unplaced} without a position at the right"
        parts.append(f"{head}: {which} below in position order{span}"
                     f"{f', {shown}' if shown else ''}")
    else:
        parts.append(f"{head}: no sections")
    if layout.atlas:
        name = "" if atlas_name == "template" else f" ({atlas_name})"
        parts.append(f"atlas{name} above at {len(layout.atlas)} position"
                     f"{'s' if len(layout.atlas) != 1 else ''}")
    lo, hi = layout.range_mm
    parts.append(f"ruler {lo:.1f}-{hi:.1f} mm from {low} (left) to {high}{angles_label(angles)}")
    parts.append(f"all at {um_per_px:.1f} um/px")
    return "; ".join(parts)


__all__ = [
    "ATLAS_UPSAMPLE", "Layout", "PER_PICTURE", "POSITIONING_MAX_WIDTH", "Positioned",
    "SectionEntry", "Slot", "paint", "plan", "position_order", "positioning_caption",
    "positioning_pictures", "ruler_range", "split", "visible",
]
