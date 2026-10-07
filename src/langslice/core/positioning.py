"""The positioning picture: the stack laid out along its slicing axis, as in ABBA.

A horizontal millimetre ruler spans the atlas range (``Workspace.position_range``).
Above it, atlas thumbnails at the requested positions, in position order,
each joined by a short line to its millimetre on the ruler. Below it, the
requested sections in stack order (``SliceState.index_original``), evenly
spaced, each joined by a line to its current position. Two lines cross
exactly when the stack order and the positions of their sections disagree
(:func:`crossing_pairs`); those lines and their labels are drawn in
:data:`CROSSING_COLOR`. The row of sections runs left to right in stack
order, or right to left when most pairs of the stack run against the ruler
(a stack numbered from posterior to anterior), so only disagreeing pairs
cross (:func:`stack_direction`).

Every thumbnail is drawn at ONE micrometres per pixel, sections and atlas
alike (:func:`langslice.core.scale.pair_um_per_px`): the largest fills a
tile, and the tile grows as the item count falls (:func:`plan`), from a
thumbnail per section of a long stack to tiles as large as the picture size
allows for one section beside two or three atlas positions.

:func:`plan` is the pure geometry (sizes in micrometres in, a
:class:`Layout` out); :func:`paint` draws a layout at a magnification and
crop, which is how :mod:`langslice.core.zoom` redraws a box of it at more
detail (the furniture scales with the tiles, so a box of the picture is the
same box at any magnification); :func:`positioning_picture` does both for a
stack.
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
from langslice.core.opening import CLAUDE_MAX_IMAGE_EDGE, tile_label
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

#: Widest positioning picture before a long stack shrinks its tiles: the
#: largest image every model lane takes in without shrinking it.
POSITIONING_MAX_WIDTH = CLAUDE_MAX_IMAGE_EDGE
#: Narrowest picture: the ruler stays readable with one or two tiles.
MIN_WIDTH = 720
#: Smallest tile: past this a long stack makes the picture wider instead.
MIN_TILE = 28
#: Space at the left and right ends, and between two tiles of a row.
MARGIN = 28
GAP = 10
PAD = 14
#: Height of the band between the atlas row and the ruler.
ATLAS_LINK_PX = 30
#: Height of the band between the ruler and the section row: the share of
#: the picture's width, between these bounds (steeper lines read better).
SECTION_LINK_SHARE = 0.075
SECTION_LINK_PX = (56, 120)
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
#: Lines that cross, and their sections' labels: order and position disagree.
CROSSING_COLOR = (255, 80, 80)
CROSSING_TEXT_COLOR = (255, 110, 110)


# --- order and crossings ----------------------------------------------------------


def stack_direction(positions: Sequence[float | None]) -> int:
    """+1 when most pairs of sections (in stack order) run up the ruler, -1
    when most run down it; +1 on a tie. Pairs at one position, or with a
    section without one, do not count."""
    up = down = 0
    for i, first in enumerate(positions):
        for second in positions[i + 1:]:
            if first is None or second is None or second == first:
                continue
            if second > first:
                up += 1
            else:
                down += 1
    return -1 if down > up else 1


def crossing_pairs(
    positions: Sequence[float | None], direction: int | None = None,
) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)``, ``i < j`` in stack order, whose positions run
    against *direction* (None: :func:`stack_direction`): exactly the lines
    of the positioning picture that cross."""
    sign = direction if direction is not None else stack_direction(positions)
    out: list[tuple[int, int]] = []
    for i, first in enumerate(positions):
        for j in range(i + 1, len(positions)):
            second = positions[j]
            if first is not None and second is not None and sign * (second - first) < 0:
                out.append((i, j))
    return out


# --- the geometry ------------------------------------------------------------------


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
    crossing: bool = False

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
    #: Micrometres per pixel of every thumbnail.
    um_per_px: float
    tile: int
    font_px: int
    small_px: int
    ruler: tuple[float, float, float]
    range_mm: tuple[float, float]
    atlas: tuple[Slot, ...]
    sections: tuple[Slot, ...]
    #: +1: the section row runs left to right in stack order; -1: right to left.
    direction: int
    #: ``(id, id)`` per crossing pair, in stack order.
    crossings: tuple[tuple[str, str], ...]
    label_step_mm: float
    tick_step_mm: float
    #: Top of the atlas labels' line and of the section labels' first line:
    #: one line per row, whatever each thumbnail's height.
    atlas_label_y: float = 0.0
    section_label_y: float = 0.0

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
    """``(label step, tick step)`` in mm: labels at least four digits apart,
    ticks halfway between labels when there is room."""
    label = next((step for step in _LABEL_STEPS if step * px_per_mm >= 5 * small_px),
                 _LABEL_STEPS[-1])
    tick = label / 2.0 if label / 2.0 * px_per_mm >= 10 else label
    return label, tick


def plan(
    sections: Sequence[SectionEntry],
    atlas: Sequence[tuple[float, tuple[float, float]]],
    *,
    range_mm: tuple[float, float],
    tile_edge: int,
    max_width: int = POSITIONING_MAX_WIDTH,
    finest_um: float = 0.0,
) -> Layout:
    """The layout of *sections* (in stack order) and *atlas* (``(mm,
    (width, height) um)`` per thumbnail, any order: drawn in position order).

    The tile is *tile_edge* while every row fits *max_width*, smaller for a
    longer row (never below :data:`MIN_TILE`); one micrometres per pixel puts
    the largest item's long edge at the tile, never finer than *finest_um*
    (a section's own pixels).
    """
    count = max(len(sections), len(atlas), 1)
    room = max_width - 2 * MARGIN - (count - 1) * GAP
    tile = int(max(MIN_TILE, min(int(tile_edge), room // count)))
    extents = [max(entry.extent_um) for entry in sections] + [max(e) for _mm, e in atlas]
    um_per_px = pair_um_per_px(extents or [tile], tile, finest=finest_um)
    inner = max(count * tile + (count - 1) * GAP, MIN_WIDTH - 2 * MARGIN)
    width = inner + 2 * MARGIN
    font_px = 14 if tile >= 110 else 12 if tile >= 56 else 11
    small_px = 12 if tile >= 56 else 11
    label_h = font_px + 5

    def centres(n: int) -> list[float]:
        pitch = inner / max(n, 1)
        return [MARGIN + pitch * (k + 0.5) for k in range(n)]

    def pitch_of(n: int) -> float:
        return inner / max(n, 1) - 4

    y = float(PAD)
    atlas_label_y = y
    atlas_slots: list[Slot] = []
    if atlas:
        ordered = sorted(atlas, key=lambda item: item[0])
        sizes = [(e[0] / um_per_px, e[1] / um_per_px) for _mm, e in ordered]
        tallest = max(h for _w, h in sizes)
        bottom = y + label_h + tallest
        for (mm, _e), (w, h), x in zip(ordered, sizes, centres(len(ordered)), strict=True):
            text = _first_fitting((f"atlas {mm:.2f} mm", f"{mm:.2f} mm", f"{mm:.1f}"),
                                  font_px, pitch_of(len(ordered)))
            atlas_slots.append(Slot(key=f"{mm:g}", centre_x=x, top=bottom - h, width=w,
                                    height=h, position_mm=float(mm), label=(text,)))
        y = bottom + ATLAS_LINK_PX
    else:
        y += 6
    ruler_y = y + 4
    lo, hi = (float(range_mm[0]), float(range_mm[1]))
    label_step, tick_step = _steps((width - 2 * MARGIN) / max(hi - lo, 1e-9), small_px)
    link = min(max(round(width * SECTION_LINK_SHARE), SECTION_LINK_PX[0]), SECTION_LINK_PX[1])
    top = ruler_y + 8 + small_px + 4 + link

    positions = [entry.position_mm for entry in sections]
    direction = stack_direction(positions)
    pairs = crossing_pairs(positions, direction)
    crossing = {index for pair in pairs for index in pair}
    xs = centres(len(sections))
    if direction < 0:
        xs = xs[::-1]
    section_slots: list[Slot] = []
    room_each = pitch_of(len(sections))
    for index, (entry, x) in enumerate(zip(sections, xs, strict=True)):
        w, h = entry.extent_um[0] / um_per_px, entry.extent_um[1] / um_per_px
        where = ((f"{entry.position_mm:.2f} mm", f"{entry.position_mm:.2f}")
                 if entry.position_mm is not None else ("no position", "none"))
        label = (_first_fitting(entry.labels, font_px, room_each),
                 _first_fitting(where, font_px, room_each))
        section_slots.append(Slot(key=entry.id, centre_x=x, top=top, width=w, height=h,
                                  position_mm=entry.position_mm, label=label,
                                  crossing=index in crossing))
    tallest = max((slot.height for slot in section_slots), default=0.0)
    section_label_y = top + tallest + 4
    height = section_label_y + (2 * (font_px + 4) if section_slots else 0) + PAD
    return Layout(
        width=int(width), height=int(math.ceil(height)), um_per_px=float(um_per_px),
        tile=tile, font_px=font_px, small_px=small_px,
        ruler=(float(MARGIN), float(width - MARGIN), float(ruler_y)), range_mm=(lo, hi),
        atlas=tuple(atlas_slots), sections=tuple(section_slots), direction=direction,
        crossings=tuple((sections[i].id, sections[j].id) for i, j in pairs),
        label_step_mm=label_step, tick_step_mm=tick_step,
        atlas_label_y=atlas_label_y, section_label_y=section_label_y,
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
    # Crossing lines last, so they lie on top; a long stack's small tiles
    # get thin lines and small dots, so the ruler stays readable.
    small_tiles = layout.tile < 80
    bold = thin if small_tiles else max(thin, round(1.5 * s))
    radius = 2.2 if small_tiles else 3.2
    for slot in sorted(layout.sections, key=lambda item: item.crossing):
        if slot.position_mm is None:
            continue
        color = CROSSING_COLOR if slot.crossing else SECTION_LINE_COLOR
        dot = CROSSING_COLOR if slot.crossing else SECTION_DOT_COLOR
        x = X(layout.x_at(slot.position_mm))
        cv2.line(canvas, _pt(X(slot.centre_x), Y(slot.top) - 3 * s), _pt(x, Y(ry)), color,
                 bold if slot.crossing else thin, cv2.LINE_AA, _SHIFT)
        cv2.circle(canvas, _pt(x, Y(ry)), round(radius * s * (1 << _SHIFT)), dot, -1,
                   cv2.LINE_AA, _SHIFT)
    out = Image.fromarray(canvas, mode="RGB")

    draw = ImageDraw.Draw(out)
    small = _font(max(6, round(layout.small_px * s)))
    font = _font(max(6, round(layout.font_px * s)))
    majors = [mm for mm, major in ticks if major]
    for mm in majors:
        number = f"{mm:g}"
        # The number is centred on its tick; the last one is followed by the unit.
        text = number + (" mm" if mm == majors[-1] else "")
        left = X(layout.x_at(mm)) - float(small.getlength(number)) / 2.0
        draw.text((left, Y(ry) + 8 * s), text, fill=TICK_TEXT_COLOR, font=small,
                  stroke_width=max(1, round(2 * s)), stroke_fill=BACKGROUND)
    for slot in layout.atlas:
        _centred(draw, slot.label[0], X(slot.centre_x), Y(layout.atlas_label_y), font,
                 TEXT_COLOR)
    line_h = (layout.font_px + 4) * s
    for slot in layout.sections:
        color = CROSSING_TEXT_COLOR if slot.crossing else TEXT_COLOR
        for row, text in enumerate(slot.label):
            _centred(draw, text, X(slot.centre_x), Y(layout.section_label_y) + row * line_h,
                     font, color)
    return out


def _centred(draw: ImageDraw.ImageDraw, text: str, x: float, y: float, font: Any,
             color: tuple[int, int, int]) -> None:
    draw.text((x - float(font.getlength(text)) / 2.0, y), text, fill=color, font=font)


# --- a stack's picture ----------------------------------------------------------------


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
    """A section's first-line labels, longest first: :func:`tile_label`,
    then without the file extension, then the index alone."""
    full = tile_label(record)
    stem = record.id.rsplit(".", 1)[0] if "." in record.id else record.id
    return (full, f"{record.index_corrected}: {stem}", str(record.index_corrected))


def positioning_picture(
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
    window: Sequence[float] = (),
    zoom_edge: int | None = None,
    shown: str = "",
) -> Positioned:
    """The positioning picture of *records* (drawn in stack order,
    ``index_original``) and the atlas at *positions_mm*, captioned.

    Sections are drawn in *look_of*'s look (*shown* words it for the
    caption), the atlas in *atlas_options*' layers at *angles*. With
    *window* (fractions of the unzoomed picture), the box is redrawn at the
    magnification that brings its long side to *zoom_edge* (None: the
    unzoomed picture's long side). Past the detail of the sections in it
    (their working copies; the atlas: its voxels) the thumbnails are
    enlarged, at most :data:`ATLAS_UPSAMPLE` times, and the caption says so.
    """
    ordered = sorted(records, key=lambda record: record.index_original)
    by_id = {record.id: record for record in ordered}
    entries: list[SectionEntry] = []
    working: dict[str, float] = {}
    finest: dict[str, float] = {}
    for record in ordered:
        extent, working[record.id], finest[record.id] = section_extent(ws, state, record)
        entries.append(SectionEntry(
            id=record.id, labels=label_candidates(record),
            position_mm=None if record.position_mm is None else float(record.position_mm),
            extent_um=extent))
    atlas = [(float(mm), atlas_extent(ws, state, float(mm), angles)) for mm in positions_mm]
    layout = plan(entries, atlas, range_mm=ws.position_range, tile_edge=tile_edge,
                  max_width=max_width, finest_um=max(finest.values(), default=0.0))

    scale = 1.0
    enlarged = 1.0
    frame = tuple(float(v) for v in window) if window else ()
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

    def draw_section(section_id: str, um_per_px: float) -> Image.Image:
        record = by_id[section_id]
        look = look_of(record)
        # Rendered at least as large as it is shown (render_slice stops at
        # the working copy), then brought to the exact scale.
        extent = max(next(e.extent_um for e in entries if e.id == section_id))
        edge = max(64, math.ceil(extent / um_per_px) + 2)
        return section_at(ws, record, um_per_px, working_um=working[section_id],
                          long_edge=edge, look=look)

    def draw_atlas(position_mm: float, um_per_px: float) -> Image.Image:
        return framed_atlas(ws, state, position_mm, atlas_options, um_per_px=um_per_px,
                            angles=angles)

    picture = paint(layout, draw_section=draw_section, draw_atlas=draw_atlas, scale=scale,
                    window=frame)
    text = positioning_caption(ws, layout, angles=angles, shown=shown,
                               atlas_name=atlas_options.atlas_name(),
                               um_per_px=layout.um_per_px / scale, zoom=scale if frame else None,
                               enlarged=enlarged)
    return Positioned(image=caption(picture, text), caption=text, layout=layout, scale=scale,
                      window=frame, um_per_px=layout.um_per_px / scale, content=picture.size)


def positioning_caption(
    ws: Workspace, layout: Layout, *, angles: Angles, shown: str, atlas_name: str,
    um_per_px: float, zoom: float | None = None, enlarged: float = 1.0,
) -> str:
    """The caption burned under a positioning picture; *enlarged*: how far
    a zoom shows its thumbnails past their source pixels."""
    low, high = ws.axis_ends
    parts: list[str] = []
    head = "positioning" + (f" zoomed x{zoom:.1f}" if zoom else "")
    if zoom and enlarged > 1.05:
        head += f" (source pixels enlarged x{enlarged:.1f})"
    if layout.sections:
        order = "left to right" if layout.direction > 0 else "right to left"
        parts.append(f"{head}: {len(layout.sections)} section"
                     f"{'s' if len(layout.sections) != 1 else ''} below in stack order "
                     f"({order}){f', {shown}' if shown else ''}")
    else:
        parts.append(head)
    if layout.atlas:
        name = "" if atlas_name == "template" else f" ({atlas_name})"
        parts.append(f"atlas{name} above at {len(layout.atlas)} position"
                     f"{'s' if len(layout.atlas) != 1 else ''}")
    parts.append(f"ruler mm from {low} (left) to {high}{angles_label(angles)}")
    parts.append(f"all at {um_per_px:.1f} um/px")
    if layout.crossings:
        count = len(layout.crossings)
        parts.append(f"red: {count} pair{'s' if count != 1 else ''} whose order and "
                     "positions disagree")
    return "; ".join(parts)


__all__ = [
    "ATLAS_UPSAMPLE", "CROSSING_COLOR", "Layout", "POSITIONING_MAX_WIDTH", "Positioned",
    "SectionEntry", "Slot", "crossing_pairs", "paint", "plan", "positioning_caption",
    "positioning_picture", "stack_direction", "visible",
]
