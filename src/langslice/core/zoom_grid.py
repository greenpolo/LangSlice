"""The coordinate grid every zoom carries.

Lines at round steps are labelled with the pixel numbers of the WHOLE
(unzoomed) picture the zoom was cut from, so a position read off the grid is
a coordinate that ``zoom`` boxes and ``set_landmarks`` endpoints take on the
zoom or on its source alike. Thin dashed lines, one colour no other drawing
uses (atlas borders are black or white cased in the other, named in the
caption; landmark markers are cyan with white and black rings, folds
magenta, a stale marker orange),
and labels on a dark chip at every edge. Drawn on the content only, never on
the caption band.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from langslice.core.captions import _font
from langslice.core.picture_geometry import SourceFrame, grid_step

#: Grid line and label colour (no other drawing uses a lime green).
GRID_COLOUR = "#7dff3a"
CHIP_COLOUR = "#101010"
DASH = 4


def _dashed(draw: ImageDraw.ImageDraw, start: tuple[float, float], stop: tuple[float, float],
            vertical: bool) -> None:
    length = int(stop[1] - start[1]) if vertical else int(stop[0] - start[0])
    for offset in range(0, max(length, 0), 2 * DASH):
        a = offset + 1
        b = min(offset + DASH, length)
        if vertical:
            draw.line([(start[0], start[1] + a), (start[0], start[1] + b)], fill=GRID_COLOUR)
        else:
            draw.line([(start[0] + a, start[1]), (start[0] + b, start[1])], fill=GRID_COLOUR)


def _label(draw: ImageDraw.ImageDraw, font, text: str, x: float, y: float,
           width: int, height: int) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 4, bottom - top + 4
    x = min(max(x, 0), width - w)
    y = min(max(y, 0), height - h)
    draw.rectangle((x, y, x + w - 1, y + h - 1), fill=CHIP_COLOUR)
    draw.text((x + 2 - left, y + 2 - top), text, fill=GRID_COLOUR, font=font)


def draw_grid(image: Image.Image, frame: SourceFrame) -> dict[str, int]:
    """Draw the grid on *image* in place, over its content (``frame.shown``
    pixels from the top left); returns ``{"step": ...}`` in source pixels."""
    width, height = (int(v) for v in frame.shown)
    low, high = frame.to_source([-0.5, -0.5]), frame.to_source([width - 0.5, height - 0.5])
    step = grid_step(float(max(high - low)))
    draw = ImageDraw.Draw(image)
    font = _font(max(11, min(16, round(min(width, height) / 45))))
    xs = [step * k for k in range(math.ceil(low[0] / step), math.floor(high[0] / step) + 1)]
    ys = [step * k for k in range(math.ceil(low[1] / step), math.floor(high[1] / step) + 1)]
    for g in xs:
        px = float(frame.to_picture([g, low[1]])[0])
        _dashed(draw, (px, 0), (px, height), vertical=True)
    for g in ys:
        py = float(frame.to_picture([low[0], g])[1])
        _dashed(draw, (0, py), (width, py), vertical=False)
    for g in xs:
        px = float(frame.to_picture([g, low[1]])[0])
        _label(draw, font, f"{g}", px + 2, 0, width, height)
        _label(draw, font, f"{g}", px + 2, height, width, height)
    for g in ys:
        py = float(frame.to_picture([low[0], g])[1])
        _label(draw, font, f"{g}", 0, py + 2, width, height)
        _label(draw, font, f"{g}", width, py + 2, width, height)
    return {"step": step}
