"""A sheet: several saved pictures tiled into one, each tile one picture.

``look`` with ``sheet`` (:func:`langslice.ops.look.look`) draws its pictures
as usual, saves each with its number, and tiles them into one picture of at
most a given long edge, for a viewer that pays per image: :func:`tile` lays
the pictures out in a grid, each scaled down whole (its own caption band
included), under a header band naming its number and what it shows, and
captions the sheet. A tile is one picture as it was drawn: nothing is drawn
over it. The sheet's recipe (:func:`recipe`) lists each tile's picture and its
box on the sheet, so a zoom on the sheet is a zoom on the picture under the
box (:func:`tile_under`, :func:`to_tile`): no detail is read from the sheet's
shrunken pixels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from langslice.core.captions import _caption_font, caption, wrap_caption

#: The recipe's renderer name (``zoom`` reads a sheet's tiles, never its pixels).
RENDERER = "sheet"
#: The most pictures one sheet tiles; a call with more makes several sheets.
MAX_TILES = 16
#: Pixels between tiles.
GAP_PX = 6
#: Height of a tile's header band (its picture number and what it shows).
HEADER_PX = 20
#: The sheet's background and its header bands' colour.
BACKGROUND = (24, 24, 24)


@dataclass(frozen=True)
class Tile:
    """One picture to tile: its image as saved, its number and a short label."""

    image: Image.Image
    number: int
    label: str


@dataclass(frozen=True)
class Sheet:
    """A tiled sheet: the captioned image and, per tile, its picture number,
    its box on the sheet (``[x0, y0, x1, y1]`` pixels of the picture's
    content, the header excluded) and the picture's own size."""

    image: Image.Image
    tiles: list[dict[str, Any]]
    scale: float


def grid(count: int, aspect: float) -> tuple[int, int]:
    """``(columns, rows)`` for *count* tiles of width/height *aspect* that
    keeps the sheet closest to square."""
    best: tuple[float, int, int] | None = None
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        ratio = (columns * aspect) / rows
        score = abs(math.log(max(ratio, 1e-6)))
        if best is None or score < best[0]:
            best = (score, columns, rows)
    assert best is not None
    return best[1], best[2]


def tile(tiles: Sequence[Tile], long_edge: int, *, title: str) -> Sheet:
    """*tiles* in one picture whose long edge is at most *long_edge* (the
    caption band below it included in the width only), captioned *title*."""
    if not tiles:
        raise ValueError("a sheet tiles at least one picture")
    cell_w = max(picture.image.width for picture in tiles)
    cell_h = max(picture.image.height for picture in tiles)
    columns, rows = grid(len(tiles), cell_w / max(cell_h + HEADER_PX, 1))
    scale = min(1.0, (long_edge - (columns + 1) * GAP_PX) / max(columns * cell_w, 1),
                (long_edge - (rows + 1) * GAP_PX - rows * HEADER_PX) / max(rows * cell_h, 1))
    scale = max(scale, 0.05)
    width = int(round(columns * cell_w * scale)) + (columns + 1) * GAP_PX
    height = int(round(rows * cell_h * scale)) + rows * HEADER_PX + (rows + 1) * GAP_PX
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = _caption_font()
    boxes: list[dict[str, Any]] = []
    for index, picture in enumerate(tiles):
        column, row = index % columns, index // columns
        x0 = GAP_PX + column * (int(round(cell_w * scale)) + GAP_PX)
        y0 = GAP_PX + row * (int(round(cell_h * scale)) + HEADER_PX + GAP_PX)
        size = (max(1, int(round(picture.image.width * scale))),
                max(1, int(round(picture.image.height * scale))))
        shrunk = picture.image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        header = wrap_caption(f"picture {picture.number}  {picture.label}", font,
                              int(round(cell_w * scale)) - 4).split("\n")[0]
        draw.text((x0 + 2, y0 + 3), header, fill=(255, 255, 255), font=font)
        top = y0 + HEADER_PX
        sheet.paste(shrunk, (x0, top))
        boxes.append({"picture": picture.number, "box": [x0, top, x0 + size[0], top + size[1]],
                      "size": [picture.image.width, picture.image.height]})
    return Sheet(image=caption(sheet, title), tiles=boxes, scale=scale)


def recipe(made: Sheet) -> dict[str, Any]:
    """The sheet's recipe: its tiles, so a zoom reads the picture under a box."""
    return {"renderer": RENDERER, "tiles": made.tiles, "scale": round(made.scale, 4),
            "shown": [made.image.width, made.image.height]}


def tile_under(recipe_: dict[str, Any], box: Sequence[float]) -> dict[str, Any] | None:
    """The tile of a sheet's recipe that holds the larger part of *box*
    (``[x0, y0, x1, y1]`` sheet pixels), or None when the box covers none."""
    best: tuple[float, dict[str, Any]] | None = None
    for held in recipe_.get("tiles") or []:
        x0, y0, x1, y1 = held["box"]
        w = min(box[2], x1) - max(box[0], x0)
        h = min(box[3], y1) - max(box[1], y0)
        if w <= 0 or h <= 0:
            continue
        if best is None or w * h > best[0]:
            best = (w * h, held)
    return None if best is None else best[1]


def to_tile(held: dict[str, Any], box: Sequence[float]) -> list[float]:
    """*box* on the sheet as a box in the pixels of the tile's own picture,
    clipped to it."""
    x0, y0, x1, y1 = held["box"]
    width, height = held["size"]
    sx = width / max(x1 - x0, 1)
    sy = height / max(y1 - y0, 1)
    out = [(max(box[0], x0) - x0) * sx, (max(box[1], y0) - y0) * sy,
           (min(box[2], x1) - x0) * sx, (min(box[3], y1) - y0) * sy]
    return [round(min(max(v, 0.0), limit), 1)
            for v, limit in zip(out, (width, height, width, height), strict=True)]


__all__ = ["MAX_TILES", "RENDERER", "Sheet", "Tile", "grid", "recipe", "tile", "tile_under",
           "to_tile"]
