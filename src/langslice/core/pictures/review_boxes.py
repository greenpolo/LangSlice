"""Review boxes drawn on a picture: the parts of a section a person marked.

:func:`draw_boxes` outlines each box, given as fractions of the picture's
content, and labels it with its comment's number, so the agent sees the
very part the person pointed at.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

__all__ = ["BOX_COLOUR", "Box", "draw_boxes"]

BOX_COLOUR = (255, 214, 10)
LABEL_COLOUR = (0, 0, 0)

Box = tuple[float, float, float, float]


def draw_boxes(
    image: Image.Image,
    content: tuple[int, int],
    boxes: Sequence[tuple[str, Box]],
) -> list[list[int]]:
    """Outline review boxes on a picture, in place.

    Parameters
    ----------
    image
        The picture, drawn on in place (its identity kept, so the note the
        core made while drawing it still finds it).
    content
        The content's width and height in pixels, at the top left (the
        caption band below is left alone).
    boxes
        ``(label, (x0, y0, x1, y1))`` per box, the corners as fractions of
        the content.

    Returns
    -------
    list of list of int
        Each box in the picture's pixels, ``[x0, y0, x1, y1]``.
    """
    width, height = content
    draw = ImageDraw.Draw(image)
    line = max(2, round(min(width, height) / 250))
    font = _font(max(12, min(22, round(min(width, height) / 35))))
    pixel_boxes: list[list[int]] = []
    for label, (x0, y0, x1, y1) in boxes:
        corners = [
            round(x0 * (width - 1)),
            round(y0 * (height - 1)),
            round(x1 * (width - 1)),
            round(y1 * (height - 1)),
        ]
        draw.rectangle(corners, outline=BOX_COLOUR, width=line)
        _label(draw, font, label, corners, content)
        pixel_boxes.append(corners)
    return pixel_boxes


def _label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text: str,
    corners: list[int],
    content: tuple[int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 8, bottom - top + 6
    x = min(max(corners[0], 0), content[0] - w)
    y = corners[1] - h
    if y < 0:
        y = corners[1]
    draw.rectangle((x, y, x + w - 1, y + h - 1), fill=BOX_COLOUR)
    draw.text((x + 4 - left, y + 3 - top), text, fill=LABEL_COLOUR, font=font)


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()
