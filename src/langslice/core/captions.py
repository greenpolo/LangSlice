"""Captions, fonts and the scale bar burned into the pictures the tools send.

Tool images reach a model as bare attachments, so the text that binds a
picture to its section rides in its pixels (:func:`caption`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#: How an atlas caption names a plane's cutting angles (:func:`angles_label`).
ANGLES_LABEL = " pitch {:.1f} yaw {:.1f}"


def angles_label(angles: tuple[float, float], template: str = ANGLES_LABEL) -> str:
    """``(pitch, yaw)`` as a caption shows them; empty for the flat plane."""
    pitch, yaw = angles
    return template.format(pitch, yaw) if (pitch or yaw) else ""


#: How a caption names the plane an atlas picture without a section is drawn
#: at when the sections' cutting angles differ (``StackState.view_angles``).
MEDIAN_ANGLES_LABEL = " at the median of the sections' cutting angles, pitch {:.1f} yaw {:.1f}"


def view_angles_label(angles: tuple[float, float], *, median: bool) -> str:
    """The angles of an atlas picture without a section: *median* (the
    sections' angles differ, :meth:`langslice.core.state.StackState.drawn_at_median`)
    says they are the median, even for the flat plane; else
    :func:`angles_label`."""
    if median:
        return MEDIAN_ANGLES_LABEL.format(*angles)
    return angles_label(angles)


#: Font size of the label strip :func:`caption` burns into an image.
CAPTION_PX = 14


@lru_cache(maxsize=1)
def _caption_font() -> Any:
    try:
        return ImageFont.load_default(size=CAPTION_PX)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def caption(image: Image.Image, text: str) -> Image.Image:
    """A COPY of *image* with *text* burned into a dark strip below it.

    Tool images reach the model as bare attachments, so the text that binds an
    image to its section or its position has to ride in the pixels. Caption
    only what is SHOWN: an image a fit measures must never be captioned, since
    the strip changes the pixels the fit reads.

    Text wider than the picture wraps (:func:`wrap_caption`) instead of
    running off its right edge, so a small picture keeps its whole label.
    """
    source = image.convert("RGB")
    font = _caption_font()
    text = wrap_caption(text, font, source.width - 6)
    probe = ImageDraw.Draw(source)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    band = int(bottom - top + 6)
    # The band sits BELOW the picture, never over it (a caption drawn on the
    # pixels covered exactly the magnified dorsal tissue an agent was
    # reading), and never above it: the picture's pixel (x, y) is then the
    # content's own, the coordinates a zoom box is given in.
    labelled = Image.new("RGB", (source.width, source.height + band), (0, 0, 0))
    labelled.paste(source, (0, 0))
    draw = ImageDraw.Draw(labelled)
    draw.text((3 - left, source.height + 3 - top), text, fill=(255, 255, 255), font=font)
    return labelled


def wrap_caption(text: str, font: Any, width: int) -> str:
    """*text* with every line broken to fit *width* pixels in *font*.

    Lines break at spaces; a single word wider than *width* breaks between
    characters. A line that already fits is left exactly as it was, so a
    caption that fit before draws the same pixels.
    """
    width = max(int(width), 24)

    def fits(piece: str) -> bool:
        return float(font.getlength(piece)) <= width

    out: list[str] = []
    for raw in text.split("\n"):
        if fits(raw):
            out.append(raw)
            continue
        line: str | None = None
        for word in raw.split(" "):
            candidate = word if line is None else f"{line} {word}"
            if fits(candidate):
                line = candidate
                continue
            if line is not None and line.strip():
                out.append(line.rstrip())
                line = word or None
            else:
                line = candidate.lstrip() or None
            while line is not None and not fits(line):
                cut = max(1, max((k for k in range(1, len(line) + 1) if fits(line[:k])),
                                 default=1))
                out.append(line[:cut])
                line = line[cut:] or None
        if line is not None and line.strip():
            out.append(line.rstrip())
    return "\n".join(out)


def _font(px: int) -> Any:
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def scale_bar_px(um_per_px: float, mm: float = 1.0) -> int:
    """Length in pixels of a *mm*-millimetre bar on a *um_per_px* canvas."""
    return int(round(mm * 1000.0 / float(um_per_px)))


def _draw_scale_bar(canvas: np.ndarray, um_per_px: float, dark: bool) -> None:
    """A 1 mm bar in the bottom-left corner, labelled. Thin, like a viewer's."""
    height, width = canvas.shape[:2]
    length = scale_bar_px(um_per_px)
    if length < 4 or length > width:
        return
    color = (235, 235, 235) if dark else (40, 40, 40)
    thickness = max(1, round(min(height, width) / 400))
    x0, y0 = 12, height - 14
    cv2.rectangle(canvas, (x0, y0), (x0 + length, y0 + thickness), color, -1)
    cv2.putText(
        canvas, "1 mm", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
    )

