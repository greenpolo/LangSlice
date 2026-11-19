"""Captions, fonts and the scale bar burned into the pictures the tools send.

Tool images reach a model as bare attachments, so the text that binds a
picture to its section rides in its pixels (:func:`caption`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANGLES_LABEL = " pitch {:.1f} yaw {:.1f}"


def angles_label(angles: tuple[float, float], template: str = ANGLES_LABEL) -> str:
    """``(pitch, yaw)`` as a caption shows them; empty for the flat plane."""
    pitch, yaw = angles
    return template.format(pitch, yaw) if (pitch or yaw) else ""


MEDIAN_ANGLES_LABEL = " at the median of the sections' cutting angles, pitch {:.1f} yaw {:.1f}"


def view_angles_label(angles: tuple[float, float], *, median: bool) -> str:
    """The angles of an atlas picture without a section: *median* (the
    sections' angles differ,
    :meth:`langslice.core.stack.stack.Stack.drawn_at_median`)
    says they are the median, even for the flat plane; else
    :func:`angles_label`."""
    if median:
        return MEDIAN_ANGLES_LABEL.format(*angles)
    return angles_label(angles)


CAPTION_PX = 14


@cache
def font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """PIL's default font at *px* pixels, kept: the one loader of every
    picture's text."""
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def caption_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The font of the caption band, :data:`CAPTION_PX` pixels."""
    return font(CAPTION_PX)


def caption(image: Image.Image, text: str, *, min_width: int = 0) -> Image.Image:
    """A COPY of *image* with *text* burned into a dark strip below it.

    Tool images reach the model as bare attachments, so the text that binds an
    image to its section or its position has to ride in the pixels. Caption
    only what is SHOWN: an image a fit measures must never be captioned, since
    the strip changes the pixels the fit reads.

    Text wider than the picture wraps (:func:`wrap_caption`) instead of
    running off its right edge, so a small picture keeps its whole label.
    *min_width* lays the strip out at least that wide: a narrow picture
    (a zoom on a tall region) then sits at the top left of a dark band of
    that width, its pixels still the content's own.
    """
    source = image.convert("RGB")
    width = max(source.width, int(min_width))
    face = caption_font()
    text = wrap_caption(text, face, width - 6)
    probe = ImageDraw.Draw(source)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=face)
    band = int(bottom - top + 6)
    labelled = Image.new("RGB", (width, source.height + band), (0, 0, 0))
    labelled.paste(source, (0, 0))
    draw = ImageDraw.Draw(labelled)
    draw.text((3 - left, source.height + 3 - top), text, fill=(255, 255, 255), font=face)
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
                cut = max(1, max((k for k in range(1, len(line) + 1) if fits(line[:k])), default=1))
                out.append(line[:cut])
                line = line[cut:] or None
        if line is not None and line.strip():
            out.append(line.rstrip())
    return "\n".join(out)


def scale_bar_px(um_per_px: float, mm: float = 1.0) -> int:
    """Length in pixels of a *mm*-millimetre bar on a *um_per_px* canvas."""
    return int(round(mm * 1000.0 / float(um_per_px)))


@dataclass(frozen=True)
class ScaleBar:
    """Where a picture's scale bar is drawn, and what it measures.

    *bar* is the filled rectangle ``(x0, y0, x1, y1)`` in picture pixels (the
    ends exclusive), *label* the box of the ``1 mm`` text above it, *length_um*
    the micrometres the bar spans, *color* its ink and *thickness* its height
    in pixels. :func:`scale_bar` makes it for a picture of a given shape and
    :func:`draw_scale_bar` draws exactly that, so a caller that composes its
    own picture reads the same place.
    """

    bar: tuple[int, int, int, int]
    label: tuple[int, int, int, int]
    length_um: float
    color: tuple[int, int, int]
    thickness: int


@dataclass(frozen=True)
class BarStyle:
    """How the scale bar sits and looks: *left_px* from the left edge and
    *bottom_px* up from the bottom edge to the bar's top, *label_gap_px*
    between the bar and its *text* (drawn at *font_scale* in OpenCV's Hershey
    font), no shorter than *min_px*, in *dark_ink* on a dark picture and
    *light_ink* on a pale one."""

    left_px: int = 12
    bottom_px: int = 14
    label_gap_px: int = 6
    min_px: int = 4
    text: str = "1 mm"
    font_scale: float = 0.4
    dark_ink: tuple[int, int, int] = (235, 235, 235)
    light_ink: tuple[int, int, int] = (40, 40, 40)


BAR_STYLE = BarStyle()


def scale_bar(shape: tuple[int, ...], um_per_px: float, dark: bool) -> ScaleBar | None:
    """The 1 mm bar on a picture of *shape* (rows, columns,
    ...) at *um_per_px*, in the bottom-left corner; None when it would be
    shorter than a few pixels or wider than the picture. *dark*: the picture
    is dark, the ink pale."""
    height, width = int(shape[0]), int(shape[1])
    length = scale_bar_px(um_per_px)
    if length < BAR_STYLE.min_px or length > width:
        return None
    thickness = max(1, round(min(height, width) / 400))
    x0, y0 = BAR_STYLE.left_px, height - BAR_STYLE.bottom_px
    (text_w, text_h), below = cv2.getTextSize(
        BAR_STYLE.text, cv2.FONT_HERSHEY_SIMPLEX, BAR_STYLE.font_scale, 1
    )
    baseline = y0 - BAR_STYLE.label_gap_px
    return ScaleBar(
        bar=(x0, y0, x0 + length + 1, y0 + thickness + 1),
        label=(x0, baseline - text_h, x0 + text_w, baseline + below + 1),
        length_um=1000.0,
        color=BAR_STYLE.dark_ink if dark else BAR_STYLE.light_ink,
        thickness=thickness,
    )


def draw_scale_bar(canvas: np.ndarray, bar: ScaleBar | None) -> None:
    """*bar* (:func:`scale_bar`) drawn on *canvas*, labelled. Thin, like a viewer's."""
    if bar is None:
        return
    x0, y0, x1, y1 = bar.bar
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), bar.color, -1)
    cv2.putText(
        canvas,
        BAR_STYLE.text,
        (x0, y0 - BAR_STYLE.label_gap_px),
        cv2.FONT_HERSHEY_SIMPLEX,
        BAR_STYLE.font_scale,
        bar.color,
        1,
        cv2.LINE_AA,
    )
