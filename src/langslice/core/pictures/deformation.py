"""The pictures of a deformable fit and of a border trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.fits.deformable import (
    DeformableRecord,
    draw_warped_borders,
    resampled_record,
)
from langslice.core.fits.deformable.plane_images import (
    native_intensity,
    native_labels,
    normalize_intensity,
)
from langslice.core.fits.deformable.render import composed_native_grid
from langslice.core.sections.channels import channel_colors
from langslice.core.workspace import Workspace

# --- the picture ------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """How one call's pictures are drawn (its display options)."""

    #: Zoom in pixels of the unzoomed picture (``display.zoom_fractions``);
    #: empty is the whole section.
    zoom: tuple[float, ...]
    #: Regions drawn strong; empty draws every border strong.
    highlight: tuple[str, ...]
    #: Regions drawn in the second color (those excluded from the fit).
    marked: tuple[str, ...]
    outlines: str
    #: The image model's trace colour (:func:`trace_picture`: plain, read back by colour).
    color: tuple[int, int, int]
    thickness: float
    atlas_opacity: float
    #: Ontology depth of the regions whose boundaries are drawn.
    depth: int
    #: Long edge of the picture (of the zoomed crop when zoomed), in pixels;
    #: never more than the fit image's own pixels.
    long_edge: int = 512
    #: Core colour of the fitted borders in :func:`picture`, ``black`` or
    #: ``white`` (cased with the other).
    core: str = "black"


def picture(
    ctx: Workspace,
    image: Image.Image,
    record: DeformableRecord,
    *,
    warped: bool,
    style: Style,
    atlas_images: tuple[str, ...],
    title: str,
    note: dict[str, Any] | None = None,
) -> Image.Image:
    """The record's borders on the section at ``style.long_edge``, captioned.

    The crop (the zoom, or the whole fit image) is shown at ``style.long_edge``
    on its long side, or at its own pixels when it has fewer: the fit image
    (:data:`FIT_LONG_EDGE`) is never upsampled. *atlas_images* (the atlas's
    images among the call's ``view.atlas_channels``) are pulled through the
    record's map and blended under the lines at ``style.atlas_opacity``. With
    *note* (:func:`langslice.core.pictures.drawn.note` fields: sections, mode)
    the picture is noted with its frame
    (:class:`~langslice.core.pictures.drawn.WarpNote`), so the job saves its
    layers.
    """
    from langslice.core.pictures import drawn as pictures_drawn
    from langslice.core.pictures.canvas import zoom_box
    from langslice.core.pictures.captions import caption
    from langslice.core.pictures.options import zoom_fractions

    width, height = record.section_size
    zoom = list(zoom_fractions(style.zoom, unzoomed_size((width, height), style.long_edge)))
    box = zoom_box(zoom, (width, height)) if zoom else (0, 0, width, height)
    crop = (box[2] - box[0], box[3] - box[1])
    factor = min(1.0, int(style.long_edge) / float(max(crop)))
    full = (max(8, int(round(width * factor))), max(8, int(round(height * factor))))
    shown = (0, 0, full[0], full[1])
    if zoom:
        shown = zoom_box(list(zoom), full)
    small = resampled_record(record, full, shown)
    base = image.convert("RGB").resize(full, Image.Resampling.LANCZOS).crop(shown)
    kinds = tuple(kind for kind in atlas_images if kind != "borders")
    if style.atlas_opacity > 0 and kinds:
        base = _blend_atlas(ctx, base, small if warped else _unwarped(small), kinds,
                            style.atlas_opacity)
    native = native_labels(ctx.atlas, record.placement)
    drawn = draw_warped_borders(
        base, small, ctx.atlas, highlight=style.highlight, marked=style.marked, warped=warped,
        outlines=style.outlines, depth=style.depth, width_px=style.thickness,
        core=style.core, native=native,
    )
    captioned = caption(drawn, title)
    if note is not None:
        pictures_drawn.note(captioned, **note, warp=pictures_drawn.WarpNote(
            record=small, band=0, warped=warped,
            highlight=tuple(style.highlight), marked=tuple(style.marked),
            outlines=style.outlines, depth=style.depth, width_px=float(style.thickness),
            native=native))
    return captioned


def unzoomed_size(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """The content size :func:`picture` and :func:`trace_picture` draw a whole
    image of *size* at: *long_edge* on its long side, never upsampled."""
    factor = min(1.0, int(long_edge) / float(max(size)))
    return (max(8, int(round(size[0] * factor))), max(8, int(round(size[1] * factor))))


def trace_picture(
    image: Image.Image, lines: np.ndarray, *, style: Style, title: str,
) -> Image.Image:
    """The image model's traced lines on the section image, at ``style.long_edge``.

    *image* and *lines* share the fit grid (:func:`traced_lines`); the zoom,
    border color and thickness are the call's, and the picture is never drawn
    past the fit image's own pixels, like :func:`picture`.
    """
    from langslice.core.pictures.canvas import zoom_box
    from langslice.core.pictures.captions import caption
    from langslice.core.pictures.options import zoom_fractions

    width, height = image.size
    zoom = list(zoom_fractions(style.zoom, unzoomed_size((width, height), style.long_edge)))
    box = zoom_box(zoom, (width, height)) if zoom else (0, 0, width, height)
    crop = (box[2] - box[0], box[3] - box[1])
    factor = min(1.0, int(style.long_edge) / float(max(crop)))
    shown = (max(8, int(round(crop[0] * factor))), max(8, int(round(crop[1] * factor))))
    base = image.convert("RGB").crop(box).resize(shown, Image.Resampling.LANCZOS)
    mask = np.asarray(lines, dtype=np.float32)[box[1]:box[3], box[0]:box[2]]
    # Area-averaged so a one-pixel line survives the shrink.
    drawn = cv2.resize(mask, shown, interpolation=cv2.INTER_AREA) > 0.02
    radius = int(round((float(style.thickness) - 1.0) / 2.0))
    if radius > 0:
        drawn = cv2.dilate(drawn.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0
    pixels = np.array(base)
    pixels[drawn] = style.color
    return caption(Image.fromarray(pixels), title)


def _unwarped(record: DeformableRecord) -> DeformableRecord:
    from dataclasses import replace

    return replace(record, field_mm=np.zeros_like(record.field_mm))


def _blend_atlas(
    ctx: Workspace, base: Image.Image, record: DeformableRecord, kinds: tuple[str, ...],
    opacity: float,
) -> Image.Image:
    """The atlas images pulled through the record's map, blended under the lines.

    One image is gray; two are added in their colours
    (:func:`langslice.core.sections.channels.channel_colors`), as everywhere
    else.
    """
    labels = native_labels(ctx.atlas, record.placement)
    nx, ny = composed_native_grid(record, 1)

    def pulled(kind: str) -> np.ndarray:
        native = native_intensity(ctx.atlas, record.placement, kind)
        native = normalize_intensity(native, labels > 0)
        return cv2.remap(native.astype(np.float32), nx, ny, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if len(kinds) == 1:
        atlas = pulled(kinds[0])[..., None] * 255.0
    else:
        atlas = sum(pulled(kind)[..., None] * np.asarray(rgb, dtype=np.float32)
                    for kind, _word, rgb in channel_colors(kinds))
    pixels = np.asarray(base, dtype=np.float32)
    mixed = pixels * (1.0 - opacity) + atlas * opacity
    return Image.fromarray(np.rint(np.clip(mixed, 0, 255)).astype(np.uint8))
