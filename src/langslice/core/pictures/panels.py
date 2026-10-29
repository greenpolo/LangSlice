"""A section alone and an atlas plane alone, framed for a picture."""

from __future__ import annotations

import logging
import math
from typing import cast

import numpy as np
from PIL import Image

from langslice.core.atlas.borders import (
    HIGHLIGHT_WIDTH,
    annotation_slice,
    region_outlines,
)
from langslice.core.atlas.images import TEMPLATE
from langslice.core.atlas.load import get_reference_slice
from langslice.core.pictures.canvas import (
    REGION_CONTEXT_ALPHA,
    _draw_polys,
    border_words,
    line_coverage,
    region_polys,
    regions_left,
    resolve_border_color,
)
from langslice.core.pictures.options import BORDERS, DisplayOptions
from langslice.core.sections.channels import channel_colors
from langslice.core.sections.images import Look
from langslice.core.sections.reading import mask_box
from langslice.core.sections.renders import render_slice
from langslice.core.space import Plane
from langslice.core.stack.section import Angles, SliceState
from langslice.core.stack.stack import StackState, plane_angles
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)
#: Intensity percentile (inside the atlas anatomy) mapped to white when an
#: atlas image other than the template is shown.
ATLAS_IMAGE_PERCENTILE = 99.5


# --- atlas images --------------------------------------------------------


def _image_plane(
    ctx: Workspace, state: StackState, kind: str, position_mm: float, labels: np.ndarray,
    angles: Angles,
) -> np.ndarray:
    """The atlas image *kind* on the native plane grid, percentile-stretched to 0..1."""
    from langslice.core.geometry.oblique import sample_oblique_plane

    values = np.asarray(sample_oblique_plane(
        ctx.atlas, position_mm, cast(Plane, state.plane), *angles, volume=kind, order=1,
    ), dtype=np.float32)
    if values.shape != labels.shape:
        values = np.asarray(Image.fromarray(values, mode="F").resize(
            (labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
    inside = values[(labels > 0) & (values > 0)]
    top = float(np.percentile(inside, ATLAS_IMAGE_PERCENTILE)) if inside.size else 0.0
    if top <= 0:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip(values / top, 0.0, 1.0)


def atlas_image_picture(
    ctx: Workspace, state: StackState, kinds: tuple[str, ...], position_mm: float,
    *, angles: Angles | None = None,
) -> Image.Image | None:
    """The atlas images *kinds* on the native plane grid; None for the template alone.

    At *angles* (:func:`langslice.core.stack.stack.plane_angles`: a section's
    own, or the stack's when None).

    ``template`` alone is left to the renderers' own reference path (the pixels
    every earlier picture showed). Any other atlas image is stretched inside
    the atlas anatomy. Two images are added in their two colours
    (:func:`langslice.core.sections.channels.channel_colors`); no image is a
    black plane (the lines alone).
    """
    images = tuple(kind for kind in kinds if kind != BORDERS)
    if images == (TEMPLATE,):
        return None
    plane = cast(Plane, state.plane)
    pitch, yaw = plane_angles(state, angles)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=pitch, yaw_deg=yaw,
    ))
    shape = labels.shape
    if not images:
        return Image.new("L", (shape[1], shape[0]), 0)

    def gray(kind: str) -> np.ndarray:
        if kind != TEMPLATE:
            return _image_plane(ctx, state, kind, position_mm, labels, (pitch, yaw))
        reference = get_reference_slice(ctx.atlas, position_mm, plane=plane,
                                        pitch_deg=pitch, yaw_deg=yaw)
        if reference.size != (shape[1], shape[0]):
            reference = reference.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
        return np.asarray(reference.convert("L"), dtype=np.float32) / 255.0

    if len(images) == 1:
        return Image.fromarray((gray(images[0]) * 255.0).astype(np.uint8), mode="L")
    total = np.zeros((*shape, 3), dtype=np.float32)
    for kind, _word, rgb in channel_colors(images):
        total += gray(kind)[..., None] * np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(np.clip(total, 0, 255).astype(np.uint8), mode="RGB")


def regions_in_plane(
    ctx: Workspace, state: StackState, position_mm: float, options: DisplayOptions,
    *, angles: Angles | None = None,
) -> list[str]:
    """The highlighted regions with at least one pixel in the plane at
    *position_mm* and *angles* (the stack's when None); a region with a side
    (``"CTX:left"``) counts only that side of the atlas picture, as it is
    drawn."""
    from langslice.core.atlas.sides import restrict, split_side

    if not options.regions:
        return []
    pitch, yaw = plane_angles(state, angles)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=cast(Plane, state.plane),
        pitch_deg=pitch, yaw_deg=yaw,
    ))
    # An atlas-only picture has no section: a side is the picture's own.
    left = regions_left(ctx.atlas, options.regions, position_mm, state.plane, pitch, yaw,
                        np.eye(2))
    return [name for name, ids in options.regions
            if restrict(np.isin(labels, list(ids)), split_side(name)[1], left).any()]


def _crop_fraction(image: Image.Image, zoom: tuple[float, ...]) -> Image.Image:
    from langslice.core.geometry.pixel_frames import cropped_geometry
    from langslice.core.pictures.canvas import zoom_box

    box = zoom_box(list(zoom), image.size)
    return cropped_geometry(image.crop(box), image, box)


# --- section and atlas pictures -----------------------------------------


def framed_section(
    ctx: Workspace, state: StackState, record: SliceState, options: DisplayOptions,
    *, long_edge: int | None = None, look: Look | str = "options",
) -> Image.Image:
    """One section as corrected, the whole image, in the call's look and zoom.

    *long_edge* None is the call's picture size (``options.long_edge``);
    *look* overrides the options'. A zoom renders the section
    larger (never past its working copy) and crops the requested fraction,
    so it magnifies.
    """
    long_edge = long_edge or options.long_edge
    drawn: Look = options.look(state, record) if look == "options" else cast(Look, look)
    whole = render_slice(ctx, record, long_edge=long_edge, look=drawn)
    if options.full_view:
        return whole
    window = options.window(whole.size)
    x0, y0, x1, y1 = window
    extent = max(abs(x1 - x0), abs(y1 - y0), 1e-3)
    larger = render_slice(
        ctx, record, long_edge=int(math.ceil(long_edge / min(1.0, extent))), look=drawn,
    )
    return _crop_fraction(larger, tuple(window))


def framed_atlas(
    ctx: Workspace, state: StackState, position_mm: float, options: DisplayOptions,
    *, long_edge: int | None = None, um_per_px: float | None = None,
    angles: Angles | None = None,
) -> Image.Image:
    """The atlas at *position_mm*, framed to its anatomy, with the call's lines.

    At *angles*: a section's own for a picture beside that section, the
    stack's view angles for a picture without a section; None, the stack's
    one angle.

    At most *long_edge* (None: ``options.long_edge``) and never upsampled past
    the plane's own voxels, unless *um_per_px*: then drawn at exactly that many
    micrometres per pixel, lines included, for a picture that puts the atlas
    beside a section drawn at the same scale
    (:mod:`langslice.core.pictures.scale`).
    """
    from langslice.core.atlas.borders import atlas_um_per_px
    from langslice.core.atlas.planes import (
        atlas_mask,
        atlas_section,
        atlas_sized,
    )
    from langslice.core.pictures.scale import resized

    edge = int(long_edge or options.long_edge)

    def sized(picture: Image.Image) -> Image.Image:
        if um_per_px is not None:
            return resized(picture, atlas_um_per_px(ctx.atlas) / float(um_per_px))
        return atlas_sized(picture, edge)

    pitch, yaw = plane_angles(state, angles)
    if options.atlas_images == ("template",) and not options.lines and options.full_view:
        from langslice.core.geometry.pixel_frames import cropped_geometry

        whole = atlas_section(ctx, state, position_mm, frame=True, angles=(pitch, yaw))
        return cropped_geometry(sized(whole), whole)
    plane = cast(Plane, state.plane)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=pitch, yaw_deg=yaw,
    ))
    picture = atlas_image_picture(ctx, state, options.atlas_channels, position_mm,
                                  angles=(pitch, yaw))
    if picture is None:
        picture = get_reference_slice(
            ctx.atlas, position_mm, plane=plane,
            pitch_deg=pitch, yaw_deg=yaw,
        )
    native = (labels.shape[1], labels.shape[0])
    if picture.size != native:
        picture = picture.resize(native, Image.Resampling.BILINEAR)
    try:
        mask = atlas_mask(ctx, state, position_mm, picture.size, angles=(pitch, yaw)) > 0
    except Exception as exc:
        logger.warning("atlas at %.3f mm: framed by its labels, no root mask (%s)",
                       position_mm, exc)
        mask = labels > 0
    box = mask_box(picture.size, mask) or (0, 0, picture.width, picture.height)
    if not options.full_view:
        from langslice.core.pictures.canvas import zoom_box

        whole = sized(picture.crop(box)).size
        inner = zoom_box(options.window(whole), (box[2] - box[0], box[3] - box[1]))
        box = (box[0] + inner[0], box[1] + inner[1], box[0] + inner[2], box[1] + inner[3])
    cropped = picture.crop(box)
    shown = sized(cropped).convert("RGB")
    from langslice.core.geometry.pixel_frames import cropped_geometry

    picture.info["pixel_to_native"] = np.eye(3).tolist()
    cropped_geometry(shown, picture, box)
    if not options.lines:
        return shown
    factor = shown.width / float(cropped.width)
    canvas = np.asarray(shown, dtype=np.uint8).copy()
    context: list[np.ndarray] = []
    if options.borders:
        context = region_outlines(ctx.atlas, position_mm, depth=options.depth, plane=plane,
                                  pitch_deg=pitch, yaw_deg=yaw)
    origin = (box[0], box[1])
    region_lines: list[np.ndarray] = []
    if options.regions:
        # An atlas-only picture has no section: a side is the picture's own.
        left = regions_left(ctx.atlas, options.regions, position_mm, plane,
                            pitch, yaw, np.eye(2))
        region_lines = region_polys(labels, options.regions, left)
    # One arrangement for the whole picture, resolved against this atlas plate's
    # own pixels (there is no section here to judge contrast against).
    drawn_polys = context + region_lines
    coverage = (
        line_coverage(canvas.shape[:2], drawn_polys, thickness=options.border_thickness,
                     origin=origin, factor=factor)
        if drawn_polys else None
    )
    core, _ = resolve_border_color(
        options.border_color, options.border_thickness, rgb=canvas, coverage=coverage,
    )
    if context:
        _draw_polys(canvas, context, core, thickness=options.border_thickness, origin=origin,
                    factor=factor, alpha=REGION_CONTEXT_ALPHA if options.regions else 1.0)
    if region_lines:
        _draw_polys(canvas, region_lines, core,
                    thickness=options.border_thickness * HIGHLIGHT_WIDTH, origin=origin,
                    factor=factor)
    result = Image.fromarray(canvas, mode="RGB")
    result.info["pixel_to_native"] = shown.info["pixel_to_native"]
    if drawn_polys:
        result.info["border_word"] = border_words(core)
    return result


def atlas_caption(state: StackState, position_mm: float, options: DisplayOptions,
                  *, angles: Angles | None = None, median: bool = False) -> str:
    """The label burned into an atlas picture drawn at *angles* (the stack's
    when None); *median*: drawn at the median of differing sections' angles,
    which the label says
    (:func:`langslice.core.pictures.captions.view_angles_label`)."""
    from langslice.core.pictures.captions import view_angles_label

    shown = view_angles_label(plane_angles(state, angles), median=median)
    name = options.atlas_name()
    extra = "" if name == TEMPLATE else f" {name}"
    if options.regions:
        extra += " regions " + ",".join(name for name, _ids in options.regions)
    return f"atlas {position_mm:.2f} mm{shown}{extra}"


__all__ = [
    "atlas_caption",
    "atlas_image_picture",
    "framed_atlas",
    "framed_section",
    "regions_in_plane",
]
