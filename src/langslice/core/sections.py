"""Sections as the run sees them: the renders every picture and fit start from.

Every path that shows or measures a section goes through :func:`render_slice`, so the
pixels the agent judges are the pixels a fit is computed on. Corrections are
applied in one order everywhere: ROTATE first, then FLIP left-right. Renders
are cached on the workspace (``render_cache``, ``render_scale``) and shared:
read them, never mutate them.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image

from langslice.core import channels
from langslice.core import section_images as images
from langslice.core.image_prep import prepare_image_for_vlm, read_working_image
from langslice.core.section_images import PREPROCESSED, Look
from langslice.core.state import SliceState
from langslice.core.workspace import Workspace

#: Working frame for transform fits and their preview panels. A COMPUTE
#: size: fits, calibration and the six stored numbers are normalized against
#: this render, whatever size the pictures are shown at.
PREVIEW_LONG_EDGE = 512

_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def drawn_look(record: SliceState, look: Look) -> Look:
    """*look* with the section image it draws resolved on *record*: None is the
    opening ``preprocessed`` (drawn through its own path at each size), a made
    image carries its digest (``{"image", "digest"}``), an opening image under
    another name ``{"image", "opening": True}``."""
    name = PREPROCESSED if look is None else look.get("image")
    if name is None:
        return look
    entry = images.entry_of(record, str(name))
    if entry is None:
        return None if name == PREPROCESSED else {"image": str(name)}
    if images.is_opening(entry):
        return None if name == PREPROCESSED else {"image": str(name), "opening": True}
    return {"image": str(name), "digest": entry.get("digest")}


def look_token(ctx: Workspace, record: SliceState, look: Look) -> str:
    """The render-cache slot of a look on *record*: the opening ``preprocessed``
    keeps ``spec.preprocess``, a made image its digest."""
    drawn = drawn_look(record, look)
    if drawn is None:
        return str(ctx.spec.preprocess)
    return json.dumps(drawn, sort_keys=True)


def render_cache_key(
    ctx: Workspace, record: SliceState, *, long_edge: int, look: Look = None,
) -> tuple[str, bool, int, int, str]:
    """The key a render is cached under: the section plus everything it shows."""
    return (record.id, record.flip, record.rotation_deg, long_edge,
            look_token(ctx, record, look))


def render_slice(
    ctx: Workspace,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    look: Look = None,
) -> Image.Image:
    """One section as the run sees it: one of its section images, sized and oriented.

    *look* None is the section's ``preprocessed`` image
    (:mod:`langslice.core.section_images`): the opening one is drawn through
    the automatic path at this size (with ``spec.preprocess == "auto"``,
    :func:`~langslice.core.image_prep.adaptive_preprocess` on the working copy
    resized here; a host's blend is the working copy itself), a remade one is
    its file resized. ``{"image": name}`` is any section image alone, in gray:
    a raw channel as the file holds it, a made one from its file.
    ``{"overlay": [raw names]}`` is several raw channels in their colours
    (:func:`langslice.core.channels.composite`); ``{"file": True}`` the working
    copy as read, in colour (a plugin's raw raster). Every look is drawn at the
    SAME size, so geometry never depends on the image drawn. Display only: the
    user's file is never touched. The whole image frame is always drawn.

    *long_edge* is a ceiling, never a target: a source smaller than it is
    returned at its own size (nothing is upsampled).

    Renders are cached on *ctx*, so the returned image is shared: read it,
    never mutate it in place.
    """
    key = render_cache_key(ctx, record, long_edge=long_edge, look=look)
    cached = ctx.render_cache.get(key)
    if cached is not None:
        return cached

    # The working copy, not the file: a whole-slide scan is read once, small.
    source, file_px_per_px = ctx.working_source(record.id)
    size = prepare_image_for_vlm(source, max_long_edge=long_edge).size
    # How many FILE pixels one render pixel spans, before any quarter-turn:
    # the section's own micrometres per pixel times this is the canvas's.
    ctx.render_scale[key] = file_px_per_px * source.width / float(size[0])
    prepped = _look_image(ctx, record, drawn_look(record, look), size, long_edge=long_edge)
    rotate = _ROTATE_OPS.get(int(record.rotation_deg) % 360)
    if rotate is not None:
        prepped = prepped.transpose(rotate)
    if record.flip:
        prepped = prepped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    from langslice.core.affine import pixel_fraction_map
    from langslice.core.landmarks import orientation

    prepped.info["pixel_to_file_fraction"] = (
        orientation(record) @ pixel_fraction_map(prepped.size)).tolist()
    ctx.render_cache[key] = prepped
    return prepped


def _gray(plane: Any) -> Image.Image:
    held = np.asarray(plane, dtype=np.uint8)
    return Image.fromarray(np.stack([held, held, held], axis=-1))


def _look_image(
    ctx: Workspace, record: SliceState, look: Look, size: tuple[int, int], *, long_edge: int,
) -> Image.Image:
    """*look* (resolved, :func:`drawn_look`) drawn unoriented at *size*, RGB."""

    def framed(image: Image.Image) -> Image.Image:
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    if look is None or look.get("opening"):
        return _gray(np.asarray(images.opening_render(ctx, record.id, long_edge)))
    if look.get("file"):  # the file as it is, no preprocessing
        image, _factor = read_working_image(ctx.image_path(record.id))
        return framed(image.convert("RGB"))
    if "overlay" in look:
        # Each channel stretched on its whole working plane, then added in
        # its colour.
        shown = [(str(name), np.asarray(images.raw_plane(ctx, record.id, str(name)),
                                        dtype=np.float32))
                 for name in look["overlay"]]
        return channels.composite(shown, framed=framed, size=size)
    return framed(_gray(images.image_plane(ctx, record, str(look["image"]))))


def canvas_um_per_px(
    ctx: Workspace,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
) -> tuple[float | None, str]:
    """``(micrometres per pixel of the RENDER, source)`` for one section.

    The render is a downsample of the file, so the file's pixel size times
    the downsample factor is the working canvas's. ``(None, "")`` when
    neither the file nor the host supplies one.
    """
    from_file, source = ctx.calibration(record.id)
    if from_file is None:
        return None, source
    key = render_cache_key(ctx, record, long_edge=long_edge)
    if key not in ctx.render_scale:
        render_slice(ctx, record, long_edge=long_edge)
    return from_file * ctx.render_scale.get(key, 1.0), source


def shown_section(
    ctx: Workspace, record: SliceState, section: Image.Image, um_per_px: float,
    look: Look = None, *, long_edge: int = PREVIEW_LONG_EDGE,
) -> tuple[Image.Image, float, tuple[float, float]]:
    """The render a PICTURE of *section* is drawn from, *long_edge* at most.

    *section* is the :data:`PREVIEW_LONG_EDGE` working frame every fit and
    every written transform is computed on, and *um_per_px* its calibration.
    At the working frame's own size with the default *look* this returns them
    unchanged with factors ``(1.0, 1.0)``; otherwise a render of the same
    section in *look* at *long_edge* (never upsampled past the working copy),
    its micrometres per pixel, and the ``(fx, fy)`` that carry working-frame
    pixels onto it. Nothing computed is drawn from here.
    """
    if long_edge == PREVIEW_LONG_EDGE and look is None:
        return section, um_per_px, (1.0, 1.0)
    shown = render_slice(ctx, record, long_edge=int(long_edge), look=look)
    fx = shown.width / float(section.width)
    fy = shown.height / float(section.height)
    return shown, um_per_px / fx, (fx, fy)


def rescale_section_matrix(matrix: Any, fx: float, fy: float) -> np.ndarray:
    """A 2x3 on a section frame, re-expressed on the same frame scaled by (fx, fy)."""
    square = np.vstack([np.asarray(matrix, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    scale = np.diag([fx, fy, 1.0])
    return (scale @ square @ np.diag([1.0 / fx, 1.0 / fy, 1.0]))[:2]
