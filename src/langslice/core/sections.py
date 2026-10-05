"""Sections as the run sees them: the renders every picture and fit start from.

Every path that shows or measures a section goes through :func:`render_slice`, so the
pixels the agent judges are the pixels a fit is computed on. Corrections are
applied in one order everywhere: ROTATE first, then FLIP left-right. Renders
are cached on the workspace (``render_cache``, ``render_scale``) and shared:
read them, never mutate them.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.appearance import Look, look_token
from langslice.core.image_prep import (
    adaptive_preprocess,
    custom_appearance,
    prepare_image_for_vlm,
    tissue_box,
)
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


def render_cache_key(
    ctx: Workspace, record: SliceState, *, long_edge: int, frame: bool, look: Look = None,
) -> tuple[str, bool, int, int, str, bool]:
    """The key a render is cached under: the section plus everything it shows.

    The default appearance keeps ``spec.preprocess`` in the look slot, so its
    keys are the ones every earlier caller computed.
    """
    return (record.id, record.flip, record.rotation_deg, long_edge,
            look_token(ctx, look), frame)


def render_slice(
    ctx: Workspace,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    frame: bool = False,
    look: Look = None,
) -> Image.Image:
    """One section as the run sees it: normalized, framed, enhanced, corrected.

    *look* None is the DEFAULT appearance: with ``spec.preprocess == "auto"``
    the section is run through
    :func:`~langslice.core.image_prep.adaptive_preprocess` — per-channel CLAHE plus a
    DAPI-weighted grayscale blend — so dim fluorescence reads like the atlas
    instead of like a black field; a host's multi-channel snapshot arrives
    already blended (``spec.host_preprocessing``). Any other look
    (:mod:`langslice.core.appearance`) is drawn from the section's raw
    channels over the SAME frame, crop and size, so geometry never depends on
    appearance. Display only: the user's file is never touched.

    *frame* crops to the tissue plus a small margin before the resize, so the
    section fills its frame about as much as a cropped atlas render does. It is
    off by default because it changes the image's coordinate frame: only the
    paths that SHOW a section to a model set it, never a fit, whose parameters
    are normalized against the render they were computed on.

    *long_edge* is a ceiling, never a target: a source smaller than it is
    returned at its own size (nothing is upsampled).

    Renders are cached on *ctx*, so the returned image is shared: read it,
    never mutate it in place.
    """
    key = render_cache_key(ctx, record, long_edge=long_edge, frame=frame, look=look)
    cached = ctx.render_cache.get(key)
    if cached is not None:
        return cached

    # The working copy, not the file: a whole-slide scan is read once, small.
    source, file_px_per_px = ctx.working_source(record.id)
    working_size = source.size
    box = tissue_box(source) if frame else None
    if box is not None:
        source = source.crop(box)
    prepped = prepare_image_for_vlm(source, max_long_edge=long_edge).image
    # How many FILE pixels one render pixel spans, before any quarter-turn:
    # the section's own micrometres per pixel times this is the canvas's.
    ctx.render_scale[key] = file_px_per_px * source.width / float(prepped.width)
    if look is not None:
        prepped = _look_image(ctx, record, look, working_size, box, prepped.size)
    elif ctx.spec.preprocess == "auto" and ctx.spec.host_preprocessing is None:
        prepped = adaptive_preprocess(prepped)
    rotate = _ROTATE_OPS.get(int(record.rotation_deg) % 360)
    if rotate is not None:
        prepped = prepped.transpose(rotate)
    if record.flip:
        prepped = prepped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ctx.render_cache[key] = prepped
    return prepped


def fine_detail(stretched: np.ndarray) -> float:
    """Fine structure of one stretched channel (0..1): the spread of what a
    3 px blur removes, over the pixels brighter than the background. Nuclei,
    layers and fibre edges score high; flat autofluorescence scores low."""
    tissue = stretched > 0.05
    if not tissue.any():
        return 0.0
    fine = stretched - cv2.GaussianBlur(stretched, (0, 0), 3.0)
    return float(fine[tissue].std())


def _look_image(
    ctx: Workspace,
    record: SliceState,
    look: dict[str, Any],
    working_size: tuple[int, int],
    box: tuple[int, int, int, int] | None,
    size: tuple[int, int],
) -> Image.Image:
    """*look* drawn from the raw channels, cropped and sized like the default."""
    names, planes = ctx.section_channels(record.id)

    def at_working(plane: np.ndarray) -> Image.Image:
        image = Image.fromarray(plane)
        if image.size != working_size:
            image = image.resize(working_size, Image.Resampling.LANCZOS)
        return image

    def framed(image: Image.Image) -> Image.Image:
        if box is not None:
            image = image.crop(box)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    def placed(plane: np.ndarray) -> np.ndarray:
        return np.asarray(framed(at_working(plane)), dtype=np.uint8)

    def named(name: str) -> np.ndarray:
        if name not in names:
            raise ValueError(f"{record.id} has no channel {name!r}; channels: {', '.join(names)}")
        return planes[names.index(name)]

    if "channel" in look:
        plane = placed(named(str(look["channel"])))
        return Image.fromarray(np.stack([plane, plane, plane], axis=-1))
    if "overlay" in look:
        # Each channel stretched on its WHOLE working plane (so a framed and
        # an unframed picture share one stretch), then added in its colour.
        from langslice.core.appearance import OVERLAY_STRETCH, channel_colors

        total = np.zeros((size[1], size[0], 3), dtype=np.float32)
        names_shown = list(look["overlay"])
        # One channel is gray; several are each added in their colour.
        colors = ([(names_shown[0], "gray", (255, 255, 255))] if len(names_shown) == 1
                  else channel_colors(names_shown))
        stretched_planes: list[np.ndarray] = []
        detail: list[float] = []
        for name, _word, _rgb in colors:
            whole = np.asarray(at_working(named(name)), dtype=np.float32)
            low, high = (float(v) for v in np.percentile(whole, OVERLAY_STRETCH))
            if high <= low:
                high = low + 1.0
            stretched = np.clip((whole - low) / (high - low), 0.0, 1.0)
            stretched_planes.append(stretched)
            detail.append(fine_detail(stretched))
        # Several channels: each is dimmed by its fine detail relative to the
        # most detailed one, so a flat autofluorescence channel (stretched to
        # full brightness on its own) cannot wash out the stain under it.
        top = max(detail) if len(colors) > 1 else 0.0
        for (_name, _word, rgb), stretched, amount in zip(
            colors, stretched_planes, detail, strict=True
        ):
            gain = amount / top if top > 0 else 1.0
            shown = np.asarray(framed(Image.fromarray((stretched * 255.0).astype(np.uint8))),
                               dtype=np.float32) / 255.0
            total += gain * shown[..., None] * np.asarray(rgb, dtype=np.float32)
        return Image.fromarray(np.clip(total, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return custom_appearance(
        [placed(plane) for plane in planes],
        channel_weights=look.get("channel_weights"),
        clahe_clip=float(look.get("clahe_clip", 4.0)),
        clahe_tiles=int(look.get("clahe_tiles", 8)),
        n4=bool(look.get("n4")),
        denoise=bool(look.get("denoise")),
    )


def canvas_um_per_px(
    ctx: Workspace,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    frame: bool = False,
) -> tuple[float | None, str]:
    """``(micrometres per pixel of the RENDER, source)`` for one section.

    The render is a downsample of the file, so the file's pixel size times
    the downsample factor is the working canvas's. ``(None, "")`` when
    neither the file nor the host supplies one.
    """
    from_file, source = ctx.calibration(record.id)
    if from_file is None:
        return None, source
    key = render_cache_key(ctx, record, long_edge=long_edge, frame=frame)
    if key not in ctx.render_scale:
        render_slice(ctx, record, long_edge=long_edge, frame=frame)
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
