"""The Simple view's pictures: a section from its main stain alone, quiet.

The Extended view draws a section's raw channels in their own colours
(:mod:`.drawing`). The Simple view draws one channel, the nuclear stain
(:func:`main_channel`), in a cool grey on the window's dark stage: the
tissue is cut out of the scanner's tile floor (:func:`tissue_mask`), its
brightest spots are compressed by a soft knee instead of clipping, and the
atlas borders are a soft light layer over it, found as the difference
between ``look``'s picture with and without them (so the scale bar, drawn
in both, goes). Read-only like every drawing of the window.

``NUCLEAR`` matches the channel names that say a channel is the nuclear
(or Nissl) stain. ``STAGE`` is the window's dark ground (``--stage``) the
pictures sit on, ``TINT`` the tissue's cool off-white at full brightness,
``LINE`` the atlas borders' colour and ``LINE_STRENGTH`` how strongly they
are laid over the tissue. ``BAR_FROM_BOTTOM``, ``BAR_LEFT``, ``BAR_GREY``,
``MIN_BAR`` and ``BAR_LABEL`` say where ``look`` draws its scale bar
(``captions._draw_scale_bar``): its row from the bottom, its left end, its
grey, the shortest bar it draws and the label's height above it. The tissue
mask is found on a copy at most ``MASK_EDGE`` pixels long. What lies on the
slide beside the tissue is left out of it (``without_debris``): a piece
apart from the largest that is under ``DEBRIS_PART`` of it and whose grain
is under ``DEBRIS_BLUR`` of the tissue's, or that reaches within
``BORDER_BAND`` of the picture's border; what is as blurred and reaches
within ``SCAN_BAND`` of the edge of the scan, where ``UNSCANNED`` or darker
is the ground the scanner left (its bright line and smears along that
edge), with ``EDGE_GLOW_RIM`` pixels around it, and what is thinner than
``THREAD`` pixels there, ``THREAD_FRINGE`` of its foot kept; and a patch whose
grain is under
``GLARE_SHARP`` of the tissue's and whose brightness is over
``GLARE_BRIGHT`` of it, at least ``GLARE_AREA`` of the picture, when over
``GLARE_APART`` of its surround (``GLARE_RING`` pixels) is not tissue, or
over ``GLARE_OUT`` of it is not and over ``GLARE_HALO`` of that still glows.
A brightfield section (dark stain on a bright slide: the median of its
scanned pixels, and of their outer ``SLIDE_RIM`` pixels, over
``BRIGHT_SLIDE``) is read by its darkness under the
slide's white, the ``SLIDE_LEVEL`` percentile.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from PIL import Image, ImageFilter
from SimpleITK import (
    GetArrayFromImage,
    GetImageFromArray,
    SmoothingRecursiveGaussian,
)

from langslice.core.pictures.look import LookRequest, look
from langslice.core.sections.images import raw_names
from langslice.core.sections.renders import render_slice
from langslice.hosts.gui.drawing import content

if TYPE_CHECKING:
    import numpy.typing as npt

    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.hosts.gui.drawing import Opened

NUCLEAR = re.compile(r"dapi|hoechst|nuc|nissl|draq|topro", re.IGNORECASE)
STAGE = (18, 18, 19)
TINT = (206, 210, 232)
LINE = (232, 234, 250)
LINE_STRENGTH = 0.55
BAR_FROM_BOTTOM = 14
BAR_LEFT = 12
BAR_GREY = 200
MIN_BAR = 4
BAR_LABEL = 22
MASK_EDGE = 512
DEBRIS_PART = 0.08
DEBRIS_BLUR = 0.5
GLARE_SHARP = 0.55
GLARE_BRIGHT = 1.15
GLARE_AREA = 0.0004
GLARE_RING = 6
GLARE_OUT = 0.08
GLARE_HALO = 0.3
GLARE_APART = 0.5
BORDER_BAND = 0.02
SCAN_BAND = 0.03
THREAD = 12
THREAD_FRINGE = 2
EDGE_GLOW_RIM = 2
UNSCANNED = 4.0
BRIGHT_SLIDE = 140.0
SLIDE_RIM = 3
SLIDE_LEVEL = 92


def main_channel(names: list[str]) -> str:
    """Choose the channel the Simple view draws.

    Parameters
    ----------
    names
        The section's raw channels, in file order.

    Returns
    -------
    str
        A channel named for a nuclear or Nissl stain; else an RGB file's
        ``blue`` (DAPI's colour); else the first.
    """
    for name in names:
        if NUCLEAR.search(name):
            return name
    return "blue" if "blue" in names else names[0]


def draw_calm_overlay(
    opened: Opened, state: Stack, section: Section, edge: int
) -> Image.Image:
    """Draw a section under its current registration, quiet.

    Parameters
    ----------
    opened
        The job, read for drawing.
    state
        The job's stack as the viewer reads it.
    section
        The section to draw.
    edge
        The picture's long edge in pixels.

    Returns
    -------
    Image.Image
        The main stain on the stage under the section's transform and
        applied deformation, the atlas borders soft over it.
    """
    channel = (main_channel(raw_names(opened.workspace, section.id)),)
    request = LookRequest(
        mode="overlay",
        sections=(section.id,),
        images=channel,
        long_edge=edge,
        atlas_layers=(),
    )
    bare = content(
        look(opened.workspace, state, request, store=opened.store)[0]
    )
    lined = content(
        look(
            opened.workspace,
            state,
            LookRequest(
                mode="overlay",
                sections=(section.id,),
                images=channel,
                long_edge=edge,
            ),
            store=opened.store,
        )[0]
    )
    return calm(bare, lined=lined)


def draw_calm_thumb(
    opened: Opened, section: Section, edge: int
) -> Image.Image:
    """Draw a section alone, oriented, from its main stain, quiet.

    Parameters
    ----------
    opened
        The job, read for drawing.
    section
        The section to draw.
    edge
        The thumbnail's long edge in pixels.

    Returns
    -------
    Image.Image
        The section turned and flipped as it is kept, on the stage.
    """
    channel = main_channel(raw_names(opened.workspace, section.id))
    picture = render_slice(
        opened.workspace, section, long_edge=edge, look={"image": channel}
    )
    return calm(picture)


def calm(
    bare: Image.Image, *, lined: Image.Image | None = None
) -> Image.Image:
    """Make one channel's picture quiet, with its borders when given.

    Parameters
    ----------
    bare
        The section without atlas borders, one channel in grey.
    lined
        The same picture with the borders, the same size; None: no borders.

    Returns
    -------
    Image.Image
        The tissue tinted on the stage, the borders a soft light layer.
    """
    grey = np.asarray(bare.convert("L")).astype(np.float32)
    if is_brightfield(bare):
        scanned = grey > UNSCANNED
        white = float(np.percentile(grey[scanned], SLIDE_LEVEL))
        grey = np.where(scanned, np.clip(white - grey, 0, 255), 0.0).astype(
            np.float32
        )
    mask = tissue_mask(grey)
    mask[scale_bar(grey)] = 0.0
    inside = grey[mask > 0.5]
    low = float(np.percentile(inside, 4)) if inside.size else 0.0
    high = float(np.percentile(inside, 99)) if inside.size else 255.0
    value = np.clip((grey - low) / max(1.0, high - low), 0, None)
    value = np.clip(1.35 * value / (1 + 0.35 * value), 0, 1) ** 0.95
    stage = np.array(STAGE, np.float32)
    tissue = value[..., None] * np.array(TINT, np.float32)
    rgb = tissue * mask[..., None] + stage * (1 - mask[..., None])
    rgb = np.maximum(rgb, stage)
    if lined is not None:
        with_lines = np.asarray(lined.convert("RGB"), dtype=np.int16)
        plain = np.asarray(bare.convert("RGB"), dtype=np.int16)
        change = cast(
            "npt.NDArray[np.int16]", np.max(np.abs(with_lines - plain), axis=2)
        )
        strength = LINE_STRENGTH * min(1.0, 0.45 + max(bare.size) / 1600)
        alpha = (np.clip(change / 110.0, 0, 1) * strength)[..., None]
        rgb = rgb * (1 - alpha) + np.array(LINE, np.float32) * alpha
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def is_brightfield(picture: Image.Image) -> bool:
    """Tell a brightfield slide (dark stain on bright glass) from darkfield.

    Parameters
    ----------
    picture
        A section's picture.

    Returns
    -------
    bool
        Whether the median of its scanned pixels, and of the rim of what
        was scanned (the glass around the tissue), is over ``BRIGHT_SLIDE``.
    """
    grey = np.asarray(picture.convert("L")).astype(np.float32)
    scanned = (grey > UNSCANNED).astype(np.uint8)
    if not scanned.any():
        return False
    padded = np.pad(scanned, SLIDE_RIM)
    core = np.asarray(cv2.erode(padded, disk(SLIDE_RIM)), dtype=np.uint8)
    rim = (scanned > 0) & (
        core[SLIDE_RIM:-SLIDE_RIM, SLIDE_RIM:-SLIDE_RIM] < 1
    )
    return (
        float(np.median(grey[scanned > 0])) > BRIGHT_SLIDE
        and float(np.median(grey[rim])) > BRIGHT_SLIDE
    )


def scale_bar(grey: npt.NDArray[np.float32]) -> tuple[slice, slice]:
    """Find the 1 mm scale bar ``look`` draws in the bottom-left corner.

    Parameters
    ----------
    grey
        The picture's one channel, 0 to 255.

    Returns
    -------
    tuple of slice
        The rows and columns of the bar and its label (empty without one).
    """
    height, width = cast("tuple[int, int]", grey.shape)
    row = height - BAR_FROM_BOTTOM
    if row < 0 or width <= BAR_LEFT:
        return slice(0, 0), slice(0, 0)
    bright = grey[row, BAR_LEFT:] >= BAR_GREY
    length = int(np.argmin(bright)) if not bright.all() else bright.size
    if length < MIN_BAR:
        return slice(0, 0), slice(0, 0)
    return slice(max(0, row - BAR_LABEL), height), slice(
        0, BAR_LEFT + max(length, 34) + 6
    )


def tissue_mask(grey: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Find the tissue: textured and brighter than the scanner's tile floor.

    Parameters
    ----------
    grey
        The picture's one channel, 0 to 255.

    Returns
    -------
    np.ndarray
        The tissue's weight, 0 to 1, its edge feathered, the picture's size.
    """
    height, width = cast("tuple[int, int]", grey.shape)
    scale = min(1.0, MASK_EDGE / max(height, width))
    small = grey
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = np.asarray(
            Image.fromarray(grey.astype(np.uint8)).resize(
                size, Image.Resampling.BILINEAR
            )
        ).astype(np.float32)
    image = GetImageFromArray(small)
    wide = 5.0 * scale + 1.0
    smooth = GetArrayFromImage(SmoothingRecursiveGaussian(image, [wide, wide]))
    values = smooth[smooth > 4]
    floor = float(np.percentile(values, 30)) if values.size else 8.0
    bright = smooth > max(12.0, floor * 1.6)
    fine = GetArrayFromImage(SmoothingRecursiveGaussian(image, [1.0, 1.0]))
    spread = 4.0 * scale + 1.0
    texture = GetArrayFromImage(
        SmoothingRecursiveGaussian(
            GetImageFromArray(np.abs(small - fine)), [spread, spread]
        )
    )
    level = (
        max(0.6, 0.12 * float(np.median(texture[bright])))
        if bright.any()
        else 0.6
    )
    binary = (bright & (texture > level)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    areas = np.asarray(stats, dtype=np.int64)[:, cv2.CC_STAT_AREA]
    large = areas >= max(4, int(small.size * 0.004))
    large[0] = False
    keep = without_debris(
        np.asarray(labels, dtype=np.int64),
        large,
        smooth,
        texture,
        bright,
        scan_edge(small),
    )
    keep = np.asarray(
        cv2.morphologyEx(keep, cv2.MORPH_CLOSE, disk(3)), dtype=np.uint8
    )
    keep = fill_holes(keep)
    keep = np.asarray(cv2.dilate(keep, disk(2)), dtype=np.uint8)
    weight = Image.fromarray(keep * 255)
    if scale < 1.0:
        weight = weight.resize((width, height), Image.Resampling.BILINEAR)
    weight = weight.filter(ImageFilter.GaussianBlur(2.0))
    return np.asarray(weight).astype(np.float32) / 255.0


def without_debris(
    labels: npt.NDArray[np.int64],
    large: npt.NDArray[np.bool_],
    smooth: npt.NDArray[np.float32],
    texture: npt.NDArray[np.float32],
    bright: npt.NDArray[np.bool_],
    edge: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Keep the tissue's pieces and leave out what lies on the slide beside it.

    Dirt, bubbles and glare are out of focus: bright, with little of the
    tissue's grain. A piece apart from the largest goes when it is small and
    either blurred or at the scan's edge, or reaches the picture's border,
    before and after the scan edge's glow and threads are taken off; a
    blurred bright patch goes when
    it stands out of the tissue (most of its surround is not tissue, or part
    of it is and that part still glows). A blurred patch inside the tissue
    stays: it is over tissue.

    Parameters
    ----------
    labels
        Each pixel's piece, zero the ground.
    large
        Which pieces are large enough to be tissue.
    smooth
        The picture, smoothed.
    texture
        The picture's fine grain, smoothed.
    bright
        Where the picture stands above the scanner's floor.
    edge
        The band along the edge of the scan.

    Returns
    -------
    np.ndarray
        One on the tissue, zero off it.
    """
    pieces = cast("list[int]", np.flatnonzero(large).tolist())
    if not pieces:
        return large[labels].astype(np.uint8)
    pixels = cast("list[int]", np.bincount(labels.ravel()).tolist())
    main = max(pieces, key=lambda piece: pixels[piece])
    sharp = (texture / (smooth + 1.0)).astype(np.float32)
    body = cast("npt.NDArray[np.bool_]", labels == main)
    focus = float(np.median(sharp[body]))
    glow = float(np.median(smooth[body]))
    height, width = cast("tuple[int, int]", labels.shape)
    keep = main_and_parts(large[labels].astype(np.uint8), sharp, focus, edge)
    blurred = sharp < DEBRIS_BLUR * focus
    keep = without_threads(without_edge_glow(keep, edge, blurred), edge)
    keep = main_and_parts(keep, sharp, focus, edge)
    glare = (
        bright & (sharp < GLARE_SHARP * focus) & (smooth > GLARE_BRIGHT * glow)
    ).astype(np.uint8)
    count, patches, stats, _ = cv2.connectedComponentsWithStats(
        glare, connectivity=8
    )
    sizes = cast(
        "list[int]",
        np.asarray(stats, dtype=np.int64)[:, cv2.CC_STAT_AREA].tolist(),
    )
    marks = np.asarray(patches, dtype=np.int64)
    tissue = keep > 0
    for patch in range(1, count):
        if sizes[patch] < height * width * GLARE_AREA:
            continue
        core = cast("npt.NDArray[np.bool_]", marks == patch)
        spread = cv2.dilate(core.astype(np.uint8), disk(GLARE_RING))
        reach = np.asarray(spread, dtype=np.uint8) > 0
        ring = reach & ~core
        beside = ring & ~tissue
        apart = np.count_nonzero(beside) / max(1, np.count_nonzero(ring))
        halo = np.count_nonzero(beside & bright) / max(
            1, np.count_nonzero(beside)
        )
        if apart > GLARE_APART or (apart > GLARE_OUT and halo > GLARE_HALO):
            keep[reach] = 0
    return keep


def main_and_parts(
    mask: npt.NDArray[np.uint8],
    sharp: npt.NDArray[np.float32],
    focus: float,
    edge: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Keep a mask's largest piece and the pieces that are tissue too.

    A piece apart from the largest goes when it is small and either blurred
    or at the edge of the scan, or when it reaches the picture's border.

    Parameters
    ----------
    mask
        One on the mask, zero off it.
    sharp
        The picture's grain for its brightness.
    focus
        The tissue's own grain for its brightness.
    edge
        The band along the edge of the scan.

    Returns
    -------
    np.ndarray
        The mask without those pieces.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 2:
        return mask
    marks = np.asarray(labels, dtype=np.int64)
    pixels = cast(
        "list[int]",
        np.asarray(stats, dtype=np.int64)[:, cv2.CC_STAT_AREA].tolist(),
    )
    main = max(range(1, count), key=lambda piece: pixels[piece])
    height, width = cast("tuple[int, int]", mask.shape)
    band = max(1, round(BORDER_BAND * min(height, width)))
    border = np.ones(mask.shape, np.bool_)
    border[band:-band, band:-band] = False
    kept = np.ones(count, np.bool_)
    kept[0] = False
    for piece in range(1, count):
        if piece == main:
            continue
        where = cast("npt.NDArray[np.bool_]", marks == piece)
        blurred = float(np.median(sharp[where])) < DEBRIS_BLUR * focus
        small = pixels[piece] < DEBRIS_PART * pixels[main]
        beside = blurred or np.count_nonzero(where & edge) > 0
        if (small and beside) or np.count_nonzero(where & border) > 0:
            kept[piece] = False
    return kept[marks].astype(np.uint8)


def without_edge_glow(
    mask: npt.NDArray[np.uint8],
    edge: npt.NDArray[np.bool_],
    blurred: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Take off a mask what is blurred and reaches the edge of the scan.

    A scanner leaves a bright line and smears along the edge of what it
    scanned, and they can touch the tissue. Blurred tissue away from that
    edge stays.

    Parameters
    ----------
    mask
        One on the mask, zero off it.
    edge
        The band along the edge of the scan.
    blurred
        Where the picture has little of the tissue's grain.

    Returns
    -------
    np.ndarray
        The mask without that glow.
    """
    soft = ((mask > 0) & blurred).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        soft, connectivity=8
    )
    marks = np.asarray(labels, dtype=np.int64)
    at_edge = cast("list[int]", np.unique(marks[edge]).tolist())
    dropped = np.zeros(count, np.bool_)
    dropped[at_edge] = True
    dropped[0] = False
    glow = cv2.dilate(dropped[marks].astype(np.uint8), disk(EDGE_GLOW_RIM))
    return np.where(np.asarray(glow, dtype=np.uint8) > 0, 0, mask).astype(
        np.uint8
    )


def without_threads(
    mask: npt.NDArray[np.uint8], edge: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint8]:
    """Take off a mask what is thin and reaches the edge of the scan.

    The scanner's bright line along that edge is sharp where it is not
    smeared. Thin tissue away from that edge stays.

    Parameters
    ----------
    mask
        One on the mask, zero off it.
    edge
        The band along the edge of the scan.

    Returns
    -------
    np.ndarray
        The mask without those threads.
    """
    thick = np.asarray(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(THREAD)), dtype=np.uint8
    )
    foot = np.asarray(cv2.dilate(thick, disk(THREAD_FRINGE)), dtype=np.uint8)
    thin = ((mask > 0) & (foot < 1)).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        thin, connectivity=8
    )
    marks = np.asarray(labels, dtype=np.int64)
    at_edge = cast("list[int]", np.unique(marks[edge]).tolist())
    dropped = np.zeros(count, np.bool_)
    dropped[at_edge] = True
    dropped[0] = False
    return np.where(dropped[marks], 0, mask).astype(np.uint8)


def scan_edge(picture: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """Find the band along the edge of what the scanner covered.

    Parameters
    ----------
    picture
        The picture's one channel, 0 to 255.

    Returns
    -------
    np.ndarray
        True within ``SCAN_BAND`` of the picture's border or of the black
        ground the scanner left around its tiles.
    """
    height, width = cast("tuple[int, int]", picture.shape)
    band = max(1, round(SCAN_BAND * min(height, width)))
    frame = np.ones(picture.shape, np.bool_)
    frame[1:-1, 1:-1] = False
    black = (picture <= UNSCANNED).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        black, connectivity=8
    )
    marks = np.asarray(labels, dtype=np.int64)
    outside = cast("list[int]", np.unique(marks[frame]).tolist())
    around = np.zeros(count, np.bool_)
    around[outside] = True
    around[0] = False
    ground = (around[marks] | frame).astype(np.uint8)
    return np.asarray(cv2.dilate(ground, disk(band)), dtype=np.uint8) > 0


def disk(radius: int) -> npt.NDArray[np.uint8]:
    """Make a disk-shaped structuring element.

    Parameters
    ----------
    radius
        The disk's radius in pixels.

    Returns
    -------
    np.ndarray
        One inside the disk, a square of side ``2 * radius + 1``.
    """
    side = 2 * radius + 1
    return np.asarray(
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side)),
        dtype=np.uint8,
    )


def fill_holes(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Fill the holes of a binary mask: what the outside cannot reach.

    Parameters
    ----------
    mask
        One on the mask, zero off it.

    Returns
    -------
    np.ndarray
        The mask with its enclosed background set to one.
    """
    height, width = cast("tuple[int, int]", mask.shape)
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    seeds = np.zeros((height + 4, width + 4), np.uint8)
    _ = cv2.floodFill(padded, seeds, (0, 0), 1)
    outside = padded[1:-1, 1:-1] - mask
    return (1 - outside).astype(np.uint8)
