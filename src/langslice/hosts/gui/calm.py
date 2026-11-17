"""The Simple view's pictures: a section from its main stain alone, quiet.

The Extended view draws a section's raw channels in their own colours
(:mod:`.drawing`). The Simple view draws one channel, the nuclear stain
(:func:`main_channel`), in a cool grey on the window's dark stage: the
tissue is cut out of the scanner's tile floor (:func:`tissue_mask`), its
brightest spots are compressed by a soft knee instead of clipping, and the
atlas borders are a soft light layer over it, found as the difference
between ``look``'s picture with and without them (so the scale bar, drawn
in both, goes). Read-only like every drawing of the window.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813 - SimpleITK's own spelling
from PIL import Image, ImageFilter

from langslice.core.pictures.look import LookRequest, look
from langslice.core.sections.images import raw_names
from langslice.core.sections.renders import render_slice
from langslice.hosts.gui.drawing import content

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.hosts.gui.drawing import Opened

#: Names that say a channel is the nuclear (or Nissl) stain.
NUCLEAR = re.compile(r"dapi|hoechst|nuc|nissl|draq|topro", re.IGNORECASE)
#: The stage the pictures sit on: the window's dark ground (``--stage``).
STAGE = (18, 18, 19)
#: The tissue's tint at full brightness, a cool off-white.
TINT = (206, 210, 232)
#: The atlas borders' colour and how strongly they are laid over the tissue.
LINE = (232, 234, 250)
LINE_STRENGTH = 0.55
#: Where ``look`` draws its scale bar (``captions._draw_scale_bar``): its row
#: from the bottom, its left end, its grey, the shortest bar it draws and the
#: label's height above it.
BAR_FROM_BOTTOM = 14
BAR_LEFT = 12
BAR_GREY = 200
MIN_BAR = 4
BAR_LABEL = 22
#: The tissue mask is found on a copy at most this many pixels long.
MASK_EDGE = 512


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
        applied deformation, the atlas borders soft over it; a section
        with no position alone, oriented, without borders.
    """
    channel = (main_channel(raw_names(opened.workspace, section.id)),)
    if section.position_mm is None:
        bare = look(
            opened.workspace,
            state,
            LookRequest(
                mode="section",
                sections=(section.id,),
                images=channel,
                long_edge=edge,
            ),
            store=opened.store,
        )
        return calm(content(bare[0]))
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
        with_lines = np.asarray(lined.convert("RGB")).astype(np.int16)
        plain = np.asarray(bare.convert("RGB")).astype(np.int16)
        alpha = np.clip(np.abs(with_lines - plain).max(axis=2) / 110.0, 0, 1)
        # Small pictures get fainter borders, else they cover the tissue.
        strength = LINE_STRENGTH * min(1.0, 0.45 + max(bare.size) / 1600)
        alpha = (alpha * strength)[..., None]
        rgb = rgb * (1 - alpha) + np.array(LINE, np.float32) * alpha
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def scale_bar(grey: np.ndarray) -> tuple[slice, slice]:
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
    height, width = grey.shape
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


def tissue_mask(grey: np.ndarray) -> np.ndarray:
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
    height, width = grey.shape
    scale = min(1.0, MASK_EDGE / max(height, width))
    small = grey
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = np.asarray(
            Image.fromarray(grey.astype(np.uint8)).resize(
                size, Image.Resampling.BILINEAR
            )
        ).astype(np.float32)
    image = sitk.GetImageFromArray(small)
    smooth = sitk.GetArrayFromImage(
        sitk.SmoothingRecursiveGaussian(image, 5.0 * scale + 1.0)
    )
    values = smooth[smooth > 4]
    floor = float(np.percentile(values, 30)) if values.size else 8.0
    bright = smooth > max(12.0, floor * 1.6)
    # Tissue is textured (nuclei); a tile floor is smooth, however bright.
    fine = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(image, 1.0))
    texture = sitk.GetArrayFromImage(
        sitk.SmoothingRecursiveGaussian(
            sitk.GetImageFromArray(np.abs(small - fine)), 4.0 * scale + 1.0
        )
    )
    level = (
        max(0.6, 0.12 * float(np.median(texture[bright])))
        if bright.any()
        else 0.6
    )
    binary = sitk.GetImageFromArray(
        (bright & (texture > level)).astype(np.uint8)
    )
    parts = sitk.RelabelComponent(
        sitk.ConnectedComponent(binary),
        minimumObjectSize=max(4, int(small.size * 0.004)),
    )
    keep = sitk.Cast(parts > 0, sitk.sitkUInt8)
    keep = sitk.BinaryFillhole(sitk.BinaryMorphologicalClosing(keep, [3, 3]))
    keep = sitk.BinaryDilate(keep, [2, 2])
    weight = Image.fromarray(
        sitk.GetArrayFromImage(keep).astype(np.uint8) * 255
    )
    if scale < 1.0:
        weight = weight.resize((width, height), Image.Resampling.BILINEAR)
    weight = weight.filter(ImageFilter.GaussianBlur(2.0))
    return np.asarray(weight).astype(np.float32) / 255.0
