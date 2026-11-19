"""The light look's pictures: a section on paper, as a brightfield slide reads.

The light look (``<html data-theme="light">``) sets the sections on a paper
ground instead of the dark stage. A brightfield section (ISH, Nissl, DAB:
dark stain on a bright slide, :func:`is_brightfield`) is drawn in its own
colours, the slide's light balanced to the paper; a darkfield section
(fluorescence) is drawn from its main stain as ink on paper, so a nuclear
stain reads as a Nissl does, its tissue cut out of the dark slide
(:func:`.calm.tissue_mask`). The brightfield slide is the paper itself, only
the ground ``look`` leaves around the section is filled in. The atlas
borders are a soft ink layer over it, found as the difference between
``look``'s picture with and without them. Read-only like every drawing of
the window.

``PAPER`` is the light look's picture ground (``--stage`` there), ``INK``
the darkfield stain's colour at full strength and ``LINE`` the atlas
borders' colour, laid over the tissue at ``LINE_STRENGTH``. A picture is
brightfield by :func:`.calm.is_brightfield`. ``SLIDE_LEVEL`` is the percentile
of the slide's pixels taken as its white; a pixel under ``STAIN_FLOOR`` of
darkness is the bare slide and is drawn as paper, the stain coming in over
``STAIN_KNEE``, so a slide's faint tint and its tiles' shading go.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from PIL import Image

from langslice.core.pictures.look import LookRequest, look
from langslice.core.sections.images import raw_names
from langslice.hosts.gui.calm import (
    UNSCANNED,
    is_brightfield,
    main_channel,
    scale_bar,
    tissue_mask,
)
from langslice.hosts.gui.drawing import (
    content,
    draw_thumb,
    raw_channels,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.hosts.gui.drawing import Opened

PAPER = (251, 250, 247)
INK = (38, 40, 58)
LINE = (44, 52, 120)
LINE_STRENGTH = 0.7
SLIDE_LEVEL = 92
STAIN_FLOOR = 0.03
STAIN_KNEE = 0.12


def draw_paper_overlay(opened: Opened, state: Stack, section: Section, edge: int) -> Image.Image:
    """Draw a section under its current registration, on paper.

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
        The section on the paper ground under its transform and applied
        deformation, the atlas borders soft ink over it.
    """
    names = raw_channels(opened, section)
    bare, lined = _with_and_without(opened, state, section, names, edge)
    if is_brightfield(bare):
        return paper(bare, lined=lined)
    channel = (main_channel(raw_names(opened.workspace, section.id)),)
    bare, lined = _with_and_without(opened, state, section, channel, edge)
    return paper(bare, lined=lined, brightfield=False)


def draw_paper_thumb(opened: Opened, section: Section, edge: int) -> Image.Image:
    """Draw a section alone, oriented, on paper.

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
        The section turned and flipped as it is kept, on the paper ground.
    """
    picture = draw_thumb(opened, section, edge)
    return paper(picture, brightfield=is_brightfield(picture))


def paper(
    bare: Image.Image,
    *,
    lined: Image.Image | None = None,
    brightfield: bool = True,
) -> Image.Image:
    """Set one picture's tissue on paper, with its borders when given.

    Parameters
    ----------
    bare
        The section without atlas borders: its own colours when
        brightfield, its main stain in grey when not.
    lined
        The same picture with the borders, the same size; None: no borders.
    brightfield
        Whether the stain is dark on a bright slide.

    Returns
    -------
    Image.Image
        The tissue on the paper ground, the borders a soft ink layer.
    """
    rgb = np.asarray(bare.convert("RGB")).astype(np.float32)
    grey = np.asarray(bare.convert("L")).astype(np.float32)
    ground = cast("npt.NDArray[np.bool_]", np.max(rgb, axis=2) <= UNSCANNED)
    paper_rgb = np.array(PAPER, np.float32)
    if brightfield:
        slide = rgb[~ground]
        white = np.percentile(slide, SLIDE_LEVEL, axis=0) if slide.size else np.full(3, 255.0)
        balanced = np.clip(rgb / np.maximum(white, 1.0), 0, 1)
        stain = 1.0 - cast("npt.NDArray[np.float32]", np.min(balanced, axis=2))
        ramp = np.clip((stain - STAIN_FLOOR) / STAIN_KNEE, 0, 1)
        strength = (ramp * ramp * (3 - 2 * ramp))[..., None]
        tissue = paper_rgb * (1 - strength) + balanced * paper_rgb * strength
        mask = (~ground).astype(np.float32)
    else:
        grey[scale_bar(grey)] = 0.0
        mask = tissue_mask(grey)
        inside = grey[mask > 0.5]
        low = float(np.percentile(inside, 4)) if inside.size else 0.0
        high = float(np.percentile(inside, 99)) if inside.size else 255.0
        value = np.clip((grey - low) / max(1.0, high - low), 0, None)
        value = np.clip(1.35 * value / (1 + 0.35 * value), 0, 1) ** 0.95
        ink = (0.88 * value)[..., None]
        tissue = paper_rgb * (1 - ink) + np.array(INK, np.float32) * ink
    weight = mask[..., None]
    out = tissue * weight + paper_rgb * (1 - weight)
    if lined is not None:
        with_lines = np.asarray(lined.convert("RGB"), dtype=np.int16)
        plain = np.asarray(bare.convert("RGB"), dtype=np.int16)
        change = cast("npt.NDArray[np.int16]", np.max(np.abs(with_lines - plain), axis=2))
        weight_of_line = LINE_STRENGTH * min(1.0, 0.45 + max(bare.size) / 1600)
        alpha = (np.clip(change / 110.0, 0, 1) * weight_of_line)[..., None]
        out = out * (1 - alpha) + np.array(LINE, np.float32) * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _with_and_without(
    opened: Opened,
    state: Stack,
    section: Section,
    images: tuple[str, ...],
    edge: int,
) -> tuple[Image.Image, Image.Image]:
    """Draw a section's overlay without and with the atlas borders."""
    pictures: list[Image.Image] = []
    for layers in ((), None):
        request = LookRequest(
            mode="overlay",
            sections=(section.id,),
            images=images,
            long_edge=edge,
            atlas_layers=layers,
        )
        looks = look(opened.workspace, state, request, store=opened.store)
        pictures.append(content(looks[0]))
    return pictures[0], pictures[1]
