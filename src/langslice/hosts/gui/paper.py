"""The light look's pictures: a section on paper, as a brightfield slide reads.

The light look (``<html data-theme="light">``) sets the sections on a paper
ground instead of the dark stage. A brightfield section (ISH, Nissl, DAB:
dark stain on a bright slide,
:func:`langslice.core.sections.tissue.is_brightfield`) is drawn in its own
colours, the slide's light balanced to the paper; a darkfield section
(fluorescence) is drawn from its main stain as ink on paper, so a nuclear
stain reads as a Nissl does, its tissue cut out of the dark slide
(:func:`langslice.core.sections.tissue.tissue_mask`) and its brightness from
the shared tone curve (:func:`.inks.tone`). The brightfield slide is the
paper itself, only the ground the canvas leaves around the section is filled
in. The atlas borders are a soft ink layer (:func:`.inks.lay_borders`) from
the core's layers. The picture has no scale bar and no caption. Read-only
like every drawing of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from PIL import Image

from langslice.core.sections.images import raw_names
from langslice.core.sections.tissue import (
    SLIDE,
    is_brightfield,
    stain_of,
    tissue_alpha,
    tissue_mask,
)
from langslice.hosts.gui.cutout import cut_out
from langslice.hosts.gui.drawing import (
    border_lines,
    main_channel,
    placed_layers,
    raw_channels,
    render_channels,
)
from langslice.hosts.gui.inks import GREY_MAX, BorderInk, Rgb, lay_borders, tone

if TYPE_CHECKING:
    import numpy.typing as npt

    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.hosts.gui.drawing import Opened

SMOOTHSTEP_CUBIC = 3.0


@dataclass(frozen=True)
class Page:
    """The light look's colours and stain curve.

    *ground* is the light look's picture ground (``--stage`` there), *ink*
    the darkfield stain's colour at full strength laid at *ink_strength*,
    *line_colour* the atlas borders' colour at *line_strength*. A brightfield
    pixel whose darkness under the slide's white is under *stain_floor* is the
    bare slide and is drawn as paper; the stain comes in over *stain_knee*
    more, so a slide's faint tint and its tiles' shading go.
    """

    ground: Rgb = (251, 250, 247)
    ink: Rgb = (38, 40, 58)
    ink_strength: float = 0.88
    line_colour: Rgb = (44, 52, 120)
    line_strength: float = 0.7
    stain_floor: float = 0.03
    stain_knee: float = 0.12

    @property
    def lines(self) -> BorderInk:
        """The borders' ink on this paper."""
        return BorderInk(self.line_colour, self.line_strength)


PAGE = Page()


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
    layers = placed_layers(opened, state, section, edge, raw_channels(opened, section))
    if is_brightfield(layers.section):
        return paper(layers.section, border_lines(layers), brightfield=True)
    channel = (main_channel(raw_names(opened.workspace, section.id)),)
    layers = placed_layers(opened, state, section, edge, channel)
    return paper(layers.section, border_lines(layers), brightfield=False)


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
    source, brightfield = _source(opened, section, edge)
    return paper(source, None, brightfield=brightfield)


def draw_paper_cutout(opened: Opened, section: Section, edge: int) -> Image.Image:
    """Draw a section's paper thumbnail cut out of its ground.

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
        RGBA: the thumbnail, its alpha the tissue.
    """
    source, brightfield = _source(opened, section, edge)
    return cut_out(paper(source, None, brightfield=brightfield), tissue_alpha(source))


def paper(
    section: Image.Image,
    lines: npt.NDArray[np.float32] | None,
    *,
    brightfield: bool,
) -> Image.Image:
    """Set one picture's tissue on paper, with its borders when given.

    Parameters
    ----------
    section
        The section without atlas borders: its own colours when brightfield,
        its main stain in grey when not.
    lines
        The borders' coverage on the same grid; None: no borders.
    brightfield
        Whether the stain is dark on a bright slide.

    Returns
    -------
    Image.Image
        The tissue on the paper ground, the borders a soft ink layer.
    """
    rgb = np.asarray(section.convert("RGB")).astype(np.float32)
    ground = cast("npt.NDArray[np.bool_]", np.max(rgb, axis=2) <= SLIDE.unscanned_level)
    paper_rgb = np.array(PAGE.ground, np.float32)
    if brightfield:
        stain = stain_of(section)
        balanced = stain.balanced if stain.balanced is not None else rgb / GREY_MAX
        ramp = np.clip((stain.signal / GREY_MAX - PAGE.stain_floor) / PAGE.stain_knee, 0, 1)
        strength = (ramp * ramp * (SMOOTHSTEP_CUBIC - 2 * ramp))[..., None]
        tissue = paper_rgb * (1 - strength) + balanced * paper_rgb * strength
        mask = (~ground).astype(np.float32)
    else:
        grey = np.asarray(section.convert("L")).astype(np.float32)
        mask = tissue_mask(grey)
        ink = (PAGE.ink_strength * tone(grey, mask))[..., None]
        tissue = paper_rgb * (1 - ink) + np.array(PAGE.ink, np.float32) * ink
    weight = mask[..., None]
    cut = (tissue * weight + paper_rgb * (1 - weight)).astype(np.float32)
    out = lay_borders(cut, lines, PAGE.lines)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _source(opened: Opened, section: Section, edge: int) -> tuple[Image.Image, bool]:
    """The section's thumbnail render and whether it is brightfield: its raw
    channels when so, else its main stain, oriented, at *edge*."""
    raw = render_channels(opened, section, edge, raw_channels(opened, section))
    if is_brightfield(raw):
        return raw, True
    channel = (main_channel(raw_names(opened.workspace, section.id)),)
    return render_channels(opened, section, edge, channel), False
