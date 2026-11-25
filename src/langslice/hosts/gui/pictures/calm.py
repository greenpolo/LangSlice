"""The Simple view's pictures: a section from its main stain alone, quiet.

The Extended view draws a section's raw channels in their own colours
(:mod:`.drawing`). The Simple view draws one channel, the nuclear stain
(:func:`.drawing.main_channel`), in a cool grey on the window's dark stage:
the tissue is cut out of the scanner's tile floor
(:func:`langslice.core.sections.tissue.tissue_mask`), its brightest spots are
compressed by the shared tone curve (:func:`.inks.tone`), and the atlas
borders are a soft light layer over it (:func:`.inks.lay_borders`), from the
core's layers: the section alone, and the borders' coverage. A brightfield
section is read by its darkness under the slide's white
(:func:`langslice.core.sections.tissue.stain_of`). The picture has no scale
bar and no caption. Read-only like every drawing of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from langslice.core.sections.images import raw_names
from langslice.core.sections.tissue import stain_of, tissue_alpha, tissue_mask
from langslice.hosts.gui.pictures.cutout import cut_out
from langslice.hosts.gui.pictures.drawing import (
    border_lines,
    main_channel,
    placed_layers,
    render_channels,
)
from langslice.hosts.gui.pictures.inks import BorderInk, Rgb, lay_borders, tone

if TYPE_CHECKING:
    import numpy.typing as npt

    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.hosts.gui.pictures.drawing import Opened


@dataclass(frozen=True)
class Stage:
    """The dark look's colours.

    *ground* is the window's dark stage (``--stage``) the pictures sit on,
    *tint* the tissue's cool off-white at full brightness, *line_colour* the
    atlas borders' colour and *line_strength* how strongly they are laid over
    the tissue.
    """

    ground: Rgb = (18, 18, 19)
    tint: Rgb = (206, 210, 232)
    line_colour: Rgb = (232, 234, 250)
    line_strength: float = 0.55

    @property
    def ink(self) -> BorderInk:
        """The borders' ink on this stage."""
        return BorderInk(self.line_colour, self.line_strength)


STAGE = Stage()


def draw_calm_overlay(opened: Opened, state: Stack, section: Section, edge: int) -> Image.Image:
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
    layers = placed_layers(opened, state, section, edge, channel)
    return calm(layers.section, border_lines(layers))


def draw_calm_thumb(opened: Opened, section: Section, edge: int) -> Image.Image:
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
    return calm(_source(opened, section, edge), None)


def draw_calm_cutout(opened: Opened, section: Section, edge: int) -> Image.Image:
    """Draw a section's quiet thumbnail cut out of its ground.

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
    source = _source(opened, section, edge)
    return cut_out(calm(source, None), tissue_alpha(source))


def calm(section: Image.Image, lines: npt.NDArray[np.float32] | None) -> Image.Image:
    """Make one channel's picture quiet, with its borders when given.

    Parameters
    ----------
    section
        The section without atlas borders, one channel in grey.
    lines
        The borders' coverage on the same grid; None: no borders.

    Returns
    -------
    Image.Image
        The tissue tinted on the stage, the borders a soft light layer.
    """
    stain = stain_of(section)
    weight = tissue_mask(stain.signal)
    ground = np.array(STAGE.ground, np.float32)
    tissue = tone(stain.signal, weight)[..., None] * np.array(STAGE.tint, np.float32)
    cover = weight[..., None]
    rgb = np.maximum(tissue * cover + ground * (1 - cover), ground)
    rgb = lay_borders(rgb, lines, STAGE.ink)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _source(opened: Opened, section: Section, edge: int) -> Image.Image:
    """The section's main stain, oriented, at *edge*."""
    channel = main_channel(raw_names(opened.workspace, section.id))
    return render_channels(opened, section, edge, (channel,))
