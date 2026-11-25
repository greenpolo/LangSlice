"""A section's picture as three parts the page lays over one another.

The viewer shows what the agent does to a section by moving one part of its
picture and not the others: the atlas borders breathe while a fit runs over
still tissue, a damage mark is drawn over both. So the window's overlay is
also served as parts, drawn once from the core's layers
(:func:`.drawing.placed_layers`) in the window's look (:mod:`.calm` or
:mod:`.paper`):

- ``tissue``: the section on the stage or the paper, no borders;
- ``borders``: the atlas borders alone, their ink at their coverage, the rest
  clear;
- ``marks``: what the job has marked on the section, the rest clear: the
  regions marked damaged (:func:`langslice.core.sections.damage.marked_area`)
  hatched in a warm ink and edged.

Laid in that order the three are the look's overlay. Read-only like every
drawing of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from langslice.core.sections.damage import marked_area
from langslice.core.sections.images import raw_names
from langslice.core.sections.tissue import is_brightfield
from langslice.hosts.gui.pictures.calm import STAGE, calm
from langslice.hosts.gui.pictures.drawing import (
    Style,
    border_lines,
    main_channel,
    placed_layers,
    raw_channels,
)
from langslice.hosts.gui.pictures.paper import PAGE, paper

if TYPE_CHECKING:
    import numpy.typing as npt

    from langslice.core.pictures.canvas import PlacementLayers
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.hosts.gui.pictures.drawing import Opened
    from langslice.hosts.gui.pictures.inks import BorderInk, Rgb


class Part(StrEnum):
    """One part of a section's overlay."""

    TISSUE = "tissue"
    BORDERS = "borders"
    MARKS = "marks"


@dataclass(frozen=True)
class Hatch:
    """How a damage mark is drawn on the window's pictures.

    Diagonal lines *spacing* pixels apart (at a 1024 px picture; in
    proportion on others, never under *min_spacing*), *width* of them inked,
    in *colour* at *alpha*, and the mark's edge in the same colour at
    *edge_alpha*.
    """

    colour: Rgb
    alpha: float = 0.7
    edge_alpha: float = 0.95
    spacing: float = 9.0
    width_share: float = 0.3
    min_spacing: int = 5
    reference_px: float = 1024.0


DARK_HATCH = Hatch((244, 132, 104))
PAPER_HATCH = Hatch((190, 64, 44))


@dataclass(frozen=True)
class Parts:
    """A section's overlay in parts: RGB tissue, RGBA borders and marks."""

    tissue: Image.Image
    borders: Image.Image
    marks: Image.Image

    def part(self, which: Part) -> Image.Image:
        """The part *which*."""
        return {Part.TISSUE: self.tissue, Part.BORDERS: self.borders, Part.MARKS: self.marks}[which]


def draw_parts(
    opened: Opened, state: Stack, section: Section, edge: int, style: Style, image: str = ""
) -> Parts:
    """Draw a section under its current registration, in parts.

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
    style
        ``paper`` for the light look, else the dark (``calm``) one.
    image
        A made image of the section (``make_image``'s name) to draw the
        tissue from, in grey; blank for its raw channels.

    Returns
    -------
    Parts
        The tissue, the borders and the marks on one pixel grid.
    """
    main = (image or main_channel(raw_names(opened.workspace, section.id)),)
    if style == Style.PAPER and image:
        layers = placed_layers(opened, state, section, edge, main)
        tissue = paper(layers.section, None, brightfield=False)
        ink, hatch = PAGE.lines, PAPER_HATCH
    elif style == Style.PAPER:
        layers = placed_layers(opened, state, section, edge, raw_channels(opened, section))
        bright = is_brightfield(layers.section)
        if not bright:
            layers = placed_layers(opened, state, section, edge, main)
        tissue = paper(layers.section, None, brightfield=bright)
        ink, hatch = PAGE.lines, PAPER_HATCH
    else:
        layers = placed_layers(opened, state, section, edge, main)
        tissue = calm(layers.section, None)
        ink, hatch = STAGE.ink, DARK_HATCH
    borders = border_layer(border_lines(layers), ink)
    marks = mark_layer(_marked(opened, state, section, layers), hatch)
    return Parts(tissue, borders, marks)


def border_layer(lines: npt.NDArray[np.float32], ink: BorderInk) -> Image.Image:
    """The borders alone: *ink*'s colour, its alpha the lines' strength.

    Laid over the tissue this is :func:`.inks.lay_borders`'s blend.
    """
    size = max(lines.shape[:2])
    strength = ink.strength * min(1.0, ink.small_floor + size / ink.full_size_px)
    alpha = np.clip(lines / ink.full_coverage, 0, 1) * strength
    rgba = np.zeros((*lines.shape[:2], 4), np.uint8)
    rgba[..., :3] = ink.colour
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def mark_layer(area: npt.NDArray[np.bool_], hatch: Hatch) -> Image.Image:
    """The marked *area* hatched and edged in *hatch*'s ink, the rest clear."""
    height, width = area.shape
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = hatch.colour
    if not area.any():
        return Image.fromarray(rgba, "RGBA")
    spacing = max(hatch.min_spacing, round(hatch.spacing * max(height, width) / hatch.reference_px))
    rows, cols = np.indices(area.shape)
    lines = area & (((rows + cols) % spacing) < max(1.0, spacing * hatch.width_share))
    inner = np.asarray(binary_erosion(area, iterations=max(1, spacing // 4)), dtype=bool)
    rim = area & ~inner
    alpha = np.where(rim, hatch.edge_alpha, np.where(lines, hatch.alpha, 0.0))
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _marked(
    opened: Opened, state: Stack, section: Section, layers: PlacementLayers
) -> npt.NDArray[np.bool_]:
    """The section's damage marks on the layers' pixel grid."""
    height, width = layers.borders.shape[:2]
    if not section.damaged:
        return np.zeros((height, width), dtype=bool)
    panel = layers.frame
    area = marked_area(opened.workspace, state, section, panel)
    x0, y0, _x1, _y1 = panel.content_box
    return area[y0 : y0 + height, x0 : x0 + width]
