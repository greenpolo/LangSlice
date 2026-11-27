"""A section's picture as three parts the page lays over one another.

The viewer shows what the agent does to a section by moving one part of its
picture and not the others: the atlas borders breathe while a fit runs over
still tissue, a damage mark is drawn over both. So the window's overlay is
served as parts, drawn once from the core's layers
(:func:`.drawing.placed_layers`):

- ``tissue``: the section as its file holds it, clear where the image does
  not reach;
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
from langslice.core.sections.tissue import SLIDE
from langslice.hosts.gui.pictures.calm import STAGE
from langslice.hosts.gui.pictures.drawing import (
    Style,
    border_lines,
    placed_layers,
    raw_channels,
)
from langslice.hosts.gui.pictures.paper import PAGE

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
    REGIONS = "regions"


@dataclass(frozen=True)
class RegionInk:
    """The outline of regions the agent's words name: the window's accent
    (``--lavender`` in each look, *dark* on a bright slide, *light* on a dark
    one) at *strength*, full from *full_coverage* of a pixel."""

    dark: Rgb = (79, 91, 181)
    light: Rgb = (164, 173, 223)
    strength: float = 0.95
    full_coverage: float = 0.43


REGION_INK = RegionInk()


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
    """A section's overlay in parts, each RGBA on one pixel grid."""

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

    The tissue is the section as its file holds it (its raw channels in their
    own colours, or a made image), nothing cut away and no tone curve, as ABBA
    shows a section: only the canvas the image does not cover (the scanner's
    black, :data:`langslice.core.sections.tissue.SLIDE`'s unscanned level) is
    clear, so the window's ground (the job's slide ground,
    :func:`slide_ground`) shows there. The borders' ink is dark on a bright
    slide and light on a dark one.

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
        Unused: the pictures are the same in both looks (kept for the cache).
    image
        A made image of the section (``make_image``'s name) to draw the
        tissue from; blank for its raw channels.

    Returns
    -------
    Parts
        The tissue, the borders and the marks on one pixel grid.
    """
    channels = (image,) if image else raw_channels(opened, section)
    layers = placed_layers(opened, state, section, edge, channels)
    rgb = np.asarray(layers.section.convert("RGB"))
    ground = slide_ground(rgb)
    bright = sum(ground) / 3 > SLIDE.bright_median
    scanned = rgb.max(axis=2) > SLIDE.unscanned_level
    tissue = Image.fromarray(np.dstack([rgb, np.where(scanned, 255, 0)]).astype(np.uint8), "RGBA")
    ink, hatch = (PAGE.lines, PAPER_HATCH) if bright else (STAGE.ink, DARK_HATCH)
    borders = border_layer(border_lines(layers), ink)
    marks = mark_layer(_marked(opened, state, section, layers), hatch)
    return Parts(tissue, borders, marks)


def draw_regions(
    opened: Opened, state: Stack, section: Section, edge: int, regions: tuple[str, ...]
) -> Image.Image:
    """Draw the outlines of atlas regions over a section, alone.

    Parameters
    ----------
    opened
        The job, read for drawing.
    state
        The job's stack as the viewer reads it.
    section
        The section the regions are outlined on.
    edge
        The picture's long edge in pixels, as its other parts.
    regions
        The regions' names or acronyms.

    Returns
    -------
    Image.Image
        RGBA on the parts' pixel grid: the regions' outlines where the section
        meets them (the core's highlighted outlines, thicker than the borders),
        in :data:`REGION_INK`, the rest clear.
    """
    layers = placed_layers(
        opened, state, section, edge, raw_channels(opened, section), highlight=regions
    )
    ground = slide_ground(np.asarray(layers.section.convert("RGB")))
    bright = sum(ground) / 3 > SLIDE.bright_median
    coverage = np.asarray(layers.highlighted, np.float32)
    alpha = np.clip(coverage / REGION_INK.full_coverage, 0, 1) * REGION_INK.strength
    rgba = np.zeros((*coverage.shape[:2], 4), np.uint8)
    rgba[..., :3] = REGION_INK.dark if bright else REGION_INK.light
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def slide_ground(rgb: npt.NDArray[np.uint8]) -> Rgb:
    """The ground of a section's picture as its file holds it: what lies along
    the picture's outer edge (the median of its outermost :data:`SLIDE`
    ``rim_px`` pixels), the glass of a brightfield slide, the dark glass of a
    darkfield one, or the black a file was saved on.
    """
    rim = SLIDE.rim_px
    edge = np.concatenate(
        [
            rgb[:rim].reshape(-1, 3),
            rgb[-rim:].reshape(-1, 3),
            rgb[:, :rim].reshape(-1, 3),
            rgb[:, -rim:].reshape(-1, 3),
        ]
    )
    ground = np.median(edge, axis=0)
    return (int(ground[0]), int(ground[1]), int(ground[2]))


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
