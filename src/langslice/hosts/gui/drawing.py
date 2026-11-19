"""The window's own pictures of a job, drawn from the core's layers.

A section under its current registration (the core's
:class:`~langslice.core.pictures.canvas.PlacementLayers`: the section on the
canvas, the atlas borders' coverage, no caption and no scale bar burned in), a
section's thumbnail and an atlas thumbnail. A look (:mod:`.calm`,
:mod:`.paper`, or here the raw one) is a function from those layers to the
window's finished picture. :mod:`.render` queues and keeps them. The window
shows a section as its file holds it, its raw channels in their own colours,
not the ``preprocessed`` image the fits read: that image's contrast
enhancement makes a scanner's tile mosaic stand out.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Protocol

import numpy as np
from PIL import Image

from langslice.core.fits.deformation import RecordStore
from langslice.core.pictures.canvas import PlacementLayers, bordered_section
from langslice.core.pictures.captions import draw_scale_bar
from langslice.core.pictures.look import LookRequest, atlas_image, overlay_layers
from langslice.core.pictures.options import MAX_OVERLAY_CHANNELS
from langslice.core.sections.images import raw_names
from langslice.core.sections.renders import render_slice
from langslice.core.sections.tissue import tissue_alpha
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace
from langslice.hosts.gui.cutout import cut_out

NUCLEAR = re.compile(r"dapi|hoechst|nuc|nissl|draq|topro", re.IGNORECASE)


class Style(StrEnum):
    """How a section's picture is drawn: raw channels, quiet, or on paper."""

    RAW = "raw"
    CALM = "calm"
    PAPER = "paper"


class Opened(Protocol):
    """A job read for drawing: its workspace and its deformation records."""

    @property
    def workspace(self) -> Workspace:
        """Hold the job's images, atlas and caches."""
        ...

    @property
    def store(self) -> RecordStore:
        """Hold the sections' deformation records."""
        ...


def raw_channels(opened: Opened, section: Section) -> tuple[str, ...]:
    """Name the channels the window draws a section from.

    Parameters
    ----------
    opened
        The job, read for drawing.
    section
        The section to draw.

    Returns
    -------
    tuple of str
        The section's raw channels, at most as many as are drawn together.
    """
    names = raw_names(opened.workspace, section.id)
    return tuple(names[:MAX_OVERLAY_CHANNELS])


def main_channel(names: list[str]) -> str:
    """Choose the channel the Simple and light looks draw a fluorescent section from.

    Parameters
    ----------
    names
        The section's raw channels, in file order.

    Returns
    -------
    str
        A channel named for a nuclear or Nissl stain (``NUCLEAR``); else an
        RGB file's ``blue`` (DAPI's colour); else the first.
    """
    for name in names:
        if NUCLEAR.search(name):
            return name
    return "blue" if "blue" in names else names[0]


def placed_layers(
    opened: Opened,
    state: Stack,
    section: Section,
    edge: int,
    channels: tuple[str, ...],
) -> PlacementLayers:
    """Draw a section's layers under its current registration.

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
    channels
        The raw channels drawn together (one is drawn in grey).

    Returns
    -------
    PlacementLayers
        The section on the canvas under its transform and applied
        deformation, and the atlas borders' coverage: no borders, caption or
        scale bar in the pixels.
    """
    request = LookRequest(mode="overlay", sections=(section.id,), images=channels, long_edge=edge)
    placed = overlay_layers(opened.workspace, state, section, request, store=opened.store)
    return placed.canvas.layers


def border_lines(layers: PlacementLayers) -> np.ndarray:
    """The atlas borders' coverage of *layers*, 0 to 1, on the section's grid.

    The context lines and the highlighted regions' outlines, before colour
    and casing.
    """
    return np.maximum(layers.borders, layers.highlighted)


def render_channels(
    opened: Opened, section: Section, edge: int, channels: tuple[str, ...]
) -> Image.Image:
    """Draw a section alone, oriented, from raw channels.

    Parameters
    ----------
    opened
        The job, read for drawing.
    section
        The section to draw.
    edge
        The picture's long edge in pixels, at most the working copy's.
    channels
        The raw channels; several are drawn together in their colours, one in
        grey.

    Returns
    -------
    Image.Image
        The section as its file holds it, turned and flipped as it is kept.
    """
    raw: dict[str, object] = {"overlay": list(channels)}
    if len(channels) == 1:
        raw = {"image": channels[0]}
    return render_slice(opened.workspace, section, long_edge=edge, look=raw)


def draw_thumb(opened: Opened, section: Section, edge: int) -> Image.Image:
    """Draw a section alone, oriented, from its raw channels.

    Parameters
    ----------
    opened
        The job, read for drawing.
    section
        The section to draw.
    edge
        The picture's long edge in pixels, at most the working copy's.

    Returns
    -------
    Image.Image
        The section as its file holds it, turned and flipped as it is kept.
    """
    return render_channels(opened, section, edge, raw_channels(opened, section))


def draw_cutout(opened: Opened, section: Section, edge: int) -> Image.Image:
    """Draw a section's raw thumbnail cut out of its ground.

    Parameters
    ----------
    opened
        The job, read for drawing.
    section
        The section to draw.
    edge
        The picture's long edge in pixels, at most the working copy's.

    Returns
    -------
    Image.Image
        RGBA: the thumbnail, its alpha the tissue.
    """
    picture = draw_thumb(opened, section, edge)
    return cut_out(picture, tissue_alpha(picture))


def draw_overlay(opened: Opened, state: Stack, section: Section, edge: int) -> Image.Image:
    """Draw a section under its current registration, in its raw channels.

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
        The section's raw channels under its transform and applied
        deformation with the atlas borders and the 1 mm scale bar, without
        the caption.
    """
    layers = placed_layers(opened, state, section, edge, raw_channels(opened, section))
    screen = bordered_section(layers)
    draw_scale_bar(screen, layers.scale_bar)
    return Image.fromarray(screen, "RGB")


def draw_atlas(opened: Opened, state: Stack, position_mm: float, edge: int) -> Image.Image:
    """Draw the atlas plane at a position, at the stack's view angles.

    Parameters
    ----------
    opened
        The job, read for drawing.
    state
        The job's stack as the viewer reads it.
    position_mm
        The position along the cutting axis, in atlas millimetres.
    edge
        The picture's long edge in pixels.

    Returns
    -------
    Image.Image
        The atlas template there.
    """
    return atlas_image(opened.workspace, state, position_mm, edge)
