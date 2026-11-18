"""The window's own pictures of a job, drawn by the core's renderers.

A section under its current registration, a section's thumbnail and an
atlas thumbnail, each without its caption band. :mod:`.render` queues and
keeps them. The window shows a section as its file holds it, its raw
channels in their own colours, not the ``preprocessed`` image the fits read:
that image's contrast enhancement makes a scanner's tile mosaic stand out.
"""

from __future__ import annotations

from typing import Protocol, cast

from PIL import Image

from langslice.core.fits.deformation import RecordStore
from langslice.core.pictures.look import LookPicture, LookRequest, look
from langslice.core.pictures.options import MAX_OVERLAY_CHANNELS
from langslice.core.sections.images import raw_names
from langslice.core.sections.renders import render_slice
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace


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
    names = raw_channels(opened, section)
    raw: dict[str, object] = {"overlay": list(names)}
    if len(names) == 1:
        raw = {"image": names[0]}
    return render_slice(opened.workspace, section, long_edge=edge, look=raw)


def draw_overlay(
    opened: Opened, state: Stack, section: Section, edge: int
) -> Image.Image:
    """Draw a section under its current registration.

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
        deformation with the atlas borders.
    """
    request = LookRequest(
        mode="overlay",
        sections=(section.id,),
        images=raw_channels(opened, section),
        long_edge=edge,
    )
    pictures = look(opened.workspace, state, request, store=opened.store)
    return content(pictures[0])


def draw_atlas(
    opened: Opened, state: Stack, position_mm: float, edge: int
) -> Image.Image:
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
    request = LookRequest(
        mode="atlas", positions_mm=(position_mm,), long_edge=edge
    )
    return content(look(opened.workspace, state, request)[0])


def content(picture: LookPicture) -> Image.Image:
    """Cut a picture's caption band off.

    Parameters
    ----------
    picture
        One of ``look``'s pictures.

    Returns
    -------
    Image.Image
        Its content: the recipe's content size, at the top left.
    """
    width, height = cast("list[int]", picture.recipe["shown"])
    return picture.image.crop((0, 0, width, height))
