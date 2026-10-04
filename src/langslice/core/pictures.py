"""The captioned pictures the viewing tools send, other than placements.

Plain PIL images with their labels burned in (tool images reach a model as
bare attachments, so every picture names itself). The doors package them:
the ADK tools through :mod:`langslice.adk.media`, MCP as image blocks.

The two reference pictures (a section tissue-framed, an atlas section) are
cached on the workspace (``Workspace.picture_cache``), keyed by everything
they draw, so a picture asked for again is not redrawn; a section's cached
caption keeps the index and flags of its first display, since filenames are
the stable identity.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from langslice.core.layers import note
from langslice.linear.appearance import Look
from langslice.linear.atlas_fetch import atlas_picture
from langslice.linear.display import (
    DisplayOptions,
    atlas_caption,
    channel_strip,
    framed_atlas,
    framed_section,
)
from langslice.linear.render import (
    caption,
    opening_edge,
    picture_edge,
    reference_slice_picture,
    render_cache_key,
    spacing_plot,
    stack_sheet,
)
from langslice.linear.state import SliceState, StackState
from langslice.linear.workspace import Workspace

logger = logging.getLogger(__name__)


def reference_section_picture(
    ws: Workspace, record: SliceState, *, long_edge: int | None = None, look: Look = None,
) -> Image.Image:
    """:func:`langslice.linear.render.reference_slice_picture`, cached per
    display state (orientation, look, size). The cached caption keeps the
    index and flags of its first display; *long_edge* None is the run's
    opening size, another size its own entry. Shared: read only."""
    long_edge = long_edge or opening_edge(ws)
    key = ("section", *render_cache_key(ws, record, long_edge=long_edge, frame=True, look=look))
    if key not in ws.picture_cache:
        ws.picture_cache[key] = reference_slice_picture(ws, record, long_edge=long_edge,
                                                        look=look)
    return ws.picture_cache[key]


def reference_atlas_picture(
    ws: Workspace, state: StackState, position_mm: float, *, long_edge: int | None = None,
) -> Image.Image:
    """:func:`langslice.linear.atlas_fetch.atlas_picture`, cached by position,
    plane, cutting angles and size. Shared: read only."""
    long_edge = long_edge or picture_edge(ws)
    key = ("atlas", state.plane, float(position_mm), state.pitch_deg, state.yaw_deg,
           int(long_edge))
    if key not in ws.picture_cache:
        ws.picture_cache[key] = atlas_picture(ws, state, position_mm, long_edge=long_edge)
    return ws.picture_cache[key]


def atlas_view_picture(
    ws: Workspace, state: StackState, position_mm: float, options: DisplayOptions,
) -> Image.Image:
    """The atlas alone at *position_mm*, tissue-framed and captioned.

    The default picture (``ara``, no lines, no zoom) is the cached reference;
    any other options draw it afresh with the call's lines and regions.
    """
    if options.atlas_images == ("ara",) and not options.lines and options.full_view:
        picture = reference_atlas_picture(ws, state, position_mm, long_edge=options.long_edge)
    else:
        picture = caption(framed_atlas(ws, state, position_mm, options),
                          atlas_caption(state, position_mm, options))
    return note(picture, mode="atlas", extra={"position_mm": float(position_mm)})


def section_label(record: SliceState, options: DisplayOptions) -> str:
    """``"<corrected index>: <filename>"`` plus what of the section is shown."""
    return f"{record.index_corrected}: {record.id}" + options.section_tag()


def channel_tile_edge(ws: Workspace, record: SliceState, options: DisplayOptions) -> int:
    """Each raw-channel tile's long edge: the strip is about one picture wide."""
    count = max(1, len(ws.section_channels(record.id)[0]))
    return max(128, min(opening_edge(ws), options.long_edge // min(count, 2)))


def section_picture(
    ws: Workspace, state: StackState, record: SliceState, options: DisplayOptions,
) -> Image.Image:
    """One section as corrected, tissue-framed, its index and id burned in.

    Mode ``channels``: the section's raw channels side by side instead, each
    unmodified and labelled with its name.
    """
    if options.mode == "channels":
        strip, _names = channel_strip(ws, state, record, options,
                                      tile_edge=channel_tile_edge(ws, record, options))
        return note(caption(strip, f"{record.index_corrected}: {record.id}  raw channels, "
                            "unmodified"), sections=(record.id,), mode="channels")
    return note(caption(framed_section(ws, state, record, options),
                        section_label(record, options)),
                sections=(record.id,), mode=options.mode)


def stack_review(
    ws: Workspace, state: StackState, options: DisplayOptions,
) -> tuple[Image.Image, Image.Image]:
    """``(sheet, plot)``: every section in written-position order, a placed
    one over the atlas at its position, captioned; then position against
    corrected index (:func:`langslice.linear.render.spacing_plot`)."""

    def atlas_under(record: SliceState) -> Any:
        if record.position_mm is None:
            return None
        try:
            return framed_atlas(ws, state, float(record.position_mm), options)
        except Exception as exc:
            logger.warning("view_stack: atlas render failed for %s: %s", record.id, exc)
            return None

    sheet = stack_sheet(
        state, ws, under=atlas_under,
        look=lambda record: options.look(state, record),
        tile_edge=options.resolution,
    )
    ids = tuple(record.id for record in state.in_order())
    return (note(sheet, sections=ids, mode="sheet"),
            note(spacing_plot(state), sections=ids, mode="spacing"))
