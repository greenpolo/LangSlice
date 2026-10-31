"""Picture sizes: the long edge of every picture the agent is shown, per level.

The working frame fits are computed on is
:data:`langslice.core.sections.renders.PREVIEW_LONG_EDGE`, never one of these.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

#: Long edge, in pixels, of every picture the agent is SHOWN, per picture
#: level (``ImageSettings.picture_level``): ``(opening, later)``. *Opening*
#: is each tile of the opening strips (every section, every atlas section
#: beneath it or in the atlas reference;
#: :mod:`langslice.core.pictures.opening`); *later* is each picture a tool
#: returns (each panel of a multi-panel picture). "auto" opens at 512 and lets
#: the agent pass ``resolution`` per call (:data:`MIN_RESOLUTION` up to the
#: driver model's own largest image, which the door passes in), 1024 when it
#: does not. The only other bound is the source: nothing is upsampled past
#: the pixels it is drawn from (a section's working copy, the atlas plane at
#: its own voxel size), so a small snapshot stays small. Never a working frame
#: (:data:`PREVIEW_LONG_EDGE`) and never the image model's input.
PICTURE_EDGES: dict[str, tuple[int, int]] = {
    "low": (256, 512),
    "medium": (384, 768),
    "high": (512, 1024),
    "auto": (512, 1024),
}
#: The level that lets the agent choose each picture's size.
AUTO_RESOLUTION = "auto"
#: Smallest long edge the agent may ask for at "auto". The largest is the
#: driver model's own maximum image (the door knows the model:
#: ``view_options.view_edge_limit``; the MCP host is Claude,
#: ``opening.CLAUDE_MAX_VIEW_EDGE``). A value outside is clamped and the
#: reply says so.
MIN_RESOLUTION = 128

#: Pictures (and, for the transform tools, sections) one call takes at the doors
#: that spend a model's context on them: LangSlice's own agent and MCP
#: (``Job.picture_cap``). The agent CLI and the library set no such cap.
MAX_IMAGES_PER_CALL = 4


def resolution_level(ctx: Workspace) -> str:
    """Return the size level of this run's pictures.

    Parameters
    ----------
    ctx
        The workspace whose settings hold the level.

    Returns
    -------
    str
        A key of :data:`PICTURE_EDGES`: the settings' level, ``low`` for one
        the table lacks.
    """
    level = ctx.settings.picture_level
    return level if level in PICTURE_EDGES else "low"


def opening_edge(ctx: Workspace) -> int:
    """Long edge of each opening-strip tile at this run's level."""
    return PICTURE_EDGES[resolution_level(ctx)][0]


def picture_edge(ctx: Workspace, requested: int | None = None) -> int:
    """Long edge of each later picture: the level's, or *requested* at "auto".

    *requested* must already be clamped (:data:`MIN_RESOLUTION` up to the
    driver model's maximum; :func:`clamp_resolution` does it); every level but
    "auto" ignores it.
    """
    level = resolution_level(ctx)
    if level == AUTO_RESOLUTION and requested:
        return int(requested)
    return PICTURE_EDGES[level][1]


def clamp_resolution(
    value: float | str, max_edge: int
) -> tuple[int | None, str]:
    """Hold a requested ``resolution`` between the smallest and largest edge.

    Parameters
    ----------
    value
        The model's ``resolution``, a number of pixels; 0 or empty is the
        default size.
    max_edge
        The largest image the driver model takes, in pixels.

    Returns
    -------
    edge : int or None
        The long edge, clamped into :data:`MIN_RESOLUTION` .. *max_edge*;
        None for the default size.
    note : str
        What the reply says when the edge was clamped; empty otherwise.

    Raises
    ------
    ValueError
        When *value* is not a finite number.
    """
    if not value:
        return None, ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("resolution must be a number of pixels")
    low, high = MIN_RESOLUTION, int(max_edge)
    edge = int(round(number))
    if edge < low or edge > high:
        clamped = min(high, max(low, edge))
        note = f"resolution {edge} is outside {low}..{high}; used {clamped}"
        return clamped, note
    return edge, ""
