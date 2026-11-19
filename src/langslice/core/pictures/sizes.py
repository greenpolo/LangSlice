"""Picture sizes: the long edge of every picture the agent is shown, per level.

The working frame fits are computed on is
:data:`langslice.core.sections.renders.PREVIEW_LONG_EDGE`, never one of these.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

PICTURE_EDGES: dict[str, tuple[int, int]] = {
    "low": (256, 512),
    "medium": (384, 768),
    "high": (512, 1024),
    "auto": (512, 1024),
}
AUTO_RESOLUTION = "auto"
MIN_RESOLUTION = 128

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


def clamp_resolution(value: float | str, max_edge: int) -> tuple[int | None, str]:
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
