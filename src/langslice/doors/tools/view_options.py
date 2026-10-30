"""The picture-size limits of the model a run talks to.

:func:`image_limit` (one image's long edge and patch budget for the run's
model lane) and :func:`view_edge_limit` (the largest picture the agent may ask
``look`` for). They read the run's model through the providers, so they live
at the doors.
"""

from __future__ import annotations

from typing import Any

from langslice.core.pictures.opening import (
    DEFAULT_IMAGE_LIMIT,
    IMAGE_LIMITS,
    VIEW_EDGES,
)
from langslice.providers.connections import canonical_provider


def image_limit(ctx: Any) -> tuple[int, int]:
    """``(long edge, patch budget)`` of one image for this run's model lane
    (:data:`langslice.core.pictures.opening.IMAGE_LIMITS`), read off the driver
    context's ``model`` (a ``provider/name`` string)."""
    return IMAGE_LIMITS.get(_provider(ctx), DEFAULT_IMAGE_LIMIT)


def _provider(ctx: Any) -> str:
    """The canonical provider of the driver context's ``model``; "" without one."""
    model = str(getattr(ctx, "model", "") or "")
    return canonical_provider(model.split("/", 1)[0]) if "/" in model else ""


def view_edge_limit(ctx: Any) -> int:
    """Largest picture the agent may ask for per call (``look``'s ``resolution``):
    the model lane's largest image edge (:func:`image_limit`). On the OpenAI
    lanes a near-square picture past ~1600 px still meets the patch budget,
    which shrinks it. A Claude lane takes Claude's (:data:`VIEW_EDGES`), as
    the MCP host does."""
    return VIEW_EDGES.get(_provider(ctx), image_limit(ctx)[0])
