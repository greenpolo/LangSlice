"""The picture-size limits of the model a run talks to.

:func:`image_limit` (one image's long edge and patch budget for the run's
model lane), :func:`view_edge_limit` (the largest picture the agent may ask
``look`` for) and :func:`clamp_resolution` (a requested ``resolution`` held
inside those limits).
"""

from __future__ import annotations

import math
from typing import Any

from langslice.core.opening import DEFAULT_IMAGE_LIMIT, IMAGE_LIMITS
from langslice.core.sizes import MIN_RESOLUTION
from langslice.providers.registry import canonical_provider


def image_limit(ctx: Any) -> tuple[int, int]:
    """``(long edge, patch budget)`` of one image for this run's model lane
    (:data:`langslice.core.opening.IMAGE_LIMITS`), read off the driver
    context's ``model`` (a ``provider/name`` string)."""
    model = str(getattr(ctx, "model", "") or "")
    provider = canonical_provider(model.split("/", 1)[0]) if "/" in model else ""
    return IMAGE_LIMITS.get(provider, DEFAULT_IMAGE_LIMIT)


def view_edge_limit(ctx: Any) -> int:
    """Largest picture the agent may ask for per call (``look``'s ``resolution``):
    the model lane's largest image edge (:func:`image_limit`). On the OpenAI
    lanes a near-square picture past ~1600 px still meets the patch budget,
    which shrinks it."""
    return image_limit(ctx)[0]


def clamp_resolution(value: Any, max_edge: int) -> tuple[int | None, str]:
    """``(long edge, note)`` for a requested ``resolution``; None for "default".

    0, None and "" are the default. A number outside :data:`MIN_RESOLUTION`
    .. *max_edge* (the driver model's largest image) is clamped into it and
    *note* says so; a value that is not a number raises ``ValueError``.
    """
    if value in (None, "", 0):
        return None, ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("resolution must be a number of pixels")
    low, high = MIN_RESOLUTION, int(max_edge)
    edge = int(round(number))
    if edge < low or edge > high:
        clamped = min(high, max(low, edge))
        return clamped, f"resolution {edge} is outside {low}..{high}; used {clamped}"
    return edge, ""
