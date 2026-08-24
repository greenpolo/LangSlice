"""Position arithmetic for the whole-brain steps.

A pure function over plain numbers, used by the positioning step's
``interpolate_between`` tool. Nothing here writes to state and nothing here
judges: it computes what the caller asked for.

Positions are expected in corrected stack order; callers normalize direction
before asking.
"""

from __future__ import annotations


def interpolate_positions(
    known: list[float | None], *, interval_mm: float
) -> list[float]:
    """Fill in unknown positions from the known ones.

    Between two known positions, spacing is distributed evenly. Outside the
    outermost known positions, steps of *interval_mm* are used. Requires at
    least one known position.
    """
    n = len(known)
    if n == 0:
        return []
    anchors = [i for i, value in enumerate(known) if value is not None]
    if not anchors:
        raise ValueError("interpolate_positions needs at least one known position")

    out = [0.0] * n
    for i in anchors:
        value = known[i]
        assert value is not None  # narrowed by the anchors comprehension
        out[i] = float(value)

    for a, b in zip(anchors, anchors[1:], strict=False):
        step = (out[b] - out[a]) / (b - a)
        for k in range(1, b - a):
            out[a + k] = out[a] + k * step

    first, last = anchors[0], anchors[-1]
    for k in range(1, first + 1):
        out[first - k] = out[first] - k * interval_mm
    for k in range(1, n - last):
        out[last + k] = out[last] + k * interval_mm
    return out
