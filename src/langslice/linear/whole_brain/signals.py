"""Advisory spacing signals for the positioning and review steps.

These are pure functions over plain numbers. They used to *be* the pipeline
(interpolate, then silently overwrite every estimate with a monotone fit);
in the node engine they are demoted to advice — an agent step asks for them,
looks at the difference, and decides. Nothing here writes to state.

Positions are expected in corrected stack order and increasing along the
slicing axis; callers normalize direction before asking.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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


def monotone_fit(
    positions: list[float],
    *,
    interval_mm: float,
    thickness_mm: float,
    weights: list[float] | None = None,
    data_delta_mm: float = 1.0,
    interval_delta_mm: float = 0.2,
    spacing_weight: float = 0.0,
) -> list[float]:
    """Fit a monotone-increasing curve through *positions*.

    Constrained optimization: pseudo-Huber data fidelity (quadratic for small
    residuals, linear for large ones, so outliers are tamed rather than
    obeyed) with a hard minimum spacing of *thickness_mm* between neighbours.
    Optional *weights* scale the per-slice data term — use it to say "trust
    these estimates more".

    Returns the fitted positions; the caller decides what, if anything, to do
    with the difference.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = len(positions)
    if n <= 1:
        return list(positions)

    raw = np.array(positions, dtype=np.float64)
    w = (
        np.ones(n, dtype=np.float64)
        if weights is None
        else np.array(weights, dtype=np.float64)
    )
    if w.shape != raw.shape:
        raise ValueError(f"weights length {w.shape} != positions length {raw.shape}")

    def _huber(residuals, delta: float):
        return delta**2 * (np.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)

    def objective(x) -> float:
        data_loss = float(np.sum(w * _huber(x - raw, data_delta_mm)))
        if spacing_weight == 0.0:
            return data_loss
        spacing_loss = float(
            np.sum(_huber(np.diff(x) - interval_mm, interval_delta_mm))
        )
        return data_loss + spacing_weight * spacing_loss

    constraints = [
        {"type": "ineq", "fun": lambda x, i=i: x[i] - x[i - 1] - thickness_mm}
        for i in range(1, n)
    ]

    x0 = np.sort(raw)
    for i in range(1, n):
        if x0[i] < x0[i - 1] + thickness_mm:
            x0[i] = x0[i - 1] + thickness_mm

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=constraints,  # type: ignore[arg-type]
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        logger.warning(
            "Monotone fit did not converge: %s (nit=%d). Using best iterate.",
            result.message,
            result.nit,
        )
    return [round(float(v), 4) for v in result.x]
