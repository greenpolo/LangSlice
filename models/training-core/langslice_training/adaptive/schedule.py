"""Shared quantile-based tolerance schedule for iSFT and RL pipelines.

Each pipeline (iSFT, single-turn RL) owns its own error buffer constructed via
:func:`make_error_buffer`.  :class:`AdaptiveSchedule` is stateless — it reads
the buffer on every call, so multiple schedule objects with different params can
observe the same buffer without racing or sharing config.

Usage (typical)::

    from langslice_training.adaptive.schedule import AdaptiveSchedule, make_error_buffer

    # At run startup — one buffer per pipeline, never shared across processes.
    buf = make_error_buffer()

    # After scoring a batch:
    for row in scored_rollouts:
        buf.append((row["abs_err_mm"], row["plane_extent_mm"],
                    row["plane"], row["atlas"], row["section_id"]))

    # During the filter phase:
    sched = AdaptiveSchedule()
    cutoff_mm = sched.quantile_cutoff_mm(buf)
    kept = [r for r in scored if r["abs_err_mm"] <= cutoff_mm]

DDP / multi-GPU note
--------------------
The buffer is single-process state.  Under DDP each rank sees only its own
observations; the schedule will diverge across ranks.  A future multi-GPU port
needs an ``all_gather`` of recent errors (or rank-0-only schedule + broadcast).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: One observation: (abs_err_mm, axis_span_mm, plane, dataset, section_id)
ErrorObservation = tuple[float, float, str, str, str]


# ---------------------------------------------------------------------------
# Buffer factory
# ---------------------------------------------------------------------------


def make_error_buffer(maxlen: int = 1024) -> deque[ErrorObservation]:
    """Construct a fresh error buffer for one consumer (iSFT or RL).

    Each pipeline owns its own buffer so their observation streams never
    silently mix.  The default ``maxlen=1024`` is deliberately larger than
    the RL default (200) so iSFT — which may run fewer rounds but score
    more rollouts per round — still warms up reliably.

    Args:
        maxlen: Maximum number of observations to retain.  Older entries
            are evicted in FIFO order once the buffer is full.

    Returns:
        An empty :class:`~collections.deque` with ``maxlen`` set.
    """
    return deque(maxlen=maxlen)


# ---------------------------------------------------------------------------
# Pure-quantile helper
# ---------------------------------------------------------------------------


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile over a pre-sorted list.

    Equivalent to ``numpy.quantile(values, q, method='linear')`` for a
    non-empty input.  Pure Python so the module has no numpy dependency.
    """
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSchedule:
    """Stateless quantile-based tolerance schedule.

    Takes a :func:`make_error_buffer` buffer as input on every call — it
    never stores one internally, so the caller decides the lifetime and
    sharing policy.

    All params have sensible defaults matching the RL-side constants in
    :mod:`single_turn_rl.adaptive_reward`.

    Attributes:
        sigma_quantile: Quantile of ``error_frac`` used for the bell's
            ``sigma_frac`` parameter.  Default 0.5 (median).
        cutoff_quantile: Quantile used for the bell's ``cutoff_frac``.
            Default 0.95.
        sigma_clamp: ``(min, max)`` bounds for the derived ``sigma_frac``.
        cutoff_clamp: ``(min, max)`` bounds for the derived ``cutoff_frac``.
            The lower bound of this clamp is overridden to be >= sigma_frac
            so the bell has a non-degenerate falloff region.
        warmup_min_observations: Minimum number of buffer entries required
            before quantile computation kicks in.
        warmup_sigma_frac: ``sigma_frac`` returned during warmup.
        warmup_cutoff_frac: ``cutoff_frac`` returned during warmup.
    """

    sigma_quantile: float = 0.5
    cutoff_quantile: float = 0.95
    sigma_clamp: tuple[float, float] = (0.005, 0.05)
    cutoff_clamp: tuple[float, float] = (0.0, 0.25)
    warmup_min_observations: int = 50
    warmup_sigma_frac: float = 0.05
    warmup_cutoff_frac: float = 0.15

    def compute(
        self,
        buffer: Sequence[ErrorObservation],
        *,
        plane: str | None = None,
    ) -> tuple[float, float]:
        """Return ``(sigma_frac, cutoff_frac)`` from recent error observations.

        Computes ``error_frac = abs_err_mm / axis_span_mm`` for every entry,
        takes the configured quantiles, and clamps each to its bounds.

        When ``plane`` is provided the buffer is filtered to matching entries.
        If the per-plane subset has fewer than ``warmup_min_observations``
        entries, the global quantile is used instead — graceful degradation
        to the legacy global behaviour when a plane is sparsely sampled.

        Returns the warmup defaults when ``len(buffer) <
        warmup_min_observations``.

        Args:
            buffer: The error-observation buffer for this pipeline.
            plane: Optional plane filter (``"coronal"``, ``"sagittal"``, …).

        Returns:
            ``(sigma_frac, cutoff_frac)`` pair.
        """
        # Per-plane path: try the filtered subset first.
        if plane is not None:
            plane_fracs = [
                abs_err / axis_span
                for abs_err, axis_span, p, *_ in buffer
                if p == plane and axis_span > 0
            ]
            if len(plane_fracs) >= self.warmup_min_observations:
                return self._derive_from_sorted(sorted(plane_fracs))

        if len(buffer) < self.warmup_min_observations:
            return self.warmup_sigma_frac, self.warmup_cutoff_frac

        error_fracs = sorted(
            abs_err / axis_span for abs_err, axis_span, *_ in buffer if axis_span > 0
        )
        if not error_fracs:
            return self.warmup_sigma_frac, self.warmup_cutoff_frac

        return self._derive_from_sorted(error_fracs)

    def _derive_from_sorted(self, sorted_fracs: list[float]) -> tuple[float, float]:
        sigma_min, sigma_max = self.sigma_clamp
        _, cutoff_max = self.cutoff_clamp

        raw_sigma = _quantile(sorted_fracs, self.sigma_quantile)
        raw_cutoff = _quantile(sorted_fracs, self.cutoff_quantile)

        sigma_frac = min(max(raw_sigma, sigma_min), sigma_max)
        # cutoff must be >= sigma so the bell has a non-degenerate falloff;
        # also capped at cutoff_max to prevent format-fail rows from flattening
        # the bell across the full axis.
        cutoff_frac = max(raw_cutoff, sigma_frac)
        cutoff_frac = min(cutoff_frac, cutoff_max)
        return sigma_frac, cutoff_frac

    def quantile_cutoff_mm(
        self,
        buffer: Sequence[ErrorObservation],
        quantile: float = 0.95,
    ) -> float:
        """Return the absolute-mm quantile cutoff for accept/reject filtering.

        Used by iSFT — accept a rollout if ``abs_err_mm <= cutoff_mm``.
        Returns ``float('inf')`` during warmup so all rollouts pass until the
        buffer has enough observations.

        Args:
            buffer: The error-observation buffer for this pipeline.
            quantile: Which percentile to use as the cutoff.  Default 0.95.

        Returns:
            Cutoff in mm, or ``float('inf')`` if fewer than
            ``warmup_min_observations`` entries are in the buffer.
        """
        if len(buffer) < self.warmup_min_observations:
            return float("inf")
        errs = sorted(obs[0] for obs in buffer)
        idx = min(int(quantile * len(errs)), len(errs) - 1)
        return errs[idx]
