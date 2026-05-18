"""Adaptive bell-curve schedule for the single-turn final-answer reward.

.. deprecated::
    The schedule logic has been lifted into :mod:`adaptive.schedule` (Phase 5).
    This module is now a thin backward-compatibility shim.  It preserves the
    original public surface (``AdaptiveRewardSchedule``,
    ``make_adaptive_terminal_reward``, ``record_error``, ``recent_errors``,
    ``clear_recent_errors``, ``_RECENT_ERRORS``) so existing callers in
    ``train_grpo.py`` and tests continue to work without modification.

    New code should import directly from :mod:`adaptive.schedule` and manage
    its own :func:`~adaptive.schedule.make_error_buffer` buffer.

AP-bin keying
-------------
Per-section difficulty observations are keyed by AP-coordinate bin rather than
by ``(plane, dataset, section_id)``. The training corpus has ~14k unique
sections seen 1x per epoch, so per-section observations don't accumulate
useful signal in a 200-entry buffer. Pooling by AP bin (sections at the same
AP coordinate look visually similar across brains) gives ~10 observations
per bin with the same buffer.

All errors stored in :data:`_RECENT_ERRORS` are *fractions of the axis span*
(``abs_err_mm / axis_span_mm``), per the standing rule that MAE and AP
coordinates are always expressed as percent-of-axis. The boundary
:func:`record_error` takes raw mm values and converts immediately.

DDP / multi-GPU note
--------------------
The rolling buffer is single-process module-level state and is only correct
under single-GPU training. Under DDP each rank would see only its own
observations and the schedule would diverge across ranks. A future multi-GPU
port needs an ``all_gather`` of recent errors (or rank-0-only schedule +
broadcast). Out of scope for this layer.
"""

from __future__ import annotations

import os
from collections import deque
from statistics import mean
from typing import Any

from langslice_training.adaptive.schedule import (
    AdaptiveSchedule as _AdaptiveSchedule,
    ErrorObservation as ErrorObservation,
    make_error_buffer as make_error_buffer,
)
from langslice_training.rl_core.rewards import normalized_bell_reward
from langslice_training.single_turn_core.rewards import (
    DEFAULT_FORMAT_PENALTY,
    DEFAULT_OUT_OF_RANGE_REWARD,
    _extract_completion_text,
    parse_position_mm,
)
from langslice_training.single_turn_core.rewards import (
    _ParseError as _ParseError,
)

# ---------------------------------------------------------------------------
# Public constants — preserved for backward compat
# ---------------------------------------------------------------------------

#: Default static fallback matching the legacy fixed-schedule reward.
DEFAULT_STATIC_FALLBACK: tuple[float, float] = (0.05, 0.15)
DEFAULT_MIN_SIGMA_FRAC: float = 0.005
DEFAULT_MAX_SIGMA_FRAC: float = 0.05
DEFAULT_SIGMA_QUANTILE: float = 0.5
DEFAULT_CUTOFF_QUANTILE: float = 0.95
DEFAULT_MAX_CUTOFF_FRAC: float = 0.25
DEFAULT_MIN_OBSERVATIONS: int = 50
DEFAULT_BUFFER_MAXLEN: int = 200

# ---------------------------------------------------------------------------
# AP-bin configuration
# ---------------------------------------------------------------------------

#: Number of AP-coordinate bins used to pool difficulty observations.
#: 20 bins → 5% of axis per bin. Tunable via the env var
#: ``LANGSLICE_AP_BIN_COUNT`` so experiments can sweep without code edits.
_AP_BIN_COUNT: int = int(os.environ.get("LANGSLICE_AP_BIN_COUNT", "20"))


def compute_ap_bin(
    ground_truth_mm: float,
    valid_range_mm: tuple[float, float],
) -> int:
    """Map a ground-truth AP coordinate to a bin index in ``[0, N_BINS-1]``.

    ``ap_pct = (gt - pos_lo) / axis_span_mm`` is clamped into ``[0, 1]``
    before scaling so out-of-range or degenerate-span inputs still produce
    a legal bin index rather than crashing the reward path.
    """
    pos_lo, pos_hi = float(valid_range_mm[0]), float(valid_range_mm[1])
    axis_span = pos_hi - pos_lo
    if axis_span <= 0.0:
        return 0
    ap_pct = (float(ground_truth_mm) - pos_lo) / axis_span
    # Clamp into [0, 1] so the bin index lands in [0, N_BINS-1] even when
    # the gt sits just outside the valid range (a few adapter rows do).
    ap_pct = max(0.0, min(1.0, ap_pct))
    bin_idx = int(ap_pct * _AP_BIN_COUNT)
    if bin_idx >= _AP_BIN_COUNT:
        bin_idx = _AP_BIN_COUNT - 1
    return bin_idx


# ---------------------------------------------------------------------------
# Module-level rolling buffer — keyed by AP bin rather than per-section.
# ---------------------------------------------------------------------------

#: Module-level rolling buffer.  Shape: ``(abs_err_pct, plane, ap_bin)``.
#: ``abs_err_pct`` is ``abs_err_mm / axis_span_mm`` ∈ ``[0, 1]``.
_RECENT_ERRORS: deque[tuple[float, str, int]] = deque(maxlen=DEFAULT_BUFFER_MAXLEN)


def record_error(
    abs_err_mm: float,
    axis_span_mm: float,
    plane: str,
    ap_bin: int,
) -> None:
    """Append one observation to the module-level rolling buffer.

    The caller passes raw mm values at the boundary (TRL hands us mm); the
    buffer stores ``abs_err_pct = abs_err_mm / axis_span_mm`` per the
    percent-of-axis convention. Observations with non-positive
    ``axis_span_mm`` are dropped — there's no meaningful percent to store.
    """
    axis = float(axis_span_mm)
    if axis <= 0.0:
        return
    abs_err_pct = float(abs_err_mm) / axis
    _RECENT_ERRORS.append((abs_err_pct, str(plane), int(ap_bin)))


def recent_errors() -> list[tuple[float, str, int]]:
    """Snapshot of the rolling buffer for tests and external monitoring."""
    return list(_RECENT_ERRORS)


def clear_recent_errors() -> None:
    """Empty the rolling buffer. Tests use this for isolation between cases."""
    _RECENT_ERRORS.clear()


# ---------------------------------------------------------------------------
# AdaptiveRewardSchedule — backward-compat wrapper
# ---------------------------------------------------------------------------

class AdaptiveRewardSchedule:
    """Quantile-based schedule for ``(sigma_frac, cutoff_frac)``.

    Backward-compat wrapper over the stateless
    :class:`~adaptive.schedule.AdaptiveSchedule`.  Uses the module-level
    ``_RECENT_ERRORS`` deque as its observation source so existing callers
    are unaffected.

    .. deprecated::
        Prefer constructing :class:`~adaptive.schedule.AdaptiveSchedule`
        directly and passing an explicit buffer from
        :func:`~adaptive.schedule.make_error_buffer`.
    """

    def __init__(
        self,
        *,
        min_sigma_frac: float = DEFAULT_MIN_SIGMA_FRAC,
        max_sigma_frac: float = DEFAULT_MAX_SIGMA_FRAC,
        sigma_quantile: float = DEFAULT_SIGMA_QUANTILE,
        cutoff_quantile: float = DEFAULT_CUTOFF_QUANTILE,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        static_fallback: tuple[float, float] = DEFAULT_STATIC_FALLBACK,
        max_cutoff_frac: float = DEFAULT_MAX_CUTOFF_FRAC,
    ) -> None:
        # Validate inputs to preserve the original contract.
        if min_sigma_frac <= 0:
            raise ValueError(f"min_sigma_frac must be positive, got {min_sigma_frac}")
        if max_sigma_frac < min_sigma_frac:
            raise ValueError(
                f"max_sigma_frac ({max_sigma_frac}) must be >= "
                f"min_sigma_frac ({min_sigma_frac})"
            )
        if max_cutoff_frac < min_sigma_frac:
            raise ValueError(
                f"max_cutoff_frac ({max_cutoff_frac}) must be >= "
                f"min_sigma_frac ({min_sigma_frac})"
            )
        if not 0.0 <= sigma_quantile <= 1.0:
            raise ValueError(f"sigma_quantile must be in [0, 1], got {sigma_quantile}")
        if not 0.0 <= cutoff_quantile <= 1.0:
            raise ValueError(f"cutoff_quantile must be in [0, 1], got {cutoff_quantile}")
        if min_observations < 1:
            raise ValueError(f"min_observations must be >= 1, got {min_observations}")
        fb_sigma, fb_cutoff = static_fallback
        if fb_sigma <= 0 or fb_cutoff <= 0:
            raise ValueError(
                f"static_fallback components must be positive, got {static_fallback}"
            )

        self.min_sigma_frac = float(min_sigma_frac)
        self.max_sigma_frac = float(max_sigma_frac)
        self.max_cutoff_frac = float(max_cutoff_frac)
        self.sigma_quantile = float(sigma_quantile)
        self.cutoff_quantile = float(cutoff_quantile)
        self.min_observations = int(min_observations)
        self.static_fallback = (float(fb_sigma), float(fb_cutoff))

        # Underlying stateless schedule; deque is passed at call time.
        self._schedule = _AdaptiveSchedule(
            sigma_quantile=self.sigma_quantile,
            cutoff_quantile=self.cutoff_quantile,
            sigma_clamp=(self.min_sigma_frac, self.max_sigma_frac),
            cutoff_clamp=(0.0, self.max_cutoff_frac),
            warmup_min_observations=self.min_observations,
            warmup_sigma_frac=float(fb_sigma),
            warmup_cutoff_frac=float(fb_cutoff),
        )

    @property
    def n_observations(self) -> int:
        """Number of entries currently in the shared rolling buffer."""
        return len(_RECENT_ERRORS)

    def current(self, plane: str | None = None) -> tuple[float, float]:
        """Return ``(sigma_frac, cutoff_frac)`` derived from recent errors.

        Delegates to :meth:`~adaptive.schedule.AdaptiveSchedule.compute`.
        The underlying schedule expects a buffer of
        ``(abs_err, axis_span, plane, ...)`` tuples and computes
        ``error_frac = abs_err / axis_span`` internally; our buffer already
        stores ``abs_err_pct`` directly, so we adapt by passing
        ``(abs_err_pct, 1.0, plane, ...)`` — the division by 1.0 leaves the
        percent untouched.
        """
        adapted: list[tuple[float, float, str, int]] = [
            (abs_err_pct, 1.0, p, ap_bin)
            for abs_err_pct, p, ap_bin in _RECENT_ERRORS
        ]
        return self._schedule.compute(adapted, plane=plane)

    def per_bin_difficulty(self, plane: str, ap_bin: int) -> float | None:
        """Mean ``abs_err_pct`` across observations matching ``(plane, ap_bin)``.

        Returns ``None`` if no observations match. This replaces the legacy
        per-section keyed difficulty: the corpus has too many distinct
        sections for a 200-entry buffer to give per-section signal, but
        pooling by AP bin gives ~10 observations per bin.
        """
        matches = [
            err_pct
            for err_pct, p, b in _RECENT_ERRORS
            if p == plane and b == ap_bin
        ]
        if not matches:
            return None
        return mean(matches)

    def per_section_difficulty(self, section_key: Any) -> float | None:
        """Removed in favor of :meth:`per_bin_difficulty`.

        The training corpus has ~14k unique sections seen 1x per epoch; a
        200-entry buffer can't give per-section signal. Observations are
        now pooled by AP bin — call :meth:`per_bin_difficulty` instead.
        """
        raise NotImplementedError(
            "per_section_difficulty was removed when the curriculum switched "
            "to AP-bin keying. Use per_bin_difficulty(plane, ap_bin) — "
            "compute the bin via compute_ap_bin(gt, valid_range)."
        )


# ---------------------------------------------------------------------------
# make_adaptive_terminal_reward — RL-specific factory (stays in shim)
# ---------------------------------------------------------------------------

def make_adaptive_terminal_reward(
    *,
    schedule: AdaptiveRewardSchedule | None = None,
    format_penalty: float = DEFAULT_FORMAT_PENALTY,
    out_of_range_reward: float = DEFAULT_OUT_OF_RANGE_REWARD,
    # Forwarded to a freshly-built schedule when ``schedule`` is None.
    min_sigma_frac: float = DEFAULT_MIN_SIGMA_FRAC,
    max_sigma_frac: float = DEFAULT_MAX_SIGMA_FRAC,
    sigma_quantile: float = DEFAULT_SIGMA_QUANTILE,
    cutoff_quantile: float = DEFAULT_CUTOFF_QUANTILE,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    static_fallback: tuple[float, float] = DEFAULT_STATIC_FALLBACK,
    max_cutoff_frac: float = DEFAULT_MAX_CUTOFF_FRAC,
):
    """Build an adaptive-reward callable matching the TRL GRPO contract.

    The returned callable matches GRPOTrainer's reward contract:
    ``func(completions, prompts=None, **dataset_columns) -> list[float]``.

    Required dataset columns (passed through by GRPOTrainer):

    * ``ground_truth_mm`` — ``list[float]``, one per row.
    * ``valid_range_mm`` — ``list[tuple[float, float]]``, one per row.

    Optional dataset columns (used only for the rolling-buffer metadata; the
    reward value does not depend on them):

    * ``plane`` — ``list[str]`` (e.g. ``"coronal"``).

    Other dataset columns (``dataset``, ``section_id``, ``atlas_name``, …)
    are forwarded by TRL but ignored by the reward — the buffer is keyed
    by AP bin computed from ``ground_truth_mm`` + ``valid_range_mm``, not
    by per-section identity.

    Behaviour matches :func:`single_turn_rl.rewards.make_terminal_reward`
    (parse → in-range bell, out-of-range → ``out_of_range_reward``, parse
    failure → ``format_penalty``) except that the bell uses
    ``schedule.current()`` instead of fixed kwargs, and every scoring call
    appends to the rolling buffer via :func:`record_error`.
    """
    sched = schedule if schedule is not None else AdaptiveRewardSchedule(
        min_sigma_frac=min_sigma_frac,
        max_sigma_frac=max_sigma_frac,
        sigma_quantile=sigma_quantile,
        cutoff_quantile=cutoff_quantile,
        min_observations=min_observations,
        static_fallback=static_fallback,
        max_cutoff_frac=max_cutoff_frac,
    )

    def adaptive_terminal_reward(
        completions: list[Any] | None = None,
        prompts: list[Any] | None = None,  # noqa: ARG001 — TRL contract
        ground_truth_mm: list[float] | None = None,
        valid_range_mm: list[tuple[float, float]] | None = None,
        plane: list[str] | None = None,
        **kwargs: Any,  # noqa: ARG001 — swallow unused dataset columns
    ) -> list[float]:
        comps = completions or []
        gts = ground_truth_mm or []
        ranges = valid_range_mm or []
        if not (len(comps) == len(gts) == len(ranges)):
            raise ValueError(
                "completions, ground_truth_mm, and valid_range_mm must be "
                f"the same length; got {len(comps)}, {len(gts)}, {len(ranges)}"
            )

        n = len(comps)
        planes = plane if plane is not None else [""] * n

        # Per-plane schedule: cache the (sigma, cutoff) per unique plane in
        # this batch so the bell scoring path doesn't re-sort the buffer
        # for every row.
        schedule_cache: dict[str, tuple[float, float]] = {}

        def _schedule_for(row_plane: str) -> tuple[float, float]:
            cached = schedule_cache.get(row_plane)
            if cached is not None:
                return cached
            params = sched.current(plane=row_plane or None)
            schedule_cache[row_plane] = params
            return params

        out: list[float] = []
        for completion, gt, vr, p in zip(
            comps, gts, ranges, planes, strict=False
        ):
            gt_f = float(gt)
            pos_lo, pos_hi = float(vr[0]), float(vr[1])
            axis_span_mm = pos_hi - pos_lo
            plane_str = str(p)
            ap_bin = compute_ap_bin(gt_f, (pos_lo, pos_hi))

            text = _extract_completion_text(completion)
            try:
                predicted = parse_position_mm(text)
            except _ParseError:
                record_error(axis_span_mm, axis_span_mm, plane_str, ap_bin)
                out.append(float(format_penalty))
                continue

            if predicted < pos_lo or predicted > pos_hi:
                oor_abs_err = min(abs(predicted - gt_f), axis_span_mm)
                record_error(oor_abs_err, axis_span_mm, plane_str, ap_bin)
                out.append(float(out_of_range_reward))
                continue

            abs_err = abs(predicted - gt_f)
            record_error(abs_err, axis_span_mm, plane_str, ap_bin)
            sigma_frac, cutoff_frac = _schedule_for(plane_str)
            out.append(
                normalized_bell_reward(
                    predicted - gt_f,
                    axis_span_mm=axis_span_mm,
                    cutoff_frac=cutoff_frac,
                    sigma_frac=sigma_frac,
                )
            )
        return out

    adaptive_terminal_reward.__name__ = "adaptive_terminal_reward"
    adaptive_terminal_reward.__qualname__ = "adaptive_terminal_reward"
    return adaptive_terminal_reward
