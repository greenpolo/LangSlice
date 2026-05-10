"""Adaptive bell-curve schedule for the single-turn final-answer reward.

The static reward in :mod:`single_turn_rl.rewards` reads a fixed ``sigma_frac``
and ``cutoff_frac`` from a TOML and uses them for every call. That works as a
warmup but stops shaping useful gradient once the policy's typical error
shrinks below ``sigma`` — the reward becomes near-flat at 1.0 and the
policy stops being pushed toward tighter localisation.

This module replaces the constants with a *schedule* that reads recent
absolute-error observations and re-derives ``(sigma_frac, cutoff_frac)`` per
call as quantiles of the running error distribution. As the policy improves,
the bell auto-tightens around the new error scale.

Implementation
--------------

* A module-level :class:`collections.deque` (``_RECENT_ERRORS``) holds the
  most recent ``maxlen`` observations as
  ``(abs_err_mm, axis_span_mm, plane, dataset, section_id)`` tuples. The
  reward closure appends to it on every call. This sidecar is mandatory:
  TRL's ``state.log_history`` only keeps aggregated metrics, not per-call
  rows, so the schedule needs its own buffer to compute quantiles.
* :class:`AdaptiveRewardSchedule` reads the deque and turns it into
  ``(sigma_frac, cutoff_frac)`` via two configurable quantiles, with min/max
  clamps and a static fallback used until the deque has warmed up.
* :func:`make_adaptive_terminal_reward` mirrors the TRL contract of
  :func:`single_turn_rl.rewards.make_terminal_reward` and substitutes the
  dynamic schedule output for the static kwargs at call time.

DDP / multi-GPU note
--------------------

The rolling buffer is single-process module-level state and is only correct
under single-GPU training. Under DDP each rank would see only its own
observations and the schedule would diverge across ranks. A future
multi-GPU port needs an ``all_gather`` of recent errors (or rank-0-only
schedule + broadcast). Out of scope for this layer.
"""

from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Any

from rlvr.rewards import normalized_bell_reward
from single_turn_rl.rewards import (
    DEFAULT_FORMAT_PENALTY,
    DEFAULT_OUT_OF_RANGE_REWARD,
    _extract_completion_text,
    parse_position_mm,
)
from single_turn_rl.rewards import (
    _ParseError as _ParseError,
)

# Default static fallback used by the schedule before warmup, picked to match
# the static defaults in :mod:`single_turn_rl.rewards` so behaviour pre-warmup
# is identical to the legacy reward.
DEFAULT_STATIC_FALLBACK: tuple[float, float] = (0.05, 0.15)
DEFAULT_MIN_SIGMA_FRAC: float = 0.005
DEFAULT_MAX_SIGMA_FRAC: float = 0.05
DEFAULT_SIGMA_QUANTILE: float = 0.5
DEFAULT_CUTOFF_QUANTILE: float = 0.95
DEFAULT_MIN_OBSERVATIONS: int = 50
DEFAULT_BUFFER_MAXLEN: int = 200


# Module-level rolling buffer. Tuple shape:
#   (abs_err_mm, axis_span_mm, plane, dataset, section_id)
_RECENT_ERRORS: deque[tuple[float, float, str, str, str]] = deque(maxlen=DEFAULT_BUFFER_MAXLEN)


def record_error(
    abs_err_mm: float,
    axis_span_mm: float,
    plane: str,
    dataset: str,
    section_id: str,
) -> None:
    """Append one observation to the module-level rolling buffer.

    ``abs_err_mm`` is the absolute coordinate error in mm; ``axis_span_mm`` is
    the valid range for the active plane (so the schedule can compute
    ``error_frac = abs_err_mm / axis_span_mm`` later). Categorical metadata
    keys the per-section difficulty lookup used by the curriculum callback.
    """
    _RECENT_ERRORS.append(
        (float(abs_err_mm), float(axis_span_mm), str(plane), str(dataset), str(section_id))
    )


def recent_errors() -> list[tuple[float, float, str, str, str]]:
    """Snapshot of the rolling buffer for tests and external monitoring."""
    return list(_RECENT_ERRORS)


def clear_recent_errors() -> None:
    """Empty the rolling buffer. Tests use this for isolation between cases."""
    _RECENT_ERRORS.clear()


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile over a pre-sorted list.

    Equivalent to ``numpy.quantile(values, q, method="linear")`` for a
    non-empty input. We reimplement here so the module has no numpy
    dependency — the existing reward stays on pure-python math too, and
    the code path is hot (called per reward batch) so pulling in numpy
    just for one quantile would be wasteful.
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


class AdaptiveRewardSchedule:
    """Quantile-based schedule for ``(sigma_frac, cutoff_frac)``.

    Reads :data:`_RECENT_ERRORS` on each :meth:`current` call, computes
    ``error_frac = abs_err_mm / axis_span_mm`` per entry, and returns the
    requested quantiles clamped to their min/max bounds. Until the deque
    holds at least ``min_observations`` entries the schedule returns
    ``static_fallback`` so early training matches the legacy fixed-schedule
    reward.

    The schedule is stateless apart from its constructor params and the
    shared module-level deque. It is safe to construct multiple instances
    with different bounds (e.g. one for monitoring, one for the live
    reward); they will all see the same observations.
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
    ) -> None:
        if min_sigma_frac <= 0:
            raise ValueError(f"min_sigma_frac must be positive, got {min_sigma_frac}")
        if max_sigma_frac < min_sigma_frac:
            raise ValueError(
                f"max_sigma_frac ({max_sigma_frac}) must be >= "
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
        self.sigma_quantile = float(sigma_quantile)
        self.cutoff_quantile = float(cutoff_quantile)
        self.min_observations = int(min_observations)
        self.static_fallback = (float(fb_sigma), float(fb_cutoff))

    @property
    def n_observations(self) -> int:
        """Number of entries currently in the shared rolling buffer."""
        return len(_RECENT_ERRORS)

    def current(self) -> tuple[float, float]:
        """Return ``(sigma_frac, cutoff_frac)`` derived from recent errors.

        Computes ``error_frac`` for every entry in the rolling buffer, takes
        the configured quantiles, clamps each to its bounds, and returns.
        Pre-warmup (fewer than ``min_observations``) returns
        ``static_fallback`` unchanged.

        ``cutoff_frac`` is clamped to be at least ``sigma_frac`` so the
        bell never collapses to a delta — equality is allowed because the
        bell formula stays well-defined and the gradient just tightens.
        """
        if len(_RECENT_ERRORS) < self.min_observations:
            return self.static_fallback

        # axis_span_mm came from valid_range_mm, which is positive by
        # construction in the reward path. Guard anyway because a hostile
        # caller of record_error could plant a zero.
        error_fracs = sorted(
            abs_err / axis_span
            for abs_err, axis_span, *_ in _RECENT_ERRORS
            if axis_span > 0
        )
        if not error_fracs:
            return self.static_fallback

        raw_sigma = _quantile(error_fracs, self.sigma_quantile)
        raw_cutoff = _quantile(error_fracs, self.cutoff_quantile)

        sigma_frac = min(max(raw_sigma, self.min_sigma_frac), self.max_sigma_frac)
        # The cutoff isn't clamped to the same band as sigma — its purpose
        # is to set the zero-reward threshold, which can sit well outside
        # the sigma envelope. We do require cutoff >= sigma so the bell
        # has a non-degenerate falloff region.
        cutoff_frac = max(raw_cutoff, sigma_frac)
        return sigma_frac, cutoff_frac

    def per_section_difficulty(
        self, section_key: tuple[str, str, str]
    ) -> float | None:
        """Mean ``error_frac`` across observations matching ``section_key``.

        ``section_key`` is ``(plane, dataset, section_id)``. Returns ``None``
        if no observations match — the curriculum callback uses ``None`` as
        "no signal yet, fall back to uniform sampling".
        """
        plane, dataset, section_id = section_key
        matches = [
            abs_err / axis_span
            for abs_err, axis_span, p, d, s in _RECENT_ERRORS
            if p == plane and d == dataset and s == section_id and axis_span > 0
        ]
        if not matches:
            return None
        return mean(matches)


def make_adaptive_terminal_reward(
    *,
    schedule: AdaptiveRewardSchedule | None = None,
    format_penalty: float = DEFAULT_FORMAT_PENALTY,
    out_of_range_reward: float = DEFAULT_OUT_OF_RANGE_REWARD,
    # Forwarded to a freshly-built schedule when ``schedule`` is None. Lets
    # callers tune the schedule from CLI/TOML without constructing the
    # object themselves; ignored when an explicit schedule is supplied.
    min_sigma_frac: float = DEFAULT_MIN_SIGMA_FRAC,
    max_sigma_frac: float = DEFAULT_MAX_SIGMA_FRAC,
    sigma_quantile: float = DEFAULT_SIGMA_QUANTILE,
    cutoff_quantile: float = DEFAULT_CUTOFF_QUANTILE,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    static_fallback: tuple[float, float] = DEFAULT_STATIC_FALLBACK,
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
    * ``dataset`` — ``list[str]`` (e.g. ``"allen_mouse_25um"``).
    * ``section_id`` — ``list[str]``, the per-row section identifier.

    Behaviour matches :func:`single_turn_rl.rewards.make_terminal_reward`
    (parse → in-range bell, out-of-range → ``out_of_range_reward``, parse
    failure → ``format_penalty``) except that the bell uses
    ``schedule.current()`` instead of fixed kwargs, and every scoring call
    appends to the rolling buffer via :func:`record_error`.

    Pass an explicit ``schedule`` to share state across multiple reward
    builds (e.g. a monitoring schedule); otherwise a fresh one is
    constructed from the per-call kwargs.
    """
    sched = schedule if schedule is not None else AdaptiveRewardSchedule(
        min_sigma_frac=min_sigma_frac,
        max_sigma_frac=max_sigma_frac,
        sigma_quantile=sigma_quantile,
        cutoff_quantile=cutoff_quantile,
        min_observations=min_observations,
        static_fallback=static_fallback,
    )

    def adaptive_terminal_reward(
        completions: list[Any] | None = None,
        prompts: list[Any] | None = None,  # noqa: ARG001 — TRL contract
        ground_truth_mm: list[float] | None = None,
        valid_range_mm: list[tuple[float, float]] | None = None,
        plane: list[str] | None = None,
        dataset: list[str] | None = None,
        section_id: list[str] | None = None,
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

        # Dynamic schedule is sampled once per batch — sigma/cutoff are not
        # row-specific, only the recorded error is.
        sigma_frac, cutoff_frac = sched.current()
        n = len(comps)
        planes = plane if plane is not None else [""] * n
        datasets = dataset if dataset is not None else [""] * n
        section_ids = section_id if section_id is not None else [""] * n

        out: list[float] = []
        for completion, gt, vr, p, d, s in zip(
            comps, gts, ranges, planes, datasets, section_ids, strict=False
        ):
            gt_f = float(gt)
            pos_lo, pos_hi = float(vr[0]), float(vr[1])
            axis_span_mm = pos_hi - pos_lo

            text = _extract_completion_text(completion)
            try:
                predicted = parse_position_mm(text)
            except _ParseError:
                # Format-fail rows have no parseable position. We still record
                # an observation so the AdaRFT curriculum can see that this
                # section is hard — skipping would make hard sections look
                # easier than they are and the curriculum would starve them.
                # Use ``axis_span_mm`` as a max-plausible-error sentinel so
                # the quantile estimator still gets a bounded contribution.
                record_error(axis_span_mm, axis_span_mm, str(p), str(d), str(s))
                out.append(float(format_penalty))
                continue

            if predicted < pos_lo or predicted > pos_hi:
                # Out-of-range rows do have a parsed prediction, but a wildly
                # off prediction could otherwise dominate the quantiles. Cap
                # the recorded absolute error at ``axis_span_mm`` so OOR rows
                # contribute meaningful per-section difficulty signal without
                # poisoning the quantile estimate.
                oor_abs_err = min(abs(predicted - gt_f), axis_span_mm)
                record_error(oor_abs_err, axis_span_mm, str(p), str(d), str(s))
                out.append(float(out_of_range_reward))
                continue

            abs_err = abs(predicted - gt_f)
            record_error(abs_err, axis_span_mm, str(p), str(d), str(s))
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
