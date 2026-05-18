"""Reward functions for the LangSlice RLVR pipeline.

Single reward: a normalized, truncated Gaussian on final-coordinate error.

    err_frac = abs(predicted_mm - truth_mm) / axis_span_mm
    reward = rescaled exp(-0.5 * (err_frac / sigma_frac)^2)

Exact hits score 1.0. Errors at or beyond ``cutoff_frac`` score 0.0. Scores
inside the cutoff are smoothly rescaled to the [0, 1] range, which keeps AP,
DV, and ML axes comparable despite their different physical spans.

Single-slice rollouts get the per-slice reward directly. Group rollouts get
the mean of per-slice rewards. A failure to submit, a wrong-kind submission,
or a wrong-count group submission yields 0.0 — no extra penalty, no shaping
terms. Format / structure / submit-count rewards from the prior version were
gameable and have been removed.
"""

from __future__ import annotations

import math
from typing import Any

from .env import LangSliceEstimateEnv

DEFAULT_CUTOFF_FRAC: float = 0.10
DEFAULT_SIGMA_FRAC: float = 0.035


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def normalized_bell_reward(
    error_mm: float,
    *,
    axis_span_mm: float,
    cutoff_frac: float = DEFAULT_CUTOFF_FRAC,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
) -> float:
    """Truncated Gaussian reward on fractional axis error.

    ``axis_span_mm`` is the valid coordinate range for the active plane, so the
    same raw-mm error is judged more strictly on shorter axes.
    """
    _validate_positive("axis_span_mm", axis_span_mm)
    _validate_positive("cutoff_frac", cutoff_frac)
    _validate_positive("sigma_frac", sigma_frac)

    err_frac = abs(float(error_mm)) / float(axis_span_mm)
    if err_frac >= cutoff_frac:
        return 0.0

    raw = math.exp(-0.5 * (err_frac / sigma_frac) ** 2)
    floor = math.exp(-0.5 * (cutoff_frac / sigma_frac) ** 2)
    return (raw - floor) / (1.0 - floor)


def make_position_reward(
    *,
    cutoff_frac: float = DEFAULT_CUTOFF_FRAC,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
):
    """Build a TRL-compatible reward function bound to reward schedule knobs.

    The returned callable matches ``GRPOTrainer.reward_funcs`` shape:
    ``func(completions, environments, **kwargs) -> list[float]``. We bind
    schedule knobs at construction time so the trainer never has to pass them
    through its kwargs path (which is reserved for dataset columns).
    """
    _validate_positive("cutoff_frac", cutoff_frac)
    _validate_positive("sigma_frac", sigma_frac)

    def position_reward(
        completions: list[Any] | None = None,  # noqa: ARG001 — TRL contract
        environments: list[LangSliceEstimateEnv] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> list[float]:
        envs = environments or []
        out: list[float] = []
        for env in envs:
            s = env._state  # noqa: SLF001 — reward funcs read private state by design
            preds = s.submitted_positions_mm
            truths = s.ground_truth_positions_mm
            if (
                preds is None
                or s.submitted_kind != s.kind
                or len(preds) != len(truths)
            ):
                out.append(0.0)
                continue
            axis_span_mm = s.pos_hi - s.pos_lo
            per_slice = [
                normalized_bell_reward(
                    p - t,
                    axis_span_mm=axis_span_mm,
                    cutoff_frac=cutoff_frac,
                    sigma_frac=sigma_frac,
                )
                for p, t in zip(preds, truths, strict=True)
            ]
            if not per_slice:
                out.append(0.0)
                continue
            if s.kind == "single":
                out.append(per_slice[0])
            else:
                out.append(sum(per_slice) / len(per_slice))
        return out

    position_reward.__name__ = "position_reward"
    position_reward.__qualname__ = "position_reward"
    return position_reward


# Default normalized reward for callers that do not bind a custom schedule.
position_reward = make_position_reward()
