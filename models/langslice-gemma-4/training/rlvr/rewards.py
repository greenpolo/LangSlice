"""Reward functions for the LangSlice RLVR pipeline.

Single reward: a gated linear ramp on absolute mm error.

    reward = max(0, 1 - |err_mm| / window_mm)

Single-slice rollouts get the per-slice reward directly. Group rollouts get
the mean of per-slice rewards. A failure to submit, a wrong-kind submission,
or a wrong-count group submission yields 0.0 — no extra penalty, no shaping
terms. Format / structure / submit-count rewards from the prior version were
gameable and have been removed.

The window is parameterised via ``window_mm`` so it can be tuned from the
TOML config without editing this file. Default 0.100 mm.
"""

from __future__ import annotations

from typing import Any

from .env import LangSliceEstimateEnv

DEFAULT_WINDOW_MM: float = 0.100


def closeness_reward(error_mm: float, window_mm: float = DEFAULT_WINDOW_MM) -> float:
    """Gated linear ramp on absolute mm error.

    1.0 at zero error, 0.0 at ``|err_mm| >= window_mm``, linear in between.
    Negative ``window_mm`` is rejected (would invert the gradient signal).
    """
    if window_mm <= 0:
        raise ValueError(f"window_mm must be positive, got {window_mm}")
    return max(0.0, 1.0 - abs(float(error_mm)) / float(window_mm))


def make_position_reward(window_mm: float = DEFAULT_WINDOW_MM):
    """Build a TRL-compatible reward function bound to ``window_mm``.

    The returned callable matches ``GRPOTrainer.reward_funcs`` shape:
    ``func(completions, environments, **kwargs) -> list[float]``. We bind the
    window at construction time so the trainer never has to pass it through
    its kwargs path (which is reserved for dataset columns).
    """
    if window_mm <= 0:
        raise ValueError(f"window_mm must be positive, got {window_mm}")

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
            per_slice = [
                closeness_reward(p - t, window_mm=window_mm)
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


# Default-window reward for callers that do not bind a custom window.
position_reward = make_position_reward(DEFAULT_WINDOW_MM)
