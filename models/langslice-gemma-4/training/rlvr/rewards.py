"""Reward functions for the LangSlice RLVR pipeline.

Each function follows TRL's ``GRPOTrainer.reward_funcs`` contract:
``func(completions, environments, **kwargs) -> list[float]``. The trainer
sums the outputs of all listed reward functions per rollout to form the
GRPO advantage signal.

The accuracy reward is the dominant signal (max 1.0); format / length
shaping rewards are small (max ~0.4 combined) so the model cannot win by
"performing the ritual" while missing the coordinate.
"""

from __future__ import annotations

from typing import Any

from .env import LangSliceEstimateEnv

# Single-slice budget set to one over the prompt's "STUDENT-SIZED STRATEGY"
# 5-step plan (1 broad + 1 narrow + 1 optional finer + submit, plus slack);
# group budget bumps to allow bracketing slice 1 and slice N independently.
SINGLE_TURN_BUDGET: int = 8
GROUP_TURN_BUDGET: int = 12

# Tolerance is atlas-relative (range fraction) but floored so a tiny atlas
# (e.g. zebrafish AP ≈ 5 mm) doesn't make the tier impossibly tight.
TOL_RANGE_FRACTION: float = 0.02
TOL_FLOOR_MM: float = 0.20


def _tier_score(err_mm: float, tol_mm: float) -> float:
    a = abs(float(err_mm))
    if a <= tol_mm:
        return 1.0
    if a <= 2.0 * tol_mm:
        return 0.5
    if a <= 4.0 * tol_mm:
        return 0.25
    return 0.0


def _tolerance_for(env: LangSliceEstimateEnv) -> float:
    s = env._state  # noqa: SLF001 — reward funcs read private env state by design
    span = s.pos_hi - s.pos_lo
    return max(TOL_RANGE_FRACTION * span, TOL_FLOOR_MM)


def position_accuracy_reward(
    completions: list[Any] | None = None,
    environments: list[LangSliceEstimateEnv] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Primary reward: tiered scoring on absolute mm error.

    Single-slice: tier on the lone submitted vs. ground-truth position.
    Group: per-slice tier scores combined as
        ``0.5 * mean + 0.5 * worst`` to penalize outlier slices.
    No submission, wrong submit kind, or wrong number of submitted positions
    all yield 0.
    """
    envs = environments or []
    out: list[float] = []
    for env in envs:
        s = env._state  # noqa: SLF001
        if s.submitted_positions_mm is None or s.submitted_kind != s.kind:
            out.append(0.0)
            continue
        tol = _tolerance_for(env)
        truths = s.ground_truth_positions_mm
        preds = s.submitted_positions_mm
        if len(preds) != len(truths):
            out.append(0.0)
            continue
        if s.kind == "single":
            out.append(_tier_score(preds[0] - truths[0], tol))
            continue
        per_slice = [_tier_score(p - t, tol) for p, t in zip(preds, truths, strict=True)]
        out.append(0.5 * (sum(per_slice) / len(per_slice)) + 0.5 * min(per_slice))
    return out


def format_compliance_reward(
    completions: list[Any] | None = None,
    environments: list[LangSliceEstimateEnv] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Small shaping reward for protocol compliance.

    + 0.2  exactly one submit call at end of trajectory
    + 0.2  submit kind matches task kind
    - 0.2  per malformed tool call along the way
    - 0.5  if no submission of any kind
    """
    envs = environments or []
    out: list[float] = []
    for env in envs:
        s = env._state  # noqa: SLF001
        score = 0.0
        if s.submit_calls == 1:
            score += 0.2
        if s.submitted_kind is not None and s.submitted_kind == s.kind:
            score += 0.2
        score -= 0.2 * s.malformed_tool_calls
        if s.submitted_positions_mm is None:
            score -= 0.5
        out.append(score)
    return out


def length_budget_reward(
    completions: list[Any] | None = None,
    environments: list[LangSliceEstimateEnv] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Small shaping reward for staying inside the per-task turn budget.

    0 within budget, ``-0.05`` per excess turn until trajectory ends. Counts
    every public-method tool call (fetch + submit) as one turn.
    """
    envs = environments or []
    out: list[float] = []
    for env in envs:
        s = env._state  # noqa: SLF001
        budget = SINGLE_TURN_BUDGET if s.kind == "single" else GROUP_TURN_BUDGET
        excess = max(0, s.turns - budget)
        out.append(-0.05 * excess)
    return out
