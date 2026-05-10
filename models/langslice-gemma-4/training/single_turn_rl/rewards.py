"""Reward function for single-turn final-answer GRPO (Lane A).

Compared to :mod:`rlvr.rewards`, this lane has no environment object — the
policy emits one completion and the reward is computed from the completion
text plus the dataset-row kwargs that TRL forwards from the dataset
(``ground_truth_mm``, ``axis_span_mm``, ``valid_range_mm``).

Three reward branches:

1. **Parse failure** (no JSON object, missing key, non-numeric, NaN, extra
   keys, surrounding prose): ``format_penalty`` (negative by default — the
   policy must learn the strict output contract).
2. **Out of range** (parsed cleanly but ``position_mm`` outside the plane's
   ``valid_range_mm``): ``out_of_range_reward`` (0.0 by default — neither
   penalize nor credit; the model just gets nothing).
3. **In range**: axis-normalized truncated-Gaussian bell on
   ``abs(position_mm - ground_truth_mm) / axis_span_mm`` via
   :func:`rlvr.rewards.normalized_bell_reward` (reused so the single-turn
   lane and the multi-turn lane agree on the underlying scoring shape).

The format-penalty branch is the main behavioral difference from the
multi-turn reward, which scored failure-to-submit as plain 0.0. Single-turn
GRPO needs an explicit format gradient because the policy starts from an SFT
checkpoint that was trained to emit prose + tool calls, not bare JSON.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from rlvr.rewards import normalized_bell_reward

DEFAULT_CUTOFF_FRAC: float = 0.15
DEFAULT_SIGMA_FRAC: float = 0.05
DEFAULT_FORMAT_PENALTY: float = -1.0
DEFAULT_OUT_OF_RANGE_REWARD: float = 0.0


_BARE_JSON_OBJECT_RE = re.compile(r"^\s*(\{.*\})\s*\Z", re.DOTALL)
_ALLOWED_KEYS: frozenset[str] = frozenset({"position_mm"})


class _ParseError(Exception):
    """Raised when a completion does not satisfy the strict-JSON contract."""


def _extract_completion_text(completion: Any) -> str:
    """Pull the assistant-text out of a TRL GRPO completion.

    GRPOTrainer hands reward funcs completions in one of two shapes depending
    on whether the dataset uses chat-format prompts:

    * **String** — when prompts are plain strings, the completion is the raw
      generated text.
    * **List of messages** — when prompts are chat-format ``[{role, content}]``
      lists, the completion is a list of one assistant message. ``content`` is
      either a string or a list of content blocks (typed dicts with ``text``
      keys); we concatenate the ``text`` blocks.

    Anything else collapses to ``str(completion)`` so the parse path can fail
    cleanly with ``_ParseError`` instead of raising at the type level.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                return "".join(parts)
    return str(completion)


def parse_position_mm(text: str) -> float:
    """Parse a strict ``{"position_mm": <number>}`` object out of ``text``.

    Tolerates leading/trailing whitespace but rejects:

    * surrounding prose or markdown fences,
    * missing or extra keys,
    * non-numeric / NaN / infinite ``position_mm``.

    Raises :class:`_ParseError` on any deviation.
    """
    match = _BARE_JSON_OBJECT_RE.match(text)
    if match is None:
        raise _ParseError("completion is not a single JSON object")
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise _ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise _ParseError("top-level JSON value is not an object")
    if "position_mm" not in obj:
        raise _ParseError("missing required key 'position_mm'")
    extra = set(obj.keys()) - _ALLOWED_KEYS
    if extra:
        raise _ParseError(f"unexpected keys: {sorted(extra)}")
    raw = obj["position_mm"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _ParseError(f"position_mm must be a number, got {type(raw).__name__}")
    value = float(raw)
    if math.isnan(value) or math.isinf(value):
        raise _ParseError(f"position_mm must be finite, got {value}")
    return value


def score_completion(
    *,
    completion: Any,
    ground_truth_mm: float,
    valid_range_mm: tuple[float, float],
    cutoff_frac: float,
    sigma_frac: float,
    format_penalty: float,
    out_of_range_reward: float,
) -> float:
    """Score one completion against its row's ground truth."""
    text = _extract_completion_text(completion)
    try:
        predicted = parse_position_mm(text)
    except _ParseError:
        return float(format_penalty)
    pos_lo, pos_hi = float(valid_range_mm[0]), float(valid_range_mm[1])
    if predicted < pos_lo or predicted > pos_hi:
        return float(out_of_range_reward)
    axis_span_mm = pos_hi - pos_lo
    return normalized_bell_reward(
        predicted - float(ground_truth_mm),
        axis_span_mm=axis_span_mm,
        cutoff_frac=cutoff_frac,
        sigma_frac=sigma_frac,
    )


def make_terminal_reward(
    *,
    cutoff_frac: float = DEFAULT_CUTOFF_FRAC,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
    format_penalty: float = DEFAULT_FORMAT_PENALTY,
    out_of_range_reward: float = DEFAULT_OUT_OF_RANGE_REWARD,
):
    """Build a TRL-shaped reward function bound to schedule + penalty knobs.

    The returned callable matches GRPOTrainer's reward contract:
    ``func(completions, prompts=None, **dataset_columns) -> list[float]``.

    Required dataset columns (passed through by GRPOTrainer):

    * ``ground_truth_mm`` — ``list[float]``, one per row.
    * ``valid_range_mm`` — ``list[tuple[float, float]]``, one per row.
    """
    if cutoff_frac <= 0:
        raise ValueError(f"cutoff_frac must be positive, got {cutoff_frac}")
    if sigma_frac <= 0:
        raise ValueError(f"sigma_frac must be positive, got {sigma_frac}")

    def terminal_reward(
        completions: list[Any] | None = None,
        prompts: list[Any] | None = None,  # noqa: ARG001 — TRL contract
        ground_truth_mm: list[float] | None = None,
        valid_range_mm: list[tuple[float, float]] | None = None,
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
        out: list[float] = []
        for completion, gt, vr in zip(comps, gts, ranges, strict=True):
            out.append(
                score_completion(
                    completion=completion,
                    ground_truth_mm=float(gt),
                    valid_range_mm=(float(vr[0]), float(vr[1])),
                    cutoff_frac=cutoff_frac,
                    sigma_frac=sigma_frac,
                    format_penalty=format_penalty,
                    out_of_range_reward=out_of_range_reward,
                )
            )
        return out

    terminal_reward.__name__ = "terminal_reward"
    terminal_reward.__qualname__ = "terminal_reward"
    return terminal_reward


# Default callable for callers that do not bind a custom schedule.
terminal_reward = make_terminal_reward()


# ---------------------------------------------------------------------------
# Adaptive-reward re-export
# ---------------------------------------------------------------------------
# The unified RL pipeline (Task 6) selects between the static reward above and
# the adaptive bell-curve schedule in ``adaptive_reward.py``. Re-export the
# adaptive factory + schedule from this module so callers (e.g. ``train_grpo``)
# can switch modes via a single import surface. The static path stays
# unchanged and is the fallback when ``--reward-mode static`` is selected.
def make_adaptive_terminal_reward(*args: Any, **kwargs: Any):  # noqa: ANN201
    """Re-export of :func:`single_turn_rl.adaptive_reward.make_adaptive_terminal_reward`.

    The actual implementation lives in :mod:`single_turn_rl.adaptive_reward`;
    we forward via a thin wrapper so a circular-import chain doesn't form
    (the adaptive module imports the static reward's parse helpers).
    """
    from .adaptive_reward import (  # noqa: PLC0415
        make_adaptive_terminal_reward as _impl,
    )

    return _impl(*args, **kwargs)


def adaptive_reward_schedule_cls():  # noqa: ANN201
    """Lazy accessor for :class:`AdaptiveRewardSchedule` (avoids the circular import)."""
    from .adaptive_reward import AdaptiveRewardSchedule  # noqa: PLC0415

    return AdaptiveRewardSchedule
