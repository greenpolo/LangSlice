"""Reward function for single-turn final-answer GRPO (Lane A).

The policy emits one completion — a Gemma 4 ``submit_estimate`` tool call —
and the reward is computed from the completion text plus the dataset-row
kwargs TRL forwards (``ground_truth_mm``, ``valid_range_mm``).

Three reward branches:

1. **Parse failure** (no ``submit_estimate`` tool call found, or it has no
   numeric ``position_mm``): ``format_penalty`` (negative by default).
2. **Out of range** (parsed but outside the plane's ``valid_range_mm``):
   ``out_of_range_reward`` (0.0 by default).
3. **In range**: axis-normalized truncated-Gaussian bell on
   ``abs(position_mm - ground_truth_mm) / axis_span_mm`` via
   :func:`rlvr.rewards.normalized_bell_reward`.

Tool-call format: Gemma 4's chat template renders ``submit_estimate`` as

    <|tool_call>call:submit_estimate{position_mm:3.5,reasoning:<|"|>...<|"|>}<tool_call|>

Authoritative pattern verified against
``transformers /models/sft-base/chat_template.jinja format_argument`` macro
+ ``vllm.tool_parsers.gemma4_utils.parse_tool_calls`` (dictsorted keys, bare
numerics, ``<|"|>`` escape tokens around strings, ``<tool_call|>`` or
``<turn|>`` close sentinel).
"""

from __future__ import annotations

import math
import re
from typing import Any

from langslice_training.rl_core.rewards import normalized_bell_reward

DEFAULT_CUTOFF_FRAC: float = 0.15
DEFAULT_SIGMA_FRAC: float = 0.05
DEFAULT_FORMAT_PENALTY: float = -1.0
DEFAULT_OUT_OF_RANGE_REWARD: float = 0.0


# Gemma 4 tool-call sentinel + name + args + close sentinel.
# `<tool_call|>` is the canonical close; some Gemma 4 outputs emit `<turn|>`
# in its place (per vllm.tool_parsers.gemma4_utils).
_GEMMA4_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\>call:(\w+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)",
    re.DOTALL,
)
# Bare ``call:NAME{...`` form — emitted by E4B-IT in thinking/reasoning mode
# when the assistant prefixes its tool call with prose. The opening
# `<|tool_call>` sentinel is omitted; the args block opens with `{` but the
# closing `}` may not be reliable (reasoning prose can contain `}`), so the
# regex captures the call NAME and lets ``_POSITION_MM_RE`` scan the
# remainder of the text for the numeric arg.
_GEMMA4_BARE_CALL_RE = re.compile(r"call:(\w+)\{", re.DOTALL)
# `position_mm` is rendered as a bare numeric by the chat template's
# format_argument macro (numbers don't get the `<|"|>` string-escape).
_POSITION_MM_RE = re.compile(
    r"position_mm\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
_SUBMIT_TOOL_NAME: str = "submit_estimate"


class _ParseError(Exception):
    """Raised when a completion does not satisfy the submit_estimate contract."""


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
    """Parse ``position_mm`` from the first Gemma 4 ``submit_estimate`` tool call.

    Tolerates surrounding text — thought channels, prose, additional non-submit
    tool calls — and picks the FIRST ``submit_estimate`` invocation. Raises
    :class:`_ParseError` if no ``submit_estimate`` appears or the parsed
    ``position_mm`` isn't a finite number.
    """
    def _parse_numeric(s: str) -> float:
        try:
            value = float(s)
        except ValueError as exc:
            raise _ParseError(f"position_mm not numeric: {s!r}") from exc
        if math.isnan(value) or math.isinf(value):
            raise _ParseError(f"position_mm must be finite, got {value}")
        return value

    # Preferred: wrapped form ``<|tool_call>call:submit_estimate{...}<tool_call|>``.
    for m in _GEMMA4_TOOL_CALL_RE.finditer(text):
        if m.group(1) != _SUBMIT_TOOL_NAME:
            continue
        pos_match = _POSITION_MM_RE.search(m.group(2))
        if pos_match is None:
            raise _ParseError("submit_estimate has no numeric position_mm")
        return _parse_numeric(pos_match.group(1))

    # Fallback: bare ``call:submit_estimate{`` form (no leading sentinel).
    # Scan the text after the opening brace for ``position_mm:N`` — the
    # reasoning prose inside the args block isn't escape-wrapped, so a
    # strict closing ``}`` match isn't reliable.
    for m in _GEMMA4_BARE_CALL_RE.finditer(text):
        if m.group(1) != _SUBMIT_TOOL_NAME:
            continue
        pos_match = _POSITION_MM_RE.search(text, m.end())
        if pos_match is None:
            raise _ParseError("submit_estimate has no numeric position_mm")
        return _parse_numeric(pos_match.group(1))

    raise _ParseError("no submit_estimate tool call found in completion")


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
