"""Unit tests for ``single_turn_rl.rewards``.

The single-turn reward parses ``position_mm`` from a Gemma 4 ``submit_estimate``
tool call and scores it via the same axis-normalized truncated-Gaussian bell
the multi-turn lane uses (``rlvr.rewards.normalized_bell_reward``). Three
branches:

* parse failure → ``format_penalty``,
* parse OK but out of range → ``out_of_range_reward``,
* in range → ``normalized_bell_reward(error_mm, axis_span_mm)``.

The Gemma 4 tool-call format is verified against the chat template
(``format_argument`` macro) and ``vllm.tool_parsers.gemma4_utils``:

    <|tool_call>call:submit_estimate{position_mm:3.5,reasoning:<|"|>...<|"|>}<tool_call|>

Numeric args are bare, string args wrapped in ``<|"|>`` escape tokens, keys
unquoted, dictsorted alphabetically. Some Gemma 4 outputs close with
``<turn|>`` instead of ``<tool_call|>``.
"""

from __future__ import annotations

import math

import pytest
from langslice_training.rl.single_turn.rewards import (
    DEFAULT_CUTOFF_FRAC,
    DEFAULT_FORMAT_PENALTY,
    DEFAULT_OUT_OF_RANGE_REWARD,
    DEFAULT_SIGMA_FRAC,
    _ParseError,
    make_terminal_reward,
    parse_position_mm,
    score_completion,
    terminal_reward,
)


def _submit(position_mm: float, reasoning: str = "Slice shows hippocampus.") -> str:
    """Render a canonical Gemma 4 submit_estimate tool call (dictsorted keys)."""
    return (
        f'<|tool_call>call:submit_estimate{{position_mm:{position_mm},'
        f'reasoning:<|"|>{reasoning}<|"|>}}<tool_call|>'
    )


# --- parse_position_mm -----------------------------------------------------


def test_parse_clean_submit_estimate() -> None:
    assert parse_position_mm(_submit(4.37)) == pytest.approx(4.37)


def test_parse_integer_position() -> None:
    assert parse_position_mm(_submit(4)) == pytest.approx(4.0)


def test_parse_negative_position() -> None:
    assert parse_position_mm(_submit(-1.5)) == pytest.approx(-1.5)


def test_parse_tolerates_surrounding_text() -> None:
    text = (
        "Some thinking text here.\n"
        + _submit(3.5)
        + "\nAnd trailing chatter."
    )
    assert parse_position_mm(text) == pytest.approx(3.5)


def test_parse_tolerates_turn_close_sentinel() -> None:
    text = '<|tool_call>call:submit_estimate{position_mm:2.1}<turn|>'
    assert parse_position_mm(text) == pytest.approx(2.1)


def test_parse_picks_first_submit_skipping_other_tool_calls() -> None:
    text = (
        '<|tool_call>call:fetch_atlas{positions_mm:[2.0,4.0]}<tool_call|>'
        + _submit(7.7)
    )
    assert parse_position_mm(text) == pytest.approx(7.7)


def test_parse_rejects_no_tool_call() -> None:
    with pytest.raises(_ParseError, match="no submit_estimate tool call"):
        parse_position_mm("just some prose without any tool call")


def test_parse_rejects_only_fetch_atlas() -> None:
    with pytest.raises(_ParseError, match="no submit_estimate tool call"):
        parse_position_mm(
            '<|tool_call>call:fetch_atlas{positions_mm:[2.0,4.0]}<tool_call|>'
        )


def test_parse_rejects_submit_estimate_with_no_position_mm() -> None:
    with pytest.raises(_ParseError, match="no numeric position_mm"):
        parse_position_mm(
            '<|tool_call>call:submit_estimate{reasoning:<|"|>maybe?<|"|>}<tool_call|>'
        )


def test_parse_rejects_nan() -> None:
    with pytest.raises(_ParseError, match="no numeric position_mm"):
        # NaN isn't a numeric literal the chat template would ever emit;
        # the regex requires digits, so it doesn't match.
        parse_position_mm(
            '<|tool_call>call:submit_estimate{position_mm:NaN}<tool_call|>'
        )


# --- score_completion ------------------------------------------------------


def _bell_at(error_mm: float, span: float = 10.0) -> float:
    err_frac = abs(error_mm) / span
    if err_frac >= DEFAULT_CUTOFF_FRAC:
        return 0.0
    raw = math.exp(-0.5 * (err_frac / DEFAULT_SIGMA_FRAC) ** 2)
    floor = math.exp(-0.5 * (DEFAULT_CUTOFF_FRAC / DEFAULT_SIGMA_FRAC) ** 2)
    return (raw - floor) / (1.0 - floor)


def _score(text: str, *, gt: float = 5.0, lo: float = 0.0, hi: float = 10.0) -> float:
    return score_completion(
        completion=text,
        ground_truth_mm=gt,
        valid_range_mm=(lo, hi),
        cutoff_frac=DEFAULT_CUTOFF_FRAC,
        sigma_frac=DEFAULT_SIGMA_FRAC,
        format_penalty=DEFAULT_FORMAT_PENALTY,
        out_of_range_reward=DEFAULT_OUT_OF_RANGE_REWARD,
    )


def test_score_exact_hit_is_one() -> None:
    assert _score(_submit(5.0)) == pytest.approx(1.0)


def test_score_in_range_uses_axis_normalized_bell() -> None:
    assert _score(_submit(5.4)) == pytest.approx(_bell_at(0.4))


def test_score_at_cutoff_is_zero() -> None:
    assert _score(_submit(6.5)) == pytest.approx(0.0)


def test_score_format_failure_returns_format_penalty() -> None:
    assert _score("totally not a tool call") == DEFAULT_FORMAT_PENALTY


def test_score_format_penalty_is_configurable() -> None:
    val = score_completion(
        completion="garbage",
        ground_truth_mm=5.0,
        valid_range_mm=(0.0, 10.0),
        cutoff_frac=DEFAULT_CUTOFF_FRAC,
        sigma_frac=DEFAULT_SIGMA_FRAC,
        format_penalty=-2.5,
        out_of_range_reward=0.0,
    )
    assert val == -2.5


def test_score_out_of_range_uses_oor_reward_not_format_penalty() -> None:
    assert _score(_submit(-1.0)) == DEFAULT_OUT_OF_RANGE_REWARD
    assert _score(_submit(1000.0)) == DEFAULT_OUT_OF_RANGE_REWARD


def test_score_out_of_range_reward_is_configurable() -> None:
    val = score_completion(
        completion=_submit(-1.0),
        ground_truth_mm=5.0,
        valid_range_mm=(0.0, 10.0),
        cutoff_frac=DEFAULT_CUTOFF_FRAC,
        sigma_frac=DEFAULT_SIGMA_FRAC,
        format_penalty=-1.0,
        out_of_range_reward=-0.25,
    )
    assert val == -0.25


def test_score_chat_format_completion_extracts_assistant_text() -> None:
    chat_completion = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": _submit(5.0)}],
        }
    ]
    val = score_completion(
        completion=chat_completion,
        ground_truth_mm=5.0,
        valid_range_mm=(0.0, 10.0),
        cutoff_frac=DEFAULT_CUTOFF_FRAC,
        sigma_frac=DEFAULT_SIGMA_FRAC,
        format_penalty=-1.0,
        out_of_range_reward=0.0,
    )
    assert val == pytest.approx(1.0)


# --- make_terminal_reward (batched) ---------------------------------------


def test_make_terminal_reward_batches_completions() -> None:
    fn = make_terminal_reward()
    rewards = fn(
        completions=[_submit(5.0), "garbage", _submit(-1.0)],
        ground_truth_mm=[5.0, 5.0, 5.0],
        valid_range_mm=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
    )
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == DEFAULT_FORMAT_PENALTY
    assert rewards[2] == DEFAULT_OUT_OF_RANGE_REWARD


def test_terminal_reward_export_is_default_callable() -> None:
    # The module-level ``terminal_reward`` is a make_terminal_reward() with defaults.
    rewards = terminal_reward(
        completions=[_submit(5.0)],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
    )
    assert rewards[0] == pytest.approx(1.0)
