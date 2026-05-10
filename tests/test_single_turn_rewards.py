"""Unit tests for ``single_turn_rl.rewards``.

The single-turn reward parses a strict ``{"position_mm": <number>}`` from each
completion and scores it via the same axis-normalized truncated-Gaussian bell
the multi-turn lane uses (``rlvr.rewards.normalized_bell_reward``). Three
branches:

* parse failure → ``format_penalty``,
* parse OK but out of range → ``out_of_range_reward``,
* in range → ``normalized_bell_reward(error_mm, axis_span_mm)``.
"""

from __future__ import annotations

import math

import pytest
from single_turn_rl.rewards import (
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

# --- parse_position_mm -----------------------------------------------------


def test_parse_clean_object() -> None:
    assert parse_position_mm('{"position_mm": 4.37}') == pytest.approx(4.37)


def test_parse_tolerates_surrounding_whitespace() -> None:
    assert parse_position_mm('   \n  {"position_mm": 1.5}  \n  ') == pytest.approx(1.5)


def test_parse_accepts_integer_value() -> None:
    assert parse_position_mm('{"position_mm": 4}') == pytest.approx(4.0)


def test_parse_rejects_prose_around_json() -> None:
    with pytest.raises(_ParseError, match="not a single JSON object"):
        parse_position_mm('Sure, here it is: {"position_mm": 4.37}')


def test_parse_rejects_markdown_fence() -> None:
    with pytest.raises(_ParseError):
        parse_position_mm('```json\n{"position_mm": 4.37}\n```')


def test_parse_rejects_extra_keys() -> None:
    with pytest.raises(_ParseError, match="unexpected keys"):
        parse_position_mm('{"position_mm": 4.37, "reasoning": "looks like CA1"}')


def test_parse_rejects_missing_key() -> None:
    with pytest.raises(_ParseError, match="missing required key"):
        parse_position_mm('{"ap_mm": 4.37}')


def test_parse_rejects_non_object_top_level() -> None:
    # The bare-object regex catches arrays / scalars before json.loads runs.
    with pytest.raises(_ParseError, match="not a single JSON object"):
        parse_position_mm("[4.37]")
    with pytest.raises(_ParseError, match="not a single JSON object"):
        parse_position_mm("4.37")


def test_parse_rejects_non_numeric_value() -> None:
    with pytest.raises(_ParseError, match="must be a number"):
        parse_position_mm('{"position_mm": "4.37"}')


def test_parse_rejects_bool_value() -> None:
    # Bools subclass int in Python; reward must reject explicitly.
    with pytest.raises(_ParseError, match="must be a number"):
        parse_position_mm('{"position_mm": true}')


def test_parse_rejects_nan_and_inf() -> None:
    with pytest.raises(_ParseError, match="must be finite"):
        parse_position_mm('{"position_mm": NaN}')
    with pytest.raises(_ParseError, match="must be finite"):
        parse_position_mm('{"position_mm": Infinity}')


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
    assert _score('{"position_mm": 5.0}') == pytest.approx(1.0)


def test_score_in_range_uses_axis_normalized_bell() -> None:
    assert _score('{"position_mm": 5.4}') == pytest.approx(_bell_at(0.4))


def test_score_at_cutoff_is_zero() -> None:
    assert _score('{"position_mm": 6.5}') == pytest.approx(0.0)


def test_score_format_failure_returns_format_penalty() -> None:
    assert _score("totally not json") == DEFAULT_FORMAT_PENALTY


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
    assert _score('{"position_mm": -1.0}') == DEFAULT_OUT_OF_RANGE_REWARD
    assert _score('{"position_mm": 1000.0}') == DEFAULT_OUT_OF_RANGE_REWARD


def test_score_out_of_range_reward_is_configurable() -> None:
    val = score_completion(
        completion='{"position_mm": -1.0}',
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
            "content": [{"type": "text", "text": '{"position_mm": 5.0}'}],
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


def test_score_chat_format_string_content() -> None:
    chat_completion = [{"role": "assistant", "content": '{"position_mm": 5.0}'}]
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


# --- make_terminal_reward + the TRL-shaped wrapper -------------------------


def test_make_terminal_reward_rejects_nonpositive_schedule() -> None:
    with pytest.raises(ValueError, match="cutoff_frac must be positive"):
        make_terminal_reward(cutoff_frac=0.0)
    with pytest.raises(ValueError, match="sigma_frac must be positive"):
        make_terminal_reward(sigma_frac=-0.05)


def test_terminal_reward_batched_call_returns_list_of_floats() -> None:
    fn = make_terminal_reward()
    out = fn(
        completions=[
            '{"position_mm": 5.0}',  # exact
            '{"position_mm": 6.5}',  # at cutoff
            "garbage",  # format fail
            '{"position_mm": -1.0}',  # OOR
        ],
        ground_truth_mm=[5.0, 5.0, 5.0, 5.0],
        valid_range_mm=[(0.0, 10.0)] * 4,
    )
    assert len(out) == 4
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == DEFAULT_FORMAT_PENALTY
    assert out[3] == DEFAULT_OUT_OF_RANGE_REWARD


def test_terminal_reward_mismatched_lengths_raise() -> None:
    fn = make_terminal_reward()
    with pytest.raises(ValueError, match="must be the same length"):
        fn(
            completions=['{"position_mm": 5.0}'],
            ground_truth_mm=[5.0, 6.0],
            valid_range_mm=[(0.0, 10.0)],
        )


def test_terminal_reward_swallows_unused_dataset_columns() -> None:
    """TRL forwards every dataset column as a kwarg; the reward must ignore extras."""
    fn = make_terminal_reward()
    out = fn(
        completions=['{"position_mm": 5.0}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        section_id=["whatever"],
        atlas_name=["allen_mouse_25um"],
        plane=["coronal"],
    )
    assert out == [pytest.approx(1.0)]


def test_terminal_reward_default_callable() -> None:
    """The module-level ``terminal_reward`` is bound to the default schedule."""
    out = terminal_reward(
        completions=['{"position_mm": 5.0}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
    )
    assert out == [pytest.approx(1.0)]


def test_terminal_reward_handles_empty_input() -> None:
    fn = make_terminal_reward()
    assert fn(completions=[], ground_truth_mm=[], valid_range_mm=[]) == []
    # Defaults to empty when called with no kwargs at all.
    assert fn() == []
