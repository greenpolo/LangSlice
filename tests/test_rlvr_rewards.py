"""Unit tests for ``rlvr.rewards`` (spec §11 verification 2).

The reward is a normalized, truncated Gaussian on final coordinate error:
    err_frac = abs(pred - truth) / axis_span_mm
    reward = rescaled exp(-0.5 * (err_frac / sigma_frac)^2)

Exact hits score 1.0. Errors at or beyond ``cutoff_frac`` score 0.0.
Single-slice rollouts get the per-slice score directly; group rollouts get
the mean. Failure modes (no submit, malformed, wrong-kind, wrong-count) → 0.0.
"""

from __future__ import annotations

import pytest
from rlvr.env import LangSliceEstimateEnv
from rlvr.rewards import (
    DEFAULT_CUTOFF_FRAC,
    DEFAULT_SIGMA_FRAC,
    make_position_reward,
    normalized_bell_reward,
    position_reward,
)


class _StubAtlasGrid:
    def range_mm(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ARG002
        from rlvr.atlas_grid import GridRange  # noqa: PLC0415

        return GridRange(pos_lo=0.0, pos_hi=13.2)

    def get_slice(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ARG002
        from PIL import Image  # noqa: PLC0415

        return Image.new("L", (4, 4)), 0.0


def _make_env(
    *,
    kind: str = "single",
    pos_lo: float = 0.0,
    pos_hi: float = 13.2,
    ground_truth: tuple[float, ...] = (5.5,),
) -> LangSliceEstimateEnv:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    e.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(pos_lo, pos_hi),
        ground_truth_positions_mm=ground_truth,
        kind=kind,
    )
    return e


# --- normalized_bell_reward (the primitive) --------------------------------


def _expected_bell(
    error_mm: float,
    *,
    axis_span_mm: float,
    cutoff_frac: float = DEFAULT_CUTOFF_FRAC,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
) -> float:
    import math

    err_frac = abs(error_mm) / axis_span_mm
    if err_frac >= cutoff_frac:
        return 0.0
    raw = math.exp(-0.5 * (err_frac / sigma_frac) ** 2)
    floor = math.exp(-0.5 * (cutoff_frac / sigma_frac) ** 2)
    return (raw - floor) / (1.0 - floor)


def test_normalized_bell_reward_exact_hit() -> None:
    assert normalized_bell_reward(0.0, axis_span_mm=10.0) == pytest.approx(1.0)


def test_normalized_bell_reward_scales_by_axis_span() -> None:
    """The same raw-mm error is more severe on a shorter plane axis."""
    long_axis = normalized_bell_reward(0.5, axis_span_mm=10.0)
    short_axis = normalized_bell_reward(0.5, axis_span_mm=5.0)
    assert long_axis == pytest.approx(_expected_bell(0.5, axis_span_mm=10.0))
    assert short_axis == 0.0
    assert long_axis > short_axis


def test_normalized_bell_reward_at_cutoff_is_zero() -> None:
    """At err_frac == cutoff_frac the truncated reward is exactly 0."""
    assert normalized_bell_reward(1.0, axis_span_mm=10.0) == pytest.approx(0.0)


def test_normalized_bell_reward_beyond_cutoff_is_zero_no_negatives() -> None:
    assert normalized_bell_reward(1.5, axis_span_mm=10.0) == 0.0
    assert normalized_bell_reward(5.0, axis_span_mm=10.0) == 0.0


def test_normalized_bell_reward_uses_absolute_value() -> None:
    assert normalized_bell_reward(-0.4, axis_span_mm=10.0) == pytest.approx(
        normalized_bell_reward(0.4, axis_span_mm=10.0)
    )


def test_normalized_bell_reward_rejects_nonpositive_parameters() -> None:
    with pytest.raises(ValueError, match="axis_span_mm must be positive"):
        normalized_bell_reward(0.05, axis_span_mm=0.0)
    with pytest.raises(ValueError, match="cutoff_frac must be positive"):
        normalized_bell_reward(0.05, axis_span_mm=10.0, cutoff_frac=0.0)
    with pytest.raises(ValueError, match="sigma_frac must be positive"):
        normalized_bell_reward(0.05, axis_span_mm=10.0, sigma_frac=-0.1)


def test_make_position_reward_rejects_nonpositive_parameters() -> None:
    with pytest.raises(ValueError, match="cutoff_frac must be positive"):
        make_position_reward(cutoff_frac=0.0)
    with pytest.raises(ValueError, match="sigma_frac must be positive"):
        make_position_reward(sigma_frac=0.0)


# --- position_reward (the TRL-shaped wrapper) ------------------------------


def test_position_reward_single_exact_hit() -> None:
    env = _make_env()
    env.submit_estimate(5.5, "exact")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(1.0)


def test_position_reward_single_uses_axis_normalized_error() -> None:
    """A 0.5mm error is nonzero on a 10mm axis but zero on a 5mm axis."""
    env = _make_env(pos_lo=0.0, pos_hi=10.0, ground_truth=(5.0,))
    env.submit_estimate(5.5, "5% off")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(_expected_bell(0.5, axis_span_mm=10.0))


def test_position_reward_single_at_cutoff_is_zero() -> None:
    env = _make_env(pos_lo=0.0, pos_hi=10.0, ground_truth=(5.0,))
    env.submit_estimate(6.0, "10% off")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(0.0)


def test_position_reward_single_short_axis_same_mm_error_is_zero() -> None:
    env = _make_env(pos_lo=0.0, pos_hi=5.0, ground_truth=(2.5,))
    env.submit_estimate(3.0, "10% off")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(0.0)


def test_position_reward_no_submit_returns_zero() -> None:
    env = _make_env()
    [score] = position_reward(environments=[env])
    assert score == 0.0


def test_position_reward_wrong_kind_submit_returns_zero() -> None:
    env = _make_env(kind="single")
    # Simulate a malformed state shape — production tools reject this, but
    # the reward must defend against any state.
    env._state.submitted_kind = "group"
    env._state.submitted_positions_mm = (5.5,)
    [score] = position_reward(environments=[env])
    assert score == 0.0


def test_position_reward_malformed_submit_returns_zero() -> None:
    env = _make_env()
    # The env's submit_estimate already rejects non-numeric position_mm and
    # records it as malformed; a reward of 0 follows from no successful submit.
    resp = env.submit_estimate("not a float", "bad")  # type: ignore[arg-type]
    assert resp["status"] == "error"
    [score] = position_reward(environments=[env])
    assert score == 0.0


def test_position_reward_group_mean_of_mixed_errors() -> None:
    """Group reward = mean of per-slice normalized bell scores."""
    env = _make_env(
        kind="group",
        pos_lo=0.0,
        pos_hi=10.0,
        ground_truth=(1.0, 2.0, 3.0, 4.0),
    )
    env.submit_group_estimate([1.0, 2.5, 3.35, 5.5], "mixed")
    [score] = position_reward(environments=[env])
    expected = (
        _expected_bell(0.0, axis_span_mm=10.0)
        + _expected_bell(0.5, axis_span_mm=10.0)
        + _expected_bell(0.35, axis_span_mm=10.0)
        + _expected_bell(1.5, axis_span_mm=10.0)
    ) / 4.0
    assert score == pytest.approx(expected)


def test_position_reward_group_wrong_count_returns_zero() -> None:
    env = _make_env(kind="group", ground_truth=(2.0, 3.0, 4.0, 5.0))
    env.submit_group_estimate([2.0, 3.0, 4.0], "missing one")
    [score] = position_reward(environments=[env])
    assert score == 0.0


def test_position_reward_group_all_perfect() -> None:
    env = _make_env(kind="group", ground_truth=(2.0, 3.0))
    env.submit_group_estimate([2.0, 3.0], "all perfect")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(1.0)


# --- custom schedule via make_position_reward ------------------------------


def test_make_position_reward_custom_cutoff_and_sigma() -> None:
    env = _make_env(pos_lo=0.0, pos_hi=10.0, ground_truth=(5.0,))
    env.submit_estimate(6.0, "10% off")
    reward = make_position_reward(cutoff_frac=0.20, sigma_frac=0.10)
    [score] = reward(environments=[env])
    assert score == pytest.approx(
        _expected_bell(
            1.0,
            axis_span_mm=10.0,
            cutoff_frac=0.20,
            sigma_frac=0.10,
        )
    )


# --- batched call shape ----------------------------------------------------


def test_position_reward_handles_batched_environments() -> None:
    e1 = _make_env()
    e1.submit_estimate(5.5, "exact")
    e2 = _make_env()  # never submits
    e3 = _make_env(kind="group", ground_truth=(2.0, 3.0, 4.0))
    e3.submit_group_estimate([2.0, 3.0, 4.0], "perfect")

    scores = position_reward(environments=[e1, e2, e3])
    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(1.0)


def test_position_reward_tolerates_no_environments_kwarg() -> None:
    assert position_reward() == []
