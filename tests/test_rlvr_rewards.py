"""Unit tests for ``rlvr.rewards`` (spec §11 verification 2).

The new reward is a single gated linear ramp on absolute mm error:
    reward = max(0, 1 - |err| / window_mm), default window 0.100 mm.
Single-slice → per-slice score directly. Group → mean of per-slice scores.
Failure modes (no submit, malformed, wrong-kind, wrong-count) → 0.0.
"""

from __future__ import annotations

import pytest
from rlvr.env import LangSliceEstimateEnv
from rlvr.rewards import (
    DEFAULT_WINDOW_MM,
    closeness_reward,
    make_position_reward,
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


# --- closeness_reward (the primitive) --------------------------------------


def test_closeness_reward_exact_hit() -> None:
    assert closeness_reward(0.0) == pytest.approx(1.0)


def test_closeness_reward_half_window_default() -> None:
    """0.050 mm error at default 0.100 mm window → 0.5."""
    assert closeness_reward(0.050) == pytest.approx(0.5)


def test_closeness_reward_half_window_custom() -> None:
    assert closeness_reward(0.025, window_mm=0.050) == pytest.approx(0.5)


def test_closeness_reward_at_window_edge_is_zero() -> None:
    """At |err| == window_mm the ramp is exactly 0."""
    assert closeness_reward(0.100) == pytest.approx(0.0)


def test_closeness_reward_beyond_window_is_zero_no_negatives() -> None:
    assert closeness_reward(0.200) == 0.0
    assert closeness_reward(5.0) == 0.0


def test_closeness_reward_uses_absolute_value() -> None:
    assert closeness_reward(-0.040) == pytest.approx(closeness_reward(0.040))


def test_closeness_reward_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="window_mm must be positive"):
        closeness_reward(0.05, window_mm=0.0)
    with pytest.raises(ValueError, match="window_mm must be positive"):
        closeness_reward(0.05, window_mm=-0.1)


def test_make_position_reward_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        make_position_reward(0.0)


# --- position_reward (the TRL-shaped wrapper) ------------------------------


def test_position_reward_single_exact_hit() -> None:
    env = _make_env()
    env.submit_estimate(5.5, "exact")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(1.0)


def test_position_reward_single_half_window() -> None:
    """Spec §11: |err|=0.050mm at default 0.100mm window → 0.5."""
    env = _make_env()
    env.submit_estimate(5.55, "half-window off")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(0.5)


def test_position_reward_single_at_window_edge_is_zero() -> None:
    env = _make_env()
    env.submit_estimate(5.5 + DEFAULT_WINDOW_MM, "edge")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(0.0)


def test_position_reward_single_beyond_window_is_zero() -> None:
    env = _make_env()
    env.submit_estimate(5.5 + 0.5, "way off")
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
    """Group reward = mean of per-slice closeness scores (no min/worst term)."""
    # GT (1.00, 2.00, 3.00, 4.00); window = 0.100.
    # Predict (1.00, 2.05, 3.025, 5.0) → errs (0, 0.05, 0.025, 1.0)
    # → closeness (1.0, 0.5, 0.75, 0.0) → mean = 0.5625.
    env = _make_env(kind="group", ground_truth=(1.0, 2.0, 3.0, 4.0))
    env.submit_group_estimate([1.0, 2.05, 3.025, 5.0], "mixed")
    [score] = position_reward(environments=[env])
    assert score == pytest.approx(0.5625)


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


# --- custom window via make_position_reward --------------------------------


def test_make_position_reward_custom_window() -> None:
    env = _make_env()
    env.submit_estimate(5.6, "0.1 off")
    reward = make_position_reward(window_mm=0.200)  # 0.1/0.2 → 0.5
    [score] = reward(environments=[env])
    assert score == pytest.approx(0.5)


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
