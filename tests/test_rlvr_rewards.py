"""Unit tests for ``rlvr.rewards`` (spec §11 verification 2).

Reward functions only read ``env._state``, so we drive them with directly
constructed ``LangSliceEstimateEnv`` instances rather than running rollouts.
"""

from __future__ import annotations

import pytest
from rlvr.env import LangSliceEstimateEnv
from rlvr.rewards import (
    GROUP_TURN_BUDGET,
    SINGLE_TURN_BUDGET,
    TOL_FLOOR_MM,
    format_compliance_reward,
    length_budget_reward,
    position_accuracy_reward,
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


# --- position_accuracy_reward ----------------------------------------------


def test_position_accuracy_perfect_single() -> None:
    env = _make_env()
    env.submit_estimate(5.6, "close")  # err=0.10mm, tol=floor 0.20mm → tier 1.0
    [score] = position_accuracy_reward(environments=[env])
    assert score == pytest.approx(1.0)


def test_position_accuracy_canned_trajectory_err_0p1mm() -> None:
    """Spec §11: ``|err|=0.1mm → position_accuracy_reward=1.0``."""
    env = _make_env()
    env.submit_estimate(5.6, "close")
    [score] = position_accuracy_reward(environments=[env])
    assert score == pytest.approx(1.0)


def test_position_accuracy_tier_breakdown() -> None:
    env = _make_env()
    # tol = max(0.02 * 13.2, 0.2) = 0.264 mm → 2*tol=0.528, 4*tol=1.056.
    env._state.submitted_kind = "single"
    env._state.submitted_positions_mm = (5.5 + 0.4,)  # in (tol, 2*tol]
    [score] = position_accuracy_reward(environments=[env])
    assert score == pytest.approx(0.5)

    env._state.submitted_positions_mm = (5.5 + 0.7,)  # in (2*tol, 4*tol]
    [score] = position_accuracy_reward(environments=[env])
    assert score == pytest.approx(0.25)

    env._state.submitted_positions_mm = (5.5 + 2.0,)  # past 4*tol
    [score] = position_accuracy_reward(environments=[env])
    assert score == pytest.approx(0.0)


def test_position_accuracy_no_submit_returns_zero() -> None:
    env = _make_env()
    [score] = position_accuracy_reward(environments=[env])
    assert score == 0.0


def test_position_accuracy_wrong_kind_submit_returns_zero() -> None:
    env = _make_env(kind="single")
    # Force-set a group submission on a single env — can't happen via tools
    # (kind check rejects), but reward must defend against any state shape.
    env._state.submitted_kind = "group"
    env._state.submitted_positions_mm = (5.5,)
    [score] = position_accuracy_reward(environments=[env])
    assert score == 0.0


def test_position_accuracy_group_aggregate() -> None:
    # Use a small range so tol = TOL_FLOOR_MM = 0.20 mm exactly.
    # 4*tol = 0.80 mm so a 1.0 mm error lands past the bottom tier.
    env = _make_env(kind="group", pos_lo=0.0, pos_hi=2.0, ground_truth=(0.5, 1.0, 1.2, 1.5))
    # Three perfect, one off by 1.0 mm → per-slice tiers [1, 1, 1, 0].
    env.submit_group_estimate([0.5, 1.0, 1.2, 2.5], "ok")
    [score] = position_accuracy_reward(environments=[env])
    # 0.5 * mean(1,1,1,0) + 0.5 * worst = 0.5 * 0.75 + 0.5 * 0 = 0.375
    assert score == pytest.approx(0.375)


def test_position_accuracy_group_wrong_count_returns_zero() -> None:
    env = _make_env(kind="group", ground_truth=(2.0, 3.0, 4.0, 5.0))
    env.submit_group_estimate([2.0, 3.0, 4.0], "missing one")
    [score] = position_accuracy_reward(environments=[env])
    assert score == 0.0


def test_position_accuracy_tolerance_uses_floor_for_small_atlas() -> None:
    env = _make_env(pos_lo=0.0, pos_hi=2.0)  # 0.02 * 2.0 = 0.04 mm < TOL_FLOOR_MM
    env.submit_estimate(env._state.ground_truth_positions_mm[0] + TOL_FLOOR_MM - 0.01, "ok")
    [score] = position_accuracy_reward(environments=[env])
    assert score == pytest.approx(1.0)


# --- format_compliance_reward ----------------------------------------------


def test_format_compliance_full_credit() -> None:
    env = _make_env()
    env.fetch_atlas([2.0, 4.0, 6.0])
    env.submit_estimate(5.6, "ok")
    [score] = format_compliance_reward(environments=[env])
    # +0.2 (one submit) + 0.2 (kind matches) - 0 (no malformed) - 0 (submitted) = 0.4
    assert score == pytest.approx(0.4)


def test_format_compliance_no_submission_penalty() -> None:
    """Spec §11: ``no submission → format_compliance_reward = -0.5``."""
    env = _make_env()
    [score] = format_compliance_reward(environments=[env])
    assert score == pytest.approx(-0.5)


def test_format_compliance_malformed_calls_subtract() -> None:
    env = _make_env(kind="single")
    # Wrong-kind submit + bad fetch payload each register one malformed call.
    env.submit_group_estimate([1.0, 2.0], "wrong kind")
    env.fetch_atlas([])
    [score] = format_compliance_reward(environments=[env])
    # submit_calls=1 → +0.2 (the attempt counts), submitted_kind=None → 0,
    # 2 malformed × -0.2 = -0.4, no real submission → -0.5. Total = -0.7.
    assert score == pytest.approx(-0.7)


def test_format_compliance_double_submit_loses_one_call_credit() -> None:
    env = _make_env()
    env.submit_estimate(5.6, "ok")
    env.submit_estimate(5.7, "again")  # second call still counted
    [score] = format_compliance_reward(environments=[env])
    # +0.2 NOT awarded (submit_calls=2), +0.2 (kind matches), -0.0 (none malformed)
    assert score == pytest.approx(0.2)


# --- length_budget_reward --------------------------------------------------


def test_length_budget_within_budget_zero() -> None:
    env = _make_env()
    env._state.turns = SINGLE_TURN_BUDGET
    [score] = length_budget_reward(environments=[env])
    assert score == 0.0


def test_length_budget_single_12_turns_minus_0p2() -> None:
    """Spec §11: ``12-turn single-slice trajectory → length_budget_reward = -0.2``."""
    env = _make_env()
    env._state.turns = 12  # budget=8, excess=4 → -0.05*4 = -0.20
    [score] = length_budget_reward(environments=[env])
    assert score == pytest.approx(-0.2)


def test_length_budget_group_uses_group_budget() -> None:
    env = _make_env(kind="group", ground_truth=(2.0, 3.0))
    env._state.turns = GROUP_TURN_BUDGET + 3  # excess=3 → -0.15
    [score] = length_budget_reward(environments=[env])
    assert score == pytest.approx(-0.15)


# --- batched call shape ----------------------------------------------------


def test_rewards_handle_batched_environments() -> None:
    e1 = _make_env()
    e1.submit_estimate(5.6, "close")
    e2 = _make_env()  # never submits
    e3 = _make_env(kind="group", ground_truth=(2.0, 3.0, 4.0))
    e3.submit_group_estimate([2.0, 3.0, 4.0], "perfect")

    accs = position_accuracy_reward(environments=[e1, e2, e3])
    fmts = format_compliance_reward(environments=[e1, e2, e3])
    lens = length_budget_reward(environments=[e1, e2, e3])
    assert len(accs) == len(fmts) == len(lens) == 3
    assert accs[0] == pytest.approx(1.0)
    assert accs[1] == 0.0
    assert accs[2] == pytest.approx(1.0)
    assert fmts[1] == pytest.approx(-0.5)


def test_rewards_tolerate_no_environments_kwarg() -> None:
    # TRL passes ``environments`` as a kwarg, but defensive default keeps the
    # functions importable / debuggable in isolation.
    assert position_accuracy_reward() == []
    assert format_compliance_reward() == []
    assert length_budget_reward() == []
