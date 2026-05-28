"""Unit tests for ``langslice_training.rl.single_turn.adaptive_reward``.

The adaptive layer wraps the static single-turn reward with a quantile-based
schedule that auto-tightens the bell as the policy improves. Tests cover:

* the rolling-buffer accessors (record / snapshot / clear / cap),
* :class:`AdaptiveRewardSchedule.current` pre- and post-warmup, including
  bound clamping and deterministic quantiles for a known distribution,
* :func:`compute_ap_bin` (round-trip + edge clamp),
* :meth:`AdaptiveRewardSchedule.per_bin_difficulty` for populated and
  unobserved bin keys,
* :func:`make_adaptive_terminal_reward` shape, side-effects on the buffer,
  and equivalence with :func:`make_terminal_reward` when the schedule is
  pinned to identical (sigma, cutoff).
"""

from __future__ import annotations

import math

import pytest
from langslice_training.rl.single_turn.adaptive_reward import (
    _AP_BIN_COUNT,
    _RECENT_ERRORS,
    DEFAULT_BUFFER_MAXLEN,
    DEFAULT_STATIC_FALLBACK,
    AdaptiveRewardSchedule,
    clear_recent_errors,
    compute_ap_bin,
    make_adaptive_terminal_reward,
    recent_errors,
    record_error,
)
from langslice_training.rl.single_turn.rewards import (
    DEFAULT_FORMAT_PENALTY,
    DEFAULT_OUT_OF_RANGE_REWARD,
    make_terminal_reward,
)


@pytest.fixture(autouse=True)
def _isolate_recent_errors() -> None:
    """Reset the module-level rolling buffer before every test."""
    clear_recent_errors()


# --- compute_ap_bin --------------------------------------------------------


def test_compute_ap_bin_anterior_pole_returns_zero() -> None:
    """gt == pos_lo → bin 0."""
    assert compute_ap_bin(0.0, (0.0, 10.0)) == 0


def test_compute_ap_bin_posterior_pole_returns_last_bin() -> None:
    """gt == pos_hi → bin N-1 (clamped from the int(1.0 * N) overflow)."""
    assert compute_ap_bin(10.0, (0.0, 10.0)) == _AP_BIN_COUNT - 1


def test_compute_ap_bin_midpoint() -> None:
    """gt at the axis midpoint → middle bin."""
    # ap_pct = 0.5 → int(0.5 * 20) = 10
    assert compute_ap_bin(5.0, (0.0, 10.0)) == _AP_BIN_COUNT // 2


def test_compute_ap_bin_clamps_below_lower_bound() -> None:
    """gt below pos_lo → bin 0 (clamped, not negative)."""
    assert compute_ap_bin(-100.0, (0.0, 10.0)) == 0


def test_compute_ap_bin_clamps_above_upper_bound() -> None:
    """gt above pos_hi → bin N-1 (clamped, not out-of-range)."""
    assert compute_ap_bin(1000.0, (0.0, 10.0)) == _AP_BIN_COUNT - 1


def test_compute_ap_bin_degenerate_axis_returns_zero() -> None:
    """Zero-span axis → bin 0 (no meaningful percent to compute)."""
    assert compute_ap_bin(5.0, (5.0, 5.0)) == 0


def test_compute_ap_bin_round_trip_uniform_grid() -> None:
    """Walking gt across the axis hits every bin in order."""
    seen: list[int] = []
    for i in range(_AP_BIN_COUNT):
        # Sample slightly above each bin's lower edge to avoid float ambiguity.
        gt = (i + 0.5) / _AP_BIN_COUNT * 10.0
        seen.append(compute_ap_bin(gt, (0.0, 10.0)))
    assert seen == list(range(_AP_BIN_COUNT))


# --- rolling buffer accessors ---------------------------------------------


def test_record_error_appends_one_entry() -> None:
    record_error(0.4, 10.0, "coronal", 5)
    snap = recent_errors()
    # abs_err_pct = 0.4 / 10.0 = 0.04
    assert snap == [(pytest.approx(0.04), "coronal", 5)]


def test_record_error_drops_degenerate_axis() -> None:
    """axis_span <= 0 → no record (avoids div-by-zero / sign flip)."""
    record_error(0.4, 0.0, "coronal", 5)
    record_error(0.4, -1.0, "coronal", 5)
    assert recent_errors() == []


def test_record_error_appends_in_order() -> None:
    record_error(0.1, 10.0, "coronal", 0)
    record_error(0.2, 10.0, "coronal", 1)
    record_error(0.3, 10.0, "coronal", 2)
    snap = recent_errors()
    assert [e[0] for e in snap] == [
        pytest.approx(0.01),
        pytest.approx(0.02),
        pytest.approx(0.03),
    ]


def test_buffer_caps_at_maxlen() -> None:
    overflow = DEFAULT_BUFFER_MAXLEN + 50
    for i in range(overflow):
        record_error(float(i), 1.0, "coronal", i % _AP_BIN_COUNT)
    snap = recent_errors()
    assert len(snap) == DEFAULT_BUFFER_MAXLEN
    # The first 50 entries should have been evicted.
    assert snap[0][0] == pytest.approx(50.0)
    assert snap[-1][0] == pytest.approx(float(overflow - 1))


def test_clear_recent_errors_empties_buffer() -> None:
    record_error(0.1, 10.0, "coronal", 0)
    record_error(0.2, 10.0, "coronal", 1)
    assert len(recent_errors()) == 2
    clear_recent_errors()
    assert recent_errors() == []
    assert len(_RECENT_ERRORS) == 0


def test_recent_errors_returns_a_snapshot_not_the_live_deque() -> None:
    """Mutating the returned list must not affect future records."""
    record_error(0.1, 10.0, "coronal", 0)
    snap = recent_errors()
    snap.clear()
    assert len(recent_errors()) == 1


# --- AdaptiveRewardSchedule.current ---------------------------------------


def test_current_returns_static_fallback_pre_warmup() -> None:
    sched = AdaptiveRewardSchedule(min_observations=10)
    for i in range(9):
        record_error(0.4, 10.0, "coronal", i)
    assert sched.current() == DEFAULT_STATIC_FALLBACK


def test_current_returns_quantiles_once_warmed() -> None:
    # Use very loose clamps so the raw quantiles flow through unchanged.
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    # error_pcts = i*0.01 for i in [0..99]
    for i in range(100):
        record_error(i * 0.01, 1.0, "coronal", i % _AP_BIN_COUNT)
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.495)
    assert cutoff == pytest.approx(0.9405)


def test_current_clamps_sigma_to_bounds() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=0.005,
        max_sigma_frac=0.05,
        max_cutoff_frac=1.0,
    )
    for i in range(20):
        record_error(0.4, 1.0, "coronal", i % _AP_BIN_COUNT)
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.05)
    assert cutoff == pytest.approx(0.4)


def test_current_clamps_sigma_up_to_min() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=0.01,
        max_sigma_frac=0.05,
    )
    for i in range(20):
        record_error(0.001, 1.0, "coronal", i % _AP_BIN_COUNT)
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.01)
    assert cutoff == pytest.approx(0.01)


def test_current_known_distribution_deterministic() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=5,
        sigma_quantile=0.0,
        cutoff_quantile=1.0,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    record_error(0.10, 1.0, "coronal", 0)
    record_error(0.20, 1.0, "coronal", 1)
    record_error(0.30, 1.0, "coronal", 2)
    record_error(0.40, 1.0, "coronal", 3)
    record_error(0.50, 1.0, "coronal", 4)
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.10)
    assert cutoff == pytest.approx(0.50)


def test_current_clamps_cutoff_to_max_cutoff_frac() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=0.25,
    )
    for i in range(50):
        record_error(10.0, 10.0, "coronal", i % _AP_BIN_COUNT)
    sigma_frac, cutoff_frac = sched.current()
    assert cutoff_frac == pytest.approx(0.25)


def test_max_cutoff_frac_default_is_025() -> None:
    sched = AdaptiveRewardSchedule()
    assert sched.max_cutoff_frac == pytest.approx(0.25)


def test_per_plane_quantile_uses_filtered_subset() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    for i in range(30):
        record_error(0.20, 1.0, "coronal", i % _AP_BIN_COUNT)
    for i in range(30):
        record_error(0.01, 1.0, "sagittal", i % _AP_BIN_COUNT)

    sigma_global, _ = sched.current()
    sigma_coronal, _ = sched.current(plane="coronal")
    sigma_sagittal, _ = sched.current(plane="sagittal")

    assert sigma_coronal == pytest.approx(0.20)
    assert sigma_sagittal == pytest.approx(0.01)
    assert sigma_sagittal < sigma_global < sigma_coronal


def test_per_plane_falls_back_to_global_when_below_min_observations() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    for i in range(50):
        record_error(0.10, 1.0, "coronal", i % _AP_BIN_COUNT)
    record_error(0.001, 1.0, "horizontal", 0)
    record_error(0.002, 1.0, "horizontal", 1)

    sigma_global, _ = sched.current()
    sigma_horizontal, _ = sched.current(plane="horizontal")
    assert sigma_horizontal == pytest.approx(sigma_global)


def test_current_no_plane_argument_returns_global_quantiles_unchanged() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=5,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    for i in range(20):
        record_error(0.05, 1.0, "coronal", i % _AP_BIN_COUNT)
    no_arg = sched.current()
    explicit_none = sched.current(plane=None)
    assert no_arg == explicit_none


def test_n_observations_property_tracks_buffer() -> None:
    sched = AdaptiveRewardSchedule()
    assert sched.n_observations == 0
    record_error(0.1, 10.0, "coronal", 0)
    record_error(0.2, 10.0, "coronal", 1)
    assert sched.n_observations == 2


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError, match="min_sigma_frac must be positive"):
        AdaptiveRewardSchedule(min_sigma_frac=0.0)
    with pytest.raises(ValueError, match="max_sigma_frac.*must be >="):
        AdaptiveRewardSchedule(min_sigma_frac=0.05, max_sigma_frac=0.01)
    with pytest.raises(ValueError, match="max_cutoff_frac.*must be >="):
        AdaptiveRewardSchedule(min_sigma_frac=0.05, max_cutoff_frac=0.001)
    with pytest.raises(ValueError, match="sigma_quantile must be in"):
        AdaptiveRewardSchedule(sigma_quantile=1.5)
    with pytest.raises(ValueError, match="cutoff_quantile must be in"):
        AdaptiveRewardSchedule(cutoff_quantile=-0.1)
    with pytest.raises(ValueError, match="min_observations must be >= 1"):
        AdaptiveRewardSchedule(min_observations=0)
    with pytest.raises(ValueError, match="static_fallback components must be positive"):
        AdaptiveRewardSchedule(static_fallback=(0.0, 0.15))


# --- per_bin_difficulty ----------------------------------------------------


def test_per_bin_difficulty_returns_running_mean() -> None:
    sched = AdaptiveRewardSchedule()
    # Three observations on (coronal, bin=5): 0.01, 0.02, 0.03 pct.
    record_error(0.1, 10.0, "coronal", 5)
    record_error(0.2, 10.0, "coronal", 5)
    record_error(0.3, 10.0, "coronal", 5)
    # Decoy entries on different bins / planes.
    record_error(9.0, 10.0, "coronal", 7)
    record_error(9.0, 10.0, "sagittal", 5)
    diff = sched.per_bin_difficulty("coronal", 5)
    assert diff == pytest.approx((0.01 + 0.02 + 0.03) / 3.0)


def test_per_bin_difficulty_multiple_bins_independent() -> None:
    """Two bins on the same plane have independent running means."""
    sched = AdaptiveRewardSchedule()
    record_error(1.0, 10.0, "coronal", 0)  # 0.1
    record_error(2.0, 10.0, "coronal", 0)  # 0.2
    record_error(5.0, 10.0, "coronal", 10)  # 0.5
    record_error(7.0, 10.0, "coronal", 10)  # 0.7
    assert sched.per_bin_difficulty("coronal", 0) == pytest.approx(0.15)
    assert sched.per_bin_difficulty("coronal", 10) == pytest.approx(0.6)


def test_per_bin_difficulty_plane_filter() -> None:
    """Same bin number on different planes does not pool."""
    sched = AdaptiveRewardSchedule()
    record_error(1.0, 10.0, "coronal", 5)
    record_error(9.0, 10.0, "sagittal", 5)
    # Coronal mean is 0.1; sagittal is 0.9 — must not mix.
    assert sched.per_bin_difficulty("coronal", 5) == pytest.approx(0.1)
    assert sched.per_bin_difficulty("sagittal", 5) == pytest.approx(0.9)


def test_per_bin_difficulty_unobserved_returns_none() -> None:
    sched = AdaptiveRewardSchedule()
    record_error(0.1, 10.0, "coronal", 0)
    assert sched.per_bin_difficulty("coronal", 19) is None


def test_per_bin_difficulty_empty_buffer_returns_none() -> None:
    sched = AdaptiveRewardSchedule()
    assert sched.per_bin_difficulty("coronal", 0) is None


def test_per_section_difficulty_raises_not_implemented() -> None:
    """The legacy per-section API points callers at the new bin method."""
    sched = AdaptiveRewardSchedule()
    with pytest.raises(NotImplementedError, match="per_bin_difficulty"):
        sched.per_section_difficulty(("coronal", "ds_a", "s001"))


# --- make_adaptive_terminal_reward ----------------------------------------


def test_make_adaptive_terminal_reward_returns_callable_with_trl_signature() -> None:
    fn = make_adaptive_terminal_reward()
    assert callable(fn)
    assert fn(completions=[], ground_truth_mm=[], valid_range_mm=[]) == []
    assert fn() == []


def test_adaptive_reward_records_error_in_deque() -> None:
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:5.4}<tool_call|>"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
    )
    assert len(out) == 1
    snap = recent_errors()
    assert len(snap) == 1
    abs_err_pct, plane, ap_bin = snap[0]
    # |5.4 - 5.0| / 10.0 = 0.04
    assert abs_err_pct == pytest.approx(0.04)
    assert plane == "coronal"
    # gt=5.0, axis (0,10) → ap_pct=0.5 → bin = int(0.5*20) = 10
    assert ap_bin == compute_ap_bin(5.0, (0.0, 10.0))


def test_adaptive_reward_format_failure_records_axis_span_sentinel() -> None:
    """Format-fail rows record the axis-span sentinel so the curriculum can
    still see the bin is hard."""
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["totally not json"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
    )
    assert out == [DEFAULT_FORMAT_PENALTY]
    snap = recent_errors()
    assert len(snap) == 1
    abs_err_pct, plane, ap_bin = snap[0]
    # axis_span_mm = 10, abs_err = 10 → pct = 1.0
    assert abs_err_pct == pytest.approx(1.0)
    assert plane == "coronal"
    assert ap_bin == compute_ap_bin(5.0, (0.0, 10.0))


def test_adaptive_reward_out_of_range_records_capped_error() -> None:
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:-1.0}<tool_call|>"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
    )
    assert out == [DEFAULT_OUT_OF_RANGE_REWARD]
    snap = recent_errors()
    assert len(snap) == 1
    abs_err_pct, plane, _ap_bin = snap[0]
    # |-1.0 - 5.0| = 6.0 → pct = 0.6
    assert abs_err_pct == pytest.approx(0.6)
    assert plane == "coronal"


def test_adaptive_reward_out_of_range_caps_extreme_predictions_at_axis_span() -> None:
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:-1000.0}<tool_call|>"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
    )
    assert out == [DEFAULT_OUT_OF_RANGE_REWARD]
    snap = recent_errors()
    assert len(snap) == 1
    abs_err_pct, _, _ = snap[0]
    # capped to axis_span / axis_span = 1.0
    assert abs_err_pct == pytest.approx(1.0)


def test_adaptive_reward_uses_supplied_schedule() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=1000,
        static_fallback=(0.05, 0.15),
    )
    fn = make_adaptive_terminal_reward(schedule=sched)
    fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
    )
    assert sched.n_observations == 1


def test_adaptive_reward_matches_static_when_schedule_returns_same_params() -> None:
    pinned_sigma = 0.05
    pinned_cutoff = 0.15

    sched = AdaptiveRewardSchedule(
        min_observations=1,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=pinned_sigma,
        max_sigma_frac=pinned_sigma,
        static_fallback=(pinned_sigma, pinned_cutoff),
    )
    for _ in range(20):
        record_error(pinned_cutoff, 1.0, "coronal", 0)
    s_now, c_now = sched.current()
    assert s_now == pytest.approx(pinned_sigma)
    assert c_now == pytest.approx(pinned_cutoff)

    static_fn = make_terminal_reward(
        sigma_frac=pinned_sigma, cutoff_frac=pinned_cutoff
    )
    adapt_fn = make_adaptive_terminal_reward(schedule=sched)
    completions = [
        "<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>",
        "<|tool_call>call:submit_estimate{position_mm:5.4}<tool_call|>",
        "<|tool_call>call:submit_estimate{position_mm:5.05}<tool_call|>",
    ]
    gts = [5.0, 5.0, 5.0]
    ranges = [(0.0, 10.0)] * 3
    static_out = static_fn(
        completions=completions, ground_truth_mm=gts, valid_range_mm=ranges
    )
    adapt_out = adapt_fn(
        completions=completions, ground_truth_mm=gts, valid_range_mm=ranges
    )
    assert len(static_out) == len(adapt_out) == 3
    for s, a in zip(static_out, adapt_out, strict=True):
        assert s == pytest.approx(a)
        assert not math.isnan(a)


def test_adaptive_reward_swallows_unused_dataset_columns() -> None:
    """TRL forwards every dataset column as a kwarg; reward must ignore extras."""
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        section_id=["s1"],
        atlas_name=["allen_mouse_25um"],
        plane=["coronal"],
        dataset=["ds_a"],
        irrelevant_extra=["foo"],
    )
    assert out == [pytest.approx(1.0)]


def test_adaptive_reward_mismatched_lengths_raise() -> None:
    fn = make_adaptive_terminal_reward()
    with pytest.raises(ValueError, match="must be the same length"):
        fn(
            completions=["<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>"],
            ground_truth_mm=[5.0, 6.0],
            valid_range_mm=[(0.0, 10.0)],
        )


def test_adaptive_reward_falls_back_when_metadata_columns_missing() -> None:
    """Reward must still work when plane is absent.

    The rolling-buffer entries get empty-string plane but the reward value
    is unaffected.
    """
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:5.0}<tool_call|>"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
    )
    assert out == [pytest.approx(1.0)]
    snap = recent_errors()
    assert len(snap) == 1
    _, plane, _ = snap[0]
    assert plane == ""


# --- per-bin adaptive sigma (chase-the-peak per coordinate range) ----------


def test_current_for_bin_discriminates_accurate_vs_inaccurate_bin() -> None:
    """With per_bin enabled, each AP bin derives sigma from its OWN error
    history: a bin the model is accurate at tightens while a bin it's weak at
    stays lenient — even within the same plane. Per-plane sigma can't do this.
    """
    sched = AdaptiveRewardSchedule(
        per_bin=True,
        min_bin_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    for _ in range(20):
        record_error(0.04, 1.0, "coronal", 5)  # accurate bin: 4% errors
    for _ in range(20):
        record_error(0.20, 1.0, "coronal", 15)  # weak bin: 20% errors
    sigma_good, _ = sched.current_for_bin("coronal", 5)
    sigma_bad, _ = sched.current_for_bin("coronal", 15)
    assert sigma_good == pytest.approx(0.04)
    assert sigma_bad == pytest.approx(0.20)
    assert sigma_good < sigma_bad


def test_current_for_bin_cold_bin_falls_back_to_plane_global() -> None:
    """A bin below min_bin_observations uses the plane-global sigma rather
    than a noisy 1-2 observation estimate."""
    sched = AdaptiveRewardSchedule(
        per_bin=True,
        min_bin_observations=10,
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    for _ in range(20):
        record_error(0.08, 1.0, "coronal", 5)
    # bin 7 has only 2 observations — below the per-bin warmup.
    record_error(0.01, 1.0, "coronal", 7)
    record_error(0.02, 1.0, "coronal", 7)
    assert sched.current_for_bin("coronal", 7) == sched.current(plane="coronal")


def test_current_for_bin_delegates_to_plane_when_per_bin_disabled() -> None:
    """per_bin=False preserves the legacy per-plane behavior for every bin."""
    sched = AdaptiveRewardSchedule(
        per_bin=False,
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    for _ in range(20):
        record_error(0.04, 1.0, "coronal", 5)
    for _ in range(20):
        record_error(0.20, 1.0, "coronal", 15)
    assert sched.current_for_bin("coronal", 5) == sched.current(plane="coronal")
    assert sched.current_for_bin("coronal", 15) == sched.current(plane="coronal")


def test_current_for_bin_clamps_to_min_sigma_floor() -> None:
    """A mastered bin's sigma floors at min_sigma_frac — the fixed precision
    'peak' the bin chases, instead of shrinking toward zero (and chasing
    label noise)."""
    sched = AdaptiveRewardSchedule(
        per_bin=True,
        min_bin_observations=10,
        sigma_quantile=0.5,
        min_sigma_frac=0.02,
        max_sigma_frac=0.15,
        max_cutoff_frac=0.25,
    )
    for _ in range(20):
        record_error(0.001, 1.0, "coronal", 5)  # near-perfect: 0.1% errors
    sigma, _ = sched.current_for_bin("coronal", 5)
    assert sigma == pytest.approx(0.02)


def test_adaptive_reward_scores_same_error_stricter_in_accurate_bin() -> None:
    """End-to-end: the same absolute error scores lower in a bin the model
    has mastered (tight sigma) than in a bin it's weak at (loose sigma)."""
    sched = AdaptiveRewardSchedule(
        per_bin=True,
        min_bin_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
        max_cutoff_frac=10.0,
    )
    good_bin = compute_ap_bin(2.5, (0.0, 10.0))
    bad_bin = compute_ap_bin(7.5, (0.0, 10.0))
    for _ in range(20):
        record_error(0.02, 1.0, "coronal", good_bin)
    for _ in range(20):
        record_error(0.20, 1.0, "coronal", bad_bin)
    fn = make_adaptive_terminal_reward(schedule=sched)
    # 0.3 mm error = 3% of the 10 mm axis, applied in each bin.
    r_good = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:2.8}<tool_call|>"],
        ground_truth_mm=[2.5],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
    )[0]
    r_bad = fn(
        completions=["<|tool_call>call:submit_estimate{position_mm:7.8}<tool_call|>"],
        ground_truth_mm=[7.5],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
    )[0]
    assert r_good < r_bad


def test_per_bin_invalid_min_bin_observations_raises() -> None:
    with pytest.raises(ValueError, match="min_bin_observations must be >= 1"):
        AdaptiveRewardSchedule(per_bin=True, min_bin_observations=0)
