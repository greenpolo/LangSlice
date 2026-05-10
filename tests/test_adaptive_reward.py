"""Unit tests for ``single_turn_rl.adaptive_reward``.

The adaptive layer wraps the static single-turn reward with a quantile-based
schedule that auto-tightens the bell as the policy improves. Tests cover:

* the rolling-buffer accessors (record / snapshot / clear / cap),
* :class:`AdaptiveRewardSchedule.current` pre- and post-warmup, including
  bound clamping and deterministic quantiles for a known distribution,
* :meth:`AdaptiveRewardSchedule.per_section_difficulty` for populated and
  unobserved section keys,
* :func:`make_adaptive_terminal_reward` shape, side-effects on the buffer,
  and equivalence with :func:`make_terminal_reward` when the schedule is
  pinned to identical (sigma, cutoff).
"""

from __future__ import annotations

import math

import pytest
from single_turn_rl.adaptive_reward import (
    _RECENT_ERRORS,
    DEFAULT_BUFFER_MAXLEN,
    DEFAULT_STATIC_FALLBACK,
    AdaptiveRewardSchedule,
    clear_recent_errors,
    make_adaptive_terminal_reward,
    recent_errors,
    record_error,
)
from single_turn_rl.rewards import (
    DEFAULT_FORMAT_PENALTY,
    DEFAULT_OUT_OF_RANGE_REWARD,
    make_terminal_reward,
)


@pytest.fixture(autouse=True)
def _isolate_recent_errors() -> None:
    """Reset the module-level rolling buffer before every test."""
    clear_recent_errors()


# --- rolling buffer accessors ---------------------------------------------


def test_record_error_appends_one_entry() -> None:
    record_error(0.4, 10.0, "coronal", "allen_mouse_25um", "s001")
    snap = recent_errors()
    assert snap == [(0.4, 10.0, "coronal", "allen_mouse_25um", "s001")]


def test_record_error_appends_in_order() -> None:
    record_error(0.1, 10.0, "coronal", "ds_a", "s001")
    record_error(0.2, 10.0, "coronal", "ds_a", "s002")
    record_error(0.3, 10.0, "coronal", "ds_a", "s003")
    snap = recent_errors()
    assert [e[0] for e in snap] == [0.1, 0.2, 0.3]


def test_buffer_caps_at_maxlen() -> None:
    # Push 1.5x the maxlen and confirm only the most recent entries survive.
    overflow = DEFAULT_BUFFER_MAXLEN + 50
    for i in range(overflow):
        record_error(float(i), 10.0, "coronal", "ds_a", f"s{i:04d}")
    snap = recent_errors()
    assert len(snap) == DEFAULT_BUFFER_MAXLEN
    # The first 50 entries should have been evicted; the surviving ones
    # start at i=50 and end at i=overflow-1.
    assert snap[0][0] == pytest.approx(50.0)
    assert snap[-1][0] == pytest.approx(float(overflow - 1))


def test_clear_recent_errors_empties_buffer() -> None:
    record_error(0.1, 10.0, "coronal", "ds_a", "s001")
    record_error(0.2, 10.0, "coronal", "ds_a", "s002")
    assert len(recent_errors()) == 2
    clear_recent_errors()
    assert recent_errors() == []
    assert len(_RECENT_ERRORS) == 0


def test_recent_errors_returns_a_snapshot_not_the_live_deque() -> None:
    """Mutating the returned list must not affect future records."""
    record_error(0.1, 10.0, "coronal", "ds_a", "s001")
    snap = recent_errors()
    snap.clear()
    assert len(recent_errors()) == 1


# --- AdaptiveRewardSchedule.current ---------------------------------------


def test_current_returns_static_fallback_pre_warmup() -> None:
    sched = AdaptiveRewardSchedule(min_observations=10)
    # 9 entries < min_observations.
    for i in range(9):
        record_error(0.4, 10.0, "coronal", "ds_a", f"s{i}")
    assert sched.current() == DEFAULT_STATIC_FALLBACK


def test_current_returns_quantiles_once_warmed() -> None:
    # Use very loose clamps so the raw quantiles flow through unchanged.
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
    )
    # error_fracs = i*0.01 for i in [0..99] = [0.00, 0.01, ..., 0.99]
    # Use axis_span = 1 so error_frac == abs_err.
    for i in range(100):
        record_error(i * 0.01, 1.0, "coronal", "ds_a", f"s{i}")
    sigma, cutoff = sched.current()
    # Linear-interp quantile of [0.00, 0.01, ..., 0.99]:
    #   q=0.5  -> pos = 0.5 * 99 = 49.5 -> 0.5*0.49 + 0.5*0.50 = 0.495
    #   q=0.95 -> pos = 0.95 * 99 = 94.05 -> 0.95*0.94 + 0.05*0.95 = 0.9405
    assert sigma == pytest.approx(0.495)
    assert cutoff == pytest.approx(0.9405)


def test_current_clamps_sigma_to_bounds() -> None:
    # Tight bounds well inside the empirical median.
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=0.005,
        max_sigma_frac=0.05,
    )
    # All errors = 0.4 mm on span 1 mm => error_frac = 0.4 (above max).
    for i in range(20):
        record_error(0.4, 1.0, "coronal", "ds_a", f"s{i}")
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.05)  # clamped to max_sigma_frac
    # cutoff stays >= sigma but otherwise unconstrained; raw quantile is 0.4.
    assert cutoff == pytest.approx(0.4)


def test_current_clamps_sigma_up_to_min() -> None:
    sched = AdaptiveRewardSchedule(
        min_observations=10,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=0.01,
        max_sigma_frac=0.05,
    )
    # All errors = 0.001 mm on span 1 mm => error_frac = 0.001 (below min).
    for i in range(20):
        record_error(0.001, 1.0, "coronal", "ds_a", f"s{i}")
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.01)  # clamped up to min_sigma_frac
    # Raw cutoff was 0.001 too, but cutoff is held >= sigma.
    assert cutoff == pytest.approx(0.01)


def test_current_known_distribution_deterministic() -> None:
    """Tight quantiles + a uniform distribution -> exact known sigma."""
    sched = AdaptiveRewardSchedule(
        min_observations=5,
        sigma_quantile=0.0,  # min observation
        cutoff_quantile=1.0,  # max observation
        min_sigma_frac=1e-9,
        max_sigma_frac=10.0,
    )
    # Observations chosen so error_frac = abs_err (span = 1).
    record_error(0.10, 1.0, "coronal", "ds_a", "s1")
    record_error(0.20, 1.0, "coronal", "ds_a", "s2")
    record_error(0.30, 1.0, "coronal", "ds_a", "s3")
    record_error(0.40, 1.0, "coronal", "ds_a", "s4")
    record_error(0.50, 1.0, "coronal", "ds_a", "s5")
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.10)
    assert cutoff == pytest.approx(0.50)


def test_n_observations_property_tracks_buffer() -> None:
    sched = AdaptiveRewardSchedule()
    assert sched.n_observations == 0
    record_error(0.1, 10.0, "coronal", "ds_a", "s1")
    record_error(0.2, 10.0, "coronal", "ds_a", "s2")
    assert sched.n_observations == 2


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError, match="min_sigma_frac must be positive"):
        AdaptiveRewardSchedule(min_sigma_frac=0.0)
    with pytest.raises(ValueError, match="must be >="):
        AdaptiveRewardSchedule(min_sigma_frac=0.05, max_sigma_frac=0.01)
    with pytest.raises(ValueError, match="sigma_quantile must be in"):
        AdaptiveRewardSchedule(sigma_quantile=1.5)
    with pytest.raises(ValueError, match="cutoff_quantile must be in"):
        AdaptiveRewardSchedule(cutoff_quantile=-0.1)
    with pytest.raises(ValueError, match="min_observations must be >= 1"):
        AdaptiveRewardSchedule(min_observations=0)
    with pytest.raises(ValueError, match="static_fallback components must be positive"):
        AdaptiveRewardSchedule(static_fallback=(0.0, 0.15))


# --- per_section_difficulty -----------------------------------------------


def test_per_section_difficulty_returns_running_mean() -> None:
    sched = AdaptiveRewardSchedule()
    key = ("coronal", "allen_mouse_25um", "s042")
    # error_frac values: 0.10, 0.20, 0.30 for the target section.
    record_error(1.0, 10.0, "coronal", "allen_mouse_25um", "s042")
    record_error(2.0, 10.0, "coronal", "allen_mouse_25um", "s042")
    record_error(3.0, 10.0, "coronal", "allen_mouse_25um", "s042")
    # Decoy entries on different sections / datasets / planes.
    record_error(9.0, 10.0, "coronal", "allen_mouse_25um", "s999")
    record_error(9.0, 10.0, "sagittal", "allen_mouse_25um", "s042")
    record_error(9.0, 10.0, "coronal", "other_ds", "s042")
    diff = sched.per_section_difficulty(key)
    assert diff == pytest.approx((0.10 + 0.20 + 0.30) / 3.0)


def test_per_section_difficulty_unobserved_returns_none() -> None:
    sched = AdaptiveRewardSchedule()
    record_error(0.1, 10.0, "coronal", "ds_a", "s_other")
    assert sched.per_section_difficulty(("coronal", "ds_a", "s_missing")) is None


def test_per_section_difficulty_empty_buffer_returns_none() -> None:
    sched = AdaptiveRewardSchedule()
    assert sched.per_section_difficulty(("coronal", "ds_a", "s001")) is None


# --- make_adaptive_terminal_reward ----------------------------------------


def test_make_adaptive_terminal_reward_returns_callable_with_trl_signature() -> None:
    fn = make_adaptive_terminal_reward()
    assert callable(fn)
    # Empty input is the cheapest TRL-shape smoke test.
    assert fn(completions=[], ground_truth_mm=[], valid_range_mm=[]) == []
    # Should also accept the no-kwarg call TRL might issue at startup.
    assert fn() == []


def test_adaptive_reward_records_error_in_deque() -> None:
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=['{"position_mm": 5.4}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
        dataset=["allen_mouse_25um"],
        section_id=["s042"],
    )
    assert len(out) == 1
    snap = recent_errors()
    assert len(snap) == 1
    abs_err, axis_span, plane, dataset, section_id = snap[0]
    assert abs_err == pytest.approx(0.4)
    assert axis_span == pytest.approx(10.0)
    assert plane == "coronal"
    assert dataset == "allen_mouse_25um"
    assert section_id == "s042"


def test_adaptive_reward_format_failure_records_axis_span_sentinel() -> None:
    """Format-fail rows have no parseable position, but we still record an
    observation so the AdaRFT curriculum can see the section is hard.

    Recording a fake zero error would drag the schedule's quantiles toward
    0 and lie about per-section difficulty. Skipping would make hard
    sections look easier than they are and the curriculum would starve
    them. The compromise is to record ``axis_span_mm`` as a bounded
    max-plausible-error sentinel.
    """
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=["totally not json"],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
        dataset=["ds_a"],
        section_id=["s1"],
    )
    assert out == [DEFAULT_FORMAT_PENALTY]
    snap = recent_errors()
    assert len(snap) == 1
    abs_err, axis_span, plane, dataset, section_id = snap[0]
    # axis_span_mm = 10.0 - 0.0 = 10.0; format-fail records abs_err == span.
    assert abs_err == pytest.approx(10.0)
    assert axis_span == pytest.approx(10.0)
    assert plane == "coronal"
    assert dataset == "ds_a"
    assert section_id == "s1"


def test_adaptive_reward_out_of_range_records_capped_error() -> None:
    """OOR rows have a parsed prediction but it's outside the valid range.

    A wildly-off OOR prediction (e.g. position_mm = -1000 on a 10mm axis)
    would otherwise dominate the quantile estimate. Cap the recorded
    absolute error at ``axis_span_mm`` so OOR rows still feed per-section
    difficulty without poisoning the schedule.
    """
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=['{"position_mm": -1.0}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
        dataset=["ds_a"],
        section_id=["s1"],
    )
    assert out == [DEFAULT_OUT_OF_RANGE_REWARD]
    snap = recent_errors()
    assert len(snap) == 1
    abs_err, axis_span, plane, dataset, section_id = snap[0]
    # |(-1.0) - 5.0| = 6.0, axis_span = 10.0, so abs_err == min(6.0, 10.0) = 6.0.
    assert abs_err == pytest.approx(6.0)
    assert axis_span == pytest.approx(10.0)
    assert plane == "coronal"
    assert dataset == "ds_a"
    assert section_id == "s1"


def test_adaptive_reward_out_of_range_caps_extreme_predictions_at_axis_span() -> None:
    """An OOR prediction far outside the axis must be capped at axis_span_mm."""
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=['{"position_mm": -1000.0}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
        plane=["coronal"],
        dataset=["ds_a"],
        section_id=["s1"],
    )
    assert out == [DEFAULT_OUT_OF_RANGE_REWARD]
    snap = recent_errors()
    assert len(snap) == 1
    abs_err, axis_span, *_ = snap[0]
    # Raw |(-1000.0) - 5.0| = 1005.0, capped to axis_span_mm = 10.0.
    assert abs_err == pytest.approx(10.0)
    assert axis_span == pytest.approx(10.0)


def test_adaptive_reward_uses_supplied_schedule() -> None:
    """A planted schedule overrides the default kwargs."""
    sched = AdaptiveRewardSchedule(
        min_observations=1000,  # never warms up here
        static_fallback=(0.05, 0.15),
    )
    fn = make_adaptive_terminal_reward(schedule=sched)
    fn(
        completions=['{"position_mm": 5.0}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
    )
    # Schedule is shared with the closure — records flow into the same buffer.
    assert sched.n_observations == 1


def test_adaptive_reward_matches_static_when_schedule_returns_same_params() -> None:
    """Sanity: when the adaptive schedule pins to (sigma, cutoff) identical to
    what the static reward uses, both rewards must produce identical scores."""
    pinned_sigma = 0.05
    pinned_cutoff = 0.15

    # Pin the adaptive schedule by setting min_observations=1 and feeding it
    # exactly one entry whose error_frac equals both quantile targets — but
    # easier: feed enough identical entries that sigma_quantile == cutoff_quantile
    # all map to the same value, then clamp to the desired band.
    sched = AdaptiveRewardSchedule(
        min_observations=1,
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        min_sigma_frac=pinned_sigma,
        max_sigma_frac=pinned_sigma,  # equal bounds force sigma exactly
        static_fallback=(pinned_sigma, pinned_cutoff),
    )
    # Plant observations whose 95th-percentile error_frac == pinned_cutoff.
    for _ in range(20):
        record_error(pinned_cutoff, 1.0, "coronal", "ds_a", "warm")
    s_now, c_now = sched.current()
    assert s_now == pytest.approx(pinned_sigma)
    assert c_now == pytest.approx(pinned_cutoff)

    # Build both rewards and a batch of inputs that exercise the in-range
    # branch (the only one whose value depends on sigma/cutoff).
    static_fn = make_terminal_reward(
        sigma_frac=pinned_sigma, cutoff_frac=pinned_cutoff
    )
    adapt_fn = make_adaptive_terminal_reward(schedule=sched)
    completions = [
        '{"position_mm": 5.0}',  # exact
        '{"position_mm": 5.4}',  # mid-bell
        '{"position_mm": 5.05}',  # tight
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
        completions=['{"position_mm": 5.0}'],
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
            completions=['{"position_mm": 5.0}'],
            ground_truth_mm=[5.0, 6.0],
            valid_range_mm=[(0.0, 10.0)],
        )


def test_adaptive_reward_falls_back_when_metadata_columns_missing() -> None:
    """Reward must still work when plane/dataset/section_id columns are absent.

    The rolling-buffer entries get empty-string keys but the reward value is
    unaffected.
    """
    fn = make_adaptive_terminal_reward()
    out = fn(
        completions=['{"position_mm": 5.0}'],
        ground_truth_mm=[5.0],
        valid_range_mm=[(0.0, 10.0)],
    )
    assert out == [pytest.approx(1.0)]
    snap = recent_errors()
    assert len(snap) == 1
    _, _, plane, dataset, section_id = snap[0]
    assert plane == ""
    assert dataset == ""
    assert section_id == ""
