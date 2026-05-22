"""Unit tests for ``langslice_training.adaptive.schedule``.

Covers:
- warmup behavior (returns defaults before min_observations)
- tightening with small errors
- widening with large errors
- quantile_cutoff_mm: inf during warmup, correct percentile after
- state isolation between independently-constructed buffers
- backward-compat shim re-exports
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))

from langslice_training.adaptive.schedule import AdaptiveSchedule, make_error_buffer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_buffer(
    buf: deque,
    abs_err: float,
    span: float,
    count: int,
    plane: str = "coronal",
    dataset: str = "ds_a",
) -> None:
    for i in range(count):
        buf.append((abs_err, span, plane, dataset, f"s{i:04d}"))


# ---------------------------------------------------------------------------
# make_error_buffer
# ---------------------------------------------------------------------------

def test_make_error_buffer_returns_empty_deque() -> None:
    buf = make_error_buffer()
    assert len(buf) == 0
    assert isinstance(buf, deque)


def test_make_error_buffer_respects_maxlen() -> None:
    buf = make_error_buffer(maxlen=10)
    assert buf.maxlen == 10


# ---------------------------------------------------------------------------
# Task E tests (exact names required by spec)
# ---------------------------------------------------------------------------

def test_warmup_returns_default_frac() -> None:
    """Buffer with fewer than warmup_min_observations entries returns warmup defaults."""
    sched = AdaptiveSchedule(warmup_min_observations=50)
    buf = make_error_buffer()
    # Add only 49 entries — below the warmup threshold.
    _fill_buffer(buf, abs_err=0.05, span=10.0, count=49)
    sigma, cutoff = sched.compute(buf)
    assert sigma == sched.warmup_sigma_frac
    assert cutoff == sched.warmup_cutoff_frac


def test_tightens_with_low_recent_errors() -> None:
    """Buffer of 100 small errors → sigma and cutoff clamped to lower bound."""
    sched = AdaptiveSchedule(
        sigma_clamp=(0.005, 0.05),
        cutoff_clamp=(0.0, 0.25),
        warmup_min_observations=50,
    )
    buf = make_error_buffer()
    # abs_err=0.05mm, span=10mm → error_frac=0.005 → hits lower sigma clamp.
    _fill_buffer(buf, abs_err=0.05, span=10.0, count=100)
    sigma, cutoff = sched.compute(buf)
    assert sigma == pytest.approx(0.005)  # clamped to lower bound
    # cutoff must be >= sigma
    assert cutoff >= sigma


def test_widens_with_high_recent_errors() -> None:
    """Buffer of 100 large errors → sigma and cutoff near upper clamps."""
    sched = AdaptiveSchedule(
        sigma_clamp=(0.005, 0.05),
        cutoff_clamp=(0.0, 0.25),
        warmup_min_observations=50,
    )
    buf = make_error_buffer()
    # abs_err=1.0mm, span=1.0mm → error_frac=1.0 → hits both upper clamps.
    _fill_buffer(buf, abs_err=1.0, span=1.0, count=100)
    sigma, cutoff = sched.compute(buf)
    assert sigma == pytest.approx(0.05)   # clamped to upper sigma bound
    assert cutoff == pytest.approx(0.25)  # clamped to upper cutoff bound


def test_quantile_cutoff_mm_returns_inf_in_warmup() -> None:
    """Small buffer → quantile_cutoff_mm returns float('inf')."""
    sched = AdaptiveSchedule(warmup_min_observations=50)
    buf = make_error_buffer()
    _fill_buffer(buf, abs_err=0.5, span=10.0, count=10)
    assert sched.quantile_cutoff_mm(buf) == float("inf")


def test_quantile_cutoff_mm_returns_95th_percentile() -> None:
    """Buffer with known distribution → cutoff matches 95th percentile."""
    sched = AdaptiveSchedule(warmup_min_observations=50)
    buf = make_error_buffer()
    # 100 entries with abs_err = 0.0, 0.01, 0.02, ..., 0.99
    for i in range(100):
        buf.append((i * 0.01, 10.0, "coronal", "ds_a", f"s{i:04d}"))

    cutoff = sched.quantile_cutoff_mm(buf, quantile=0.95)
    # 95th percentile of [0.00, 0.01, ..., 0.99] (100 evenly spaced points):
    # index = int(0.95 * 100) = 95 → errs[95] = 0.95
    assert cutoff == pytest.approx(0.95)


def test_separate_buffers_dont_share_state() -> None:
    """Two make_error_buffer() instances are independent — appending to one
    does not affect the other."""
    buf_a = make_error_buffer()
    buf_b = make_error_buffer()
    buf_a.append((0.1, 10.0, "coronal", "ds_a", "s001"))
    assert len(buf_a) == 1
    assert len(buf_b) == 0  # completely independent


def test_single_turn_adaptive_reward_exports_work() -> None:
    """Canonical single-turn module exports AdaptiveRewardSchedule and helpers."""
    from langslice_training.rl.single_turn.adaptive_reward import (  # noqa: PLC0415
        AdaptiveRewardSchedule,
        clear_recent_errors,
        make_adaptive_terminal_reward,
        recent_errors,
        record_error,
    )
    assert callable(AdaptiveRewardSchedule)
    assert callable(make_adaptive_terminal_reward)
    assert callable(record_error)
    assert callable(recent_errors)
    assert callable(clear_recent_errors)
    # Smoke test: the shim's schedule still delegates to AdaptiveSchedule.
    clear_recent_errors()
    sched = AdaptiveRewardSchedule(min_observations=5, static_fallback=(0.05, 0.15))
    # Pre-warmup must return static_fallback.
    sigma, cutoff = sched.current()
    assert sigma == pytest.approx(0.05)
    assert cutoff == pytest.approx(0.15)
    clear_recent_errors()


# ---------------------------------------------------------------------------
# Additional schedule tests
# ---------------------------------------------------------------------------

def test_compute_post_warmup_returns_quantiles() -> None:
    """After warmup, compute() returns quantile-derived values."""
    sched = AdaptiveSchedule(
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        sigma_clamp=(1e-9, 10.0),
        cutoff_clamp=(0.0, 10.0),
        warmup_min_observations=10,
        warmup_sigma_frac=0.05,
        warmup_cutoff_frac=0.15,
    )
    buf = make_error_buffer()
    # error_fracs = [0.00, 0.01, ..., 0.99] (span=1 so frac==abs_err)
    for i in range(100):
        buf.append((i * 0.01, 1.0, "coronal", "ds_a", f"s{i:04d}"))
    sigma, cutoff = sched.compute(buf)
    # q=0.5 of 100 points: pos=49.5 → 0.495
    # q=0.95 of 100 points: pos=94.05 → 0.9405
    assert sigma == pytest.approx(0.495)
    assert cutoff == pytest.approx(0.9405)


def test_plane_filter_per_plane() -> None:
    """Per-plane filter returns the plane-specific quantile."""
    sched = AdaptiveSchedule(
        sigma_quantile=0.5,
        cutoff_quantile=0.95,
        sigma_clamp=(1e-9, 10.0),
        cutoff_clamp=(0.0, 10.0),
        warmup_min_observations=10,
    )
    buf = make_error_buffer()
    # Coronal errors: frac=0.20
    for i in range(30):
        buf.append((0.20, 1.0, "coronal", "ds_a", f"c{i}"))
    # Sagittal errors: frac=0.01
    for i in range(30):
        buf.append((0.01, 1.0, "sagittal", "ds_b", f"s{i}"))

    sigma_cor, _ = sched.compute(buf, plane="coronal")
    sigma_sag, _ = sched.compute(buf, plane="sagittal")
    assert sigma_cor == pytest.approx(0.20)
    assert sigma_sag == pytest.approx(0.01)


def test_quantile_cutoff_mm_with_custom_quantile() -> None:
    """quantile_cutoff_mm respects the quantile parameter."""
    sched = AdaptiveSchedule(warmup_min_observations=10)
    buf = make_error_buffer()
    for i in range(100):
        buf.append((float(i), 1000.0, "coronal", "ds_a", f"s{i:04d}"))
    # 50th percentile of [0..99]: int(0.50*100)=50 → errs[50]=50
    cutoff_50 = sched.quantile_cutoff_mm(buf, quantile=0.50)
    assert cutoff_50 == pytest.approx(50.0)
