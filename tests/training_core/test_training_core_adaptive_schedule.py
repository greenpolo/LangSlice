from __future__ import annotations

from langslice_training.adaptive.schedule import AdaptiveSchedule, _quantile


def test_quantile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _quantile(values, 0.5) == 2.5
    assert _quantile(values, 0.25) == 1.75


def test_quantile_cutoff_warmup_is_infinite() -> None:
    sched = AdaptiveSchedule(warmup_min_observations=3)
    buffer = [(0.3, 10.0, "coronal", "atlas", "s1")]
    assert sched.quantile_cutoff_mm(buffer) == float("inf")
