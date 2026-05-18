"""Sanity checks for the frozen empirical-distribution constants.

These tests validate that the distribution tables in
``langslice_traces._empirical`` are well-formed (weights sum to 1.0), that the
calibrated SIGMA_ANCHOR_MM lies in a reasonable range, and that the ``sample``
/ ``sample_roundness`` helpers are deterministic + statistically faithful.

Regenerate the constants by running
``models/langslice-traces/scripts/extract_empirical_distributions.py`` if the
SFT corpus is materially changed; these tests should still pass against the
regenerated values.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
from langslice_traces import sample, sample_roundness
from langslice_traces._empirical import (
    OBSERVED_INTEGER_RATE_STEP0,
    OBSERVED_N,
    OBSERVED_R_STEP0_GT,
    P_CENTER_OFFSET_STEP1,
    P_N_TOOL_STEPS,
    P_NFETCH_STEP0,
    P_NFETCH_STEP1,
    P_ROUNDNESS_STEP0,
    P_ROUNDNESS_STEP1,
    P_SPAN_STEP0,
    P_SPAN_STEP1,
    SIGMA_ANCHOR_MM,
)

_SUM_TOLERANCE = 1e-6


def _sum_weights(dist) -> float:
    return sum(w for _, w in dist)


def test_p_n_tool_steps_sums_to_one() -> None:
    assert abs(_sum_weights(P_N_TOOL_STEPS) - 1.0) < _SUM_TOLERANCE


def test_p_nfetch_step0_sums_to_one() -> None:
    assert abs(_sum_weights(P_NFETCH_STEP0) - 1.0) < _SUM_TOLERANCE


def test_p_nfetch_step1_sums_to_one() -> None:
    assert abs(_sum_weights(P_NFETCH_STEP1) - 1.0) < _SUM_TOLERANCE


def test_p_span_step0_sums_to_one() -> None:
    assert abs(_sum_weights(P_SPAN_STEP0) - 1.0) < _SUM_TOLERANCE


def test_p_span_step1_sums_to_one() -> None:
    assert abs(_sum_weights(P_SPAN_STEP1) - 1.0) < _SUM_TOLERANCE


def test_p_center_offset_step1_sums_to_one() -> None:
    assert abs(_sum_weights(P_CENTER_OFFSET_STEP1) - 1.0) < _SUM_TOLERANCE


def test_p_roundness_step0_sums_to_one() -> None:
    assert abs(sum(P_ROUNDNESS_STEP0.values()) - 1.0) < _SUM_TOLERANCE


def test_p_roundness_step1_sums_to_one() -> None:
    assert abs(sum(P_ROUNDNESS_STEP1.values()) - 1.0) < _SUM_TOLERANCE


def test_sigma_anchor_in_reasonable_range() -> None:
    """Calibrated sigma should sit in the regime where r ~ 0.85-0.99.

    Tighter than ~0.5 mm and the corpus would look almost perfectly anchored;
    looser than ~3 mm and r would degrade below 0.8.
    """
    assert 0.5 <= SIGMA_ANCHOR_MM <= 3.0


def test_observed_r_step0_gt_is_corpus_consistent() -> None:
    """Step-0 center is strongly correlated with the eventual GT submit.

    The 1716-row corpus shows r ~ 0.98. If extraction was wrong this would
    land outside the band.
    """
    assert 0.96 < OBSERVED_R_STEP0_GT < 0.99


def test_observed_n_matches_corpus_size() -> None:
    assert OBSERVED_N == 1716


def test_observed_integer_rate_step0_high() -> None:
    """Roughly half of step-0 positions should be whole-mm in the corpus."""
    assert OBSERVED_INTEGER_RATE_STEP0 > 0.4


def test_sample_deterministic() -> None:
    """Same seed yields the same value."""
    dist = [("a", 0.1), ("b", 0.3), ("c", 0.6)]
    assert sample(random.Random(0), dist) == sample(random.Random(0), dist)
    assert sample(random.Random(42), dist) == sample(random.Random(42), dist)


def test_sample_returns_only_known_values() -> None:
    dist = [("a", 0.1), ("b", 0.9)]
    rng = random.Random(0)
    for _ in range(100):
        assert sample(rng, dist) in {"a", "b"}


def test_sample_covers_distribution() -> None:
    """Over 10000 draws, observed frequencies match weights within 0.02."""
    dist = [("a", 0.1), ("b", 0.3), ("c", 0.6)]
    rng = random.Random(123)
    counts: Counter[str] = Counter()
    n = 10_000
    for _ in range(n):
        counts[sample(rng, dist)] += 1
    for value, weight in dist:
        observed = counts[value] / n
        assert abs(observed - weight) < 0.02, f"value={value!r} observed={observed} weight={weight}"


def test_sample_roundness_deterministic() -> None:
    classes = {"int": 0.5, "half": 0.1, "tenth": 0.3, "raw": 0.1}
    assert sample_roundness(random.Random(7), classes) == sample_roundness(
        random.Random(7), classes
    )


def test_sample_roundness_returns_valid_class() -> None:
    rng = random.Random(0)
    for _ in range(200):
        out = sample_roundness(rng, P_ROUNDNESS_STEP0)
        assert out in {"int", "half", "tenth", "raw"}


def test_sample_with_non_normalized_weights() -> None:
    """Helper should tolerate weights that don't sum to 1.0 exactly."""
    dist = [("a", 2.0), ("b", 8.0)]
    rng = random.Random(0)
    counts: Counter[str] = Counter()
    n = 5_000
    for _ in range(n):
        counts[sample(rng, dist)] += 1
    # Expect 20%/80% split within 0.02.
    assert abs(counts["a"] / n - 0.2) < 0.02
    assert abs(counts["b"] / n - 0.8) < 0.02


@pytest.mark.parametrize(
    "dist",
    [P_N_TOOL_STEPS, P_NFETCH_STEP0, P_NFETCH_STEP1],
)
def test_int_distributions_have_positive_weights(dist) -> None:
    for value, weight in dist:
        assert weight > 0.0
        assert isinstance(value, int)


@pytest.mark.parametrize(
    "dist",
    [P_SPAN_STEP0, P_SPAN_STEP1, P_CENTER_OFFSET_STEP1],
)
def test_float_distributions_have_positive_weights(dist) -> None:
    for value, weight in dist:
        assert weight > 0.0
        assert isinstance(value, float)


def test_roundness_classes_present() -> None:
    expected = {"int", "half", "tenth", "raw"}
    assert set(P_ROUNDNESS_STEP0.keys()) == expected
    assert set(P_ROUNDNESS_STEP1.keys()) == expected
