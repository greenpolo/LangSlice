"""Tests for ``curriculum.sampler.WeightedRowDataset``.

Phase B contract: with uniform weights, ``WeightedRandomSampler`` should
produce a sampling distribution statistically indistinguishable from
``RandomSampler``. The Kolmogorov-Smirnov test below verifies that
property over 1000 samples (p > 0.01).

The trainer subclasses (``CurriculumGRPOTrainer`` /
``CurriculumSFTTrainer``) need TRL installed at import time, so we don't
exercise them here — the hot path under test is the sampler-level
behaviour, which is what the trainer subclasses delegate to.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from langslice_training.adaptive.curriculum.sampler import WeightedRowDataset
from torch.utils.data import RandomSampler, WeightedRandomSampler


def _stub_rows(n: int) -> list[dict[str, Any]]:
    """Build minimal rows that satisfy ``RowDataset.__init__``'s key scan.

    ``RowDataset.__getitem__`` decodes images via PIL and we don't want to
    pay that cost in tests, so we write tests around ``__len__`` and the
    ``_weights`` field directly. The sampler itself only consumes the
    dataset length; index dereferences happen later in the DataLoader.
    """
    return [
        {
            "kind": "single",
            "_image_paths": ("nonexistent.png",),
            "_apply_clahes": (False,),
            "_atlas_long_edge": 256,
            "row_id": i,
        }
        for i in range(n)
    ]


def test_weights_default_to_one() -> None:
    """A fresh ``WeightedRowDataset`` exposes ``_weights`` initialised to 1.0."""
    ds = WeightedRowDataset(_stub_rows(7))
    assert ds._weights == [1.0] * 7
    assert len(ds) == 7


def test_set_weights_updates_field() -> None:
    """``set_weights`` replaces ``_weights`` element-wise."""
    ds = WeightedRowDataset(_stub_rows(4))
    ds.set_weights([0.5, 1.5, 2.0, 0.0])
    assert ds._weights == [0.5, 1.5, 2.0, 0.0]


def test_set_weights_rejects_wrong_length() -> None:
    ds = WeightedRowDataset(_stub_rows(4))
    try:
        ds.set_weights([1.0, 2.0])
    except ValueError as exc:
        assert "len(dataset)" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for wrong-length weights")


def test_set_weights_rejects_negative_values() -> None:
    ds = WeightedRowDataset(_stub_rows(3))
    try:
        ds.set_weights([1.0, -0.5, 1.0])
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for negative weights")


def test_set_weights_rejects_all_zero() -> None:
    ds = WeightedRowDataset(_stub_rows(3))
    try:
        ds.set_weights([0.0, 0.0, 0.0])
    except ValueError as exc:
        assert "> 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for all-zero weights")


def _ks_two_sample(a: list[int], b: list[int]) -> float:
    """Two-sample Kolmogorov-Smirnov test, returning a *p-value approximation*.

    We avoid pulling scipy into the test deps for one stat. The standard
    asymptotic formula is::

        D = max |F_a(x) - F_b(x)|
        p ≈ 2 * exp(-2 * D^2 * (n*m / (n+m)))

    Returns the approximate two-sided p-value.
    """
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    universe = sorted(set(a_sorted) | set(b_sorted))
    n_a = len(a_sorted)
    n_b = len(b_sorted)
    cum_a = 0
    cum_b = 0
    i_a = 0
    i_b = 0
    d = 0.0
    for x in universe:
        while i_a < n_a and a_sorted[i_a] <= x:
            i_a += 1
        while i_b < n_b and b_sorted[i_b] <= x:
            i_b += 1
        cum_a = i_a
        cum_b = i_b
        diff = abs(cum_a / n_a - cum_b / n_b)
        if diff > d:
            d = diff
    import math

    factor = (n_a * n_b) / (n_a + n_b)
    return 2.0 * math.exp(-2.0 * d * d * factor)


def test_uniform_weights_indistinguishable_from_random_sampler() -> None:
    """With uniform weights, WeightedRandomSampler ≈ RandomSampler.

    Sample 1000 indices from a 100-row dataset under each sampler with
    fixed seeds, then run a 2-sample Kolmogorov-Smirnov test on the index
    distributions. The p-value should be > 0.01 (i.e. we can't reject the
    null hypothesis that the two samples are drawn from the same
    distribution). Setting ``replacement=True`` on the RandomSampler so
    both share the with-replacement convention.
    """
    n_rows = 100
    n_samples = 1000

    ds = WeightedRowDataset(_stub_rows(n_rows))
    g_w = torch.Generator().manual_seed(42)
    g_r = torch.Generator().manual_seed(43)

    weighted = WeightedRandomSampler(
        weights=ds._weights,
        num_samples=n_samples,
        replacement=True,
        generator=g_w,
    )
    random = RandomSampler(
        ds,
        replacement=True,
        num_samples=n_samples,
        generator=g_r,
    )

    weighted_idxs = list(weighted)
    random_idxs = list(random)

    assert len(weighted_idxs) == n_samples
    assert len(random_idxs) == n_samples
    assert all(0 <= i < n_rows for i in weighted_idxs)
    assert all(0 <= i < n_rows for i in random_idxs)

    p = _ks_two_sample(weighted_idxs, random_idxs)
    assert p > 0.01, f"K-S p-value too low: {p:.4f}"

    # Spot-check: each row should appear roughly n_samples/n_rows = 10 times.
    # A Bernoulli-trial deviation past ±5x is very unlikely under uniform.
    counts_w = Counter(weighted_idxs)
    counts_r = Counter(random_idxs)
    expected = n_samples / n_rows
    assert all(abs(counts_w[i] - expected) < 5 * expected for i in range(n_rows))
    assert all(abs(counts_r[i] - expected) < 5 * expected for i in range(n_rows))


def test_skewed_weights_actually_skew_sampling() -> None:
    """Sanity check: when weights are NOT uniform, the sampler reflects them.

    Phase C will rely on this. With weights ``[0, 0, ..., 0, 1]`` only the
    last index should ever be drawn.
    """
    n_rows = 10
    ds = WeightedRowDataset(_stub_rows(n_rows))
    weights = [0.0] * n_rows
    weights[-1] = 1.0
    ds.set_weights(weights)

    g = torch.Generator().manual_seed(7)
    sampler = WeightedRandomSampler(
        weights=ds._weights,
        num_samples=200,
        replacement=True,
        generator=g,
    )
    idxs = list(sampler)
    assert all(i == n_rows - 1 for i in idxs)
