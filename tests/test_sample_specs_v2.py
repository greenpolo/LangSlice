"""Unit tests for ``_sample_specs_v2`` in ``iSFT/iterate.py``.

These tests use a lightweight in-memory setup:
- Fake allocation examples (dataclass with the fields _sample_specs_v2 reads).
- In-memory SeenLedger with manifest_root=None (degenerate fingerprint).
- Deterministic seeded random.Random instances.

No GPU, Docker, or real manifest files needed.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_TRAINING = (
    Path(__file__).resolve().parents[1]
    / "models" / "langslice-gemma-4" / "training"
)
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from iSFT.seen_ledger import SeenLedger  # noqa: E402
from iSFT.iterate import _sample_specs_v2  # noqa: E402


# ---------------------------------------------------------------------------
# Fake allocation example (minimal fields consumed by _sample_specs_v2)
# ---------------------------------------------------------------------------

@dataclass
class FakeExample:
    section_id: str
    atlas_name: str = "fake_atlas"
    plane: str = "coronal"
    ap_mm: float = 1.0
    subject_id: str = "subj_1"
    image_path: Any = Path("/fake/image.jpg")


def _make_pool(
    n: int,
    bin_label: str,
    *,
    plane: str = "coronal",
    prefix: str = "sec",
    ap_base: float = 0.0,
) -> list[FakeExample]:
    """Make n fake examples all assigned to the same bin label."""
    return [
        FakeExample(
            section_id=f"{prefix}_{bin_label}_{i}",
            plane=plane,
            ap_mm=ap_base + i * 0.1,
        )
        for i in range(n)
    ]


def _make_ledger(tmp_path: Path) -> SeenLedger:
    ledger = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=None)
    ledger.load()
    return ledger


def _build_section_bins(
    examples: list[FakeExample],
    label: str,
) -> dict[str, tuple[str, str] | int]:
    return {ex.section_id: (ex.plane, label) for ex in examples}


# ---------------------------------------------------------------------------
# Tests — _sample_specs_v2
# ---------------------------------------------------------------------------

class TestSampleSpecsV2:
    def test_high_weight_bins_sampled_first(self, tmp_path: Path) -> None:
        """Bins with higher weight are drained first.

        Three bins: weight 0.5 (10 sections), 0.3 (10 sections), 0.2 (10 sections).
        Request 15. Expect all 10 from the weight-0.5 bin and 5 from weight-0.3.
        """
        pool_a = _make_pool(10, "q5", prefix="high")   # weight 0.5
        pool_b = _make_pool(10, "q4", prefix="mid")    # weight 0.3
        pool_c = _make_pool(10, "q1", prefix="low")    # weight 0.2

        allocation = pool_a + pool_b + pool_c

        section_bins: dict[str, tuple[str, str] | int] = {}
        section_bins.update(_build_section_bins(pool_a, "q5"))
        section_bins.update(_build_section_bins(pool_b, "q4"))
        section_bins.update(_build_section_bins(pool_c, "q1"))

        bin_weights = {"q5": 0.5, "q4": 0.3, "q1": 0.2}

        ledger = _make_ledger(tmp_path)
        rng = random.Random(42)

        result = _sample_specs_v2(
            allocation=allocation,
            bin_weights=bin_weights,
            section_bins=section_bins,
            seen_ledger=ledger,
            n_target=15,
            rng=rng,
        )

        assert len(result) == 15

        result_ids = {ex.section_id for ex in result}
        high_ids = {ex.section_id for ex in pool_a}
        mid_ids = {ex.section_id for ex in pool_b}
        low_ids = {ex.section_id for ex in pool_c}

        # All 10 from high-weight bin should be present.
        assert high_ids.issubset(result_ids), (
            f"Expected all 10 high-weight sections; got {result_ids & high_ids}"
        )
        # 5 from mid-weight bin.
        assert len(result_ids & mid_ids) == 5
        # None from low-weight bin.
        assert len(result_ids & low_ids) == 0

    def test_skips_seen_sections(self, tmp_path: Path) -> None:
        """Pre-populated ledger entries are not returned."""
        pool = _make_pool(10, "q3")
        section_bins = _build_section_bins(pool, "q3")
        bin_weights = {"q3": 1.0}

        ledger = _make_ledger(tmp_path)
        # Mark first 5 as already seen.
        for ex in pool[:5]:
            ledger.mark_seen(ex.section_id, round_idx=0, source="synthetic")

        rng = random.Random(0)
        result = _sample_specs_v2(
            allocation=pool,
            bin_weights=bin_weights,
            section_bins=section_bins,
            seen_ledger=ledger,
            n_target=10,
            rng=rng,
        )

        result_ids = {ex.section_id for ex in result}
        seen_ids = {ex.section_id for ex in pool[:5]}

        # None of the pre-seen IDs should appear.
        assert result_ids.isdisjoint(seen_ids), (
            f"Found pre-seen IDs in result: {result_ids & seen_ids}"
        )
        # At most 5 unseen sections available.
        assert len(result) == 5

    def test_advances_on_bin_exhaustion(self, tmp_path: Path) -> None:
        """When the first bin runs dry, sampling continues from the next bin."""
        pool_a = _make_pool(5, "q5", prefix="a")   # only 5 sections
        pool_b = _make_pool(10, "q1", prefix="b")  # 10 sections

        allocation = pool_a + pool_b
        section_bins: dict[str, tuple[str, str] | int] = {}
        section_bins.update(_build_section_bins(pool_a, "q5"))
        section_bins.update(_build_section_bins(pool_b, "q1"))

        bin_weights = {"q5": 2.0, "q1": 0.5}  # q5 has higher weight → first

        ledger = _make_ledger(tmp_path)
        rng = random.Random(7)

        result = _sample_specs_v2(
            allocation=allocation,
            bin_weights=bin_weights,
            section_bins=section_bins,
            seen_ledger=ledger,
            n_target=8,
            rng=rng,
        )

        assert len(result) == 8

        result_ids = {ex.section_id for ex in result}
        a_ids = {ex.section_id for ex in pool_a}
        b_ids = {ex.section_id for ex in pool_b}

        # All 5 from pool_a should be included (it's exhausted).
        assert a_ids.issubset(result_ids)
        # Remaining 3 come from pool_b.
        assert len(result_ids & b_ids) == 3

    def test_marks_returned_sections_seen(self, tmp_path: Path) -> None:
        """After calling _sample_specs_v2, returned IDs appear in the ledger."""
        pool = _make_pool(8, "q2")
        section_bins = _build_section_bins(pool, "q2")
        bin_weights = {"q2": 1.0}

        ledger = _make_ledger(tmp_path)
        rng = random.Random(99)

        # The sampler marks sections seen ONLY when we explicitly call mark_seen.
        # _sample_specs_v2 returns candidates but does NOT mark them; the caller
        # (in _phase_rollouts / _phase_append_iterative) marks them.
        # Verify this contract: ledger should be empty after sampling.
        result = _sample_specs_v2(
            allocation=pool,
            bin_weights=bin_weights,
            section_bins=section_bins,
            seen_ledger=ledger,
            n_target=5,
            rng=rng,
        )
        assert len(result) == 5
        # Ledger not yet marked by _sample_specs_v2 itself.
        assert ledger.seen_count() == 0

        # Simulate the caller marking them.
        for ex in result:
            ledger.mark_seen(ex.section_id, round_idx=0, source="real_rollout")

        assert ledger.seen_count() == 5
        for ex in result:
            assert ledger.is_seen(ex.section_id)

    def test_empty_allocation_returns_empty(self, tmp_path: Path) -> None:
        """Empty allocation → empty result, no crash."""
        ledger = _make_ledger(tmp_path)
        rng = random.Random(0)
        result = _sample_specs_v2(
            allocation=[],
            bin_weights={},
            section_bins={},
            seen_ledger=ledger,
            n_target=10,
            rng=rng,
        )
        assert result == []

    def test_all_sections_already_seen_returns_empty(self, tmp_path: Path) -> None:
        """All sections pre-seen → returns empty list (all pools exhausted)."""
        pool = _make_pool(5, "q1")
        section_bins = _build_section_bins(pool, "q1")
        bin_weights = {"q1": 1.0}

        ledger = _make_ledger(tmp_path)
        for ex in pool:
            ledger.mark_seen(ex.section_id, round_idx=0, source="synthetic")

        rng = random.Random(0)
        result = _sample_specs_v2(
            allocation=pool,
            bin_weights=bin_weights,
            section_bins=section_bins,
            seen_ledger=ledger,
            n_target=5,
            rng=rng,
        )
        assert result == []

    def test_integer_bin_keys_supported(self, tmp_path: Path) -> None:
        """section_bins with int values (legacy shape) work without crashing."""
        pool = _make_pool(6, "q3", prefix="int_key")
        # Use integer bin indices instead of (plane, label) tuples.
        section_bins: dict[str, tuple[str, str] | int] = {
            ex.section_id: 2  # bin index 2 → "q3"
            for ex in pool
        }
        bin_weights = {"q3": 1.5}

        ledger = _make_ledger(tmp_path)
        rng = random.Random(11)
        result = _sample_specs_v2(
            allocation=pool,
            bin_weights=bin_weights,
            section_bins=section_bins,
            seen_ledger=ledger,
            n_target=4,
            rng=rng,
        )
        assert len(result) == 4
