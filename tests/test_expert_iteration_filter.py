"""Unit tests for tools.expert_iteration.filter (pure functions, no I/O)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package import works regardless of where pytest is invoked
# from. The training-container session sets cwd=/workspace/LangSlice, but
# CI may run from a worktree.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tools.expert_iteration.filter import best_of_n, threshold_accept  # noqa: E402


def _row(section_id: str, gen_idx: int, abs_err_mm: float,
         plane_extent_mm: float = 13.2) -> dict:
    return {
        "section_id": section_id,
        "generation_idx": gen_idx,
        "abs_err_mm": abs_err_mm,
        "plane_extent_mm": plane_extent_mm,
        "plane": "coronal",
        "atlas": "allen_mouse_25um",
    }


# ──────────────────────────────────────────────────────────────────────────
# best_of_n
# ──────────────────────────────────────────────────────────────────────────

def test_best_of_n_picks_lowest_error_per_section() -> None:
    rollouts = [
        _row("a", 0, 0.5),
        _row("a", 1, 0.1),    # winner for "a"
        _row("a", 2, 0.3),
        _row("b", 0, 0.9),    # only "b" rollout
        _row("c", 0, 0.2),    # winner for "c"
        _row("c", 1, 0.4),
    ]
    out = best_of_n(rollouts)

    assert len(out) == 3
    by_sid = {r["section_id"]: r for r in out}
    assert by_sid["a"]["generation_idx"] == 1
    assert by_sid["b"]["generation_idx"] == 0
    assert by_sid["c"]["generation_idx"] == 0


def test_best_of_n_preserves_first_seen_section_order() -> None:
    rollouts = [
        _row("c", 0, 0.5),
        _row("a", 0, 0.4),
        _row("b", 0, 0.6),
        _row("a", 1, 0.1),  # later, but a was seen first
    ]
    out = best_of_n(rollouts)
    order = [r["section_id"] for r in out]
    assert order == ["c", "a", "b"]


def test_best_of_n_empty_input() -> None:
    assert best_of_n([]) == []


def test_best_of_n_tie_keeps_first_occurrence() -> None:
    rollouts = [
        _row("a", 0, 0.3),  # tied first
        _row("a", 1, 0.3),  # tied second
    ]
    out = best_of_n(rollouts)
    assert len(out) == 1
    # best_of_n updates only when strictly lower err, so first tie wins.
    assert out[0]["generation_idx"] == 0


# ──────────────────────────────────────────────────────────────────────────
# threshold_accept
# ──────────────────────────────────────────────────────────────────────────

def test_threshold_accept_keeps_rows_within_pct_of_extent() -> None:
    # threshold_pct=0.015 of 13.2mm extent ~= 0.198mm tolerance.
    # We avoid the exact boundary because 0.015*13.2 is not bit-equal to 0.198
    # in IEEE-754 floats — a sub-epsilon error vs 0.20 is the safer test.
    rollouts = [
        _row("a", 0, 0.10),    # accept
        _row("a", 1, 0.30),    # reject
        _row("b", 0, 0.197),   # comfortably under tol — accept
        _row("c", 0, 0.20),    # over tol — reject
    ]
    out = threshold_accept(rollouts, threshold_pct=0.015)

    sids = sorted({(r["section_id"], r["generation_idx"]) for r in out})
    assert sids == [("a", 0), ("b", 0)]


def test_threshold_accept_uses_per_row_extent() -> None:
    """Different planes have different extents — single threshold_pct works."""
    rollouts = [
        _row("a", 0, 0.20, plane_extent_mm=13.2),  # 1.5% = 0.198 → reject
        _row("b", 0, 0.20, plane_extent_mm=20.0),  # 1.5% = 0.300 → accept
    ]
    out = threshold_accept(rollouts, threshold_pct=0.015)
    sids = {r["section_id"] for r in out}
    assert sids == {"b"}


def test_threshold_accept_zero_extent_is_dropped() -> None:
    rollouts = [_row("a", 0, 0.0, plane_extent_mm=0.0)]
    assert threshold_accept(rollouts, threshold_pct=0.015) == []


def test_threshold_accept_rejects_negative_threshold() -> None:
    import pytest
    with pytest.raises(ValueError):
        threshold_accept([], threshold_pct=-0.01)


def test_threshold_accept_can_keep_multiple_generations_per_section() -> None:
    """Unlike best_of_n, threshold_accept does not collapse to one-per-section."""
    rollouts = [
        _row("a", 0, 0.05),
        _row("a", 1, 0.10),
        _row("a", 2, 0.50),
    ]
    out = threshold_accept(rollouts, threshold_pct=0.015)
    # Threshold = 0.198mm of 13.2mm extent. Generations 0+1 pass; 2 fails.
    assert {r["generation_idx"] for r in out} == {0, 1}
