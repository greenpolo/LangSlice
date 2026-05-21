"""Unit tests for adaptive accept/reject threshold wiring in iSFT.

Tests the _kept_rollouts_for_filter() helper in iterate.py, exercising:
- disabled (default) → byte-identical to static threshold_accept path
- enabled with all-small-error buffer → tight cutoff, only best rollouts pass
- enabled with all-large-error buffer → loose cutoff, more rollouts pass
- warmup pass-through → all rollouts accepted when buffer < warmup_min_n

No real model runs, vLLM, or Docker. Uses synthetic scored-rollout dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))

from adaptive.schedule import make_error_buffer  # noqa: E402

# Direct import of the function under test (not via module; it's a module-level
# function that we import directly so tests never touch vLLM or Docker).
from iSFT.iterate import _kept_rollouts_for_filter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    section_id: str,
    gen_idx: int,
    abs_err_mm: float,
    plane_extent_mm: float = 13.2,
    plane: str = "coronal",
    predicted_mm: float = 5.0,
) -> dict:
    """Minimal scored-rollout row for filter tests."""
    return {
        "section_id": section_id,
        "generation_idx": gen_idx,
        "abs_err_mm": abs_err_mm,
        "plane_extent_mm": plane_extent_mm,
        "plane": plane,
        "atlas": "allen_mouse_25um",
        "predicted_mm": predicted_mm,
    }


def _fill_buffer(buf, abs_err: float, span: float, count: int) -> None:
    for i in range(count):
        buf.append((abs_err, span, "coronal", "ds_a", f"s{i:04d}"))


# ---------------------------------------------------------------------------
# test_adaptive_accept_disabled_byte_identical
# ---------------------------------------------------------------------------

def test_adaptive_accept_disabled_byte_identical() -> None:
    """With adaptive_buffer=None, filter output is byte-identical to static path."""
    rollouts = [
        _row("a", 0, 0.10),   # 0.10 <= 0.015*13.2=0.198 → accept
        _row("a", 1, 0.30),   # reject
        _row("b", 0, 0.05),   # accept
        _row("c", 0, 0.25),   # reject
    ]
    # Static path (baseline)
    from iSFT.filter import threshold_accept  # noqa: PLC0415
    static_expected = threshold_accept(rollouts, threshold_pct=0.015)

    # _kept_rollouts_for_filter with adaptive_buffer=None must match.
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="threshold-accept",
        threshold_pct=0.015,
        adaptive_buffer=None,
    )
    # Same section_ids + generation_idx pairs in same order.
    assert [(r["section_id"], r["generation_idx"]) for r in result] == [
        (r["section_id"], r["generation_idx"]) for r in static_expected
    ]


def test_adaptive_accept_disabled_best_of_n_byte_identical() -> None:
    """best-of-n mode with no adaptive buffer works unchanged."""
    rollouts = [
        _row("a", 0, 0.5),
        _row("a", 1, 0.1),  # winner
        _row("b", 0, 0.3),
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="best-of-n",
        threshold_pct=0.015,
        adaptive_buffer=None,
    )
    assert len(result) == 2
    by_sid = {r["section_id"]: r for r in result}
    assert by_sid["a"]["generation_idx"] == 1


# ---------------------------------------------------------------------------
# test_adaptive_accept_tightens_on_low_mae
# ---------------------------------------------------------------------------

def test_adaptive_accept_tightens_on_low_mae() -> None:
    """Pre-populate buffer with small errors; only tight rollouts pass."""
    buf = make_error_buffer()
    # 100 observations with abs_err=0.05mm → 95th percentile cutoff ≈ 0.05mm
    _fill_buffer(buf, abs_err=0.05, span=10.0, count=100)

    rollouts = [
        _row("a", 0, 0.04),   # under 0.05 cutoff → pass
        _row("a", 1, 0.06),   # over 0.05 cutoff → fail
        _row(
            "b", 0, 0.05
        ),   # exactly at (or just over) cutoff; int(0.95*100)=95 → errs[95]=0.05 → pass
        _row("c", 0, 0.20),   # well over → fail
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="threshold-accept",
        threshold_pct=0.5,  # generous static threshold — adaptive should tighten
        adaptive_buffer=buf,
        adaptive_quantile=0.95,
        adaptive_warmup_n=50,
    )
    kept_ids = {(r["section_id"], r["generation_idx"]) for r in result}
    # c(0.20mm) must be rejected
    assert ("c", 0) not in kept_ids
    # a(0.04mm) must pass
    assert ("a", 0) in kept_ids


def test_adaptive_accept_rejects_outliers_tighter_than_static() -> None:
    """Adaptive threshold based on small-error buffer is tighter than a static 5% threshold."""
    buf = make_error_buffer()
    _fill_buffer(buf, abs_err=0.02, span=10.0, count=100)

    # Static 5% of 13.2mm = 0.66mm tolerance → would accept everything
    rollouts = [
        _row("x", 0, 0.01),  # well inside adaptive cutoff → accept
        _row("x", 1, 0.50),  # outside adaptive cutoff (>0.02) → reject
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="threshold-accept",
        threshold_pct=0.05,
        adaptive_buffer=buf,
        adaptive_quantile=0.95,
        adaptive_warmup_n=50,
    )
    kept = {r["generation_idx"] for r in result}
    assert 0 in kept        # x gen 0 passes
    assert 1 not in kept    # x gen 1 rejected by adaptive threshold


# ---------------------------------------------------------------------------
# test_adaptive_accept_widens_on_high_mae
# ---------------------------------------------------------------------------

def test_adaptive_accept_widens_on_high_mae() -> None:
    """Buffer with large errors gives a loose cutoff that doesn't starve corpus."""
    buf = make_error_buffer()
    # 100 observations with abs_err=2.0mm → 95th percentile ≈ 2.0mm
    _fill_buffer(buf, abs_err=2.0, span=10.0, count=100)

    rollouts = [
        _row("a", 0, 0.5),   # well under 2.0mm → accept
        _row("b", 0, 1.5),   # under 2.0mm → accept
        _row("c", 0, 2.1),   # slightly over → may reject
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="threshold-accept",
        threshold_pct=0.001,  # very tight static threshold — adaptive should be looser
        adaptive_buffer=buf,
        adaptive_quantile=0.95,
        adaptive_warmup_n=50,
    )
    kept_ids = {r["section_id"] for r in result}
    # a and b should pass (0.5mm and 1.5mm are under 2.0mm cutoff)
    assert "a" in kept_ids
    assert "b" in kept_ids


# ---------------------------------------------------------------------------
# Warmup: all rollouts pass
# ---------------------------------------------------------------------------

def test_warmup_passthrough_all_accepted() -> None:
    """When buffer < warmup_min_n, cutoff = inf → all rollouts pass."""
    buf = make_error_buffer()
    _fill_buffer(buf, abs_err=0.01, span=10.0, count=10)  # below warmup=50

    rollouts = [
        _row("a", 0, 5.0),   # very large error
        _row("b", 0, 3.0),
        _row("c", 0, 0.1),
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="threshold-accept",
        threshold_pct=0.0,  # would reject everything if used
        adaptive_buffer=buf,
        adaptive_quantile=0.95,
        adaptive_warmup_n=50,
    )
    # All must pass because we're in warmup → cutoff = inf
    assert len(result) == 3


# ---------------------------------------------------------------------------
# best-of-n mode is unaffected by adaptive_buffer
# ---------------------------------------------------------------------------

def test_adaptive_accept_best_of_n_still_picks_lowest() -> None:
    """best-of-n with adaptive buffer still picks lowest error per section."""
    buf = make_error_buffer()
    _fill_buffer(buf, abs_err=0.01, span=10.0, count=100)  # tight buffer

    rollouts = [
        _row("a", 0, 5.0),   # high error
        _row("a", 1, 0.01),  # best rollout for "a"
        _row("b", 0, 0.5),
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="best-of-n",
        threshold_pct=0.015,
        adaptive_buffer=buf,
        adaptive_quantile=0.95,
        adaptive_warmup_n=50,
    )
    by_sid = {r["section_id"]: r for r in result}
    # best-of-n is not filtered by adaptive cutoff — picks lowest per section.
    assert by_sid["a"]["generation_idx"] == 1
    assert "b" in by_sid


# ---------------------------------------------------------------------------
# None-predicted rows excluded before adaptive filtering
# ---------------------------------------------------------------------------

def test_none_predicted_excluded() -> None:
    """Rows without predicted_mm are excluded before any filter path."""
    buf = make_error_buffer()
    _fill_buffer(buf, abs_err=0.1, span=10.0, count=100)

    rollouts = [
        _row("a", 0, 0.05),
        {"section_id": "b", "generation_idx": 0, "abs_err_mm": 0.05,
         "plane_extent_mm": 13.2, "plane": "coronal", "atlas": "x",
         "predicted_mm": None},   # excluded
    ]
    result = _kept_rollouts_for_filter(
        rollouts,
        filter_mode="threshold-accept",
        threshold_pct=0.5,
        adaptive_buffer=buf,
        adaptive_quantile=0.95,
        adaptive_warmup_n=50,
    )
    sids = {r["section_id"] for r in result}
    assert "b" not in sids
    assert "a" in sids
