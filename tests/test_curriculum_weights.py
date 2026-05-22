"""Tests for ``langslice_training.adaptive.curriculum.weights``.

The formula under test is::

    w_bin = (mae_bin / baseline_mae) ** alpha

with optional 0.5-EMA smoothing against ``prev_weights`` BEFORE the
``max_weight_change`` ratio cap is applied, and a 0.1×baseline floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langslice_training.adaptive.curriculum.weights import (
    compute_weights,
    read_per_bin_mae,
    read_weights_json,
    update_weighted_dataset,
    write_weights_json,
)

# ──────────────────────────────────────────────────────────────────────────
# read_per_bin_mae
# ──────────────────────────────────────────────────────────────────────────


def test_read_per_bin_mae_skips_empty_bins(tmp_path: Path) -> None:
    """Bins with ``n=0`` (no rows) must be skipped, not error out."""
    summary = {
        "per_coord_bin": {
            "coronal": {
                "q1": {"n": 12, "mae_mm": 0.32},
                "q2": {"n": 0},  # empty — should be skipped
                "q3": {"n": 8, "mae_mm": 0.51},
            },
            "sagittal": {
                "q1": {"n": 5, "mae_mm": 0.40},
            },
        }
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    out = read_per_bin_mae(path)
    assert out == {
        ("coronal", "q1"): 0.32,
        ("coronal", "q3"): 0.51,
        ("sagittal", "q1"): 0.40,
    }


def test_read_per_bin_mae_handles_missing_block(tmp_path: Path) -> None:
    """Summaries without ``per_coord_bin`` (degenerate) yield empty dict."""
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"overall": {"n": 0}}))
    assert read_per_bin_mae(path) == {}


# ──────────────────────────────────────────────────────────────────────────
# compute_weights — formula
# ──────────────────────────────────────────────────────────────────────────


def test_formula_alpha_one() -> None:
    """``alpha=1.0``: weights are the ratio mae_bin / baseline_mae."""
    per_bin = {("coronal", "q1"): 0.20, ("coronal", "q3"): 0.40}
    section_bins = {
        "s_easy": ("coronal", "q1"),
        "s_hard": ("coronal", "q3"),
    }
    # baseline defaults to mean = 0.30
    weights = compute_weights(per_bin, section_bins)
    assert weights["s_easy"] == pytest.approx(0.20 / 0.30)
    assert weights["s_hard"] == pytest.approx(0.40 / 0.30)


def test_formula_alpha_quadratic() -> None:
    """``alpha=2.0``: ratio is squared."""
    per_bin = {("coronal", "q1"): 0.20, ("coronal", "q3"): 0.40}
    section_bins = {
        "s_easy": ("coronal", "q1"),
        "s_hard": ("coronal", "q3"),
    }
    weights = compute_weights(per_bin, section_bins, alpha=2.0)
    assert weights["s_easy"] == pytest.approx((0.20 / 0.30) ** 2)
    assert weights["s_hard"] == pytest.approx((0.40 / 0.30) ** 2)


def test_explicit_baseline_mae() -> None:
    """``baseline_mae`` overrides the default-mean behaviour."""
    per_bin = {("coronal", "q1"): 0.30, ("coronal", "q3"): 0.60}
    section_bins = {"s_q1": ("coronal", "q1"), "s_q3": ("coronal", "q3")}
    weights = compute_weights(per_bin, section_bins, baseline_mae=0.30)
    assert weights["s_q1"] == pytest.approx(1.0)
    assert weights["s_q3"] == pytest.approx(2.0)


def test_section_without_bin_gets_baseline() -> None:
    """Sections whose bin isn't in per_bin_mae get the baseline weight 1.0
    (subject to the floor)."""
    per_bin = {("coronal", "q1"): 0.30}
    section_bins = {
        "s_known": ("coronal", "q1"),
        "s_orphan": ("coronal", "q5"),  # no eval data for this bin
    }
    weights = compute_weights(per_bin, section_bins)
    assert weights["s_known"] == pytest.approx(1.0)  # ratio = 0.30 / 0.30
    assert weights["s_orphan"] == pytest.approx(1.0)


def test_legacy_int_bin_idx_works() -> None:
    """Plain int bin indices average MAE across planes per q_idx."""
    per_bin = {
        ("coronal", "q1"): 0.20,
        ("sagittal", "q1"): 0.30,  # avg with coronal q1 = 0.25
        ("coronal", "q3"): 0.50,
    }
    section_bins: dict[str, tuple[str, str] | int] = {
        "s_q0_int": 0,  # avg of (coronal q1, sagittal q1) = 0.25
        "s_q2_int": 2,  # only coronal q3 = 0.50
    }
    # baseline = mean(0.20, 0.30, 0.50) = 1.0/3 ≈ 0.333
    weights = compute_weights(per_bin, section_bins)
    baseline = (0.20 + 0.30 + 0.50) / 3
    assert weights["s_q0_int"] == pytest.approx(0.25 / baseline)
    assert weights["s_q2_int"] == pytest.approx(0.50 / baseline)


# ──────────────────────────────────────────────────────────────────────────
# Smoothing + cap + floor
# ──────────────────────────────────────────────────────────────────────────


def test_ema_smoothing_with_prev() -> None:
    """w_smoothed = 0.5 * w_computed + 0.5 * w_prev (default smoothing)."""
    per_bin = {("coronal", "q1"): 0.40}
    section_bins = {"s": ("coronal", "q1")}
    # baseline = 0.40 → raw computed weight = 1.0
    prev = {"s": 4.0}
    weights = compute_weights(
        per_bin, section_bins,
        prev_weights=prev,
        smoothing=0.5,
        max_weight_change=10.0,  # cap is loose here so EMA dominates
    )
    # smoothed = 0.5*1.0 + 0.5*4.0 = 2.5; ratio to prev = 0.625, within cap
    assert weights["s"] == pytest.approx(2.5)


def test_max_weight_change_cap_upward() -> None:
    """When the new weight would exceed ``cap × prev``, clamp upward."""
    # raw computed = 10.0 (10x baseline); prev = 1.0 → smoothed = 5.5
    # ratio_to_prev = 5.5; cap = 3 → clamp to prev*3 = 3.0
    per_bin = {("coronal", "q1"): 1.00}  # baseline=1.0, ratio=1.0... not a spike
    # Build a real spike: hard-set baseline to make the ratio big.
    per_bin = {("coronal", "q1"): 10.0}
    section_bins = {"s": ("coronal", "q1")}
    weights = compute_weights(
        per_bin, section_bins,
        baseline_mae=1.0,
        prev_weights={"s": 1.0},
        max_weight_change=3.0,
    )
    # raw = 10.0; smoothed = 0.5*10 + 0.5*1 = 5.5; capped at 1*3 = 3.0
    assert weights["s"] == pytest.approx(3.0)


def test_max_weight_change_cap_downward() -> None:
    """Symmetric cap: new weight can't drop below prev / cap either."""
    per_bin = {("coronal", "q1"): 0.05}  # 1/20 of baseline
    section_bins = {"s": ("coronal", "q1")}
    # First confirm "no cap when ratio doesn't exceed it":
    weights_loose = compute_weights(
        per_bin, section_bins,
        baseline_mae=1.0,
        prev_weights={"s": 1.0},
        max_weight_change=3.0,
        floor_fraction=0.0,
    )
    # raw = 0.05; smoothed = 0.5*0.05 + 0.5*1.0 = 0.525; ratio = 0.525
    # 1/cap = 0.333... so 0.525 > 0.333, no cap.
    assert weights_loose["s"] == pytest.approx(0.525)
    weights2 = compute_weights(
        per_bin, section_bins,
        baseline_mae=1.0,
        prev_weights={"s": 1.0},
        max_weight_change=1.5,
        floor_fraction=0.0,
        smoothing=0.0,  # disable smoothing so raw 0.05 reaches the cap
    )
    # smoothed = 0.05; ratio = 0.05; 1/cap = 1/1.5 ≈ 0.667 → clamp to 0.667
    assert weights2["s"] == pytest.approx(1.0 / 1.5)


def test_floor_keeps_weights_alive() -> None:
    """No section is permitted to go below ``floor_fraction × baseline``."""
    per_bin = {("coronal", "q1"): 0.001}  # tiny error
    section_bins = {"s": ("coronal", "q1")}
    weights = compute_weights(
        per_bin, section_bins,
        baseline_mae=1.0,
        floor_fraction=0.1,
        smoothing=0.0,
    )
    assert weights["s"] == pytest.approx(0.1)


def test_no_prev_weights_skips_cap_and_smoothing() -> None:
    """Without prev_weights the raw computed value flows through (subject to floor)."""
    per_bin = {("coronal", "q1"): 5.0}
    section_bins = {"s": ("coronal", "q1")}
    weights = compute_weights(
        per_bin, section_bins,
        baseline_mae=1.0,
        max_weight_change=2.0,  # would have capped if prev_weights was set
        prev_weights=None,
    )
    assert weights["s"] == pytest.approx(5.0)


# ──────────────────────────────────────────────────────────────────────────
# update_weighted_dataset
# ──────────────────────────────────────────────────────────────────────────


def test_update_weighted_dataset_maps_section_ids_to_rows() -> None:
    """Rows whose ``section_id`` matches the weights map get those weights."""
    from langslice_training.adaptive.curriculum.sampler import WeightedRowDataset

    rows = [
        {"section_id": "a", "kind": "single",
         "_image_paths": ("nonexistent.png",), "_apply_clahes": (False,),
         "_atlas_long_edge": 256},
        {"section_id": "b", "kind": "single",
         "_image_paths": ("nonexistent.png",), "_apply_clahes": (False,),
         "_atlas_long_edge": 256},
        {"section_id": "c", "kind": "single",
         "_image_paths": ("nonexistent.png",), "_apply_clahes": (False,),
         "_atlas_long_edge": 256},
    ]
    ds = WeightedRowDataset(rows)
    n_matched = update_weighted_dataset(ds, {"a": 2.5, "b": 0.5})
    assert n_matched == 2
    # Row "c" wasn't in the map → baseline weight 1.0
    assert ds._weights == [2.5, 0.5, 1.0]


def test_update_weighted_dataset_baseline_for_missing() -> None:
    """The baseline_weight kwarg controls the default for unmatched rows."""
    from langslice_training.adaptive.curriculum.sampler import WeightedRowDataset

    rows = [
        {"section_id": "a", "kind": "single",
         "_image_paths": ("x.png",), "_apply_clahes": (False,),
         "_atlas_long_edge": 256},
        {"section_id": "x", "kind": "single",
         "_image_paths": ("x.png",), "_apply_clahes": (False,),
         "_atlas_long_edge": 256},
    ]
    ds = WeightedRowDataset(rows)
    update_weighted_dataset(ds, {"a": 5.0}, baseline_weight=0.3)
    assert ds._weights == [5.0, 0.3]


# ──────────────────────────────────────────────────────────────────────────
# round-trip JSON
# ──────────────────────────────────────────────────────────────────────────


def test_write_then_read_weights_json(tmp_path: Path) -> None:
    """Write + read round-trips. Metadata is dropped on read."""
    weights = {"a": 1.5, "b": 0.5, "c": 1.0}
    path = tmp_path / "weights.json"
    write_weights_json(path, weights, metadata={"round": 3})

    out = read_weights_json(path)
    assert out == pytest.approx(weights)
    # metadata is ignored on read but preserved in the file:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["_metadata"]["round"] == 3


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


def test_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        compute_weights({}, {}, alpha=0.0)


def test_invalid_smoothing_raises() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        compute_weights({}, {}, smoothing=-0.1)


def test_invalid_max_weight_change_raises() -> None:
    with pytest.raises(ValueError, match="max_weight_change"):
        compute_weights({}, {}, max_weight_change=0.5)


def test_invalid_floor_fraction_raises() -> None:
    with pytest.raises(ValueError, match="floor_fraction"):
        compute_weights({}, {}, floor_fraction=-0.1)
