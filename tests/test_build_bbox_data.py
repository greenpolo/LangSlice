"""Tests for build_bbox_data orchestrator (stage 'sample')."""
from __future__ import annotations

import build_bbox_data
import numpy as np


def test_pick_source_prefers_real_when_eligible():
    coverage_index = {
        "AL1A": {
            "Hippocampal Formation": {
                "atlas": "allen_mouse_25um",
                "orientation": "coronal",
                "section_ids": ["AL1A:s001", "AL1A:s005", "AL1A:s010", "AL1A:s015"],
            }
        }
    }
    rng = np.random.default_rng(42)
    decision = build_bbox_data.pick_source(
        atlas="allen_mouse_25um",
        orientation="coronal",
        landmark="Hippocampal Formation",
        coverage_index=coverage_index,
        rng=rng,
    )
    assert decision["source_type"] == "real_histology"
    assert decision["source_brain"] == "AL1A"


def test_pick_source_falls_through_when_real_thin():
    coverage_index = {
        "AL1A": {
            "Hippocampal Formation": {
                "atlas": "allen_mouse_25um",
                "orientation": "coronal",
                "section_ids": ["AL1A:s001"],  # only 1 -- below >=4 gate
            }
        }
    }
    rng = np.random.default_rng(42)
    decision = build_bbox_data.pick_source(
        atlas="allen_mouse_25um",
        orientation="coronal",
        landmark="Hippocampal Formation",
        coverage_index=coverage_index,
        rng=rng,
    )
    assert decision["source_type"] in {"augmented_atlas", "reference_atlas"}
    assert decision["source_brain"] is None


def test_pick_source_respects_per_brain_region_cap():
    coverage_index = {
        "AL1A": {
            "Hippocampal Formation": {
                "atlas": "allen_mouse_25um",
                "orientation": "coronal",
                "section_ids": ["AL1A:s001", "AL1A:s005", "AL1A:s010", "AL1A:s015"],
            }
        }
    }
    rng = np.random.default_rng(42)
    decision = build_bbox_data.pick_source(
        atlas="allen_mouse_25um",
        orientation="coronal",
        landmark="Hippocampal Formation",
        coverage_index=coverage_index,
        rng=rng,
        source_counts={("AL1A", "Hippocampal Formation"): 3},
    )
    assert decision["source_type"] in {"augmented_atlas", "reference_atlas"}
    assert decision["source_brain"] is None


def test_sample_section_count_is_in_range():
    rng = np.random.default_rng(0)
    counts = [build_bbox_data.sample_section_count(rng) for _ in range(2000)]
    assert min(counts) == 4
    assert max(counts) == 8
    assert 4 in counts and 8 in counts and 5 in counts and 6 in counts and 7 in counts


def test_sample_spacings_mm_are_independent_and_in_range():
    rng = np.random.default_rng(0)
    samples = [build_bbox_data.sample_spacings_mm(rng, n_gaps=5) for _ in range(2000)]
    flat = [gap for gaps in samples for gap in gaps]
    assert all(len(gaps) == 5 for gaps in samples)
    assert min(flat) >= 0.2
    assert max(flat) <= 0.8
    assert any(len({round(gap, 6) for gap in gaps}) > 1 for gaps in samples)


def test_sample_anchor_returns_none_when_extent_too_narrow():
    """If region's mm extent can't fit minimum span ((4-1)x0.2=0.6 mm), reject."""
    rng = np.random.default_rng(0)
    anchor = build_bbox_data.sample_anchor_mm(
        rng=rng, region_mm_min=2.0, region_mm_max=2.5,
        spacings_mm=[0.2, 0.2, 0.2],
    )
    assert anchor is None


def test_sample_anchor_returns_value_when_extent_fits():
    rng = np.random.default_rng(0)
    anchor = build_bbox_data.sample_anchor_mm(
        rng=rng, region_mm_min=2.0, region_mm_max=6.0,
        spacings_mm=[0.2, 0.3, 0.4],
    )
    # sum([0.2, 0.3, 0.4]) = 0.9 mm span; valid anchor range is [2.0, 6.0-0.9=5.1]
    assert anchor is not None
    assert 2.0 <= anchor <= 5.1
