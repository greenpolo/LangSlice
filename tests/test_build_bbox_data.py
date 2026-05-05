"""Tests for build_bbox_data orchestrator (stage 'sample')."""
from __future__ import annotations

import json
from pathlib import Path

import build_bbox_data
import numpy as np
from PIL import Image


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


def test_stage_sample_writes_draft_manifest(tmp_path: Path, monkeypatch):
    """End-to-end: stage 1 produces a valid draft manifest from a stubbed coverage index."""
    coverage_index_path = tmp_path / "coverage.json"
    coverage_index_path.write_text(json.dumps({}), encoding="utf-8")

    out_dir = tmp_path / "bbox_data"

    # Minimal stub: monkeypatch the heavy renderers to return a tiny atlas slice
    # and a fixed bbox. The orchestrator's real I/O is too heavy for unit tests.
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(
        stage_sample, "_iter_viable_tuples",
        lambda *a, **kw: iter([
            ("allen_mouse_25um", "coronal", "Hippocampal Formation"),
        ]),
    )
    monkeypatch.setattr(
        stage_sample, "_render_example_atlas",
        lambda *a, **kw: {
            "id": "bbox_000001",
            "atlas": "allen_mouse_25um",
            "atlas_version": "CCFv3",
            "orientation": "coronal",
            "region": "Hippocampal Formation",
            "source_type": "augmented_atlas",
            "source_brain": None,
            "modality": "dapi",
            "is_hemisphere": False,
            "section_image_paths": [
                str(tmp_path / f"sec_{i}.png") for i in range(4)
            ],
            "section_positions_mm": [3.0, 3.4, 3.8, 4.2],
            "bboxes": [
                {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]},
            ] * 4,
        },
    )
    monkeypatch.setattr(
        stage_sample, "_render_overlay_strip",
        lambda *a, **kw: tmp_path / "overlay.png",
    )

    args = build_bbox_data.parse_args([
        "--stage", "sample",
        "--target-total", "1",
        "--seed", "0",
        "--out-dir", str(out_dir),
        "--coverage-index", str(coverage_index_path),
    ])
    rc = stage_sample.run_stage_sample(args)
    assert rc == 0

    manifest_path = out_dir / "draft_manifest.jsonl"
    assert manifest_path.exists()
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["region"] == "Hippocampal Formation"
    assert len(record["bboxes"]) == 4


def _spacings_stub(rng, n_gaps):  # noqa: ARG001
    return [0.2, 0.3, 0.4][:n_gaps]


def _spacings_stub_choose(rng, n_gaps):  # noqa: ARG001
    return [0.2, 0.4, 0.6][:n_gaps]


def test_render_example_real_returns_record(tmp_path: Path, monkeypatch):
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(build_bbox_data, "sample_section_count", lambda rng: 4)
    monkeypatch.setattr(build_bbox_data, "sample_spacings_mm", _spacings_stub)

    # Stub the registration helper + atlas annotation so we don't need a real
    # brain on disk.
    def fake_voxel(i, j):
        return np.array([i, 0.0, j], dtype=np.float64)

    fake_records = []
    for idx, pos in enumerate([3.0, 3.2, 3.5, 3.9]):
        fake_records.append({
            "section_id": f"FAKE:s{idx:03d}",
            "image_path": tmp_path / "fake_real_section.png",
            "image_shape": (96, 96),
            "pixel_to_voxel": fake_voxel,
            "midline_x_at_row": (lambda _i: 48.0),
            "is_hemisphere": False,
            "position_mm": pos,
        })
    by_id = {r["section_id"]: r for r in fake_records}

    def _load_record_stub(**kw):
        return by_id[kw["section_id"]]

    monkeypatch.setattr(stage_sample, "_load_real_section_record", _load_record_stub)
    # Ensure the test image exists.
    Image.new("RGB", (96, 96), (50, 50, 50)).save(tmp_path / "fake_real_section.png")

    fake_annotation = np.zeros((1, 96, 96), dtype=np.int32)
    fake_annotation[0, 20:50, 10:40] = 1   # left
    fake_annotation[0, 20:50, 60:90] = 1   # right
    fake_atlas = type("FA", (), {
        "annotation": fake_annotation,
        "resolution": (25.0, 25.0, 25.0),
        "metadata": {"version": "CCFv3"},
    })()
    monkeypatch.setattr(stage_sample, "_load_atlas_cached", lambda name: fake_atlas)

    rec = stage_sample._render_example_real(
        atlas_name="allen_mouse_25um", orientation="coronal",
        landmark="Hippocampal Formation", region_ids={1},
        brain_id="FAKE", section_ids=list(by_id),
        out_dir=tmp_path, example_id="bbox_test",
        rng=np.random.default_rng(0),
    )
    assert rec is not None
    assert rec["source_type"] == "real_histology"
    assert rec["source_brain"] == "FAKE"
    assert len(rec["bboxes"]) == 4
    for bbox in rec["bboxes"]:
        assert "left" in bbox and "right" in bbox


def test_render_example_real_scales_bbox_after_thumbnail(tmp_path: Path, monkeypatch):
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(build_bbox_data, "sample_section_count", lambda rng: 4)
    monkeypatch.setattr(build_bbox_data, "sample_spacings_mm", _spacings_stub)

    Image.new("RGB", (1600, 800), (50, 50, 50)).save(tmp_path / "wide_section.png")

    def _wide_voxel(i, j):
        return np.array([i / 10.0, 0.0, j / 10.0])

    fake_records = []
    for idx, pos in enumerate([0.0, 0.2, 0.5, 0.9]):
        fake_records.append({
            "section_id": f"FAKE:s{idx:03d}",
            "image_path": tmp_path / "wide_section.png",
            "image_shape": (800, 1600),
            "pixel_to_voxel": _wide_voxel,
            "midline_x_at_row": (lambda _i: 800.0),
            "is_hemisphere": True,
            "position_mm": pos,
        })
    by_id = {r["section_id"]: r for r in fake_records}

    def _load_record_stub(**kw):
        return by_id[kw["section_id"]]

    monkeypatch.setattr(stage_sample, "_load_real_section_record", _load_record_stub)
    fake_annotation = np.zeros((1, 100, 200), dtype=np.int32)
    fake_annotation[0, 20:60, 20:80] = 1
    fake_atlas = type("FA", (), {
        "annotation": fake_annotation,
        "metadata": {"version": "CCFv3"},
    })()
    monkeypatch.setattr(stage_sample, "_load_atlas_cached", lambda name: fake_atlas)

    rec = stage_sample._render_example_real(
        atlas_name="allen_mouse_25um", orientation="coronal",
        landmark="Hippocampal Formation", region_ids={1},
        brain_id="FAKE", section_ids=list(by_id),
        out_dir=tmp_path, example_id="bbox_scale",
        rng=np.random.default_rng(0),
    )
    assert rec is not None
    # 1600x800 thumbnails to 768x384; bbox coordinates must be in that frame.
    assert all(0 <= coord <= 768 for bbox in rec["bboxes"] for coord in bbox)


def test_choose_real_section_subset_samples_n_and_per_gap_spacings(monkeypatch):
    import _stage_sample as stage_sample  # type: ignore

    monkeypatch.setattr(build_bbox_data, "sample_section_count", lambda rng: 4)
    monkeypatch.setattr(build_bbox_data, "sample_spacings_mm", _spacings_stub_choose)
    monkeypatch.setattr(build_bbox_data, "sample_anchor_mm", lambda **kw: 0.0)

    records = [
        {"section_id": f"s{i}", "position_mm": i * 0.2}
        for i in range(20)
    ]
    chosen = stage_sample._choose_real_section_subset(records, rng=np.random.default_rng(4))
    assert chosen is not None
    assert 4 <= len(chosen) <= 8
    gaps = [
        b["position_mm"] - a["position_mm"]
        for a, b in zip(chosen, chosen[1:], strict=False)
    ]
    assert all(0.2 <= gap <= 0.8 for gap in gaps)
    assert len({round(gap, 6) for gap in gaps}) > 1
