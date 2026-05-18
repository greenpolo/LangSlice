from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from iSFT.atlas_signature import (
    RegionFeatures,
    _extract_slice_features,
    _score_regions,
    compose_visible_clues,
)


@dataclass
class _FakeAtlas:
    atlas_name: str = "allen_mouse_25um"
    orientation: str = "asr"
    resolution: tuple[float, float, float] = (250.0, 100.0, 100.0)

    def __post_init__(self) -> None:
        self.reference = np.zeros((8, 20, 20), dtype=np.uint16)
        ann = np.zeros((8, 20, 20), dtype=np.int32)

        for ap in range(8):
            width = 3 + ap
            ann[ap, 5:8, 2 : 2 + width] = 10

        ann[:, 12:14, 12:14] = 20
        ann[:, 15:17, 15:17] = 20
        ann[:, 1:3, 15:17] = 30

        self.annotation = ann
        self.structures = {
            10: {"id": 10, "name": "Dentate gyrus", "acronym": "DG"},
            20: {"id": 20, "name": "Lateral ventricle", "acronym": "LV"},
            30: {"id": 30, "name": "Corpus callosum", "acronym": "cc"},
            "DG": {"id": 10, "name": "Dentate gyrus", "acronym": "DG"},
            "VL": {"id": 20, "name": "Lateral ventricle", "acronym": "VL"},
            "cc": {"id": 30, "name": "Corpus callosum", "acronym": "cc"},
        }

    def get_structure_descendants(self, acronym: str) -> list[str]:
        _ = acronym
        return []


def test_extract_slice_features_geometry_metrics() -> None:
    ann = np.zeros((10, 10), dtype=np.int32)
    ann[1:3, 1:7] = 1
    ann[6:8, 1:3] = 2
    ann[6:8, 5:7] = 2

    feats = _extract_slice_features(ann)
    f1 = feats[1]
    f2 = feats[2]

    assert pytest.approx(f1.area_fraction, rel=1e-6) == 12.0 / 100.0
    assert 0.35 < f1.centroid_x < 0.45
    assert 0.10 < f1.centroid_y < 0.30
    assert f1.major_axis_len > f1.minor_axis_len
    assert f1.elongation > 1.8
    assert abs(f1.orientation_deg) < 20.0
    assert 0.0 < f1.compactness <= 1.0

    assert f2.component_count == 2
    assert pytest.approx(f2.largest_component_fraction, rel=1e-6) == 0.5


def test_score_regions_prefers_real_geometry_over_tiny_barcode() -> None:
    base = {
        10: RegionFeatures(
            area_fraction=0.22,
            centroid_x=0.40,
            centroid_y=0.35,
            major_axis_len=0.20,
            minor_axis_len=0.08,
            elongation=2.5,
            orientation_deg=8.0,
            compactness=0.52,
            component_count=1,
            largest_component_fraction=1.0,
        ),
        99: RegionFeatures(
            area_fraction=0.001,
            centroid_x=0.90,
            centroid_y=0.90,
            major_axis_len=0.01,
            minor_axis_len=0.01,
            elongation=1.0,
            orientation_deg=0.0,
            compactness=0.2,
            component_count=1,
            largest_component_fraction=1.0,
        ),
    }
    comparisons = {
        -0.50: {
            10: RegionFeatures(
                area_fraction=0.12,
                centroid_x=0.32,
                centroid_y=0.35,
                major_axis_len=0.15,
                minor_axis_len=0.08,
                elongation=1.9,
                orientation_deg=4.0,
                compactness=0.56,
                component_count=1,
                largest_component_fraction=1.0,
            ),
            99: RegionFeatures(
                area_fraction=0.0,
                centroid_x=0.0,
                centroid_y=0.0,
                major_axis_len=0.0,
                minor_axis_len=0.0,
                elongation=1.0,
                orientation_deg=0.0,
                compactness=0.0,
                component_count=0,
                largest_component_fraction=0.0,
            ),
        },
        0.50: {
            10: RegionFeatures(
                area_fraction=0.30,
                centroid_x=0.47,
                centroid_y=0.35,
                major_axis_len=0.23,
                minor_axis_len=0.08,
                elongation=2.9,
                orientation_deg=10.0,
                compactness=0.48,
                component_count=1,
                largest_component_fraction=1.0,
            ),
            99: RegionFeatures(
                area_fraction=0.004,
                centroid_x=0.91,
                centroid_y=0.90,
                major_axis_len=0.012,
                minor_axis_len=0.01,
                elongation=1.2,
                orientation_deg=1.0,
                compactness=0.18,
                component_count=1,
                largest_component_fraction=1.0,
            ),
        },
    }
    names = {10: "Dentate gyrus", 99: "barcode speck"}

    ranked = _score_regions(base, comparisons, names, top_k=2)
    assert ranked[0].region_id == 10
    assert ranked[0].score > ranked[1].score
    assert ranked[0].local_change_magnitude > 0.0


def test_compose_visible_clues_prefix_and_no_coordinate_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_atlas = _FakeAtlas()
    monkeypatch.setattr("iSFT.atlas_signature.load_atlas", lambda _name: fake_atlas)

    text = compose_visible_clues(
        "fake_mouse_250um",
        "coronal",
        1.00,
        comparison_positions=[0.50, 1.50],
        top_k=4,
    )

    assert text.startswith("Visible clues:")
    assert text != "Visible clues: atlas signature unavailable."
    assert "1.00" not in text
    assert "0.50" not in text
    assert "1.50" not in text


def test_compose_visible_clues_prioritizes_resolved_human_landmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_atlas = _FakeAtlas()
    monkeypatch.setattr("iSFT.atlas_signature.load_atlas", lambda _name: fake_atlas)

    text = compose_visible_clues("allen_mouse_25um", "coronal", 1.00, top_k=2)

    assert "Lateral ventricle" in text
    assert "Dentate gyrus" in text


def test_compose_visible_clues_defaults_to_sixteen_clues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_atlas = _FakeAtlas()
    for rid in range(100, 120):
        fake_atlas.annotation[:, rid - 100, 0:2] = rid
        fake_atlas.structures[rid] = {"id": rid, "name": f"Region {rid}", "acronym": f"R{rid}"}
    monkeypatch.setattr("iSFT.atlas_signature.load_atlas", lambda _name: fake_atlas)

    text = compose_visible_clues("fake_mouse_250um", "coronal", 1.00)

    assert text.count(";") + 1 == 16


def test_compose_visible_clues_fallback_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_name: str) -> object:
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr("iSFT.atlas_signature.load_atlas", _raise)
    text = compose_visible_clues("missing", "coronal", 1.25)
    assert text == "Visible clues: atlas signature unavailable."
