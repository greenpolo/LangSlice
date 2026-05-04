"""Tests for the landmark loader and resolver."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))

import landmarks  # type: ignore  # noqa: E402


@pytest.fixture
def tmp_landmarks_json(tmp_path: Path) -> Path:
    payload = {
        "version": "0.1.0",
        "landmarks_by_orientation": {
            "coronal": [
                {"name": "Hippocampal Formation"},
                {"name": "Anterior Commissure"},
            ],
            "sagittal": [
                {"name": "Dentate Gyrus"},
            ],
            "horizontal": [],
        },
    }
    p = tmp_path / "landmarks.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_landmarks_for_orientation_returns_list(tmp_landmarks_json: Path):
    loader = landmarks.LandmarkLoader(landmarks_path=tmp_landmarks_json)
    coronal = loader.landmarks_for_orientation("coronal")
    assert coronal == ["Hippocampal Formation", "Anterior Commissure"]
    assert loader.landmarks_for_orientation("sagittal") == ["Dentate Gyrus"]
    assert loader.landmarks_for_orientation("horizontal") == []


def test_landmarks_for_unknown_orientation_raises(tmp_landmarks_json: Path):
    loader = landmarks.LandmarkLoader(landmarks_path=tmp_landmarks_json)
    with pytest.raises(KeyError):
        loader.landmarks_for_orientation("axial")


@pytest.fixture
def tmp_atlas_map_json(tmp_path: Path) -> Path:
    payload = {
        "Hippocampal Formation": {
            "allen_mouse_25um": {
                "acronym": "HPF",
                "include_descendants": True,
            },
        },
        "Anterior Commissure": {
            "allen_mouse_25um": {
                "acronym": "act",
                "include_descendants": False,
            },
        },
    }
    p = tmp_path / "landmark_atlas_map.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _fake_atlas_with_tree() -> object:
    """Build a tiny fake BrainGlobe atlas exposing the real API surface:
    structures / lookup_df / get_structure_descendants(structure)."""
    atlas = MagicMock()
    atlas.structures = {
        "HPF": {"acronym": "HPF", "id": 1089},
        "CA1": {"acronym": "CA1", "id": 382},
        "DG": {"acronym": "DG", "id": 726},
        "act": {"acronym": "act", "id": 908},
    }
    atlas.lookup_df = None

    def descendants(structure: str | dict) -> list[str]:
        acronym = structure["acronym"] if isinstance(structure, dict) else structure
        if acronym == "HPF":
            return ["CA1", "DG"]
        return []

    atlas.get_structure_descendants.side_effect = descendants
    return atlas


def test_resolve_landmark_with_descendants(
    tmp_landmarks_json: Path, tmp_atlas_map_json: Path
):
    loader = landmarks.LandmarkLoader(
        landmarks_path=tmp_landmarks_json, atlas_map_path=tmp_atlas_map_json
    )
    atlas = _fake_atlas_with_tree()
    ids = loader.resolve("Hippocampal Formation", atlas, atlas_name="allen_mouse_25um")
    assert ids == {1089, 382, 726}


def test_resolve_landmark_without_descendants(
    tmp_landmarks_json: Path, tmp_atlas_map_json: Path
):
    loader = landmarks.LandmarkLoader(
        landmarks_path=tmp_landmarks_json, atlas_map_path=tmp_atlas_map_json
    )
    atlas = _fake_atlas_with_tree()
    ids = loader.resolve("Anterior Commissure", atlas, atlas_name="allen_mouse_25um")
    assert ids == {908}


def test_resolve_unmapped_landmark_returns_empty(
    tmp_landmarks_json: Path, tmp_atlas_map_json: Path
):
    loader = landmarks.LandmarkLoader(
        landmarks_path=tmp_landmarks_json, atlas_map_path=tmp_atlas_map_json
    )
    atlas = _fake_atlas_with_tree()
    ids = loader.resolve("Dentate Gyrus", atlas, atlas_name="allen_mouse_25um")
    assert ids == set()
