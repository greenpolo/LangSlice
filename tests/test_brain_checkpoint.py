import json
from pathlib import Path

from langslice_harness.whole_brain.checkpoint import load_checkpoint, save_checkpoint
from langslice_harness.whole_brain.types import BrainEstimationConfig, SlicePosition


def _cfg() -> BrainEstimationConfig:
    return BrainEstimationConfig(
        image_folder="/tmp",
        atlas_name="allen_mouse_25um",
        thickness_um=50,
        interval_um=200,
        n_anchors=2,
        max_parallel=4,
        z_axis="AP",
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    path = str(tmp_path / "checkpoint.json")
    slices = [
        SlicePosition("a.tif", 0, 1.0, "anchor", True),
        SlicePosition("b.tif", 1, 1.2, "interpolated", False),
    ]
    save_checkpoint(path, _cfg(), slices)
    loaded_slices = load_checkpoint(path)
    assert len(loaded_slices) == 2
    assert loaded_slices[0].filename == "a.tif"
    assert loaded_slices[0].position_mm == 1.0
    assert loaded_slices[0].locked is True
    assert loaded_slices[1].locked is False


def test_incremental_save(tmp_path: Path):
    """Saving again with updated slices overwrites the file."""
    path = str(tmp_path / "checkpoint.json")
    slices = [SlicePosition("a.tif", 0, 1.0, "anchor", True)]
    save_checkpoint(path, _cfg(), slices)

    slices.append(SlicePosition("b.tif", 1, 1.2, "refined", True))
    save_checkpoint(path, _cfg(), slices)

    loaded = load_checkpoint(path)
    assert len(loaded) == 2


def test_load_nonexistent_returns_empty(tmp_path: Path):
    path = str(tmp_path / "nonexistent.json")
    loaded = load_checkpoint(path)
    assert loaded == []


def test_checkpoint_json_is_human_readable(tmp_path: Path):
    path = str(tmp_path / "checkpoint.json")
    slices = [SlicePosition("a.tif", 0, 1.0, "anchor", True)]
    save_checkpoint(path, _cfg(), slices)
    with open(path) as f:
        data = json.load(f)
    assert "slices" in data
    assert data["slices"][0]["filename"] == "a.tif"
