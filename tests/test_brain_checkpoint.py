"""Checkpoint save/load."""

import json
from pathlib import Path

from langslice.linear.whole_brain.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from langslice.linear.whole_brain.state import SliceState, StackState


def _state() -> StackState:
    return StackState(
        atlas="allen_mouse_25um",
        interval_mm=0.2,
        slices=[
            SliceState(id="a.tif", index_original=0, index_corrected=0, position_mm=1.0),
            SliceState(id="b.tif", index_original=1, index_corrected=1),
        ],
        completed_nodes=["ingest"],
    )


def test_default_path_lives_beside_the_images(tmp_path: Path):
    assert default_checkpoint_path(str(tmp_path)) == str(tmp_path / "brain_estimate.json")


def test_save_and_load_roundtrip(tmp_path: Path):
    path = str(tmp_path / "brain_estimate.json")
    save_checkpoint(_state(), path)
    loaded = load_checkpoint(path)
    assert loaded == _state()


def test_save_overwrites_and_leaves_no_temp_files(tmp_path: Path):
    path = str(tmp_path / "brain_estimate.json")
    state = _state()
    save_checkpoint(state, path)
    state.mark_complete("survey")
    save_checkpoint(state, path)

    loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.completed_nodes == ["ingest", "survey"]
    assert [p.name for p in tmp_path.iterdir()] == ["brain_estimate.json"]


def test_load_missing_returns_none(tmp_path: Path):
    assert load_checkpoint(str(tmp_path / "nope.json")) is None


def test_checkpoint_json_is_human_readable(tmp_path: Path):
    path = str(tmp_path / "brain_estimate.json")
    save_checkpoint(_state(), path)
    data = json.loads(Path(path).read_text())
    assert data["slices"][0]["id"] == "a.tif"
    assert data["completed_nodes"] == ["ingest"]
