"""Real nodes (ingest, emit) and an end-to-end run over the stub graph."""

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from langslice.linear.whole_brain.engine import build_context, run_brain
from langslice.linear.whole_brain.nodes import emit, ingest
from langslice.linear.whole_brain.state import BrainConfig, StackState


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    """Enough of a BrainGlobe atlas for the space/position helpers."""

    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)


def _fake_atlas_loader(name: str) -> object:
    return _FakeAtlas()


def _make_stack(folder: Path, n: int = 5) -> list[str]:
    """Tiny generated PNGs with non-lexicographic numbering."""
    names = [f"slice_{i}.png" for i in (1, 2, 3, 10, 11)][:n]
    for index, name in enumerate(names):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return names


def _config(folder: Path, **kwargs) -> BrainConfig:
    return BrainConfig(image_folder=str(folder), **kwargs)


def test_ingest_populates_state_and_contact_sheet(tmp_path: Path):
    names = _make_stack(tmp_path)
    state = StackState()
    ctx = build_context(
        _config(tmp_path), emit=lambda _m: None, atlas_loader=_fake_atlas_loader
    )

    assert asyncio.run(ingest(state, ctx)) == ""

    assert [s.id for s in state.slices] == names
    assert [s.index_original for s in state.slices] == [0, 1, 2, 3, 4]
    assert [s.index_corrected for s in state.slices] == [0, 1, 2, 3, 4]
    assert all(s.position_mm is None and not s.flip for s in state.slices)
    assert state.atlas == "allen_mouse_25um"
    assert state.interval_mm == 0.2
    assert state.thickness_mm == 0.05
    assert "0.00-13.18 mm" in state.notes[0]  # (528 - 1) * 0.025 mm along AP

    sheet = Path(state.contact_sheet)
    assert sheet.name == "contact_sheet.png"
    assert sheet.parent == tmp_path
    with Image.open(sheet) as img:
        assert img.size[0] > 0 and img.size[1] > 0


def test_ingest_rejects_empty_folder(tmp_path: Path):
    ctx = build_context(
        _config(tmp_path), emit=lambda _m: None, atlas_loader=_fake_atlas_loader
    )
    with pytest.raises(ValueError, match="No slice images"):
        asyncio.run(ingest(StackState(), ctx))


def test_ingest_rejects_unknown_plane(tmp_path: Path):
    _make_stack(tmp_path, n=1)
    ctx = build_context(
        _config(tmp_path, plane="oblique"),
        emit=lambda _m: None,
        atlas_loader=_fake_atlas_loader,
    )
    with pytest.raises(ValueError, match="Unsupported plane"):
        asyncio.run(ingest(StackState(), ctx))


def test_emit_writes_the_state_serialization(tmp_path: Path):
    out = tmp_path / "results.json"
    ctx = build_context(
        _config(tmp_path, out=str(out)),
        emit=lambda _m: None,
        atlas_loader=_fake_atlas_loader,
    )
    state = StackState(atlas="allen_mouse_25um", notes=["hi"])

    asyncio.run(emit(state, ctx))

    assert json.loads(out.read_text()) == state.to_dict()


def test_run_brain_end_to_end_then_resume(tmp_path: Path, monkeypatch):
    from tests.fakes import (
        EllipseAtlas,
        ellipse_section,
        install_fake_adk_model_clean_stack,
    )

    # Every agent step is real now; script them to submit at once. The stack
    # reaches the positioning step unplaced, so the fake positions it in one
    # set_positions call. The transform step's affine runs for real against
    # the synthetic ellipse atlas.
    names = _make_stack(tmp_path)
    install_fake_adk_model_clean_stack(
        monkeypatch,
        positions={name: 2.0 + index for index, name in enumerate(names)},
    )
    for index, name in enumerate(names):
        ellipse_section(angle=4.0 * index).save(tmp_path / name)
    ellipse_atlas = EllipseAtlas()
    messages: list[str] = []
    config = _config(tmp_path, out=str(tmp_path / "results.json"))

    state = asyncio.run(
        run_brain(config, emit=messages.append, atlas_loader=lambda _n: ellipse_atlas)
    )

    assert state.completed_nodes == [
        "ingest", "survey", "fix", "seed", "position", "transforms", "review", "emit",
    ]
    assert len(state.slices) == 5
    checkpoint = json.loads((tmp_path / "brain_estimate.json").read_text())
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["slices"] == checkpoint["slices"]
    assert (tmp_path / "contact_sheet.png").exists()
    # Every section leaves the graph positioned and with a transform proposal.
    assert all(s.position_mm is not None for s in state.slices)
    assert all(s.affine is not None and len(s.affine) == 6 for s in state.slices)
    assert any("5 affine fit(s)" in note for note in state.notes)
    # The positioning step really submitted: its summary and the end anchors
    # it had to defend are both on the record.
    assert any("position: Placed the stack" in note for note in state.notes)
    assert any("position: end anchor" in note for note in state.notes)
    # ...and they were checked against the atlas, not waved through.
    assert not any("end-anchor check skipped" in note for note in state.notes)
    assert any("review: Stack is consistent end to end." in n for n in state.notes)
    assert any("skipping 'fix'" in m for m in messages)  # the stack came back clean

    # Second run resumes: every node is already complete, nothing re-runs.
    resumed_messages: list[str] = []
    resumed = asyncio.run(
        run_brain(
            config, emit=resumed_messages.append, atlas_loader=lambda _n: ellipse_atlas
        )
    )
    assert resumed.completed_nodes == state.completed_nodes
    assert sum("already complete" in m for m in resumed_messages) == 8

    # --fresh re-runs the whole graph.
    fresh_messages: list[str] = []
    refreshed = asyncio.run(
        run_brain(
            _config(tmp_path, out=str(tmp_path / "results.json"), resume=False),
            emit=fresh_messages.append,
            atlas_loader=lambda _n: ellipse_atlas,
        )
    )
    assert not any("already complete" in m for m in fresh_messages)
    # The contact sheet written by the first run must not be re-ingested.
    assert len(refreshed.slices) == 5
