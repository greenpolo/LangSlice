"""Real nodes (ingest, emit) and an end-to-end run over the stub graph."""

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from langslice.linear import APResult
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


async def _fake_anchor_estimate(*, image_path: str, **_kwargs) -> APResult:
    """Stand-in for the single-slice worker: 1 mm per section, no model call."""
    index = int(Path(image_path).stem.split("_")[-1])
    return APResult(position_mm=float(index), reasoning="fake")


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
    from tests.fakes import install_fake_adk_model_clean_stack

    # survey and position are real agents now; script them to submit at once,
    # and stub the single-slice worker seed uses for its anchor estimates.
    install_fake_adk_model_clean_stack(monkeypatch)
    monkeypatch.setattr(
        "langslice.linear.whole_brain.seeding.run_slice_estimation",
        _fake_anchor_estimate,
    )
    _make_stack(tmp_path)
    messages: list[str] = []
    config = _config(tmp_path, out=str(tmp_path / "results.json"))

    state = asyncio.run(
        run_brain(config, emit=messages.append, atlas_loader=_fake_atlas_loader)
    )

    assert state.completed_nodes == [
        "ingest", "survey", "fix", "seed", "position", "transforms", "review", "emit",
    ]
    assert len(state.slices) == 5
    checkpoint = json.loads((tmp_path / "brain_estimate.json").read_text())
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["slices"] == checkpoint["slices"]
    assert (tmp_path / "contact_sheet.png").exists()
    # Every section leaves the graph positioned.
    assert all(s.position_mm is not None for s in state.slices)
    # transforms and review are still stubs; fix was skipped.
    assert sum("not implemented" in m for m in messages) == 2
    assert any("skipping 'fix'" in m for m in messages)

    # Second run resumes: every node is already complete, nothing re-runs.
    resumed_messages: list[str] = []
    resumed = asyncio.run(
        run_brain(config, emit=resumed_messages.append, atlas_loader=_fake_atlas_loader)
    )
    assert resumed.completed_nodes == state.completed_nodes
    assert sum("already complete" in m for m in resumed_messages) == 8
    assert not any("not implemented" in m for m in resumed_messages)

    # --fresh re-runs the whole graph.
    fresh_messages: list[str] = []
    refreshed = asyncio.run(
        run_brain(
            _config(tmp_path, out=str(tmp_path / "results.json"), resume=False),
            emit=fresh_messages.append,
            atlas_loader=_fake_atlas_loader,
        )
    )
    assert not any("already complete" in m for m in fresh_messages)
    # The contact sheet written by the first run must not be re-ingested.
    assert len(refreshed.slices) == 5
