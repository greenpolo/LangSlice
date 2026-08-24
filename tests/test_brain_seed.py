"""The seed node: it no longer places anything.

Prescriptive anchor seeding is gone — picking key sections needs atlas
knowledge the node does not have, so the stack reaches the positioning agent
unplaced and that agent chooses its own strategy. All the node does is say so
(and, one day, run DeepSlice).
"""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from langslice.linear.whole_brain.deepslice import deepslice_available, run_deepslice
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import ingest, seed
from langslice.linear.whole_brain.state import BrainConfig, StackState

_DEEPSLICE = "langslice.linear.whole_brain.nodes.deepslice_available"


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


def _ingested(folder: Path, n: int = 9, **kwargs) -> tuple[StackState, object]:
    for index in range(n):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(
            folder / f"slice_{index:02d}.png"
        )
    ctx = build_context(
        BrainConfig(image_folder=str(folder), **kwargs),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    return state, ctx


# --- the seed node -------------------------------------------------------


def test_seed_passes_the_stack_through_unplaced(tmp_path: Path):
    messages: list[str] = []
    state, ctx = _ingested(tmp_path, n=5)
    ctx.emit = messages.append  # type: ignore[attr-defined]

    assert asyncio.run(seed(state, ctx)) == ""

    assert all(s.position_mm is None for s in state.slices)
    assert all(s.position_source == "" for s in state.slices)
    assert any("no automatic seeding available" in note for note in state.notes)
    # The note is read back by the positioning step, so it states a fact only.
    assert any("the stack is unplaced" in note for note in state.notes)
    assert not any("strategy" in note for note in state.notes)
    assert any("unplaced" in message for message in messages)


def test_seed_notes_deepslice_when_it_is_installed(tmp_path: Path, monkeypatch):
    state, ctx = _ingested(tmp_path, n=3)
    monkeypatch.setattr(_DEEPSLICE, lambda: True)

    assert asyncio.run(seed(state, ctx)) == ""

    assert any("deepslice is installed but not wired" in n for n in state.notes)
    assert all(s.position_mm is None for s in state.slices)


def test_seed_node_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    ctx = build_context(
        BrainConfig(image_folder=str(tmp_path)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()

    assert asyncio.run(seed(state, ctx)) == ""

    assert state.notes == []


# --- the DeepSlice seam --------------------------------------------------


def test_deepslice_is_not_installed_and_not_implemented():
    assert deepslice_available() is False
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        run_deepslice(StackState(), None)  # type: ignore[arg-type]
