"""The checkpoint write-observer hook: fires per save, scoped, never breaks
the write. Pure Python — no agent, no model.
"""

from __future__ import annotations

from pathlib import Path

from langslice.linear.checkpoint import observe_checkpoints, save_checkpoint
from langslice.linear.state import SliceState, StackState


def _state() -> StackState:
    return StackState(slices=[SliceState(id="a.png", index_original=0, index_corrected=0)])


def test_observer_fires_with_the_state_after_every_save(tmp_path: Path):
    path = str(tmp_path / "linear_state.json")
    seen: list[StackState] = []
    state = _state()

    with observe_checkpoints(seen.append):
        save_checkpoint(state, path)
        state.notes.append("second write")
        save_checkpoint(state, path)

    assert len(seen) == 2
    assert seen[0] is state and seen[1] is state
    assert seen[1].notes == ["second write"]


def test_observer_is_removed_on_context_exit(tmp_path: Path):
    path = str(tmp_path / "linear_state.json")
    seen: list[StackState] = []

    with observe_checkpoints(seen.append):
        save_checkpoint(_state(), path)
    save_checkpoint(_state(), path)  # outside the context: not observed

    assert len(seen) == 1


def test_an_observer_exception_does_not_propagate_or_break_the_write(tmp_path: Path):
    path = str(tmp_path / "linear_state.json")

    def flaky(_state: StackState) -> None:
        raise RuntimeError("ABBA hiccup")

    with observe_checkpoints(flaky):
        save_checkpoint(_state(), path)  # must not raise

    assert Path(path).exists()


def test_two_observers_can_be_attached_at_once(tmp_path: Path):
    path = str(tmp_path / "linear_state.json")
    a: list[StackState] = []
    b: list[StackState] = []

    with observe_checkpoints(a.append), observe_checkpoints(b.append):
        save_checkpoint(_state(), path)

    assert len(a) == 1 and len(b) == 1
