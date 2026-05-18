"""Unit tests for iSFT.state — phase tracker + resume."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))

from iSFT.state import (  # noqa: E402
    PHASES,
    RunState,
    init_or_resume,
    load_state,
    save_state,
    state_path,
)


def test_phases_constant_includes_expected_steps() -> None:
    expected = (
        "sampled", "rollouts", "scored", "filtered", "appended",
        "unioned", "trained", "evaluated", "curriculum", "done",
    )
    assert PHASES == expected


def test_fresh_state_starts_at_round_zero_with_no_phase() -> None:
    s = RunState(run_id="run-001", rounds_total=3)
    assert s.round == 0
    assert s.phase is None
    assert s.is_phase_done(0, "sampled") is False
    assert s.is_phase_done(0, "rollouts") is False
    assert s.next_phase(0) == "sampled"


def test_mark_phase_advances_within_round() -> None:
    s = RunState(run_id="r", rounds_total=2)
    s.mark_phase(0, "rollouts", rollouts_manifest="x.json")
    assert s.phase == "rollouts"
    assert s.is_phase_done(0, "rollouts") is True
    assert s.is_phase_done(0, "sampled") is True   # earlier phase implied done
    assert s.is_phase_done(0, "scored") is False
    assert s.next_phase(0) == "scored"
    assert s.artifacts["0"]["rollouts_manifest"] == "x.json"


def test_advance_round_archives_summary_and_resets_phase() -> None:
    s = RunState(run_id="r", rounds_total=3)
    s.mark_phase(0, "done")
    s.advance_round(summary={"n_kept": 100})
    assert s.round == 1
    assert s.phase is None
    assert len(s.history) == 1
    assert s.history[0]["round"] == 0
    assert s.history[0]["summary"] == {"n_kept": 100}


def test_earlier_round_is_always_done() -> None:
    s = RunState(run_id="r", rounds_total=3)
    s.round = 2
    s.phase = "rollouts"
    assert s.is_phase_done(0, "evaluated") is True
    assert s.is_phase_done(1, "evaluated") is True


def test_later_round_is_never_done() -> None:
    s = RunState(run_id="r", rounds_total=3)
    s.round = 1
    s.phase = "evaluated"
    assert s.is_phase_done(2, "sampled") is False
    assert s.next_phase(2) == "sampled"


def test_next_phase_after_done_returns_done() -> None:
    s = RunState(run_id="r", rounds_total=2)
    s.mark_phase(0, "done")
    assert s.next_phase(0) == "done"


def test_unknown_phase_raises() -> None:
    s = RunState(run_id="r", rounds_total=1)
    with pytest.raises(ValueError):
        s.mark_phase(0, "frobnicated")
    # is_phase_done short-circuits on phase=None, so it only raises for an
    # unknown phase string when there's actually a recorded phase to compare.
    s.mark_phase(0, "rollouts")
    with pytest.raises(ValueError):
        s.is_phase_done(0, "nope")


# ──────────────────────────────────────────────────────────────────────────
# Disk I/O
# ──────────────────────────────────────────────────────────────────────────

def test_save_then_load_round_trips(tmp_path: Path) -> None:
    s = RunState(run_id="run-xyz", rounds_total=4)
    s.mark_phase(0, "scored", scored_path="round_0_scored.json")
    save_state(s, tmp_path)

    loaded = load_state(tmp_path)
    assert loaded is not None
    assert loaded.run_id == "run-xyz"
    assert loaded.rounds_total == 4
    assert loaded.round == 0
    assert loaded.phase == "scored"
    assert loaded.artifacts["0"]["scored_path"] == "round_0_scored.json"


def test_load_state_missing_returns_none(tmp_path: Path) -> None:
    assert load_state(tmp_path) is None


def test_load_state_corrupt_returns_none_with_warning(tmp_path: Path) -> None:
    # Write garbage to the state path.
    p = state_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json", encoding="utf-8")
    # Should not raise; should return None so the driver starts fresh.
    assert load_state(tmp_path) is None


def test_init_or_resume_creates_fresh_state(tmp_path: Path) -> None:
    s = init_or_resume(tmp_path, run_id="abc", rounds_total=5)
    assert s.run_id == "abc"
    assert s.rounds_total == 5
    assert s.round == 0
    assert s.phase is None
    assert state_path(tmp_path).is_file()


def test_init_or_resume_picks_up_existing_state(tmp_path: Path) -> None:
    s = RunState(run_id="abc", rounds_total=3)
    s.mark_phase(0, "rollouts")
    s.mark_phase(0, "scored")
    save_state(s, tmp_path)

    resumed = init_or_resume(tmp_path, run_id="abc", rounds_total=3)
    assert resumed.round == 0
    assert resumed.phase == "scored"
    assert resumed.is_phase_done(0, "scored") is True


def test_init_or_resume_extends_rounds_total(tmp_path: Path) -> None:
    """Restarting with --rounds 5 after an initial --rounds 3 should extend."""
    s = RunState(run_id="abc", rounds_total=3)
    save_state(s, tmp_path)
    resumed = init_or_resume(tmp_path, run_id="abc", rounds_total=5)
    assert resumed.rounds_total == 5


def test_state_json_is_atomic(tmp_path: Path) -> None:
    """A successful save_state should leave only state.json (no .tmp residue)."""
    s = RunState(run_id="r", rounds_total=1)
    save_state(s, tmp_path)
    files = list(tmp_path.iterdir())
    names = {f.name for f in files}
    assert "state.json" in names
    # No leftover tempfiles.
    for n in names:
        assert not n.startswith(".state.")


# ──────────────────────────────────────────────────────────────────────────
# Resume logic — verify each phase resumes from the right next-step
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("completed,expected_next", [
    (None, "sampled"),
    ("sampled", "rollouts"),
    ("rollouts", "scored"),
    ("scored", "filtered"),
    ("filtered", "appended"),
    ("appended", "unioned"),
    ("unioned", "trained"),
    ("trained", "evaluated"),
    ("evaluated", "curriculum"),
    ("curriculum", "done"),
    ("done", "done"),
])
def test_resume_from_each_phase_picks_correct_next(
    completed: str | None, expected_next: str,
) -> None:
    s = RunState(run_id="r", rounds_total=1)
    if completed is not None:
        s.mark_phase(0, completed)
    assert s.next_phase(0) == expected_next


def test_history_preserved_across_save_load(tmp_path: Path) -> None:
    s = RunState(run_id="r", rounds_total=3)
    s.mark_phase(0, "done")
    s.advance_round(summary={"foo": "bar"})
    save_state(s, tmp_path)

    loaded = load_state(tmp_path)
    assert loaded is not None
    assert len(loaded.history) == 1
    assert loaded.history[0]["summary"] == {"foo": "bar"}
    assert loaded.round == 1


def test_to_dict_is_json_serializable() -> None:
    s = RunState(run_id="r", rounds_total=2)
    s.mark_phase(0, "rollouts", manifest="x.json")
    blob = s.to_dict()
    # Round-trip through json (ensures all fields are jsonable).
    assert json.loads(json.dumps(blob)) == blob
