"""The job's write observers (:meth:`Job.observe`): fired per checkpoint,
scoped, never breaking the write. Pure Python — no agent, no model."""

from __future__ import annotations

from pathlib import Path

from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.job.job import Job
from langslice.job.layout import JobLayout


def _job(tmp_path: Path) -> Job:
    state = StackState(slices=[SliceState(id="a.png", index_original=0, index_corrected=0)])
    spec = JobSpec(image_folder=str(tmp_path), preprocess="none")
    return Job(state, spec, layout=JobLayout.for_images(tmp_path))


def _write(job: Job, note: str) -> None:
    before = job.snapshot()
    job.state.notes.append(note)
    job.commit(before)


def test_an_observer_fires_with_the_state_after_every_write(tmp_path: Path):
    job = _job(tmp_path)
    seen: list[list[str]] = []
    with job.observe(lambda state: seen.append(list(state.notes))):
        _write(job, "first")
        _write(job, "second")
    assert seen == [["first"], ["first", "second"]]


def test_an_observer_is_removed_on_context_exit(tmp_path: Path):
    job = _job(tmp_path)
    seen: list[StackState] = []
    with job.observe(seen.append):
        _write(job, "observed")
    _write(job, "not observed")
    assert len(seen) == 1


def test_an_observer_exception_does_not_break_the_write(tmp_path: Path):
    job = _job(tmp_path)

    def flaky(_state: StackState) -> None:
        raise RuntimeError("display failed")

    with job.observe(flaky):
        _write(job, "kept")
    assert Path(job.checkpoint_path).exists()
    assert job.state.notes == ["kept"]


def test_two_observers_can_be_attached_at_once(tmp_path: Path):
    job = _job(tmp_path)
    a: list[StackState] = []
    b: list[StackState] = []
    with job.observe(a.append), job.observe(b.append):
        _write(job, "both")
    assert len(a) == 1 and len(b) == 1
