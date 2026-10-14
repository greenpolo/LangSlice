"""The job's write observers (:meth:`Job.observe`): fired per checkpoint,
scoped, never breaking the write. Pure Python — no agent, no model."""

from __future__ import annotations

from pathlib import Path

from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.job.job import Job
from langslice.job.layout import JobLayout, job_folder_for


def _job(tmp_path: Path) -> Job:
    state = StackState(slices=[SliceState(id="a.png", index_original=0, index_corrected=0)])
    spec = JobSpec(image_folder=str(tmp_path), preprocess="none")
    return Job(state, spec, layout=JobLayout(job_folder_for(tmp_path), tmp_path))


def _write(job: Job, note: str) -> None:
    before = job.snapshot()
    job.state.notes.append(note)
    job.commit(before)


def test_observers_fire_per_write_and_are_removed_on_exit(tmp_path: Path):
    job = _job(tmp_path)
    seen: list[list[str]] = []
    other: list[StackState] = []
    with job.observe(lambda state: seen.append(list(state.notes))), job.observe(other.append):
        _write(job, "first")
        _write(job, "second")
    _write(job, "not observed")
    assert seen == [["first"], ["first", "second"]]
    assert len(other) == 2


def test_an_observer_exception_does_not_break_the_write(tmp_path: Path):
    job = _job(tmp_path)

    def flaky(_state: StackState) -> None:
        raise RuntimeError("display failed")

    with job.observe(flaky):
        _write(job, "kept")
    assert Path(job.checkpoint_path).exists()
    assert job.state.notes == ["kept"]
