"""The job's background work (``langslice.job.background``): notices handed
out once, running work listed, waits that give way under the job's lock,
each landing its own undo step, and the packaged trace's stale-input rule."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.job.background import DONE, FAILED, Landed
from langslice.job.job import DEFAULT_POSITION, Job
from langslice.ops import submit as ops_submit
from langslice.ops.inputs import STALE_INPUT, section_inputs
from langslice.ops.traces import land_trace
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _open(tmp_path: Path, n: int = 3, **spec_kwargs: Any) -> tuple[Job, Any]:
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    return Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path), ctx


def _note_step(job: Job, text: str) -> None:
    """One undo step: a note on the stack."""
    with job.writing():
        before = job.snapshot()
        job.state.notes.append(text)
        job.commit(before)


def test_running_work_is_listed_and_its_notice_handed_out_once(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])
    release = threading.Event()

    def land() -> Landed:
        assert release.wait(5)
        _note_step(job, "landed")
        return Landed(status=DONE, text="the note landed.", result={"note": "landed"})

    work = job.background.start("note", ["s0.png"], land)
    assert work.id == "w1" and work.status == "running"
    assert job.background.running() == [work]
    assert job.background.running_for("s0.png") is work
    assert job.background.running_for("s1.png") is None
    assert job.background.notices() == []  # nothing finished yet
    release.set()
    job.background.wait_all()
    assert job.background.running() == []
    notices = job.background.notices()
    assert [item.id for item in notices] == ["w1"]
    assert notices[0].notice == "w1 note of s0.png finished: the note landed."
    assert notices[0].result == {"note": "landed"}
    assert job.background.notices() == []  # handed out once
    summary = job.background.get("w1").summary()  # type: ignore[union-attr]
    assert summary["status"] == "done" and summary["sections"] == ["s0.png"]


def test_a_landing_is_its_own_undo_step(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])
    depth = len(job.undo_stack)

    def land() -> Landed:
        _note_step(job, "landed")
        return Landed(status=DONE, text="ok")

    job.background.start("note", ["s0.png"], land, wait_for_images=False)
    job.background.wait_all()
    assert len(job.undo_stack) == depth + 1
    assert "landed" in job.state.notes
    assert job.undo()
    assert "landed" not in job.state.notes


def test_a_failure_is_a_failed_notice(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])

    def land() -> Landed:
        raise RuntimeError("the engine stopped")

    job.background.start("note", ["s1.png"], land)
    job.background.start("note", ["s2.png"], lambda: Landed(status=FAILED, text="no reply."))
    job.background.wait_all()
    notices = {item.sections[0]: item for item in job.background.notices()}
    assert notices["s1.png"].status == "failed"
    assert notices["s1.png"].notice == ("w1 note of s1.png failed: RuntimeError: the engine "
                                        "stopped")
    assert notices["s2.png"].notice == "w2 note of s2.png failed: no reply."


def test_pictures_are_saved_with_numbers(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])
    picture = Image.new("RGB", (20, 10), (200, 10, 10))
    job.background.start("note", ["s0.png"],
                         lambda: Landed(status=DONE, text="drawn.", pictures=[picture]))
    job.background.wait_all()
    job.views.flush()
    (work,) = job.background.notices()
    assert len(work.pictures) == 1 and work.images == [picture]
    assert work.notice.endswith(f"Pictures #{work.pictures[0]}.")
    record = job.views.lookup(work.pictures[0])
    assert record is not None and record.tool == "note"


def test_submit_waits_and_the_work_gives_way_under_the_lock(tmp_path: Path):
    """submit runs under the job's lock (the doors hold it) and waits for the
    background work, whose landing needs that lock: the work thread gives
    way and the landing runs on the waiting thread."""
    job, _ = _open(tmp_path, tasks=["position"])
    assert all(record.position_source == DEFAULT_POSITION for record in job.state.slices)
    release = threading.Event()
    landed_on: list[str] = []

    def land() -> Landed:
        assert release.wait(5)
        with job.writing():
            before = job.snapshot()
            for index, record in enumerate(job.state.in_order()):
                record.position_mm = 2.0 + 0.2 * index
            job.commit(before)
        landed_on.append(threading.current_thread().name)
        return Landed(status=DONE, text="positions written.")

    work = job.background.start("positions", ["s0.png", "s1.png", "s2.png"], land)
    timer = threading.Timer(0.3, release.set)
    with job.writing():  # as the door holds it around submit
        timer.start()
        done = ops_submit.submit(job, summary="placed")
    assert done.summary == "placed" and job.state.submitted
    assert landed_on == [threading.current_thread().name]
    assert job.background.get(work.id).status == "done"  # type: ignore[union-attr]
    assert not any(record.position_source for record in job.state.slices)


def test_a_trace_landing_on_a_changed_section_is_stale(tmp_path: Path):
    """The packaged trace's landing applies nothing to a section whose
    inputs moved since the trace was asked for (STALE_INPUT)."""
    job, ctx = _open(tmp_path, tasks=["nonlinear"])
    record = job.state.slices[0]
    record.position_mm = 2.0
    record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    record.image_correction = {"status": "ok", "geometry_fingerprint": "g"}
    expected = section_inputs(job.state, record, deformation=True)
    _note_step(job, "a write elsewhere")
    assert section_inputs(job.state, record, deformation=True) == expected
    record.position_mm = 2.5  # the section moved while the image call ran
    depth = len(job.undo_stack)
    landed = land_trace(job, ctx, record.id, expected)
    assert landed.status == FAILED and landed.result["error"] == STALE_INPUT
    assert landed.text.startswith(f"{STALE_INPUT}: ")
    assert record.deformation is None and len(job.undo_stack) == depth
    record.image_correction = {"status": "error", "message": "no image"}
    record.position_mm = 2.0
    failed = land_trace(job, ctx, record.id, expected)
    assert failed.result["error"] == "TRACE_FAILED"
    assert "no image" in failed.text


def test_close_waits_for_the_work(tmp_path: Path):
    job, _ = _open(tmp_path, tasks=["position"])
    release = threading.Event()
    threading.Timer(0.2, release.set).start()
    job.background.start("note", ["s0.png"],
                         lambda: Landed(status=DONE, text="ok") if release.wait(5) else
                         Landed(status=FAILED, text="timeout"))
    job.close()
    assert job.background.running() == []
