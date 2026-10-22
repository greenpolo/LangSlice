"""What an engine prints goes to the job's log, and a fit's row says how long
it waited for the machine's GPU queue and whether it ran again after the GPU
ran out of memory."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from langslice.core.deformable import engines
from langslice.core.engine_output import engine_output
from langslice.ops.deformable import gpu_rows


def test_engine_output_sends_descriptor_2_to_the_log_and_back(tmp_path: Path):
    """What the engine and the programs it starts print on standard error
    lands in the log; nested blocks share it; standard error comes back."""
    log = tmp_path / "logs" / "engine.log"
    before = os.fstat(2).st_ino
    with engine_output(log, "a fit"):
        os.write(2, b"from the engine\n")
        with engine_output(log, "another"):
            subprocess.run([sys.executable, "-c", "import sys; sys.stderr.write('from gcc\\n')"],
                           check=True)
        os.write(2, b"still the engine\n")
    assert os.fstat(2).st_ino == before
    text = log.read_text()
    assert "a fit" in text and "another" in text
    assert text.index("from the engine") < text.index("from gcc") < text.index("still the")
    with engine_output(None):  # a job that persists nothing: stderr as it is
        assert os.fstat(2).st_ino == before


def test_gpu_turn_reports_how_long_it_waited(tmp_path: Path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(engines, "GPU_LOCK", lock)
    with engines.gpu_turn("cpu") as turn:
        assert turn.waited_s == 0.0
    if sys.platform == "win32":
        return
    held = threading.Event()

    def hold() -> None:
        with open(lock, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            held.set()
            time.sleep(0.4)
            fcntl.flock(handle, fcntl.LOCK_UN)

    other = threading.Thread(target=hold)
    other.start()
    held.wait(5)
    with engines.gpu_turn("cuda") as turn:
        assert turn.waited_s >= 0.2
    other.join()


def test_a_fits_row_says_its_wait_and_retries_in_words(tmp_path: Path):
    from langslice.core import deformation
    from langslice.core.deformable import finish_fit, prepare_warp
    from langslice.core.deformable.engines import EngineResult
    from langslice.core.deformable.fit import WorkingGrid
    from tests.test_landmarks import setup

    assert gpu_rows({"runtime_s": 3.0}) == {}
    assert gpu_rows({"gpu_wait_s": 0.4}) == {}  # under a second: not worth a word
    rows = gpu_rows({"gpu_wait_s": 42.4, "oom_retries": 1})
    assert rows["gpu_wait_s"] == 42
    assert rows["retried"].startswith("The GPU ran out of memory once (another process held "
                                      "it); the fit waited and ran again")
    assert "2 times" in gpu_rows({"oom_retries": 2})["retried"]
    # The engine's numbers reach the record's engine block, which the row reads.
    job, ws, record = setup(tmp_path)
    try:
        grid = deformation.fit_grid(job.state, ws, record)
        size = grid.image.size
        prepared = prepare_warp(ws.atlas, grid.placement, size, deformation.DETAIL_LEVEL)
        prepared.grid = WorkingGrid(size, grid.placement.section_mm_per_px, size)
        zero = np.zeros((size[1], size[0], 2), np.float32)
        fitted = finish_fit(prepared, EngineResult(
            field_mm=zero, inverse_field_mm=zero, inverse_source="test",
            native_parameters={}, runtime_s=1.0, engine_version="test",
            gpu_wait_s=12.34, oom_retries=1))
    finally:
        job.close()
    assert fitted.engine["gpu_wait_s"] == 12.3 and fitted.engine["oom_retries"] == 1
    assert gpu_rows(fitted.engine)["gpu_wait_s"] == 12
