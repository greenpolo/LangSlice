"""What an engine prints goes to the job's log, and a fit's row says how long
it waited for the machine's GPU queue and whether it ran again after the GPU
ran out of memory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from langslice.core.fits.engine_log import engine_output
from langslice.tools.deformable import gpu_rows


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
    text = log.read_text(encoding="utf-8")
    assert "a fit" in text and "another" in text
    assert text.index("from the engine") < text.index("from gcc") < text.index("still the")
    with engine_output(None):
        assert os.fstat(2).st_ino == before


def test_a_fits_row_says_its_wait_and_retries_in_words(tmp_path: Path):
    from langslice.core.fits import deformation
    from langslice.core.fits.deformable import finish_fit, prepare_warp
    from langslice.core.fits.deformable.engines import EngineResult
    from langslice.core.fits.deformable.fit import WorkingGrid
    from tests.test_landmarks import setup

    assert gpu_rows({"runtime_s": 3.0}) == {}
    assert gpu_rows({"gpu_wait_s": 0.4}) == {}
    rows = gpu_rows({"gpu_wait_s": 42.4, "oom_retries": 1})
    assert rows["gpu_wait_s"] == 42
    assert rows["retried"].startswith("The GPU ran out of memory once (another process held "
                                      "it); the fit waited and ran again")
    assert "2 times" in gpu_rows({"oom_retries": 2})["retried"]
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
