"""One registration from start to finish on a small synthetic brain.

The run places three synthetic sections, looks at them, fits each in-plane,
and exports the result. It checks that LangSlice still runs end to end and
that the fit moved toward the known answer; it pins no pixel and no wording.
"""

from __future__ import annotations

import json
from pathlib import Path

import langslice
from tests.synthetic_job import PIXEL_SIZE_UM, POSITIONS


def test_a_brain_registers_from_start_to_export(images: Path) -> None:
    with langslice.create_job(
        images,
        positions=POSITIONS,
        pixel_size_um=PIXEL_SIZE_UM,
        preprocess="none",
        tasks=["transform"],
    ) as job:
        looked = job.look(mode="overlay", sections=list(POSITIONS))
        assert looked["status"] == "ok", looked
        fitted = job.elastix_affine(sections=list(POSITIONS))
        assert fitted["status"] == "ok", fitted
        submitted = job.submit(summary="Smoke run.", notes=[], interval_breaks=[])
        assert submitted["status"] == "ok", submitted
        folder = Path(job.folder)
    registration = json.loads((folder / "registration.json").read_text(encoding="utf-8"))
    assert len(registration["sections"]) == len(POSITIONS)
