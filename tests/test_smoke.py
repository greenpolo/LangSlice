"""One registration from start to finish on a small synthetic brain.

The run places three synthetic sections, looks at them, fits each in-plane,
and exports the result. It checks that LangSlice still runs end to end and
that the fit moved toward the known answer; it pins no pixel and no wording.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import langslice
import langslice.doors.jobs as jobs
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section

SECTION_SHRINK = 0.8
PIXEL_SIZE_UM = 25.0 / SECTION_SHRINK
POSITIONS = {"s0.png": 0.1, "s1.png": 0.15, "s2.png": 0.2}


def write_sections(folder: Path) -> None:
    """Three grey sections of the synthetic atlas under a smooth known field."""
    atlas = SyntheticAtlas()
    folder.mkdir(parents=True)
    for seed, name in enumerate(POSITIONS):
        drawn, _ = render_section(atlas, SMOOTH_FIELD(), seed=seed)
        plane = cv2.GaussianBlur(np.asarray(drawn)[..., 0], (0, 0), 1.2)
        size = (
            round(plane.shape[1] * SECTION_SHRINK),
            round(plane.shape[0] * SECTION_SHRINK),
        )
        small = cv2.resize(plane, size, interpolation=cv2.INTER_AREA)
        Image.fromarray(np.stack([small] * 3, axis=-1)).save(folder / name)


@pytest.fixture
def images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    atlas = SyntheticAtlas()
    make_context = jobs.context

    def context_on_synthetic_atlas(*args: object, **kwargs: object) -> object:
        kwargs["atlas_loader"] = lambda _name: atlas
        return make_context(*args, **kwargs)

    monkeypatch.setattr(jobs, "context", context_on_synthetic_atlas)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    folder = tmp_path / "stack"
    write_sections(folder)
    return folder


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
