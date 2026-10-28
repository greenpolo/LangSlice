"""The opening of a large angled stack spline-filters the atlas volume once.

Each section of an angled stack is drawn on its own oblique atlas plane,
sampled cubically. ``map_coordinates`` spline-filters its whole input on
every cubic call unless given a prefiltered volume, which once made the
opening of a 100-section WHS rat stack take 41 minutes: this test counts the
filter runs for the opening of 100 angled sections (no timing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from langslice.agent.engine import build_context, build_seed_message
from langslice.core import oblique
from langslice.core.spec import JobSpec
from langslice.job.job import ingest
from tests.fakes import ellipse_section


class SmallAtlas:
    """An ellipsoid brain in a small volume."""

    atlas_name = "fake_small_25um"
    orientation = "asr"
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}

    def __init__(self, shape: tuple[int, int, int] = (66, 40, 57)) -> None:
        z, y, x = (np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in shape)
        inside = (z[:, None, None] ** 2 / 0.8 + y[None, :, None] ** 2 / 0.7
                  + x[None, None, :] ** 2 / 0.8) < 1.0
        self.annotation = inside.astype(np.uint32)
        texture = (np.sin(np.arange(shape[2]) / 3.0)[None, None, :] * 40.0).astype(np.float32)
        self.template = np.where(inside, 150.0 + texture, 0.0).astype(np.float32)


def test_opening_of_100_angled_sections_filters_the_volume_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = SmallAtlas()
    for index in range(100):
        ellipse_section((120, 90), (45, 33)).save(tmp_path / f"s{index:03d}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 0.2 + 0.008 * index
        record.cutting_angles_deg = {"pitch": 4.0, "yaw": -3.0}
    monkeypatch.setattr(oblique, "_filtered_cache", {})
    calls = []
    real = oblique._spline_filter_threaded

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(oblique, "_spline_filter_threaded", counting)
    seed = build_seed_message(state, ctx, "The job.")
    assert any(part.inline_data is not None for part in seed.parts or [])
    assert len(calls) == 1
