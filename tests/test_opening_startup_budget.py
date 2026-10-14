"""The opening of a large angled stack is drawn within a time budget.

Each section of an angled stack is drawn on its own oblique atlas plane,
sampled cubically. ``map_coordinates`` spline-filters its whole input on
every cubic call unless given a prefiltered volume, which once made the
opening of a 100-section WHS rat stack take 41 minutes: this test holds the
opening of 100 angled sections on an atlas of the Allen 25 um volume's
shape within :data:`BUDGET_S`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from langslice.agent.engine import build_context, build_seed_message
from langslice.core import oblique
from langslice.core.spec import JobSpec
from langslice.job.job import ingest
from tests.fakes import ellipse_section

#: Seconds for the whole opening; about 1 s on an 8-core workstation, and
#: several minutes when the volume is filtered once per section.
BUDGET_S = 20.0


class AllenSizedAtlas:
    """An ellipsoid brain in a volume the shape of the Allen 25 um atlas."""

    atlas_name = "fake_allen_sized_25um"
    orientation = "asr"
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}

    def __init__(self, shape: tuple[int, int, int] = (528, 320, 456)) -> None:
        z, y, x = (np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in shape)
        inside = (z[:, None, None] ** 2 / 0.8 + y[None, :, None] ** 2 / 0.7
                  + x[None, None, :] ** 2 / 0.8) < 1.0
        self.annotation = inside.astype(np.uint32)
        texture = (np.sin(np.arange(shape[2]) / 3.0)[None, None, :] * 40.0).astype(np.float32)
        self.template = np.where(inside, 150.0 + texture, 0.0).astype(np.float32)


def test_opening_of_100_angled_sections_within_budget(tmp_path: Path) -> None:
    atlas = AllenSizedAtlas()
    for index in range(100):
        ellipse_section((400, 300), (150, 110)).save(tmp_path / f"s{index:03d}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 2.0 + 0.08 * index
        record.cutting_angles_deg = {"pitch": 4.0, "yaw": -3.0}
    oblique._filtered_cache.clear()
    start = time.perf_counter()
    seed = build_seed_message(state, ctx)
    elapsed = time.perf_counter() - start
    assert any(part.inline_data is not None for part in seed.parts or [])
    assert elapsed < BUDGET_S, f"opening took {elapsed:.1f} s"
