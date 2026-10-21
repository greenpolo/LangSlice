"""Section masks (``core.section_masks``): the EM sorting at the current
registration, the files and state entries, staleness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langslice.agent.engine import build_context
from langslice.core import section_masks
from langslice.core.deformable.masks import EM_MATCH_THRESHOLD, dilate
from langslice.core.deformable.record import DeformableRecord
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.core.state import SliceState, StackState
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from langslice.job.layout import JobLayout
from tests.deformable_synthetic import SyntheticAtlas, bump_field, render_section

ID = "s0.png"
HOLE = (150, 112, 8)


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _hole(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    yy, xx = np.indices((height, width))
    cx, cy, r = HOLE
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2


def _setup(folder: Path, atlas: SyntheticAtlas, *, hole: bool):
    image, _ = render_section(atlas, bump_field([]), remove=_hole((340, 260)) if hole else None)
    image.save(folder / ID)
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["position", "transform", "nonlinear"], inputs={"pixel_size_um": 25.0},
        transform=TransformSpec(angles=True), nonlinear=NonlinearSpec(provider="openai-oauth"),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    record = state.slices[0]
    record.position_mm = 0.1
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": 25.0, "source": "host"}}
    box = build_tools(state, ctx, spec)
    return state, ctx, record, box.job, np.asarray(image.convert("L"))


# --- the EM sorting ----------------------------------------------------------------


def test_the_sorting_leaves_a_hole_out_and_keeps_the_tissue(tmp_path: Path, atlas):
    state, ctx, record, job, gray = _setup(tmp_path, atlas, hole=True)
    weights, facts = section_masks.em_sorting(
        ctx, state, record, job.deformations, channel_image=gray, atlas_image="nissl", params={})
    assert weights.dtype == np.float32 and weights.shape == gray.shape
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    hole = _hole(gray.shape[::-1])
    interior = ~dilate(~hole, 4)
    assert (weights[interior] < EM_MATCH_THRESHOLD).mean() > 0.9
    ring = dilate(hole, 14) & ~dilate(hole, 8)
    assert (weights[ring] >= EM_MATCH_THRESHOLD).mean() > 0.9
    assert set(facts["params"]) == set(section_masks.SORTING_KEYS)
    assert 0.0 < facts["left_out"] < 0.5


def test_a_clean_section_loses_little(tmp_path: Path, atlas):
    state, ctx, record, job, gray = _setup(tmp_path, atlas, hole=False)
    weights, facts = section_masks.em_sorting(
        ctx, state, record, job.deformations, channel_image=gray, atlas_image="template",
        params={})
    assert facts["left_out"] < 0.1
    again, again_facts = section_masks.em_sorting(
        ctx, state, record, job.deformations, channel_image=gray, atlas_image="template",
        params={})
    assert np.array_equal(weights, again) and facts == again_facts


def test_sorting_widths_and_means_can_be_given(tmp_path: Path, atlas):
    state, ctx, record, job, gray = _setup(tmp_path, atlas, hole=True)
    _, facts = section_masks.em_sorting(
        ctx, state, record, job.deformations, channel_image=gray, atlas_image="template",
        params={"match_sd": 0.2, "background_mean": 0.9})
    assert facts["params"]["match_sd"] == pytest.approx(0.2)
    assert facts["params"]["background_mean"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="Unknown sorting key"):
        section_masks.em_sorting(ctx, state, record, job.deformations, channel_image=gray,
                                 atlas_image="template", params={"threshold": 0.5})
    with pytest.raises(ValueError, match="has no image"):
        section_masks.em_sorting(ctx, state, record, job.deformations, channel_image=gray,
                                 atlas_image="gfp", params={})


# --- files and entries ---------------------------------------------------------------


def _meta(state: StackState, record: SliceState, **over: Any) -> dict[str, Any]:
    return {"kind": "em", "channel": "preprocessed", "atlas_image": "template",
            "params": {"match_sd": 0.1}, "inputs": section_masks.registration_digest(state, record),
            "left_out": 0.04, **over}


def test_a_mask_writes_a_float16_npz_and_reads_back(tmp_path: Path):
    layout = JobLayout(tmp_path / "job", tmp_path)
    layout.ensure()
    state = StackState(slices=[SliceState(ID, 0, 0)])
    record = state.slices[0]
    weights = np.random.default_rng(0).random((30, 40)).astype(np.float32)
    entry = section_masks.write_mask(layout, record, "hole", weights, _meta(state, record))
    assert record.masks["hole"] == entry
    assert entry["file"] == "sections/s0/masks/hole.npz" and entry["kind"] == "em"
    with np.load(layout.folder / entry["file"]) as data:
        assert data["weights"].dtype == np.float16 and data["weights"].shape == (30, 40)
        assert data["grid"].tolist() == [40, 30]
        assert str(data["made"])
    view = section_masks.read_mask(layout, state, record, "hole")
    assert view.weights.dtype == np.float32 and not view.stale
    assert np.array_equal(view.weights, weights.astype(np.float16).astype(np.float32))
    assert view.kind == "em" and view.params == {"match_sd": 0.1} and view.left_out == 0.04
    assert section_masks.section_weights(view) is view.weights
    assert section_masks.section_weights(None) is None
    # The state entry round-trips through JSON.
    assert StackState.from_dict(json.loads(json.dumps(state.to_dict()))) == state


def test_a_mask_made_again_keeps_the_file_an_undone_state_points_at(tmp_path: Path):
    layout = JobLayout(tmp_path / "job", tmp_path)
    layout.ensure()
    state = StackState(slices=[SliceState(ID, 0, 0)])
    record = state.slices[0]
    first = section_masks.write_mask(layout, record, "m", np.zeros((8, 8)), _meta(state, record))
    second = section_masks.write_mask(layout, record, "m", np.ones((8, 8)), _meta(state, record))
    assert first["file"] != second["file"]
    with np.load(layout.folder / first["file"]) as data:
        assert data["weights"].max() == 0
    assert not list((layout.folder / "sections" / "s0" / "masks").glob("*.partial*"))


def test_bad_masks_are_refused(tmp_path: Path):
    layout = JobLayout(tmp_path / "job", tmp_path)
    layout.ensure()
    state = StackState(slices=[SliceState(ID, 0, 0)])
    record = state.slices[0]
    for name in ("none", "a b", "", "x" * 41, "../up"):
        with pytest.raises(ValueError, match="mask name"):
            section_masks.write_mask(layout, record, name, np.zeros((4, 4)), _meta(state, record))
    with pytest.raises(ValueError, match="kind"):
        section_masks.write_mask(layout, record, "m", np.zeros((4, 4)),
                                 _meta(state, record, kind="threshold"))
    with pytest.raises(ValueError, match="finite"):
        section_masks.write_mask(layout, record, "m", np.full((4, 4), np.nan), _meta(state, record))
    with pytest.raises(KeyError):
        section_masks.read_mask(layout, state, record, "missing")
    assert not record.masks


def test_a_mask_goes_stale_when_the_registration_moves(tmp_path: Path):
    layout = JobLayout(tmp_path / "job", tmp_path)
    layout.ensure()
    state = StackState(atlas="a", slices=[SliceState(ID, 0, 0, position_mm=1.0)])
    record = state.slices[0]
    section_masks.write_mask(layout, record, "m", np.ones((6, 6)), _meta(state, record))
    assert section_masks.mask_rows(state, record) == [{"name": "m", "kind": "em", "stale": False}]
    record.position_mm = 1.5
    view = section_masks.read_mask(layout, state, record, "m")
    assert view.stale
    assert section_masks.mask_rows(state, record)[0]["stale"] is True
    record.position_mm = 1.0
    assert not section_masks.read_mask(layout, state, record, "m").stale
    record.damaged_regions = ["TH"]  # the damage marks are part of what the mask was made at
    assert section_masks.read_mask(layout, state, record, "m").stale


def test_the_status_rows_list_a_sections_masks(tmp_path: Path):
    from langslice.core.status import status_rows

    layout = JobLayout(tmp_path / "job", tmp_path)
    layout.ensure()
    state = StackState(slices=[SliceState(ID, 0, 0, position_mm=1.0), SliceState("s1.png", 1, 1)])
    section_masks.write_mask(layout, state.slices[0], "hole", np.ones((6, 6)),
                             _meta(state, state.slices[0]))
    rows = status_rows(state)
    assert rows[0]["masks"] == [{"name": "hole", "kind": "em", "stale": False}]
    assert "masks" not in rows[1]


def test_the_fit_weights_are_the_applied_records_saved_sorting(tmp_path: Path, atlas):
    state, ctx, record, job, _gray = _setup(tmp_path, atlas, hole=False)
    assert section_masks.fit_weights(job.deformations, state, record) is None  # no deformation

    class _Store:
        def __init__(self, held: DeformableRecord | None) -> None:
            self.held = held

        def current(self, *_args: Any) -> DeformableRecord | None:
            return self.held

    held = object.__new__(DeformableRecord)
    held.weights = np.full((4, 4), 0.5, dtype=np.float16)
    sorting = section_masks.fit_weights(_Store(held), state, record)  # type: ignore[arg-type]
    assert sorting is not None and sorting.dtype == np.float32 and sorting.mean() == 0.5
    held.weights = None
    assert section_masks.fit_weights(_Store(held), state, record) is None  # type: ignore[arg-type]
