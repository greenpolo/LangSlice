"""Raw channels: drawn alone as the file holds them, several together in their
colours with automatic contrast (``core.channels``), read from the file's own
intensities; and what every fit and the image model read: the section image
named, ``preprocessed`` by default (``core.section_images``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import channels
from langslice.core import section_images as images
from langslice.core.image_prep import IntensityRange, working_intensity_ranges
from langslice.core.sections import render_slice
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.job.job import Job
from tests.deformable_synthetic import SyntheticAtlas
from tests.test_section_images import _rgb_section

POSITION = 0.1
UM = 25.0
RECIPE = images.validate_recipe(weights={"green": 1.0}, clahe_clip=2.0, clahe_tiles=4)


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _open(tmp_path: Path, atlas: SyntheticAtlas, n: int = 2) -> tuple[Job, Any]:
    for index in range(n):
        _rgb_section(index).save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   tasks=["position", "transform", "nonlinear"],
                   inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    for record in job.state.slices:
        record.position_mm = POSITION
    return job, ctx


def _remake(job: Job, ctx: Any, record: SliceState) -> None:
    """``preprocessed`` made again from :data:`RECIPE` (as ``make_image`` does)."""
    array, _ = images.make(ctx, record.id, record.images, RECIPE)
    images.write_image(job.layout, record, images.PREPROCESSED, array, {"recipe": RECIPE})


# --- raw channels ------------------------------------------------------------------


def test_one_raw_channel_is_drawn_as_the_file_holds_it_and_several_in_colours(
        tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    record = job.state.slices[0]
    source = np.asarray(_rgb_section(0))
    green = np.asarray(render_slice(ctx, record, long_edge=160, look={"image": "green"}))
    assert np.array_equal(green[..., 0], source[..., 1])
    assert (green[..., 0] == green[..., 2]).all()
    together = np.asarray(render_slice(ctx, record, long_edge=160,
                                       look={"overlay": ["red", "green", "blue"]}))
    assert together.shape == source.shape and not (together[..., 0] == together[..., 1]).all()
    assert channels.colour_words(["red", "green", "blue"]) == "red + green + blue"
    assert channels.colour_words(["DAPI", "GFP"]) == "DAPI green + GFP magenta"


def _sixteen_bit(path: Path) -> np.ndarray:
    """Two 16-bit pages; page 1 rises from 1000 to 41000 across the columns."""
    ramp = np.tile(np.linspace(1000, 41000, 200), (100, 1)).astype(np.uint16)
    other = np.full((100, 200), 7, dtype=np.uint16)
    other[40:60, 50:150] = 3000
    tifffile.imwrite(path, np.stack([ramp, other]), photometric="minisblack")
    return ramp


def test_a_page_is_read_from_its_own_intensities(tmp_path: Path, atlas):
    ramp = _sixteen_bit(tmp_path / "s0.tif")
    ranges = working_intensity_ranges(tmp_path / "s0.tif")
    assert [(r.dtype, r.low, r.high) for r in ranges] == [
        ("uint16", 1000.0, 41000.0), ("uint16", 7.0, 3000.0)]
    assert ranges[0].dtype_min == 0.0 and ranges[0].dtype_max == 65535.0
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    names, planes = ctx.section_channels("s0.tif")
    assert names == ("ch1", "ch2")
    # Each working plane is its page stretched from its own minimum to maximum.
    held = channels.intensity_ranges(ctx, "s0.tif")["ch1"]
    np.testing.assert_allclose(held.to_file(planes[0]), ramp, atol=(40000 / 255.0) + 1)


def test_an_eight_bit_file_reads_as_its_own_intensities(tmp_path: Path):
    _rgb_section().save(tmp_path / "a.png")
    (only,) = working_intensity_ranges(tmp_path / "a.png")
    assert only == IntensityRange("uint8", 0.0, 255.0, 0.0, 255.0)
    assert only.to_plane(51.0) == 51.0


def test_automatic_limits_read_the_background_and_the_tissue():
    """A dark background: black at the background level, full colour at the
    tissue's 95th percentile; a light background keeps the percentiles."""
    rng = np.random.default_rng(1)
    plane = rng.normal(12.0, 2.0, (200, 300)).astype(np.float32)
    plane[50:150, 60:240] = rng.uniform(30.0, 90.0, (100, 180))
    plane[:20, :40] = 0.0  # an unscanned tile
    ((low, high),) = channels.auto_limits([plane])
    assert 10.0 < low < 14.0
    assert high == pytest.approx(np.percentile(plane[50:150, 60:240], 95), abs=1.0)
    bright = 255.0 - plane
    ((low, high),) = channels.auto_limits([bright])
    assert (low, high) == pytest.approx(tuple(np.percentile(bright, (1.0, 99.5))))


# --- what the algorithms read ------------------------------------------------------


class _Stop(Exception):
    pass


def test_the_elastix_affine_reads_the_section_image_named(tmp_path: Path, atlas, monkeypatch):
    from langslice.core import deformable, deformation
    from langslice.core import transform as tr

    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    seen: list[np.ndarray] = []

    def capture(image, *_args, **_kwargs):
        seen.append(np.asarray(image).copy())
        raise _Stop

    monkeypatch.setattr(deformable, "prepare_fit", capture)
    start = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    with pytest.raises(_Stop):
        tr.elastix_affine(state, ctx, record, start)
    grid = deformation.fit_grid(state, ctx, record, transform={"params": start})
    assert np.array_equal(seen[0], np.asarray(grid.image))  # the opening preprocessed
    _remake(job, ctx, record)
    with pytest.raises(_Stop):
        tr.elastix_affine(state, ctx, record, start)
    wanted = render_slice(ctx, record, long_edge=deformation.FIT_LONG_EDGE)
    assert np.array_equal(seen[1], np.asarray(wanted))
    assert not np.array_equal(seen[1], seen[0])
    with pytest.raises(_Stop):
        tr.elastix_affine(state, ctx, record, start, section_image="blue")
    blue = render_slice(ctx, record, long_edge=deformation.FIT_LONG_EDGE, look={"image": "blue"})
    assert np.array_equal(seen[2], np.asarray(blue))


def test_the_fireants_syn_fit_reads_the_section_image(tmp_path: Path, atlas, monkeypatch):
    """What FireANTs SyN is handed is the section image on the fit grid, and its
    cache key changes when the image is made again."""
    from langslice.core import deformation
    from langslice.ops import deformable as ops_deformable

    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": UM, "source": "host"}}
    seen: list[Any] = []
    monkeypatch.setattr(deformation, "run_jobs", lambda _ws, jobs: seen.extend(jobs))
    monkeypatch.setattr(ops_deformable, "refuse_without_engine", lambda: None)
    ops_deformable.fireants_syn(job, ctx, [record.id])
    _remake(job, ctx, job.state.slices[0])
    ops_deformable.fireants_syn(job, ctx, [record.id])
    record = job.state.slices[0]
    wanted = render_slice(ctx, record, long_edge=deformation.FIT_LONG_EDGE)
    assert len(seen) == 2
    assert np.array_equal(np.asarray(seen[1].image), np.asarray(wanted))
    assert not np.array_equal(np.asarray(seen[0].image), np.asarray(seen[1].image))
    assert seen[0].key != seen[1].key
    assert seen[1].image_identity["digest"] == record.images["preprocessed"]["digest"]


def test_the_image_model_reads_the_preprocessed_image(tmp_path: Path, monkeypatch):
    from langslice.core.handoff import LinearRegistrationInput
    from langslice.core.nonlinear import registration_tool as tool
    from langslice.core.nonlinear.upright import Upright
    from langslice.providers.registry import ImageModel, resolve_image_model

    original = Image.new("RGB", (60, 40), (40, 70, 90))
    original.save(tmp_path / "section.png")
    spec = JobSpec(image_folder=str(tmp_path), atlas="test", preprocess="none")
    ctx = SimpleNamespace(spec=spec, atlas=object(),
                          image_path=lambda _: str(tmp_path / "section.png"))
    record = SliceState(id="section.png", index_original=0, index_corrected=0,
                        position_mm=4, transform={"params": [1, 0, 0, 0, 1, 0]})
    state = StackState(atlas="test", slices=[record])
    labels = np.zeros((40, 60), dtype=np.int64)
    labels[5:30, 5:25] = 100000001
    placement = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    metadata: dict[str, Any] = {}

    def prepared(*_args, **_kwargs):
        return LinearRegistrationInput(original, placement, "test", 4, "coronal", 0, 0,
                                       dict(metadata))

    monkeypatch.setattr(tool, "prepare_linear_registration", prepared)
    monkeypatch.setattr(tool, "annotation_slice", lambda *a, **k: labels)
    monkeypatch.setattr(tool, "prepare_canvas", lambda image, **k: (image, np.eye(3)))
    monkeypatch.setattr(tool, "upright", lambda *a, **k: Upright())
    monkeypatch.setattr(tool, "correction_fingerprint", lambda *a, **k: "geometry")
    model = ImageModel("openai-oauth", resolve_image_model("openai-oauth").model,
                       lambda request: SimpleNamespace(image=original, route="test"))
    calls = tmp_path / "calls"
    tool.start_correction(state, ctx, record.id, image_model=model, calls_dir=calls)
    # preprocessed made again: a new call key
    metadata["section_image"] = {"name": "preprocessed", "digest": "ab12cd34"}
    tool.start_correction(state, ctx, record.id, image_model=model, calls_dir=calls)
    assert len([path for path in calls.iterdir() if path.is_dir()]) == 2
