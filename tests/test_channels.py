"""The channel model: raw channels displayed with properties that never reach
a fit, and one preprocessed channel per section that every fit and the image
model read (``core.channels``, ``core.appearance``, ``ops.appearance``)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import appearance as looks
from langslice.core import channels
from langslice.core.display import default_options
from langslice.core.handoff import prepare_linear_registration
from langslice.core.image_prep import IntensityRange, working_intensity_ranges
from langslice.core.sections import render_slice
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.job.checkpoint import FORMAT_KEY, STATE_FORMAT_VERSION, load_checkpoint
from langslice.job.job import Job
from langslice.ops import appearance as ops_appearance
from langslice.ops.refusal import Refused
from tests.deformable_synthetic import SyntheticAtlas

POSITION = 0.1
UM = 25.0
RECIPE = {"channel_weights": [0.0, 1.0, 0.0], "clahe_clip": 2.0, "clahe_tiles": 4,
          "n4": False, "denoise": False}


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _rgb_section(seed: int = 0) -> Image.Image:
    """Tissue whose three colour planes carry different structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:120, :160]
    body = ((yy - 60) / 50.0) ** 2 + ((xx - 80) / 70.0) ** 2 < 1
    red = np.where(body, 90 + 60 * np.sin(xx / 6.0), 5)
    green = np.where(body, 40 + 150 * ((yy // 15 + xx // 15) % 2), 5)
    blue = np.where(body, rng.uniform(20, 60, body.shape), 5)
    return Image.fromarray(np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8))


def _open(tmp_path: Path, atlas: SyntheticAtlas, n: int = 2) -> tuple[Job, Any]:
    for index in range(n):
        _rgb_section(index).save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    for record in job.state.slices:
        record.position_mm = POSITION
    return job, ctx


def _pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image).copy()


# --- display properties --------------------------------------------------------


def test_display_properties_change_pictures_and_never_the_preprocessed_channel(
    tmp_path: Path, atlas,
):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    preprocessed = _pixels(looks.preprocessed_image(ctx, state, record, long_edge=512))
    default = _pixels(render_slice(ctx, record, long_edge=512))
    overlay = {"overlay": ["red", "green"]}
    plain = _pixels(render_slice(ctx, record, long_edge=512,
                                 look=channels.with_properties(state, overlay)))

    done = ops_appearance.set_channel_properties(
        job, ctx, "red", contrast_limits=[20, 120], gamma=0.5, colormap="magenta")
    assert done.changed and len(job.undo_stack) == 1
    assert done.properties == {"contrast_limits": [20.0, 120.0], "gamma": 0.5,
                               "colormap": "magenta"}
    assert done.sections == ["s0.png", "s1.png"]
    assert done.intensities is not None and done.intensities["dtype"] == "uint8"
    assert done.intensities["dtype_range"] == [0.0, 255.0]
    assert state.appearance["channels"]["red"]["gamma"] == 0.5

    # The pictures of raw channels change ...
    shown = channels.with_properties(state, overlay)
    assert shown is not None and shown["properties"]["red"]["colormap"] == "magenta"
    assert not np.array_equal(_pixels(render_slice(ctx, record, long_edge=512, look=shown)),
                              plain)
    # ... and nothing a fit or the image model reads does.
    ctx.render_cache.clear()
    assert np.array_equal(_pixels(looks.preprocessed_image(ctx, state, record, long_edge=512)),
                          preprocessed)
    assert np.array_equal(_pixels(render_slice(ctx, record, long_edge=512)), default)
    assert looks.preprocessed_settings(state, record.id) is None

    # An argument left out keeps its value; reset clears them all.
    again = ops_appearance.set_channel_properties(job, ctx, "red", gamma=2.0)
    assert again.properties is not None and again.properties["contrast_limits"] == [20.0, 120.0]
    assert ops_appearance.set_channel_properties(job, ctx, "red", reset=True).properties is None
    assert "channels" not in state.appearance
    assert not ops_appearance.set_channel_properties(job, ctx, "red", reset=True).changed
    assert len(job.undo_stack) == 3  # the unchanged call took no step
    job.undo()
    assert state.appearance["channels"]["red"]["gamma"] == 2.0


def test_display_properties_persist_into_a_later_session(tmp_path: Path, atlas):
    """Set once, a channel's display stays: checkpointed with the job, and a
    job reopened from the folder draws and captions with it."""
    from langslice.core.look import LookRequest, look

    job, ctx = _open(tmp_path, atlas)
    ops_appearance.set_channel_properties(job, ctx, "red", gamma=0.5, colormap="magenta")
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None
    held = saved.appearance["channels"]["red"]
    assert (held["gamma"], held["colormap"]) == (0.5, "magenta")
    later = build_context(job.spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    reopened = Job.open(job.spec, later, folder=later.job_folder,
                        results_path=later.results_path)
    assert reopened.state.appearance["channels"]["red"]["gamma"] == 0.5
    picture = look(later, reopened.state, LookRequest("section", sections=("s0.png",)))[0]
    assert "red gamma 0.5 magenta" in picture.caption


def test_bad_channel_properties_are_refused_with_nothing_written(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    with pytest.raises(Refused) as unknown:
        ops_appearance.set_channel_properties(job, ctx, "dapi", gamma=2.0)
    assert unknown.value.code == "UNKNOWN_CHANNEL"
    assert unknown.value.details["channels"] == ["blue", "green", "red"]
    for bad in ({"contrast_limits": [120, 20]}, {"contrast_limits": [1]},
                {"gamma": 0.0}, {"colormap": "plaid"}):
        with pytest.raises(Refused) as refused:
            ops_appearance.set_channel_properties(job, ctx, "red", **bad)
        assert refused.value.code == "BAD_ARGS"
    assert job.state.appearance == {} and not job.undo_stack


def test_raw_channel_pictures_and_captions_restate_the_properties(tmp_path: Path, atlas):
    from langslice.core.look import LookRequest, look

    job, ctx = _open(tmp_path, atlas)
    state = job.state
    request = LookRequest("section", sections=(state.slices[0].id,), channels=("green",))
    before = _pixels(look(ctx, state, request)[0].image)
    ops_appearance.set_channel_properties(job, ctx, "green", contrast_limits=[0, 100])
    after = look(ctx, state, request)[0]
    assert not np.array_equal(_pixels(after.image), before)
    assert "green 0-100 gamma 1" in after.caption
    assert channels.describe("green", channels.channel_properties(state, "green")) == (
        "green 0-100 gamma 1")
    overlay = default_options("section")
    tagged = type(overlay)(**{**overlay.__dict__, "channels": ("red", "green"), "version": ""})
    assert "green 0-100 gamma 1" in tagged.section_tag(state)
    assert tagged.section_tag() == "  [raw red + green]"


# --- file intensities -----------------------------------------------------------


def _sixteen_bit(path: Path) -> np.ndarray:
    """Two 16-bit pages; page 1 rises from 1000 to 41000 across the columns."""
    ramp = np.tile(np.linspace(1000, 41000, 200), (100, 1)).astype(np.uint16)
    other = np.full((100, 200), 7, dtype=np.uint16)
    other[40:60, 50:150] = 3000
    tifffile.imwrite(path, np.stack([ramp, other]), photometric="minisblack")
    return ramp


def test_contrast_limits_are_file_intensities(tmp_path: Path, atlas):
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

    state = StackState(atlas="x", slices=[SliceState(id="s0.tif", index_original=0,
                                                     index_corrected=0)])
    channels.set_properties(state, "ch1", channels.ChannelProperties((1000.0, 21000.0)))
    look = channels.with_properties(state, {"channel": "ch1"})
    drawn = np.asarray(render_slice(ctx, state.slices[0], long_edge=512, look=look))[..., 0]
    row = drawn[50].astype(float)
    raw = ramp[50].astype(float)
    expected = np.clip((raw - 1000.0) / 20000.0, 0, 1) * 255.0
    assert np.abs(row - expected).max() <= 3.0  # 8-bit working planes
    assert row[raw >= 21000].min() == 255 and row[0] == 0
    summary = channels.channel_summary(ctx, ["s0.tif"], "ch1")
    assert summary is not None and summary["dtype"] == "uint16"
    assert 1000 <= summary["percentiles"]["1"] < 2000
    assert 40000 < summary["percentiles"]["99.5"] <= 41000


def test_an_eight_bit_file_reads_as_its_own_intensities(tmp_path: Path):
    _rgb_section().save(tmp_path / "a.png")
    (only,) = working_intensity_ranges(tmp_path / "a.png")
    assert only == IntensityRange("uint8", 0.0, 255.0, 0.0, 255.0)
    assert only.to_plane(51.0) == 51.0


# --- the preprocessed channel ----------------------------------------------------


def test_the_default_preprocessed_channel_is_the_default_render(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    record.transform = {"params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
    assert (looks.preprocessed_image(ctx, state, record, long_edge=512)
            is render_slice(ctx, record, long_edge=512, frame=False))
    default = prepare_linear_registration(state, ctx, record.id, long_edge=512)
    shown = prepare_linear_registration(state, ctx, record.id, long_edge=512,
                                        preprocessed=True)
    assert np.array_equal(np.asarray(shown.image), np.asarray(default.image))
    assert "preprocessed" not in shown.metadata

    looks.set_settings(state, looks.PREPROCESSED, None, dict(RECIPE))
    recipe = prepare_linear_registration(state, ctx, record.id, long_edge=512,
                                         preprocessed=True)
    assert not np.array_equal(np.asarray(recipe.image), np.asarray(default.image))
    assert recipe.metadata["preprocessed"] == RECIPE
    # Same frame, size and placement: only the pixels differ.
    assert recipe.image.size == default.image.size
    np.testing.assert_array_equal(recipe.atlas_to_slice, default.atlas_to_slice)
    # Geometry is still measured on the default render.
    plain = prepare_linear_registration(state, ctx, record.id, long_edge=512)
    assert np.array_equal(np.asarray(plain.image), np.asarray(default.image))


def test_recipes_override_per_section_and_reset(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    options = default_options("section")
    done = ops_appearance.set_preprocessed(job, ctx, None, clahe_clip=2.0,
                                           shown=state.in_order()[:1], options=options)
    assert done.written.in_force == {"preprocessed": {"stack": {
        "channel_weights": None, "clahe_clip": 2.0, "clahe_tiles": 8,
        "n4": False, "denoise": False}}}
    assert len(done.pictures) == 1 and done.pictures[0].before_settings is None
    assert done.pictures[0].after_settings == looks.preprocessed_settings(state, "s0.png")
    assert not np.array_equal(np.asarray(done.pictures[0].before),
                              np.asarray(done.pictures[0].after))

    ops_appearance.set_preprocessed(job, ctx, ["s1.png"], channel_weights=[0, 1, 0])
    stack = looks.preprocessed_settings(state, "s0.png")
    assert stack is not None and stack["clahe_clip"] == 2.0
    override = looks.preprocessed_settings(state, "s1.png")
    assert override is not None and override["channel_weights"] == [0.0, 1.0, 0.0]
    ops_appearance.set_preprocessed(job, ctx, ["s1.png"], reset=True)
    assert looks.preprocessed_settings(state, "s1.png") == stack
    ops_appearance.set_preprocessed(job, ctx, None, reset=True)
    assert looks.preprocessed_settings(state, "s0.png") is None
    assert state.appearance == {} and len(job.undo_stack) == 4

    refusals = {
        "UNKNOWN_SLICE_IDS": dict(section_ids=["nope.png"]),
        "CHANNEL_COUNT_MISMATCH": dict(section_ids=None, channel_weights=[1, 1]),
        "BAD_ARGS": dict(section_ids=None, clahe_clip=-1.0),
    }
    for code, kwargs in refusals.items():
        with pytest.raises(Refused) as refused:
            ops_appearance.set_preprocessed(job, ctx, **kwargs)
        assert refused.value.code == code
    assert state.appearance == {} and len(job.undo_stack) == 4


def test_old_saved_states_load_fit_as_preprocessed_and_drop_view(tmp_path: Path):
    state = StackState(atlas="x", slices=[SliceState(id="a.png", index_original=0,
                                                     index_corrected=0)])
    fit = {"stack": dict(RECIPE), "sections": {}}
    view = {"stack": {**RECIPE, "clahe_clip": 9.0}, "sections": {}}
    shown = {"red": {"contrast_limits": [1, 2], "gamma": 1.0, "colormap": None}}
    state.appearance = {"fit": fit, "view": view, "channels": shown}
    path = tmp_path / "state.json"
    path.write_text(json.dumps({FORMAT_KEY: STATE_FORMAT_VERSION, **state.to_dict()}))
    loaded = load_checkpoint(str(path))
    assert loaded is not None
    assert loaded.appearance == {"preprocessed": fit, "channels": shown}
    assert looks.preprocessed_settings(loaded, "a.png") == RECIPE

    # In memory, a state still holding "fit" reads it as the preprocessed
    # channel, and the next write moves it under its own name.
    assert looks.preprocessed_settings(state, "a.png") == RECIPE
    looks.set_settings(state, "fit", ["a.png"], None)
    assert "fit" not in state.appearance
    assert state.appearance["preprocessed"]["stack"] == RECIPE
    assert looks.migrated({"preprocessed": 1, "fit": 2, "view": 3}) == {"preprocessed": 1}


# --- what the algorithms read ------------------------------------------------------


class _Stop(Exception):
    pass


def test_the_elastix_affine_reads_the_preprocessed_channel(tmp_path: Path, atlas, monkeypatch):
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
    for _ in range(2):
        with pytest.raises(_Stop):
            tr.elastix_affine(state, ctx, record, start)
        looks.set_settings(state, looks.PREPROCESSED, None, dict(RECIPE))
    grid = deformation.fit_grid(state, ctx, record, transform={"params": start})
    default = np.asarray(grid.image)
    assert np.array_equal(seen[0], default)  # the default recipe: the default render
    wanted = looks.preprocessed_image(ctx, state, record, long_edge=deformation.FIT_LONG_EDGE)
    assert np.array_equal(seen[1], np.asarray(wanted.resize(grid.image.size,
                                                            Image.Resampling.BILINEAR)
                                              if wanted.size != grid.image.size else wanted))
    assert not np.array_equal(seen[1], default)
    # Display properties are not read.
    channels.set_properties(state, "green", channels.ChannelProperties(gamma=3.0))
    with pytest.raises(_Stop):
        tr.elastix_affine(state, ctx, record, start)
    assert np.array_equal(seen[2], seen[1])


def test_the_ants_syn_fit_reads_the_preprocessed_channel(tmp_path: Path, atlas, monkeypatch):
    """What ANTs SyN is handed is the section's preprocessed channel on the fit
    grid: a recipe changes it, a raw channel's display properties do not."""
    from langslice.core import deformation
    from langslice.ops import deformable as ops_deformable

    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": UM, "source": "host"}}
    seen: list[np.ndarray] = []
    monkeypatch.setattr(deformation, "run_jobs", lambda _ws, jobs: seen.extend(
        np.asarray(item.image).copy() for item in jobs))
    monkeypatch.setattr(ops_deformable, "refuse_without_ants", lambda: None)
    ops_deformable.ants_syn(job, ctx, [record.id])
    looks.set_settings(state, looks.PREPROCESSED, None, dict(RECIPE))
    ops_deformable.ants_syn(job, ctx, [record.id])
    grid = deformation.fit_grid(state, ctx, record)
    wanted = looks.preprocessed_image(ctx, state, record, long_edge=deformation.FIT_LONG_EDGE)
    if wanted.size != grid.image.size:
        wanted = wanted.resize(grid.image.size, Image.Resampling.BILINEAR)
    assert len(seen) == 2
    assert np.array_equal(seen[1], np.asarray(wanted))
    assert not np.array_equal(seen[0], seen[1])
    # Display properties are not read.
    channels.set_properties(state, "green", channels.ChannelProperties(gamma=3.0))
    ops_deformable.ants_syn(job, ctx, [record.id])
    assert np.array_equal(seen[2], seen[1])


def test_the_image_model_reads_the_preprocessed_channel(tmp_path: Path, monkeypatch):
    from langslice.core.handoff import LinearRegistrationInput
    from langslice.core.nonlinear import registration_tool as tool
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
    asked: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    def prepared(*_args, **kwargs):
        asked.append(kwargs)
        return LinearRegistrationInput(original, placement, "test", 4, "coronal", 0, 0,
                                       dict(metadata))

    monkeypatch.setattr(tool, "prepare_linear_registration", prepared)
    monkeypatch.setattr(tool, "annotation_slice", lambda *a, **k: labels)
    monkeypatch.setattr(tool, "_merge_classified", lambda ids, atlas: ids)
    monkeypatch.setattr(tool, "prepare_canvas", lambda image, **k: (image, image.size, 0, 0, 0))
    monkeypatch.setattr(tool, "correction_fingerprint", lambda *a, **k: "geometry")
    model = ImageModel("openai-oauth", resolve_image_model("openai-oauth").model,
                       lambda request: SimpleNamespace(image=original, route="test"))
    calls = tmp_path / "calls"
    tool.start_correction(state, ctx, record.id, image_model=model, calls_dir=calls)
    assert asked[-1]["preprocessed"] is True
    metadata["preprocessed"] = dict(RECIPE)  # a recipe: a new call key
    tool.start_correction(state, ctx, record.id, image_model=model, calls_dir=calls)
    assert len([path for path in calls.iterdir() if path.is_dir()]) == 2
    monkeypatch.setattr(tool, "outlined_atlas_template", lambda *a, **k: original)
    tool.start_atlas_correction(state, ctx, record.id, image_model=model, calls_dir=calls)
    assert asked[-1]["preprocessed"] is True
