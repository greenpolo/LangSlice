"""Derived channels: the registry on ``StackState.appearance``, the recipe steps,
``set_channel`` and ``look`` drawing a derived channel and a mask outline."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from langslice.core import appearance as looks
from langslice.core.display import default_options
from langslice.core.look import LookError, LookRequest, look
from langslice.core.sections import render_slice
from langslice.core.state import StackState
from langslice.ops import appearance as ops_appearance
from langslice.ops import look as ops_look
from langslice.ops.refusal import Refused
from tests.deformable_synthetic import SyntheticAtlas
from tests.test_channels import RECIPE, _open

OLD_RECIPE = dict(RECIPE)


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _pixels(image) -> np.ndarray:
    return np.asarray(image).copy()


def test_an_older_preprocessed_entry_migrates_and_renders_the_same(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    state.appearance = {"preprocessed": {"stack": dict(OLD_RECIPE),
                                         "sections": {"s1.png": {**OLD_RECIPE, "n4": True}}}}
    # A job saved by the older code loads as it was and reads through the registry.
    loaded = StackState.from_dict(state.to_dict())
    assert looks.preprocessed_settings(loaded, record.id) == OLD_RECIPE
    assert looks.derived_settings(loaded, "preprocessed", "s1.png")["n4"] is True
    assert looks.derived_names(loaded) == ["preprocessed"]
    migrated = looks.migrate(state.appearance)
    assert "preprocessed" not in migrated
    assert migrated["derived"]["preprocessed"]["stack"] == OLD_RECIPE
    assert "preprocessed" in state.appearance  # migrate copies

    before = _pixels(looks.preprocessed_image(ctx, state, record, long_edge=512))
    assert np.array_equal(before, _pixels(render_slice(ctx, record, long_edge=512,
                                                       look=dict(OLD_RECIPE))))
    # Writing anything moves the entry to its new home; the render does not change.
    looks.set_derived(state, "mine", None, looks.validate_recipe(invert=True))
    assert "preprocessed" not in state.appearance
    assert state.appearance["derived"]["preprocessed"]["stack"] == OLD_RECIPE
    after = _pixels(looks.channel_image(ctx, state, record, "preprocessed", long_edge=512))
    assert np.array_equal(before, after)
    assert "mine" in looks.derived_names(state)


def test_set_derived_stack_and_override(tmp_path: Path, atlas):
    job, _ctx = _open(tmp_path, atlas)
    state = job.state
    stack = looks.validate_recipe(clahe_clip=2.0)
    over = looks.validate_recipe(weights={"red": 1.0})
    looks.set_derived(state, "tone", None, stack)
    looks.set_derived(state, "tone", ["s1.png"], over)
    assert looks.derived_settings(state, "tone", "s0.png") == stack
    assert looks.derived_settings(state, "tone", "s1.png") == over
    looks.set_derived(state, "tone", None, looks.validate_recipe(clahe_clip=3.0))
    assert looks.derived_settings(state, "tone", "s1.png") == over  # kept
    looks.set_derived(state, "tone", ["s1.png"], None)
    assert looks.derived_settings(state, "tone", "s1.png")["clahe_clip"] == 3.0
    looks.set_derived(state, "tone", None, None)  # another name: deleted
    assert state.appearance == {}
    assert looks.derived_settings(state, "tone", "s0.png") is None
    # preprocessed: a stack reset keeps overrides, as before
    looks.set_derived(state, "preprocessed", ["s1.png"], over)
    looks.set_derived(state, "preprocessed", None, stack)
    looks.set_derived(state, "preprocessed", None, None)
    assert looks.derived_settings(state, "preprocessed", "s1.png") == over
    assert looks.derived_settings(state, "preprocessed", "s0.png") is None


def test_stretch_and_invert_recipe(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    plain = looks.validate_recipe(weights={"red": 1.0})
    toned = looks.validate_recipe(weights={"red": 1.0}, contrast_limits=[0.2, 0.8], gamma=2.0)
    inverted = looks.validate_recipe(weights={"red": 1.0}, invert=True)
    base = np.asarray(render_slice(ctx, record, long_edge=256, look=plain), dtype=np.float32)
    inv = np.asarray(render_slice(ctx, record, long_edge=256, look=inverted), dtype=np.float32)
    assert np.abs((255.0 - base) - inv).max() <= 1.0
    tone = np.asarray(render_slice(ctx, record, long_edge=256, look=toned), dtype=np.float32)
    expect = np.clip((base / 255.0 - 0.2) / 0.6, 0, 1) ** 2 * 255.0
    assert np.abs(expect - tone).max() <= 1.0
    # weights name channels by name, not position
    green = looks.validate_recipe(weights={"green": 1.0})
    assert not np.array_equal(_pixels(render_slice(ctx, record, long_edge=256, look=plain)),
                              _pixels(render_slice(ctx, record, long_edge=256, look=green)))
    with pytest.raises(ValueError):
        looks.validate_recipe(contrast_limits=[0.9, 0.1])
    with pytest.raises(ValueError):
        looks.weights_for(looks.validate_recipe(weights={"nope": 1.0}), ["red", "green"])


def test_channel_image_of_a_raw_name_is_the_raw_render(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    assert looks.channel_names(state, ctx, record) == ["red", "green", "blue", "preprocessed"]
    raw = looks.channel_image(ctx, state, record, "green", long_edge=256)
    assert raw is render_slice(ctx, record, long_edge=256, look={"channel": "green"})
    default = looks.channel_image(ctx, state, record, "preprocessed", long_edge=256)
    assert default is render_slice(ctx, record, long_edge=256)
    with pytest.raises(ValueError, match="red, green, blue, preprocessed"):
        looks.channel_image(ctx, state, record, "nope", long_edge=256)


def test_set_channel_writes_raw_properties_and_derived_recipes(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    options = default_options("section")
    raw = ops_appearance.set_channel(job, ctx, "green", gamma=0.8)
    assert raw.kind == "raw" and raw.properties is not None and raw.properties["gamma"] == 0.8
    assert raw.channels == ["red", "green", "blue", "preprocessed"]
    for kwargs in ({"sections": ["s0.png"]}, {"weights": {"red": 1}}, {"invert": True},
                   {"n4": True}, {"clahe_clip": 1.0}):
        with pytest.raises(Refused) as refused:
            ops_appearance.set_channel(job, ctx, "green", **kwargs)
        assert refused.value.code == "BAD_ARGS"
    done = ops_appearance.set_channel(job, ctx, "nuclei", weights={"red": 1.0}, invert=True,
                                      shown=state.in_order()[:1], options=options)
    assert done.kind == "derived" and not done.deleted
    assert done.channels[-2:] == ["preprocessed", "nuclei"]
    assert done.in_force["stack"]["invert"] is True and len(done.pictures) == 1
    assert not np.array_equal(_pixels(done.pictures[0].before), _pixels(done.pictures[0].after))
    over = ops_appearance.set_channel(job, ctx, "nuclei", ["s1.png"], weights={"green": 1.0})
    assert over.in_force["sections"]["s1.png"]["weights"] == {"green": 1.0}
    for kwargs in ({"colormap": "cyan"}, {"weights": {"nope": 1.0}}, {"gamma": 99.0},
                   {"contrast_limits": [0.8, 0.2]}):
        with pytest.raises(Refused) as refused:
            ops_appearance.set_channel(job, ctx, "nuclei", **kwargs)
        assert refused.value.code == "BAD_ARGS"
    with pytest.raises(Refused) as refused:
        ops_appearance.set_channel(job, ctx, "nuclei", ["nope.png"])
    assert refused.value.code == "UNKNOWN_SLICE_IDS"
    ops_appearance.set_channel(job, ctx, "nuclei", ["s1.png"], reset=True)
    assert looks.derived_settings(state, "nuclei", "s1.png")["invert"] is True
    gone = ops_appearance.set_channel(job, ctx, "nuclei", reset=True)
    assert gone.deleted and "nuclei" not in looks.derived_names(state)
    default = ops_appearance.set_channel(job, ctx, "preprocessed", clahe_clip=2.0)
    assert default.in_force["stack"]["clahe_clip"] == 2.0 and not default.deleted
    ops_appearance.set_channel(job, ctx, "preprocessed", reset=True)
    assert looks.preprocessed_settings(state, "s0.png") is None


def test_look_draws_a_derived_channel_alone(tmp_path: Path, atlas):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    ops_appearance.set_channel(job, ctx, "nuclei", weights={"red": 1.0}, invert=True)
    pictures = look(ctx, state, LookRequest(mode="section", sections=("s0.png",),
                                            channels=("nuclei",)))
    assert "[nuclei channel]" in pictures[0].caption or "nuclei channel" in pictures[0].caption
    assert "inverted" in pictures[0].caption
    with pytest.raises(LookError) as mixed:
        look(ctx, state, LookRequest(mode="section", sections=("s0.png",),
                                     channels=("nuclei", "red")))
    assert mixed.value.code == "MIXED_CHANNELS"
    with pytest.raises(LookError) as unknown:
        look(ctx, state, LookRequest(mode="section", sections=("s0.png",), channels=("nope",)))
    assert unknown.value.code == "UNKNOWN_CHANNEL" and "nuclei" in str(unknown.value)
    # ``preprocessed`` is shown as before
    pre = look(ctx, state, LookRequest(mode="section", sections=("s0.png",),
                                       channels=("preprocessed",)))
    assert "preprocessed channel" in pre[0].caption


def _fake_masks(monkeypatch, weights: np.ndarray):
    """A stand-in for ``langslice.core.section_masks`` when chunk A has not merged."""
    if importlib.util.find_spec("langslice.core.section_masks") is not None:
        pytest.skip("the real section_masks module is present; covered by its own tests")
    module = types.ModuleType("langslice.core.section_masks")

    def read_mask(layout, state, record, name):
        if name not in record.masks:
            raise KeyError(name)
        return types.SimpleNamespace(name=name, kind="em", weights=weights, stale=False)

    module.read_mask = read_mask  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langslice.core.section_masks", module)
    import langslice.core as core_package
    monkeypatch.setattr(core_package, "section_masks", module, raising=False)


def test_look_outlines_a_mask_and_refuses_unknown_ones(tmp_path: Path, atlas, monkeypatch):
    job, ctx = _open(tmp_path, atlas)
    state = job.state
    record = state.slices[0]
    yy, xx = np.mgrid[:120, :160]
    weights = (((yy - 60) / 30.0) ** 2 + ((xx - 80) / 40.0) ** 2 < 1).astype(np.float32)
    _fake_masks(monkeypatch, weights)
    # SliceState.masks arrives with chunk A; the look reads it by name only.
    monkeypatch.setattr(record, "masks", {"em1": {"kind": "em"}}, raising=False)
    plain = ops_look.look(job, ctx, "section", ["s0.png"])
    drawn = ops_look.look(job, ctx, "section", ["s0.png"], masks=["em1"])
    assert drawn.masks[0]["name"] == "em1" and drawn.masks[0]["drawn"] is True
    assert not np.array_equal(_pixels(plain.pictures[0]), _pixels(drawn.pictures[0]))
    for kwargs, code in (({"masks": ["nope"]}, "UNKNOWN_MASK"),):
        with pytest.raises(Refused) as refused:
            ops_look.look(job, ctx, "section", ["s0.png"], **kwargs)
        assert refused.value.code == code
    with pytest.raises(Refused) as refused:
        ops_look.look(job, ctx, "atlas", positions_mm=[0.1], masks=["em1"])
    assert refused.value.code == "BAD_ARGS"


def test_look_masks_are_unknown_without_the_module(tmp_path: Path, atlas, monkeypatch):
    job, ctx = _open(tmp_path, atlas)
    with pytest.raises(Refused) as refused:
        ops_look.look(job, ctx, "section", ["s0.png"], masks=["em1"])
    assert refused.value.code == "UNKNOWN_MASK"
