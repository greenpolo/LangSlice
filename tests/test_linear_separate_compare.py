"""Separate positioning references reuse cached pictures, not composed canvases.

The side-by-side comparison is no longer a tool's view mode; the operation
behind it (``ops.views.view_placement`` with ``side_by_side`` options) and
the core's reference pictures it reuses are what these pin.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.display import default_options
from langslice.core.jpeg import encode_jpeg
from langslice.core.pictures import reference_atlas_picture, reference_section_picture
from langslice.core.scale import reference_scale
from langslice.core.sizes import picture_edge
from langslice.core.spec import JobSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from langslice.ops import views
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import tool_named as _tool

_ATLAS = SlabAtlas()


def _box(folder: Path, n: int = 5):
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            folder / f"s{index}.png")
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    state = ingest(spec, ctx)
    return state, ctx, build_tools(state, ctx, spec)


def _images(pictures):
    """What a door sends: each picture in the doors' JPEG encoding."""
    return [encode_jpeg(picture) for picture in pictures if not isinstance(picture, str)]


def test_separate_comparison_reuses_cached_pictures_and_maps_each_pair(tmp_path):
    state, ctx, box = _box(tmp_path)
    position = 3.0
    s0, s1 = state.by_id("s0.png"), state.by_id("s1.png")
    edge = picture_edge(ctx)
    result = views.view_placement(
        box.job, ctx, [(s0, position), (s0, position), (s1, position)],
        default_options("side_by_side", atlas_channels=("template",), long_edge=edge))
    pictures = result.pictures
    images = _images(pictures)
    assert len(images) == 5
    # Each section at the one scale it shares with every atlas beside it.
    records = [s0, s1]
    scales = {record.id: reference_scale(ctx, state, record, edge) for record in records}
    shown = {record.id: reference_section_picture(ctx, record, long_edge=edge,
                                                  scale=scales[record.id]) for record in records}
    assert pictures[0] is shown["s0.png"]
    assert pictures[3] is shown["s1.png"]
    assert scales["s0.png"] == scales["s1.png"]
    # One atlas picture, drawn at the sections' own (here shared) cutting angles.
    assert s0.angles == s1.angles
    assert pictures[1] is pictures[2] is pictures[4] is reference_atlas_picture(
        ctx, state, position, long_edge=edge, angles=s0.angles,
        um_per_px=scales["s0.png"][0])
    assert images[1] == images[2] == images[4]
    assert [pair.indexes for pair in result.shown] == [
        {"section": 0, "atlas": 1}, {"section": 0, "atlas": 2},
        {"section": 3, "atlas": 4},
    ]


def test_reference_reuse_survives_reorder_but_not_orientation_or_preprocess(tmp_path):
    state, ctx, box = _box(tmp_path)
    original = _images([reference_section_picture(ctx, state.by_id("s0.png"))])[0]
    record = state.by_id("s0.png")
    # The order follows the positions: written back to front, the stack is reversed.
    written = _tool(box, "position_sections")(
        [{"id": r.id, "position_mm": 10.0 - index} for index, r in enumerate(state.in_order())],
        view=False)
    assert written["reordered"] and written["order"][0] == "s4.png"
    assert _images([reference_section_picture(ctx, record)])[0] == original
    record.rotation_deg = 90
    turned = _images([reference_section_picture(ctx, record)])[0]
    assert turned != original
    before = len(ctx.picture_cache)
    record.flip = True
    reference_section_picture(ctx, record)
    assert len(ctx.picture_cache) == before + 1
    before = len(ctx.picture_cache)
    ctx.spec.preprocess = "auto"
    reference_section_picture(ctx, record)
    assert len(ctx.picture_cache) == before + 1


def test_atlas_reuse_keys_on_exact_position_and_cutting_angles(tmp_path, monkeypatch):
    state, ctx, _box_value = _box(tmp_path)
    first = _images([reference_atlas_picture(ctx, state, 3.0)])[0]
    assert _images([reference_atlas_picture(ctx, state, 3.0)])[0] == first
    # Fake atlas does not support oblique interpolation; isolate cache identity.
    from langslice.core import atlas_fetch
    original = atlas_fetch.atlas_section
    image = original(ctx, state, 3.0, frame=True)
    calls = []

    def render(_ctx, _state, position, **_kwargs):
        calls.append(position)
        return image

    monkeypatch.setattr(atlas_fetch, "atlas_section", render)
    state.cutting_angles_deg = {"pitch": 2.0, "yaw": 0.0}
    assert _images([reference_atlas_picture(ctx, state, 3.0)])[0] != first
    reference_atlas_picture(ctx, state, 3.0001)
    reference_atlas_picture(ctx, state, 3.0001)
    assert calls == [3.0, 3.0001]
