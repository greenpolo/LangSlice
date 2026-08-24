"""Fluorescence preprocessing in the whole-brain visual path.

Everything an agent looks at goes through ``render_slice``: the per-slice
``view_slices`` images, the contact sheet, and the transform-step previews.
With ``preprocess="auto"`` (the default) that render runs
:func:`langslice.image_prep.adaptive_preprocess` — per-channel CLAHE plus a
DAPI-weighted grayscale blend — so a dim fluorescent section reads like the
atlas. It is display only: the file on disk is never rewritten.
"""

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from langslice.linear.whole_brain._step_common import render_slice
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import build_contact_sheet, ingest
from langslice.linear.whole_brain.state import BrainConfig, StackState


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


def _fluorescent_section(size: tuple[int, int] = (120, 90)) -> Image.Image:
    """A dim, blue-dominant section on a black field, like a DAPI channel."""
    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    tissue = ((xx - width / 2) / (width / 3)) ** 2 + (
        (yy - height / 2) / (height / 3)
    ) ** 2 < 1.0
    arr[..., 2][tissue] = 40  # dim DAPI-ish blue
    arr[..., 0][tissue] = 12  # a little tracer red
    return Image.fromarray(arr, mode="RGB")


def _stack(folder: Path, n: int = 3, **kwargs) -> tuple[StackState, object]:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        _fluorescent_section().save(folder / f"slice_{index:02d}.png")
    ctx = build_context(
        BrainConfig(image_folder=str(folder), **kwargs),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    return state, ctx


def test_render_slice_enhances_fluorescence_when_preprocess_is_auto(tmp_path: Path):
    state, ctx = _stack(tmp_path, preprocess="auto")
    record = state.in_order()[0]

    rendered = np.asarray(render_slice(ctx, record).convert("RGB"), dtype=np.int16)
    raw = np.asarray(_fluorescent_section().convert("RGB"), dtype=np.int16)

    # adaptive_preprocess blends the channels into one grey value...
    assert np.array_equal(rendered[..., 0], rendered[..., 2])
    # ...and lifts the dim tissue well clear of the raw signal.
    assert rendered.mean() > raw.mean() * 2


def test_render_slice_leaves_the_section_alone_when_preprocess_is_none(tmp_path: Path):
    state, ctx = _stack(tmp_path, preprocess="none")
    record = state.in_order()[0]

    rendered = np.asarray(render_slice(ctx, record).convert("RGB"), dtype=np.int16)

    # Untouched: still blue-dominant, still dim.
    assert rendered[..., 2].max() > rendered[..., 0].max()
    assert rendered.max() == 40


def test_preprocessing_never_rewrites_the_users_file(tmp_path: Path):
    state, ctx = _stack(tmp_path, preprocess="auto")
    path = tmp_path / state.in_order()[0].id
    before = path.read_bytes()

    render_slice(ctx, state.in_order()[0])
    build_contact_sheet(state, ctx)

    assert path.read_bytes() == before


def test_contact_sheet_geometry_is_the_same_either_way(tmp_path: Path):
    auto_state, auto_ctx = _stack(tmp_path / "auto", preprocess="auto")
    raw_state, raw_ctx = _stack(tmp_path / "raw", preprocess="none")

    with Image.open(build_contact_sheet(auto_state, auto_ctx)) as auto_sheet:
        auto_size, auto_mean = auto_sheet.size, np.asarray(auto_sheet).mean()
    with Image.open(build_contact_sheet(raw_state, raw_ctx)) as raw_sheet:
        raw_size, raw_mean = raw_sheet.size, np.asarray(raw_sheet).mean()

    assert auto_size == raw_size
    # Same grid, brighter tissue.
    assert auto_mean > raw_mean
