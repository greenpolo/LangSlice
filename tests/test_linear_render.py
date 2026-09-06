"""Rendering: correction order (rotate then flip) and the status table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from langslice.linear.engine import build_context, ingest
from langslice.linear.render import render_slice, status_rows, status_text
from langslice.linear.spec import JobSpec
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _ctx(folder: Path):
    return build_context(
        JobSpec(image_folder=str(folder), model="fake-model", preprocess="none"),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _ATLAS,
    )


def _corner_image(path: Path) -> None:
    """A 40x40 field with one bright square in the top-left corner."""
    array = np.zeros((40, 40, 3), dtype=np.uint8)
    array[2:10, 2:10] = 255
    Image.fromarray(array).save(path)


def test_render_rotates_before_flipping(tmp_path: Path):
    _corner_image(tmp_path / "a.png")
    ctx = _ctx(tmp_path)
    state = ingest(ctx.spec, ctx)
    record = state.slices[0]

    def corner(image: Image.Image) -> tuple[int, int]:
        array = np.asarray(image.convert("L"))
        ys, xs = np.nonzero(array > 128)
        return int(ys.mean() < array.shape[0] / 2), int(xs.mean() < array.shape[1] / 2)

    assert corner(render_slice(ctx, record, long_edge=40)) == (1, 1)  # top-left

    record.rotation_deg = 90  # counter-clockwise: top-left -> bottom-left
    assert corner(render_slice(ctx, record, long_edge=40)) == (0, 1)

    record.flip = True  # then mirrored: bottom-left -> bottom-right
    assert corner(render_slice(ctx, record, long_edge=40)) == (0, 0)


def test_render_cache_keys_on_every_correction(tmp_path: Path):
    _corner_image(tmp_path / "a.png")
    ctx = _ctx(tmp_path)
    state = ingest(ctx.spec, ctx)
    record = state.slices[0]

    flat = render_slice(ctx, record, long_edge=40)
    record.rotation_deg = 180
    rotated = render_slice(ctx, record, long_edge=40)
    assert flat is not rotated
    assert len(ctx.render_cache) == 2


def test_status_rows_carry_the_whole_row(tmp_path: Path):
    for index in range(3):
        _corner_image(tmp_path / f"s{index}.png")
    ctx = _ctx(tmp_path)
    state = ingest(ctx.spec, ctx)
    state.slices[0].position_mm = 1.0
    state.slices[1].position_mm = 1.5
    state.slices[1].flip = True
    state.slices[2].damaged = True
    state.slices[2].damage_note = "half the section is gone"
    state.slices[0].transform = {"kind": "silhouette", "params": [], "iou": 0.9}

    rows = status_rows(state)
    assert [row["id"] for row in rows] == ["s0.png", "s1.png", "s2.png"]
    assert rows[0]["spacing_to_next_mm"] == 0.5
    assert rows[1]["spacing_to_next_mm"] is None  # nothing placed after it
    assert rows[1]["flip"] is True
    assert rows[2]["position_mm"] is None
    assert rows[2]["damage_note"] == "half the section is gone"
    assert rows[0]["transform"] == "silhouette"

    text = status_text(state)
    assert "s2.png" in text and "unplaced" in text and "damaged" in text
