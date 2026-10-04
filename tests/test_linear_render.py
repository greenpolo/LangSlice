"""Rendering: correction order (rotate then flip) and the status table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from langslice.core.captions import caption
from langslice.core.sections import render_slice
from langslice.core.status import status_rows, status_text
from langslice.linear.engine import build_context
from langslice.linear.job import ingest
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
    state.slices[0].transform = {
        "kind": "silhouette",
        "params": [],
        "iou": 0.9,
        "mirrored": True,
    }

    rows = status_rows(state)
    assert [row["id"] for row in rows] == ["s0.png", "s1.png", "s2.png"]
    assert rows[0]["delta_to_next_mm"] == 0.5
    assert rows[1]["delta_to_next_mm"] is None  # nothing placed after it
    assert rows[1]["flip"] is True
    assert rows[2]["position_mm"] is None
    assert rows[2]["damage_note"] == "half the section is gone"
    assert rows[0]["transform"] == "silhouette"
    assert rows[0]["transform_iou"] == 0.9
    assert rows[0]["transform_mirrored"] is True
    assert rows[1]["transform_mirrored"] is None

    text = status_text(state)
    assert "s2.png" in text and "unplaced" in text and "damaged" in text
    assert "delta_to_next_mm" in text
    assert "iou=0.900" in text and "mirrored=True" in text


def test_the_delta_to_the_next_section_is_signed(tmp_path: Path):
    for index in range(2):
        _corner_image(tmp_path / f"s{index}.png")
    ctx = _ctx(tmp_path)
    state = ingest(ctx.spec, ctx)
    state.slices[0].position_mm = 5.0
    state.slices[1].position_mm = 4.0  # the stack runs backwards
    assert status_rows(state)[0]["delta_to_next_mm"] == -1.0


def test_caption_labels_a_copy_without_touching_the_original():
    source = Image.new("RGB", (120, 80), (10, 10, 10))
    labelled = caption(source, "atlas 3.20 mm")

    assert labelled is not source
    # The caption is a band ABOVE the picture: same width, taller, pixels untouched.
    assert labelled.size[0] == source.size[0] and labelled.size[1] > source.size[1]
    assert np.asarray(source).max() == 10  # the original is untouched

    array = np.asarray(labelled)
    strip, below = array[:18, :90], array[30:, :]
    assert strip.max() > 200  # bright text in a dark box, top-left
    assert below.max() == 10  # nothing outside the strip changed


def test_compact_rows_drops_null_and_empty_fields_only():
    from langslice.core.status import compact_rows

    rows = [
        {"index": 0, "id": "a", "position_mm": None, "flip": False, "caveats": [],
         "damage_note": "", "transform_iou": 0.0}
    ]
    assert compact_rows(rows) == [{"index": 0, "id": "a", "flip": False, "transform_iou": 0.0}]
    assert rows[0]["position_mm"] is None  # the input is not mutated


def test_render_scale_counts_file_pixels_through_the_working_copy(tmp_path: Path):
    """A large file is rendered from a smaller working copy; the recorded scale
    (and so the canvas calibration) still counts pixels of the FILE."""
    from langslice.core.sections import render_cache_key
    from langslice.image_prep import WORKING_MAX_EDGE

    width = WORKING_MAX_EDGE * 2
    Image.fromarray(np.full((width // 2, width, 3), 90, dtype=np.uint8)).save(tmp_path / "big.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    record = ingest(spec, ctx).slices[0]

    render = render_slice(ctx, record, long_edge=512)
    assert ctx.working_source(record.id)[0].width == WORKING_MAX_EDGE
    key = render_cache_key(ctx, record, long_edge=512, frame=False)
    assert render.width == 512
    assert abs(ctx.render_scale[key] - width / 512) < 1e-9
