"""image_resolution scales what the agent is shown, never what is computed."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.atlas.render import MODEL_MIN_LONG_EDGE, atlas_um_per_px, model_long_edge
from langslice.linear.engine import build_context, build_seed_message, ingest
from langslice.linear.render import (
    PREVIEW_LONG_EDGE,
    render_slice,
    shown_section,
)
from langslice.linear.spec import JobSpec
from langslice.linear.toolbox import build_tools
from langslice.linear.transform import calibrate
from tests.test_linear_physical import TwoRegionAtlas

_UM_PER_PX = 5.0


def _section() -> Image.Image:
    """A 6 x 4.5 mm field at 5 um/px with a bright 4 x 3 mm block of tissue."""
    arr = np.zeros((900, 1200, 3), dtype=np.uint8)
    arr[150:750, 200:1000] = 120
    arr[300:500, 400:700, 1] = 220
    return Image.fromarray(arr)


def _run(folder: Path, resolution: str) -> tuple[Any, Any, dict[str, Any]]:
    folder.mkdir()
    for index in range(2):
        _section().save(folder / f"s{index}.tif", dpi=(25400 / _UM_PER_PX,) * 2)
    spec = JobSpec(image_folder=str(folder), model="fake", preprocess="none",
                   image_resolution=resolution)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: TwoRegionAtlas())
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 0.2 + 0.1 * index
    box = build_tools(state, ctx, spec)
    return state, ctx, {tool.__name__: tool for tool in box.tools}


def _sizes(result: dict[str, Any]) -> list[tuple[int, int]]:
    return [Image.open(io.BytesIO(part.inline_data.data)).size
            for part in result[TOOL_MEDIA_PARTS_KEY] if part.inline_data]


def _widths(sizes: list[tuple[int, int]]) -> list[int]:
    # Widths: a caption adds a band of fixed height above every picture.
    return [size[0] for size in sizes]


def _entry(**extra: Any) -> dict[str, Any]:
    return {"id": "s0.tif", "rotation_deg": 4.0, "scale_x": 1.1, "scale_y": 0.95,
            "translate_x_mm": 0.12, "translate_y_mm": -0.05, "pivot": "tissue", **extra}


def test_the_size_rule_is_unchanged_at_scale_one_and_multiplies_above():
    atlas = TwoRegionAtlas()

    def old_rule(size, um, cap):
        long = max(1, int(max(size)))
        edge = min(long, int(cap))
        if um and um > 0:
            edge = min(edge, int(round(long * float(um) / atlas_um_per_px(atlas))))
        return max(min(long, MODEL_MIN_LONG_EDGE), edge)

    for size in [(90, 60), (512, 300), (1200, 900), (3000, 2000)]:
        for um in [None, 2.5, 7.3, 25.0, 60.0]:
            for cap in (256, 512, 768):
                assert model_long_edge(size, um, atlas, cap=cap) == old_rule(size, um, cap)
                doubled = model_long_edge(size, um, atlas, cap=cap, scale=2.0)
                assert doubled <= max(size)  # never upsampled
                assert doubled >= model_long_edge(size, um, atlas, cap=cap)
    assert model_long_edge((3000, 2000), 2.5, atlas, cap=512, scale=2.0) == 600
    assert model_long_edge((3000, 2000), 2.5, atlas, cap=512) == 300


def test_low_draws_from_the_working_render_itself(tmp_path: Path):
    state, ctx, _tools = _run(tmp_path / "low", "low")
    record = state.slices[0]
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    shown, um, factors = shown_section(ctx, record, section, 9.0)
    assert shown is section and um == 9.0 and factors == (1.0, 1.0)


@pytest.mark.parametrize("resolution, multiple", [("medium", 1.5), ("high", 2.0)])
def test_pictures_grow_while_fits_and_transforms_stay(tmp_path: Path, resolution, multiple):
    low_state, low_ctx, low = _run(tmp_path / "low", "low")
    big_state, big_ctx, big = _run(tmp_path / resolution, resolution)

    def seed_long(state, ctx):
        parts = build_seed_message(state, ctx).parts or []
        return [Image.open(io.BytesIO(p.inline_data.data)).size[0]
                for p in parts if p.inline_data]

    ratios = [b / a for a, b in zip(seed_long(low_state, low_ctx),
                                    seed_long(big_state, big_ctx), strict=True)]
    assert all(r == pytest.approx(multiple, rel=0.05) for r in ratios)

    pairs = [
        (low["view_slices"](["s0.tif"]), big["view_slices"](["s0.tif"])),
        (low["fetch_atlas"]([0.2]), big["fetch_atlas"]([0.2])),
        (low["compare_placement"]([{"id": "s0.tif"}], mode="overlay"),
         big["compare_placement"]([{"id": "s0.tif"}], mode="overlay")),
        (low["view_stack"](), big["view_stack"]()),
    ]
    low_fit = low["fit_affine"](["s0.tif"], "silhouette")
    big_fit = big["fit_affine"](["s0.tif"], "silhouette")
    pairs.append((low_fit, big_fit))
    low_adjust = low["adjust_transforms"]([_entry(mode="ab")])
    big_adjust = big["adjust_transforms"]([_entry(mode="ab")])
    pairs.append((low_adjust, big_adjust))
    for small, large in pairs:
        for a, b in zip(_widths(_sizes(small)), _widths(_sizes(large)), strict=True):
            # The plot beside the contact sheet is not a picture of tissue.
            if a == 768 and b == 768:
                continue
            assert b / a == pytest.approx(multiple, rel=0.06), (a, b)

    # What is computed does not move: calibration, the fit, the stored numbers.
    for record_low, record_big in zip(low_state.slices, big_state.slices, strict=True):
        section_low = render_slice(low_ctx, record_low, long_edge=PREVIEW_LONG_EDGE)
        section_big = render_slice(big_ctx, record_big, long_edge=PREVIEW_LONG_EDGE)
        assert section_low.size == section_big.size
        assert calibrate(low_state, low_ctx, record_low, section_low) == calibrate(
            big_state, big_ctx, record_big, section_big)
    assert low_fit["results"] == big_fit["results"]
    assert low_state.slices[0].transform == big_state.slices[0].transform
    assert low_adjust["results"][0]["physical"] == big_adjust["results"][0]["physical"]


def test_a_larger_picture_shows_the_same_map(tmp_path: Path, monkeypatch):
    """Drawn larger, the adjusted section lands where the small picture puts it."""
    from langslice.linear import render, toolbox

    drawn: list[np.ndarray] = []

    def capture(*args, **kwargs):
        frames: list[dict[str, Any]] = []
        images, iou = render.physical_views(*args, **kwargs, frames=frames)
        x0, y0, x1, y1 = frames[0]["content_box"]
        drawn.append(np.asarray(images[0].convert("L"))[y0:y1, x0:x1])
        return images, iou

    monkeypatch.setattr(toolbox, "physical_views", capture)
    _low_state, _low_ctx, low = _run(tmp_path / "low", "low")
    _big_state, _big_ctx, big = _run(tmp_path / "high", "high")
    entry = _entry(mode="section", pivot=[0.3, 0.6], rotation_deg=12.0)
    low["adjust_transforms"]([entry])
    big["adjust_transforms"]([entry])

    def tissue(body: np.ndarray) -> tuple[float, float, float]:
        ys, xs = np.nonzero(body > 60)
        return xs.mean() / body.shape[1], ys.mean() / body.shape[0], len(xs) / body.size

    small, large = tissue(drawn[0]), tissue(drawn[1])
    assert drawn[1].shape[1] / drawn[0].shape[1] == pytest.approx(2.0, rel=0.02)
    assert small == pytest.approx(large, abs=0.005)
