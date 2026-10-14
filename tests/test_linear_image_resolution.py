"""image_resolution sizes what the agent is shown, never what is computed.

Each level is two long edges (``core.sizes.PICTURE_EDGES``): the opening
strips' tiles and every later picture (``look``'s, and the picture each change
tool shows of what it wrote, ``ops.look.show_result``); "auto" adds a
``resolution`` argument to ``look``, clamped to the driver model's largest
image. Nothing is upsampled past its source.
"""

from __future__ import annotations

import hashlib
import inspect
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context, build_seed_message
from langslice.core.captions import CAPTION_PX, _caption_font, caption, wrap_caption
from langslice.core.opening import CLAUDE_MAX_VIEW_EDGE, COLUMN_GAP, section_tile, strip_layout
from langslice.core.sections import render_slice
from langslice.core.sizes import MIN_RESOLUTION, PICTURE_EDGES, opening_edge
from langslice.core.spec import JobSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.media import package_result
from langslice.doors.tools.toolbox import build_tools
from langslice.doors.tools.view_options import image_limit
from langslice.job.job import ingest
from tests.test_linear_physical import TwoRegionAtlas

#: A 6 x 4.5 mm field at 2.5 um/px (2400 x 1800 px) with 4 x 3 mm of tissue:
#: large enough that every level's sizes are reached without upsampling.
_UM_PER_PX = 2.5


def _section(size: tuple[int, int] = (2400, 1800)) -> Image.Image:
    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[height // 6 : height * 5 // 6, width // 6 : width * 5 // 6] = 120
    arr[height // 3 : height * 5 // 9, width // 3 : width * 7 // 12, 1] = 220
    arr[height * 11 // 18 : height * 13 // 18, width * 5 // 8 : width * 3 // 4, 0] = 250
    return Image.fromarray(arr)


class Tools(dict):
    """The run's tools by name, and the box they came from."""

    box: Any = None


_BUILT: dict[Any, Any] = {}


def _run(folder: Path, resolution: str, *, size: tuple[int, int] = (2400, 1800),
         count: int = 2, um_per_px: float = _UM_PER_PX,
         tasks: tuple[str, ...] = ("position", "transform"), fresh: bool = False,
         ) -> tuple[Any, Any, Tools, JobSpec]:
    # Building 2400 x 1800 TIFFs dominates; a stack is read-only here except for
    # the hand transform every shared-stack test writes with the same numbers,
    # so each configuration is built once per process. *fresh*: a stack of the
    # caller's own, for a test that writes other numbers.
    key = (resolution, size, count, um_per_px, tasks)
    if key in _BUILT and not fresh:
        return _BUILT[key]
    folder.mkdir()
    for index in range(count):
        _section(size).rotate(7 * index, fillcolor=(0, 0, 0)).save(
            folder / f"s{index}.tif", resolution_unit=3, x_resolution=10000 / um_per_px,
            y_resolution=10000 / um_per_px)
    spec = JobSpec(image_folder=str(folder), model="fake", preprocess="none",
                   image_resolution=resolution, tasks=list(tasks))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: TwoRegionAtlas())
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 0.2 + 0.1 * index
    box = build_tools(state, ctx, spec)
    tools = Tools({tool.__name__: tool for tool in box.tools})
    tools.box = box
    if fresh:
        return state, ctx, tools, spec
    _BUILT[key] = state, ctx, tools, spec
    return _BUILT[key]


def _images(result: dict[str, Any]) -> list[Image.Image]:
    # What the ADK agent receives: the door's JPEG parts.
    return [Image.open(io.BytesIO(part.inline_data.data))
            for part in package_result(result)[TOOL_MEDIA_PARTS_KEY] if part.inline_data]


def _widths(result: dict[str, Any]) -> list[int]:
    # Widths: a caption adds a band below every picture; the tissue here is
    # wider than tall, so a picture's width is its long edge.
    return [image.width for image in _images(result)]


def _entry(**extra: Any) -> dict[str, Any]:
    return {"id": "s0.tif", "rotation_deg": 4.0, "scale_x": 1.1, "scale_y": 0.95,
            "translate_x_mm": 0.12, "translate_y_mm": -0.05, **extra}


def _seed_sections(state: Any, ctx: Any) -> list[Image.Image]:
    """The opening strip(s), and each section's tile as the strip draws it."""
    parts = build_seed_message(state, ctx).parts or []
    strips = [Image.open(io.BytesIO(p.inline_data.data)) for p in parts if p.inline_data]
    _count, tile = strip_layout(image_limit(ctx)[0], opening_edge(ctx))
    assert strips[0].width == len(state.slices) * tile + (len(state.slices) - 1) * COLUMN_GAP
    return [section_tile(ctx, state, record, tile) for record in state.in_order()]


# --- each level's two sizes ------------------------------------------------


@pytest.mark.parametrize("level", ["low", "medium", "high", "auto"])
def test_each_level_opens_at_one_size_and_shows_later_pictures_at_another(
    tmp_path: Path, level: str,
):
    opening, later = PICTURE_EDGES[level]
    state, ctx, tools, _spec = _run(tmp_path / level, level)

    # Each tile is the opening size, shrunk by at most a few pixels to fill the strip.
    assert all(opening - 8 <= image.width <= opening for image in _seed_sections(state, ctx))
    assert _widths(tools["look"]("section", sections=["s0.tif"])) == [later]
    assert _widths(tools["look"]("overlay", sections=["s0.tif"])) == [later]
    # A change tool's picture of what it wrote is the later size too (elastix
    # finds nothing to fit on this flat synthetic tissue, so the hand tool draws).
    assert _widths(tools["interactive_transform"]([_entry()])) == [later]


def test_a_small_snapshot_is_never_upsampled(tmp_path: Path):
    """A 300 x 225 px snapshot stays at most its own pixels at every size."""
    state, ctx, tools, _spec = _run(tmp_path / "small", "high", size=(300, 225),
                                    um_per_px=20.0)
    whole = render_slice(ctx, state.slices[0], long_edge=4096)
    assert max(whole.size) <= 300
    assert _seed_sections(state, ctx)[0].width == whole.width
    assert _widths(tools["look"]("section", sections=["s0.tif"])) == [whole.width]
    (overlay,) = _images(tools["look"]("overlay", sections=["s0.tif"]))
    assert overlay.width < PICTURE_EDGES["high"][1]


# --- auto -------------------------------------------------------------------


def test_auto_sizes_each_call_and_clamps(tmp_path: Path):
    state, ctx, tools, spec = _run(tmp_path / "auto", "auto")
    look = tools["look"]
    assert "resolution" in inspect.signature(look).parameters
    plain = look("section", sections=["s0.tif"])
    assert _widths(plain) == [512] and "resolution_note" not in plain
    chosen = look("section", sections=["s0.tif"], resolution=1200)
    assert _widths(chosen) == [1200] and "resolution_note" not in chosen
    # The cap is the driver model's largest image: the
    # OpenAI lanes' 2048 px here, the run's model being no other lane.
    big = look("overlay", sections=["s0.tif"], resolution=5000)
    assert _widths(big) == [2048]
    assert "2048" in big["resolution_note"]
    small = look("section", sections=["s0.tif"], resolution=20)
    assert _widths(small) == [MIN_RESOLUTION] and str(MIN_RESOLUTION) in small["resolution_note"]
    refused = look("section", sections=["s0.tif"], resolution="large")
    assert refused["error"] == "BAD_ARGS"
    # A change tool's picture is the level's later size: it takes no resolution.
    assert _widths(tools["interactive_transform"]([_entry()])) == [512]
    # The cap is the driver model's own: Claude's for the MCP host.
    claude = build_tools(state, ctx, spec, max_view_edge=CLAUDE_MAX_VIEW_EDGE)
    assert claude.max_view_edge == CLAUDE_MAX_VIEW_EDGE == 2000
    tool = next(tool for tool in claude.tools if tool.__name__ == "look")
    big = tool("overlay", sections=["s0.tif"], resolution=5000)
    assert _widths(big) == [2000] and "2000" in big["resolution_note"]


# --- what is computed does not move ------------------------------------------


#: A hand transform about the canvas centre on this stack (re-pinned when the
#: per-entry pivot was deleted): the stored numbers are the same at every level.
_PINNED_S1_ADJUST = [1.0973204552858067, 0.04970148754268928, -0.05351097141424801,
                     -0.10230949482471711, 0.947685847746833, 0.06620071242783103]


def _fixed_stack(folder: Path, level: str) -> tuple[Any, Any, Any]:
    """The stack the pinned numbers were computed on (two 2400 x 1800 sections)."""
    from langslice.job.job import Job

    folder.mkdir()
    for index in range(2):
        arr = np.zeros((1800, 2400, 3), dtype=np.uint8)
        arr[300:1500, 400:2000] = 120
        arr[600:1000, 800:1400, 1] = 220
        arr[1100:1300, 1500:1800, 0] = 250
        Image.fromarray(arr).rotate(7 * index, fillcolor=(0, 0, 0)).save(
            folder / f"s{index}.tif", resolution_unit=3, x_resolution=10000 / 2.5,
            y_resolution=10000 / 2.5)
    spec = JobSpec(image_folder=str(folder), model="fake", preprocess="none",
                   image_resolution=level)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: TwoRegionAtlas())
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 0.2 + 0.1 * index
    job = Job(state, spec, layout=ctx.layout, results_path=ctx.results_path)
    job.workspace = ctx
    return state, ctx, job


@pytest.mark.parametrize("level", ["low", "auto"])
def test_written_transforms_are_the_numbers_from_before(tmp_path: Path, level: str):
    from langslice.ops import transforms

    state, ctx, job = _fixed_stack(tmp_path / level, level)
    transforms.interactive_transform(job, ctx, [{
        "id": "s1.tif", "rotation_deg": 4.0, "scale_x": 1.1, "scale_y": 0.95,
        "translate_x_mm": 0.12, "translate_y_mm": -0.05,
        "shear": 0.0}])
    assert state.by_id("s1.tif").transform["params"] == _PINNED_S1_ADJUST


def _digest(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def test_the_image_model_and_deformable_fit_inputs_do_not_depend_on_the_level(tmp_path: Path):
    from langslice.core import deformation
    from langslice.core.handoff import prepare_linear_registration

    seen: dict[str, tuple[str, str, Any]] = {}
    for level in ("low", "auto"):
        state, ctx, tools, _ = _run(tmp_path / level, level)
        tools["interactive_transform"]([_entry()], view=False)
        prepared = prepare_linear_registration(state, ctx, "s0.tif")
        grid = deformation.fit_grid(state, ctx, state.by_id("s0.tif"))
        seen[level] = (_digest(prepared.image), _digest(grid.image), prepared.metadata)
    assert seen["low"] == seen["auto"]


def test_a_larger_picture_shows_the_same_map(tmp_path: Path, monkeypatch):
    """Drawn larger, the transformed section lands where the small picture puts it."""
    from langslice.core import canvas, placement

    drawn: list[np.ndarray] = []

    def capture(*args, **kwargs):
        images = canvas.physical_views(*args, **kwargs)
        x0, y0, x1, y1 = kwargs["panel_frames"][0].content_box
        drawn.append(np.asarray(images[0].convert("L"))[y0:y1, x0:x1])
        return images

    monkeypatch.setattr(placement, "physical_views", capture)
    _low_state, _low_ctx, low, _ = _run(tmp_path / "low", "low", fresh=True)
    _big_state, _big_ctx, big, _ = _run(tmp_path / "high", "high", fresh=True)
    entry = _entry(rotation_deg=12.0)
    low["interactive_transform"]([entry])
    big["interactive_transform"]([entry])
    assert len(drawn) == 2  # each call's picture of what it wrote

    def tissue(body: np.ndarray) -> tuple[float, float, float]:
        ys, xs = np.nonzero(body > 60)
        return xs.mean() / body.shape[1], ys.mean() / body.shape[0], len(xs) / body.size

    small, large = tissue(drawn[0]), tissue(drawn[1])
    assert drawn[1].shape[1] / drawn[0].shape[1] == pytest.approx(2.0, rel=0.02)
    assert small == pytest.approx(large, abs=0.005)


# --- captions ------------------------------------------------------------------


def test_a_long_caption_wraps_instead_of_running_off_a_small_picture():
    font = _caption_font()
    text = "12: section_08_overlay.tif  9.00 mm (+0.40 to next)  [damaged: dorsal cortex torn]"
    picture = Image.new("RGB", (160, 120), (40, 40, 40))
    labelled = caption(picture, text)
    wrapped = wrap_caption(text, font, picture.width - 6)
    lines = wrapped.split("\n")
    assert len(lines) > 2
    assert all(font.getlength(line) <= picture.width - 6 for line in lines)
    # Nothing is dropped: the words come back in order.
    assert " ".join(wrapped.split()) == " ".join(text.split())
    # The band grows to hold every line; the picture keeps its width.
    assert labelled.width == picture.width
    assert labelled.height - picture.height >= len(lines) * CAPTION_PX
    # A word wider than the picture breaks between characters.
    assert all(font.getlength(line) <= 30 for line in
               wrap_caption("section_08_overlay.tif", font, 30).split("\n"))
