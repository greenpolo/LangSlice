"""``make_image`` through the tool door, and a script's ``add_image``: made section
images are files with entries, remade for some sections, deleted, undone; a fit
names the image it read by file and digest; look draws ``preprocessed`` by default."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import section_images as images
from langslice.core.sections import render_slice
from langslice.core.spec import JobSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import Job
from tests.linear_tool_helpers import load_checkpoint
from tests.linear_tool_helpers import tool_named as _tool
from tests.test_section_images import _rgb_section

UM = 25.0


@pytest.fixture(scope="module")
def atlas() -> Any:
    from tests.deformable_synthetic import SyntheticAtlas

    return SyntheticAtlas()


def _box(tmp_path: Path, atlas: Any, n: int = 2) -> tuple[Any, Any, Any]:
    """``(job, ctx, box)``: *n* three-channel sections, opened as every door opens a job."""
    for index in range(n):
        _rgb_section(index).save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model",
                   tasks=["position", "transform", "nonlinear"], inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    box = build_tools(job.state, ctx, spec, job=job)
    for record in job.state.slices:
        record.position_mm = 0.1
        record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            "calibration": {"section_um_per_px": UM, "source": "host"}}
    return job, ctx, box


def test_make_image_writes_a_file_per_section_and_shows_it(tmp_path: Path, atlas):
    job, ctx, box = _box(tmp_path, atlas)
    made = _tool(box, "make_image")("reds", weights={"red": 1}, clahe_clip=2)
    assert made["status"] == "ok", made
    assert made["image"] == "reds" and made["sections"] == "every section"
    assert made["recipe"] == {"weights": {"red": 1.0}, "clahe_clip": 2.0, "clahe_tiles": 8}
    assert made["section_images"] == ["red", "green", "blue", "preprocessed", "reds"]
    # One picture per section, drawn as look's section mode draws the image.
    assert len(made["pictures"]) == 2 and len(made[TOOL_MEDIA_PARTS_KEY]) == 2
    assert all("section image reds (weights red 1, CLAHE 2 on 8x8 tiles)" in picture["caption"]
               for picture in made["pictures"])
    assert {record.tool for record in job.views.records()} == {"make_image"}
    for record in job.state.slices:
        entry = record.images["reds"]
        assert (job.folder / entry["file"]).is_file()
        assert entry["source"]["recipe"]["weights"] == {"red": 1.0}
    assert len(job.undo_stack) == 1
    listed = _tool(box, "list_files")("sections/s0/images")
    assert "reds.png" in listed["text"] and "preprocessed.png" in listed["text"]
    status = _tool(box, "status")()["section_images"]
    assert status["reds"] == made["recipe"]


def test_remade_for_some_sections_and_deleted_and_undone(tmp_path: Path, atlas):
    job, ctx, box = _box(tmp_path, atlas)
    tool = _tool(box, "make_image")
    tool("blues", weights={"blue": 1}, view=False)
    first = job.state.by_id("s1.png").images["blues"]["file"]
    again = tool("blues", sections=["s1"], weights={"blue": 1}, invert=True, view=False)
    assert again["sections"] == ["s1.png"]
    assert job.state.by_id("s1.png").images["blues"]["file"].endswith("blues.2.png")
    assert (job.folder / first).is_file()  # files are never deleted
    held = _tool(box, "status")()["section_images"]["blues"]
    assert held["varies"]["s1.png"]["invert"] is True and "invert" not in held
    gone = tool("blues", sections=["s0.png"], delete=True)
    assert gone["deleted"] is True and gone["sections"] == ["s0.png"]
    assert _tool(box, "status")()["section_images"]["blues"]["except"] == ["s0.png"]
    assert len(job.undo_stack) == 3
    _tool(box, "undo")(steps=3)
    assert "blues" not in job.state.by_id("s0.png").images
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None and "blues" not in saved.by_id("s1.png").images


def test_make_image_refuses_before_writing(tmp_path: Path, atlas):
    job, _ctx, box = _box(tmp_path, atlas)
    tool = _tool(box, "make_image")
    assert tool("red", weights={"green": 1})["error"] == "BAD_ARGS"  # a raw channel's name
    assert tool("bad name!")["error"] == "BAD_ARGS"
    assert tool("none")["error"] == "BAD_ARGS"
    assert tool("x", weights={"nope": 1})["error"] == "UNKNOWN_IMAGE"
    assert tool("x", weights={"x": 1})["error"] == "BAD_ARGS"  # an image from itself
    assert tool("x", clahe_clip=-1)["error"] == "BAD_ARGS"
    assert tool("x", sections=["nope.png"])["error"] == "UNKNOWN_SLICE_IDS"
    assert tool("x", delete=True)["error"] == "UNKNOWN_IMAGE"
    assert tool("preprocessed", delete=True)["error"] == "BAD_ARGS"
    assert not job.undo_stack


def test_preprocessed_remade_then_made_with_the_defaults_is_the_opening_again(
        tmp_path: Path, atlas):
    job, ctx, box = _box(tmp_path, atlas)
    record = job.state.by_id("s0.png")
    opening = dict(record.images["preprocessed"])
    default = np.asarray(render_slice(ctx, record, long_edge=160)).copy()
    _tool(box, "make_image")("preprocessed", weights={"green": 1}, view=False)
    record = job.state.by_id("s0.png")
    assert record.images["preprocessed"]["file"].endswith("preprocessed.2.png")
    changed = np.asarray(render_slice(ctx, record, long_edge=160))
    assert not np.array_equal(changed, default)
    # The look's default draws it, and the fits read it.
    looked = _tool(box, "look")("section", sections=["s0.png"])
    assert "section image preprocessed (weights green 1" in looked["pictures"][0]["caption"]
    _tool(box, "make_image")("preprocessed", view=False)
    record = job.state.by_id("s0.png")
    assert record.images["preprocessed"]["file"] == opening["file"]
    assert record.images["preprocessed"]["source"] == {"opening": "adaptive"}
    assert np.array_equal(np.asarray(render_slice(ctx, record, long_edge=160)), default)


def test_look_draws_preprocessed_by_default_and_raw_channels_when_named(tmp_path: Path, atlas):
    _job, _ctx, box = _box(tmp_path, atlas)
    look = _tool(box, "look")
    plain = look("section", sections=["s0.png"])
    assert "section image preprocessed" in plain["pictures"][0]["caption"]
    pixels = np.asarray(plain[TOOL_MEDIA_PARTS_KEY][0])
    assert (pixels[..., 0] == pixels[..., 1]).all()  # one image, in gray
    together = look("section", sections=["s0.png"], images=["red", "green", "blue"])
    assert "raw red + green + blue" in together["pictures"][0]["caption"]
    assert look("section", images=["red", "preprocessed"])["error"] == "MIXED_IMAGES"
    overlay = look("overlay", sections=["s0.png"])
    assert "section image preprocessed" in overlay["pictures"][0]["caption"]


def test_a_fit_names_the_section_image_it_read(tmp_path: Path, atlas):
    pytest.importorskip("itk")
    job, _ctx, box = _box(tmp_path, atlas)
    _tool(box, "make_image")("greens", weights={"green": 1}, view=False)
    fit = _tool(box, "elastix_affine")(["s0.png"], section_image="greens", view=False)
    assert fit["status"] == "ok", fit
    used = fit["results"][0]["inputs"]["section_image"]
    entry = job.state.by_id("s0.png").images["greens"]
    assert used == {"name": "greens", "kind": "made", "file": entry["file"],
                    "digest": entry["digest"]}
    raw = _tool(box, "elastix_affine")(["s1.png"], section_image="red", view=False)
    assert raw["results"][0]["inputs"]["section_image"] == {
        "name": "red", "kind": "raw", "file": "s1.png", "digest": None}
    assert _tool(box, "elastix_affine")(["s0.png"], section_image="nope")["error"] == \
        "UNKNOWN_IMAGE"


def test_remaking_an_image_makes_the_section_stale_for_a_long_verb(tmp_path: Path, atlas):
    from langslice.ops.inputs import section_inputs

    job, _ctx, box = _box(tmp_path, atlas)
    record = job.state.by_id("s0.png")
    before = section_inputs(job.state, record)
    _tool(box, "make_image")("preprocessed", sections=["s0.png"], weights={"red": 1},
                             view=False)
    assert section_inputs(job.state, job.state.by_id("s0.png")) != before
    other = job.state.by_id("s1.png")
    assert section_inputs(job.state, other) == section_inputs(job.state, other)


def test_a_script_adds_an_image_the_fits_and_look_read(tmp_path: Path, atlas):
    import langslice

    for index in range(2):
        _rgb_section(index).save(tmp_path / f"s{index}.png")
    with langslice.create_job(tmp_path, tasks=["transform", "nonlinear"],
                              positions={"s0.png": 0.1, "s1.png": 0.1},
                              pixel_size_um=UM, atlas_loader=lambda _n: atlas) as job:
        c = job.context
        shape = c.working_grid("s0.png")
        raw = c.section_raster("s0.png", raw=True, images=["red"]).array[..., 0]
        assert raw.shape == shape
        hema = 1.0 - raw.astype(np.float32) / 255.0
        entry = c.add_image("s0.png", "hema", hema, note="one minus red")
        assert entry["source"] == {"script": "one minus red"}
        assert (Path(job.folder) / entry["file"]).is_file()
        assert "hema" in c.image_names("s0.png") and "hema" not in c.image_names("s1.png")
        held = np.asarray(Image.open(Path(job.folder) / entry["file"]))
        assert np.abs(held.astype(int) - (255 - raw.astype(int))).max() <= 1
        frame = c.fit_frame("s0.png", image="hema")
        assert frame.section_image_ref["name"] == "hema"
        assert frame.section_image_ref["digest"] == entry["digest"]
        looked = job.look(mode="section", sections=["s0.png"], images=["hema"])
        assert "made by a script: one minus red" in looked["pictures"][0]["caption"]
        from langslice.core.plugins import ToolError

        with pytest.raises(ToolError) as refused:
            c.add_image("s0.png", "red", hema)
        assert refused.value.code == "BAD_ARGS"
        with pytest.raises(ToolError):
            c.add_image("s0.png", "small", np.zeros((4, 4), np.uint8))
        summary = job.status()["section_images"]["hema"]
        assert summary == {"script": "one minus red", "except": ["s1.png"]}
        assert images.PREPROCESSED in job.status()["section_images"]


def test_a_fit_picture_is_drawn_on_the_section_image_it_read(tmp_path: Path, atlas):
    pytest.importorskip("itk")
    _job, _ctx, box = _box(tmp_path, atlas)
    _tool(box, "make_image")("greens", weights={"green": 1}, view=False)
    fit = _tool(box, "elastix_affine")(["s0.png"], section_image="greens")
    assert fit["status"] == "ok", fit
    assert "section image greens" in fit["pictures"][0]["caption"]
    plain = _tool(box, "elastix_affine")(["s1.png"])
    assert "section image preprocessed" in plain["pictures"][0]["caption"]
