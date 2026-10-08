"""The fit tools through the tool door: elastix_affine, ants_syn and the
packaged trace_borders (a stand-in image model), with their pictures, their
background notices and the stale-deformation sweep.

The atlas is the deformable tests' synthetic one (50 um voxels, a section
rendered from it at 25 um/px): real enough for Elastix and ANTs to fit, and
the shape the Elastix tests use. ANTs cases skip without antspyx.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import deformation
from langslice.core.nonlinear.types import GeneratedSegmentation
from langslice.core.sizes import picture_edge
from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from langslice.providers.registry import resolve_image_model
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, bump_field, render_section
from tests.linear_tool_helpers import ToolContext
from tests.linear_tool_helpers import tool_named as _tool

needs_ants = pytest.mark.skipif(not deformation.ants_available(), reason="needs antspyx")
IDENTITY = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "calibration": {"section_um_per_px": 25.0, "source": "host"}}


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    """Fits in this process at the coarse detail level, as the deformable tests run."""
    monkeypatch.setattr(deformation, "USE_PROCESS_POOL", False)
    monkeypatch.setattr(deformation, "DETAIL_LEVEL", "coarse")


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


class StubModel:
    """The run's image model, its call answered with the drawn borders (the
    attachment with more colour); *gate* holds the reply back until set."""

    def __init__(self, spec: JobSpec) -> None:
        self.calls: list[Any] = []
        self.gate = threading.Event()
        self.gate.set()
        resolved = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)
        self.model = dataclasses.replace(resolved, call=self.generate)

    def generate(self, request: Any) -> GeneratedSegmentation:
        self.calls.append(request)
        assert self.gate.wait(10)
        images = [request.slice_image, *request.reference_images]

        def colour(image: Image.Image) -> float:
            pixels = np.asarray(image.convert("RGB")).astype(np.int16)
            return float(np.abs(pixels[..., 0] - pixels[..., 2]).mean())

        drawn = max(images, key=colour)
        return GeneratedSegmentation(image=drawn.convert("RGB"), provider=request.provider,
                                     model=str(request.model), route="test-stub")


def _run(tmp_path: Path, atlas: SyntheticAtlas, *, n: int = 1, transformed: bool = False,
         tasks: tuple[str, ...] = ("transform", "nonlinear"), provider: str = "none",
         **spec_kwargs: Any):
    """``(state, ctx, box, stub)``: *n* sections deformed off the atlas by a
    smooth field, at 0.1 mm (with *transformed*, under the identity)."""
    for seed in range(n):
        image, _labels = render_section(atlas, SMOOTH_FIELD() if seed else bump_field([]),
                                        seed=seed)
        image.save(tmp_path / f"s{seed}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   tasks=list(tasks), inputs={"pixel_size_um": 25.0},
                   nonlinear=NonlinearSpec(provider=provider), **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for record in state.slices:
        record.position_mm = 0.1
        if transformed:
            record.transform = dict(IDENTITY)
    stub = StubModel(spec) if provider != "none" else None
    box = build_tools(state, ctx, spec, image_model=stub.model if stub else None)
    return state, ctx, box, stub


# --- elastix_affine -----------------------------------------------------------------


def test_elastix_affine_fits_writes_one_step_and_shows_each_section(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas, n=2)
    result = _tool(box, "elastix_affine")()
    assert result["status"] == "ok", result
    rows = result["results"]
    assert [row["id"] for row in rows] == ["s0.png", "s1.png"]
    for row in rows:
        assert row["status"] == "ok" and row["iou"] > 0.9
        assert "params" not in row  # the six raw numbers stay host-side
        assert set(row["physical"]) == {"rotation_deg", "scale_x", "scale_y", "shear",
                                        "translate_x_mm", "translate_y_mm", "pivot"}
        assert row["calibration"] == {"section_um_per_px": 25.0, "source": "host"}
    assert all(record.transform["kind"] == "elastix" for record in state.slices)
    assert len(box.job.undo_stack) == 1
    # A picture per fitted section, under its new transform with the borders.
    assert len(result["pictures"]) == 2 and len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    assert "s0.png overlay" in result["pictures"][0]["caption"]
    assert "elastix transform" in result["pictures"][0]["caption"]
    assert [r.tool for r in box.job.views.records()] == ["elastix_affine"] * 2
    _tool(box, "undo")()
    assert all(record.transform is None for record in state.slices)


def test_a_restricted_fit_zooms_its_picture_to_the_regions(tmp_path: Path, atlas):
    state, ctx, box, _ = _run(tmp_path, atlas)
    result = _tool(box, "elastix_affine")(["s0.png"], ["TH"])
    assert result["status"] == "ok", result
    (row,) = result["results"]
    assert row["regions"]["include"] == ["TH"]
    box_fractions = row["restrict_box"]
    assert len(box_fractions) == 4 and 0 <= box_fractions[0] < box_fractions[2] <= 1
    (record,) = box.job.views.records()
    assert record.recipe["args"]["zoom"] == box_fractions
    assert state.slices[0].transform["kind"] == "elastix"
    # The region comes out at the picture size, its caption at that width.
    (picture,) = result[TOOL_MEDIA_PARTS_KEY]
    edge = picture_edge(ctx)
    assert picture.width == edge
    assert max(record.recipe["shown"]) == edge


def test_elastix_affine_without_a_picture_and_its_refusals(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas)
    fit = _tool(box, "elastix_affine")
    nope = fit(["nope.png"])
    assert nope["error"] == "UNKNOWN_SLICE_IDS" and nope["unknown"] == ["nope.png"]
    assert nope["filenames"] == ["s0.png"]
    assert fit(["s0.png"], ["XYZ"])["error"] == "UNKNOWN_REGIONS"
    unavailable = fit(["s0.png"], [], "nissl")
    assert unavailable["error"] == "FIT_ATLAS_UNAVAILABLE"
    assert unavailable["atlas_image"] == ["template"]
    assert fit(["s0.png"], [], "spline")["error"] == "BAD_ARGS"
    assert state.slices[0].transform is None and box.job.undo_stack == []
    quiet = fit(["s0.png"], view=False)
    assert quiet["status"] == "ok" and quiet["results"][0]["status"] == "ok"
    assert "pictures" not in quiet and TOOL_MEDIA_PARTS_KEY not in quiet


def test_a_fit_that_turns_a_section_far_reports_the_turn():
    from langslice.ops.transforms import LARGE_TURN_DEG, _turn_deg

    def fitted(rotation: float) -> dict:
        return {"physical": {"rotation_deg": rotation}}

    assert _turn_deg(None, fitted(-179.9)) > LARGE_TURN_DEG
    assert _turn_deg(fitted(170.0), fitted(-175.0)) == pytest.approx(15.0)
    assert _turn_deg(fitted(10.0), fitted(-20.0)) < LARGE_TURN_DEG


# --- ants_syn --------------------------------------------------------------------------


def test_ants_syn_is_refused_without_antspyx(tmp_path: Path, atlas, monkeypatch):
    import langslice.ops.deformable as ops_deformable

    state, _, box, _ = _run(tmp_path, atlas, transformed=True)
    monkeypatch.setattr(ops_deformable, "ants_ready", lambda: False)
    result = _tool(box, "ants_syn")(["s0.png"])
    assert result["status"] == "error" and result["error"] == "ANTS_MISSING"
    assert "antspyx" in result["message"]
    assert state.slices[0].deformation is None and box.job.undo_stack == []


@needs_ants
def test_ants_syn_builds_on_the_current_registration_and_shows_it(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas, n=2, transformed=True)
    fit = _tool(box, "ants_syn")
    result = fit(["s1.png"])
    assert result["status"] == "ok", result
    (row,) = result["results"]
    assert row["status"] == "ok" and row["written"] is True and row["steps"] == 1
    assert set(row["displacement_mm"]) == {"max", "median"}
    assert result["changed"][0]["deformation_steps"] == 1
    (picture,) = result["pictures"]
    assert "s1.png overlay" in picture["caption"] and "deformation drawn" in picture["caption"]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1
    saved = box.job.views.lookup(picture["id"])
    assert saved is not None and saved.tool == "ants_syn"
    # The next fit composes onto it; view=False draws nothing.
    again = fit(["s1.png"], ["TH"], view=False)
    assert again["results"][0]["steps"] == 2
    assert "pictures" not in again and TOOL_MEDIA_PARTS_KEY not in again
    assert len(state.by_id("s1.png").deformation["steps"]) == 2
    _tool(box, "undo")()
    assert len(state.by_id("s1.png").deformation["steps"]) == 1


@needs_ants
def test_a_changed_placement_clears_the_deformation_in_the_same_step(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas, n=2, transformed=True)
    assert _tool(box, "ants_syn")(["s1.png"], view=False)["status"] == "ok"
    turned = _tool(box, "interactive_transform")([{"id": "s1.png", "rotation_deg": 4.0}],
                                                 view=False)
    assert turned["deformation_cleared"] == ["s1.png"]
    assert state.by_id("s1.png").deformation is None
    _tool(box, "undo")()  # the placement and the deformation come back together
    assert state.by_id("s1.png").deformation is not None
    assert state.by_id("s1.png").transform["kind"] == "interactive"
    # With its picture: the deformation goes first, so the picture shows the
    # section as it now stands.
    shown = _tool(box, "interactive_transform")([{"id": "s1.png", "rotation_deg": 4.0}])
    assert shown["deformation_cleared"] == ["s1.png"]
    assert "no deformation" in shown["pictures"][0]["caption"]


@needs_ants
def test_submit_takes_left_linear_in_a_nonlinear_run(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas, n=2, transformed=True, tasks=("nonlinear",))
    _tool(box, "ants_syn")(["s1.png"], view=False)
    submit = _tool(box, "submit")
    refused = submit("done", [], [], left_linear=[{"id": "s1.png", "reason": "fitted"}],
                     tool_context=ToolContext())
    assert refused["error"] == "HAS_DEFORMATION" and refused["ids"] == ["s1.png"]
    stray = submit("done", [], [], left_linear=[{"id": "s0.png", "why": "flat"}])
    assert stray["error"] == "UNKNOWN_ARGUMENTS"
    assert stray["problems"][0]["argument"] == "left_linear[0]"
    done = submit("done", [], [], left_linear=[{"id": "s0.png", "reason": "torn tissue"}],
                  tool_context=ToolContext())
    assert done["status"] == "ok", done
    assert done["left_linear"] == {"s0.png": "torn tissue"}
    assert state.submitted is True


# --- trace_borders ----------------------------------------------------------------------


def test_trace_borders_comes_only_with_an_image_model(tmp_path: Path, atlas):
    _, _, box, _ = _run(tmp_path, atlas, transformed=True)
    assert "trace_borders" not in box.names and "ants_syn" in box.names


@needs_ants
def test_trace_borders_starts_background_work_and_its_notice_opens_a_later_reply(
    tmp_path: Path, atlas,
):
    state, _, box, stub = _run(tmp_path, atlas, transformed=True, tasks=("nonlinear",),
                               provider="openai-oauth")
    assert stub is not None
    stub.gate.clear()
    trace = _tool(box, "trace_borders")
    started = trace("s0.png")
    assert started["status"] == "started" and started["work"] == "w1"
    assert started["id"] == "s0.png" and started["trace"]["status"] == "running"
    assert "submit waits for it" in started["message"]
    # status lists it while it runs, and asking again starts nothing.
    (running,) = _tool(box, "status")()["background_running"]
    assert running["id"] == "w1" and running["kind"] == "trace_borders"
    again = trace("s0.png")
    assert again["status"] == "running" and again["work"] == "w1"
    stub.gate.set()
    box.job.background.wait_all()

    reply = _tool(box, "status")()
    assert list(reply)[0] == "background"  # the notice comes first
    (notice,) = reply["background"]
    assert notice.startswith("w1 trace_borders of s0.png finished")
    work_pictures = [entry for entry in reply["pictures"] if entry.get("work") == "w1"]
    assert len(work_pictures) == 2 and len(reply[TOOL_MEDIA_PARTS_KEY]) == 2
    assert len(state.by_id("s0.png").deformation["steps"]) == 1
    # Handed out once.
    assert "background" not in _tool(box, "status")()

    # The same trace at the same placement is already fitted: nothing again.
    landed = trace("s0.png")
    assert landed["status"] == "ok" and landed["landed"] is True
    assert landed["trace"]["cached"] is True
    assert len(stub.calls) == 1
    assert box.job.background.running() == []
