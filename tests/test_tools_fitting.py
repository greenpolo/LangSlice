"""The fit tools through the tool door: elastix_affine, fireants_syn and the
packaged trace_borders (a stand-in image model), with their refusals, their
background notices and the stale-deformation sweep.

The atlas is the deformable tests' synthetic one (50 um voxels, a section
rendered from it at 25 um/px): real enough for Elastix and FireANTs to fit, and
the shape the Elastix tests use. FireANTs cases skip without torch and fireants.
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

needs_engine = pytest.mark.skipif(not deformation.engine_available(), reason="needs FireANTs")
IDENTITY = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "calibration": {"section_um_per_px": 25.0, "source": "host"}}


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


def test_elastix_affine_refuses_then_fits_once_and_shows_the_region(tmp_path: Path, atlas):
    state, ctx, box, _ = _run(tmp_path, atlas)
    fit = _tool(box, "elastix_affine")
    nope = fit(["nope.png"])
    assert nope["error"] == "UNKNOWN_SLICE_IDS" and nope["unknown"] == ["nope.png"]
    assert nope["filenames"] == ["s0.png"]
    assert fit(["s0.png"], ["XYZ"])["error"] == "UNKNOWN_REGIONS"
    unavailable = fit(["s0.png"], [], "nissl")
    assert unavailable["error"] == "BAD_ARGS"
    assert unavailable["atlas_image"] == ["template"]
    assert fit(["s0.png"], [], "spline")["error"] == "BAD_ARGS"
    assert fit([])["error"] == "BAD_ARGS"
    many = fit(["s0.png"] * 5)
    assert many == {"status": "error", "error": "TOO_MANY_SECTIONS", "max_sections": 4,
                    "requested": 5}
    assert state.slices[0].transform is None and box.job.undo_stack == []

    # One real fit, restricted to a region: one undo step, a picture zoomed to it.
    result = fit(["s0.png"], ["TH"])
    assert result["status"] == "ok", result
    (row,) = result["results"]
    assert row["status"] == "ok" and "params" not in row  # raw numbers stay host-side
    assert set(row["physical"]) == {"rotation_deg", "scale_x", "scale_y", "shear",
                                    "translate_x_mm", "translate_y_mm", "pivot"}
    assert row["calibration"] == {"section_um_per_px": 25.0, "source": "host"}
    assert row["regions"]["include"] == ["TH"]
    box_fractions = row["restrict_box"]
    assert len(box_fractions) == 4 and 0 <= box_fractions[0] < box_fractions[2] <= 1
    (record,) = box.job.views.records()
    assert record.tool == "elastix_affine"
    assert record.recipe["args"]["zoom"] == box_fractions
    assert state.slices[0].transform["kind"] == "elastix"
    assert len(box.job.undo_stack) == 1
    (picture,) = result[TOOL_MEDIA_PARTS_KEY]
    edge = picture_edge(ctx)
    assert picture.width == edge and max(record.recipe["shown"]) == edge
    _tool(box, "undo")()
    assert state.slices[0].transform is None


# --- fireants_syn --------------------------------------------------------------------------


@needs_engine
def test_fireants_syn_builds_on_the_current_registration_and_shows_it(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas, n=2, transformed=True)
    fit = _tool(box, "fireants_syn")
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
    assert saved is not None and saved.tool == "fireants_syn"
    # The next fit composes onto it; view=False draws nothing.
    again = fit(["s1.png"], ["TH"], view=False)
    assert again["results"][0]["steps"] == 2
    assert "pictures" not in again and TOOL_MEDIA_PARTS_KEY not in again
    assert len(state.by_id("s1.png").deformation["steps"]) == 2
    _tool(box, "undo")()
    assert len(state.by_id("s1.png").deformation["steps"]) == 1
    # A changed placement clears the deformation in the same step; undo restores both.
    turned = _tool(box, "interactive_transform")([{"id": "s1.png", "rotation_deg": 4.0}],
                                                 view=False)
    assert turned["deformation_cleared"] == ["s1.png"]
    assert state.by_id("s1.png").deformation is None
    _tool(box, "undo")()
    assert state.by_id("s1.png").deformation is not None
    assert state.by_id("s1.png").transform["kind"] == "interactive"


def test_submit_needs_no_deformation_in_a_nonlinear_run(tmp_path: Path, atlas):
    state, _, box, _ = _run(tmp_path, atlas, n=2, transformed=True, tasks=("nonlinear",))
    submit = _tool(box, "submit")
    stray = submit("done", [], [], left_linear=[{"id": "s0.png", "reason": "flat"}])
    assert stray["error"] == "UNKNOWN_ARGUMENTS"
    done = submit("done", [], [], tool_context=ToolContext())
    assert done["status"] == "ok", done
    assert state.submitted is True


# --- trace_borders ----------------------------------------------------------------------


def test_trace_borders_comes_only_with_an_image_model(tmp_path: Path, atlas):
    _, _, box, _ = _run(tmp_path, atlas, transformed=True)
    assert "trace_borders" not in box.names and "fireants_syn" in box.names


@needs_engine
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
