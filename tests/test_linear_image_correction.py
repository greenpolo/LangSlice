"""The optional image-model tool stores annotations without changing placement."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from langslice.agent.prompt import build_job_statement
from langslice.core import handoff
from langslice.core.spec import DEFAULT_TASKS, JobSpec, NonlinearSpec
from langslice.core.state import StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import _tool_target_ids, build_tools
from langslice.job.checkpoint import load_checkpoint
from langslice.job.job import apply_host_inputs
from tests.linear_tool_helpers import stack as _stack
from tests.linear_tool_helpers import tool_named as _tool


def _placed(tmp_path: Path):
    state, ctx, spec = _stack(tmp_path, n=1, placed=True, tasks=["nonlinear"])
    state.slices[0].transform = {
        "kind": "interactive", "params": [1, 0, .1, 0, 1, .2],
        "calibration": {"section_um_per_px": 25, "source": "host"},
    }
    return state, ctx, spec


def test_image_tool_is_opt_in_and_takes_id_prompt_and_regions(tmp_path: Path):
    from google.adk.tools import FunctionTool

    state, ctx, spec = _stack(tmp_path, n=1)
    assert spec.tasks == list(DEFAULT_TASKS)
    assert "trace_borders" not in build_tools(state, ctx, spec).names
    spec.tasks = ["nonlinear"]
    box = build_tools(state, ctx, spec)
    schema = FunctionTool(_tool(box, "trace_borders"))._get_declaration()
    declaration = schema.model_dump()
    parameters = declaration["parameters"] or declaration["parameters_json_schema"]
    assert set(parameters["properties"]) == {"section", "prompt", "restrict_to"}
    assert not {"elastix_affine", "interactive_transform", "position_sections"} & set(
        box.names)
    restored = JobSpec.from_dict(spec.to_dict())
    assert restored.nonlinear == NonlinearSpec()


def test_image_correction_runs_in_background_and_submit_waits(tmp_path: Path, monkeypatch):
    import threading

    from langslice.core.nonlinear import registration_tool

    state, ctx, spec = _placed(tmp_path)
    transform = copy.deepcopy(state.slices[0].transform)
    spec.nonlinear = NonlinearSpec(provider="openai-api", image_model="test-model")
    release = threading.Event()
    calls = []

    def fake_start(actual_state, actual_ctx, section_id, **kwargs):
        assert actual_state is state and actual_ctx is ctx
        calls.append((section_id, kwargs))
        running = {"status": "running", "geometry_fingerprint": "geometry",
                   "prompt_edited": bool(kwargs["prompt"])}

        def job():
            assert release.wait(5)
            return {**running, "status": "ok"}

        return running, job

    monkeypatch.setattr(registration_tool, "start_correction", fake_start)
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "geometry")
    box = build_tools(state, ctx, spec)
    result = _tool(box, "trace_borders")(state.slices[0].id, "Edited prompt.")
    assert result["status"] == "started" and result["work"] == "w1", result
    # The door resolved the spec's provider and model once; the operation got it.
    model = calls[0][1].pop("image_model")
    assert (model.provider, model.model) == ("openai-api", "test-model")
    assert calls == [("s0.png", {
        "prompt": "Edited prompt.",
        "calls_dir": Path(ctx.job_folder) / "sections" / "s0" / "image_correction",
        "include": (), "exclude": (),
    })]
    # The tool returns while the image call is still running, with no images.
    assert result["trace"]["status"] == "running" and TOOL_MEDIA_PARTS_KEY not in result
    again = _tool(box, "trace_borders")("s0")  # the filename without its extension
    assert again["status"] == "running" and again["work"] == "w1"
    assert len(calls) == 1
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved.slices[0].image_correction == state.slices[0].image_correction
    restored = StackState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.slices[0].image_correction == state.slices[0].image_correction
    assert state.slices[0].transform == transform
    _tool(box, "undo")()
    assert state.slices[0].image_correction is None
    _tool(box, "redo")()
    assert state.slices[0].image_correction["status"] == "running"
    release.set()
    # submit waits for the work; the stand-in's trace has no lines to fit.
    assert _tool(box, "submit")("Done", [], [])["error"] == "MISSING_DEFORMATIONS"
    assert box.job.background.running() == []
    assert _tool(box, "submit")("Done", [], [], left_linear=[
        {"id": "s0.png", "reason": "The placement already fits."}])["status"] == "ok"
    assert state.slices[0].image_correction["status"] == "ok"
    assert load_checkpoint(ctx.checkpoint_path).slices[0].image_correction["status"] == "ok"
    assert state.slices[0].transform == transform
    assert _tool_target_ids(state, "trace_borders", {"section": "s0"}) == ["s0.png"]
    assert _tool_target_ids(state, "trace_borders", {"section": "0"}) == []


def test_result_of_an_undone_correction_does_not_land(tmp_path: Path, monkeypatch):
    from langslice.core.nonlinear import registration_tool

    state, ctx, spec = _placed(tmp_path)
    running = {"status": "running", "geometry_fingerprint": "geometry"}
    monkeypatch.setattr(registration_tool, "start_correction",
                        lambda *a, **k: (running, lambda: {**running, "status": "ok"}))
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "geometry")
    box = build_tools(state, ctx, spec)
    _tool(box, "trace_borders")("0")
    _tool(box, "undo")()
    assert box.job.settle_image_corrections() is False
    assert state.slices[0].image_correction is None
    box.job.background.wait_all()
    assert state.slices[0].deformation is None


@pytest.mark.parametrize("result", [None, {"status": "error"}, {
    "status": "ok", "geometry_fingerprint": "previous-placement",
}])
def test_submit_needs_no_completed_correction(tmp_path, monkeypatch, result):
    """Tracing is the agent's choice: with or without a current trace, submit
    asks only for the deformation (or the section named in left_linear)."""
    state, ctx, spec = _placed(tmp_path)
    state.slices[0].image_correction = result
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "current")
    box = build_tools(state, ctx, spec)
    response = _tool(box, "submit")("Done", [], [])
    assert response["error"] == "MISSING_DEFORMATIONS"
    assert _tool(box, "submit")("Done", [], [], left_linear=[
        {"id": "s0.png", "reason": "The placement already fits."}])["status"] == "ok"
    assert state.submitted


@pytest.mark.parametrize("key,code", [("keep_warp", "KEEPS_HOST_WARP"),
                                      ("nonlinear_skip", "NONLINEAR_SKIPPED")])
def test_sections_kept_out_of_nonlinear_need_no_trace_and_no_deformation(
        tmp_path, monkeypatch, key, code):
    """A section the host keeps out of Nonlinear is refused by the traces and
    skipped by both submit gates, so the run can still submit."""
    from langslice.ops import traces

    state, ctx, spec = _stack(
        tmp_path, n=2, placed=True, tasks=["nonlinear"], inputs={key: ["s1.png"]},
        nonlinear=NonlinearSpec(provider="openai-api", image_model="test-model"))
    apply_host_inputs(state, spec)
    for record in state.slices:
        record.transform = {"kind": "interactive", "params": [1, 0, .1, 0, 1, .2],
                            "calibration": {"section_um_per_px": 25, "source": "host"}}
    state.slices[0].image_correction = {"status": "ok", "geometry_fingerprint": "current"}
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "current")
    box = build_tools(state, ctx, spec)
    assert _tool(box, "trace_borders")("s1.png")["error"] == code
    rows = traces.trace_from_atlas(box.job, ctx, ["s1.png"], image_model=None).rows  # type: ignore[arg-type]
    assert [row["error"] for row in rows] == [code]
    assert _tool(box, "submit")("Done", [], [], left_linear=[
        {"id": "s0.png", "reason": "The placement already fits."}])["status"] == "ok"
    assert state.slices[1].deformation is None


def test_image_tool_reports_missing_placement_without_checkpoint_mutation(tmp_path, monkeypatch):
    from langslice.core.nonlinear import registration_tool

    state, ctx, spec = _stack(tmp_path, n=1, tasks=["nonlinear"])

    def missing(*args, **kwargs):
        raise ValueError("s0.png requires a position and linear transform")

    monkeypatch.setattr(registration_tool, "start_correction", missing)
    monkeypatch.setattr(handoff, "correction_fingerprint", missing)
    box = build_tools(state, ctx, spec)
    response = _tool(box, "trace_borders")("s0.png")
    assert response["error"] == "INVALID_LINEAR_PLACEMENT"
    assert state.slices[0].image_correction is None
    assert not box.job.undo_stack
    assert _tool(box, "submit")("Done", [], [])["error"] == "MISSING_DEFORMATIONS"


def test_host_inputs_preserve_complete_transform_and_do_not_alias_it(tmp_path):
    state, _, spec = _placed(tmp_path)
    supplied = {"params": [1, 0, 0, 0, 1, 0], "physical": {"scale_x": .1}}
    spec.inputs = {"transforms": {"s0.png": supplied}}
    apply_host_inputs(state, spec)
    assert state.slices[0].transform == supplied
    supplied["physical"]["scale_x"] = .9
    assert state.slices[0].transform["physical"]["scale_x"] == .1
    spec.inputs = {"transforms": {"missing.png": supplied}}
    with pytest.raises(ValueError, match="unknown section"):
        apply_host_inputs(state, spec)


def test_nonlinear_only_prompt_describes_fixed_supplied_placement(tmp_path):
    state, ctx, spec = _placed(tmp_path)
    prompt = build_job_statement(
        spec, state, tool_names=build_tools(state, ctx, spec).names,
        species="mouse", pos_lo=0, pos_hi=10, axis_ends=("anterior", "posterior"),
    )
    prompt = " ".join(prompt.split())
    assert ("- Existing linear transforms are supplied and fixed for this run, "
            "orientation included.") in prompt
    assert "transforms are not part of this run" not in prompt

