"""The optional image-model tool stores annotations without changing placement."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.linear.checkpoint import load_checkpoint
from langslice.linear.engine import apply_host_inputs
from langslice.linear.prompt import build_job_statement
from langslice.linear.spec import DEFAULT_TASKS, JobSpec, NonlinearSpec
from langslice.linear.state import StackState
from langslice.linear.toolbox import _tool_target_ids, build_tools
from tests.test_linear_toolbox import _stack, _tool


def _placed(tmp_path: Path):
    state, ctx, spec = _stack(tmp_path, n=1, placed=True, tasks=["nonlinear"])
    state.slices[0].transform = {
        "kind": "interactive", "params": [1, 0, .1, 0, 1, .2],
        "calibration": {"section_um_per_px": 25, "source": "host"},
    }
    return state, ctx, spec


def test_image_tool_is_opt_in_and_has_only_id_and_notes(tmp_path: Path):
    from google.adk.tools import FunctionTool

    state, ctx, spec = _stack(tmp_path, n=1)
    assert spec.tasks == list(DEFAULT_TASKS)
    assert "correct_slice_borders" not in build_tools(state, ctx, spec).names
    spec.tasks = ["nonlinear"]
    box = build_tools(state, ctx, spec)
    schema = FunctionTool(_tool(box, "correct_slice_borders"))._get_declaration()
    declaration = schema.model_dump()
    parameters = declaration["parameters"] or declaration["parameters_json_schema"]
    assert set(parameters["properties"]) == {"id", "additional_notes"}
    assert not {"fit_affine", "adjust_transforms", "search_atlas", "reject"} & set(box.names)
    restored = JobSpec.from_dict(spec.to_dict())
    assert restored.nonlinear == NonlinearSpec()


def test_image_correction_stores_notes_images_and_undo_without_changing_transform(
    tmp_path: Path, monkeypatch,
):
    from langslice import registration_tool

    state, ctx, spec = _placed(tmp_path)
    transform = copy.deepcopy(state.slices[0].transform)
    spec.nonlinear = NonlinearSpec(provider="openai-api", image_model="test-model")
    raw = tmp_path / "raw.png"
    lines = tmp_path / "lines.png"
    Image.new("RGB", (40, 30), "blue").save(raw)
    Image.new("RGB", (40, 30), "yellow").save(lines)
    calls = []

    def fake_correct(actual_state, actual_ctx, section_id, **kwargs):
        assert actual_state is state and actual_ctx is ctx
        calls.append((section_id, kwargs))
        return {
            "status": "ok", "geometry_fingerprint": "geometry",
            "additional_notes": kwargs["additional_notes"],
            "artifact_paths": {"raw_reply": str(raw), "lines_on_original": str(lines)},
        }

    monkeypatch.setattr(registration_tool, "correct_slice", fake_correct)
    monkeypatch.setattr(registration_tool, "correction_fingerprint", lambda *_: "geometry")
    box = build_tools(state, ctx, spec)
    result = _tool(box, "correct_slice_borders")("0", "Left fragment is displaced.")
    assert calls == [("s0.png", {
        "additional_notes": "Left fragment is displaced.",
        "out": Path(ctx.results_path).parent / "nonlinear",
        "provider": "openai-api", "image_model": "test-model",
    })]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    assert result["image_indexes"] == {"raw_reply": 0, "lines_on_original": 1}
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved.slices[0].image_correction == state.slices[0].image_correction
    restored = StackState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.slices[0].image_correction == state.slices[0].image_correction
    assert state.slices[0].transform == transform
    _tool(box, "undo")()
    assert state.slices[0].image_correction is None
    assert state.slices[0].transform == transform
    _tool(box, "redo")()
    assert state.slices[0].image_correction["status"] == "ok"
    assert _tool(box, "submit")("Done", [], [])["status"] == "ok"
    assert _tool_target_ids(state, "correct_slice_borders", {"id": "0"}) == ["s0.png"]


@pytest.mark.parametrize("result", [None, {"status": "error"}, {
    "status": "ok", "geometry_fingerprint": "previous-placement",
}])
def test_submit_requires_completed_correction_at_current_geometry(tmp_path, monkeypatch, result):
    from langslice import registration_tool

    state, ctx, spec = _placed(tmp_path)
    state.slices[0].image_correction = result
    monkeypatch.setattr(registration_tool, "correction_fingerprint", lambda *_: "current")
    box = build_tools(state, ctx, spec)
    response = _tool(box, "submit")("Done", [], [])
    assert response["error"] == "MISSING_IMAGE_CORRECTIONS"
    assert response["sections"][0]["id"] == "s0.png"
    assert not state.submitted


def test_image_tool_reports_missing_placement_without_checkpoint_mutation(tmp_path, monkeypatch):
    from langslice import registration_tool

    state, ctx, spec = _stack(tmp_path, n=1, tasks=["nonlinear"])

    def missing(*args, **kwargs):
        raise ValueError("s0.png requires a position and linear transform")

    monkeypatch.setattr(registration_tool, "correct_slice", missing)
    monkeypatch.setattr(registration_tool, "correction_fingerprint", missing)
    box = build_tools(state, ctx, spec)
    response = _tool(box, "correct_slice_borders")("s0.png")
    assert response["error"] == "INVALID_LINEAR_PLACEMENT"
    assert state.slices[0].image_correction is None
    assert not box.undo_stack
    refusal = _tool(box, "submit")("Done", [], [])
    assert "requires a position" in refusal["sections"][0]["reason"]


def test_host_inputs_preserve_complete_transform_and_do_not_alias_it(tmp_path):
    state, _, spec = _placed(tmp_path)
    supplied = {"params": [1, 0, 0, 0, 1, 0], "spline": {"source": [[.1, .2]]}}
    spec.inputs = {"transforms": {"s0.png": supplied}}
    apply_host_inputs(state, spec)
    assert state.slices[0].transform == supplied
    supplied["spline"]["source"][0][0] = .9
    assert state.slices[0].transform["spline"]["source"][0][0] == .1
    spec.inputs = {"transforms": {"missing.png": supplied}}
    with pytest.raises(ValueError, match="unknown section"):
        apply_host_inputs(state, spec)


def test_nonlinear_only_prompt_describes_fixed_supplied_placement(tmp_path):
    state, ctx, spec = _placed(tmp_path)
    prompt = build_job_statement(
        spec, state, tool_names=build_tools(state, ctx, spec).names,
        species="mouse", pos_lo=0, pos_hi=10, axis_ends=("anterior", "posterior"),
    )
    assert "Existing linear transforms are supplied and fixed" in prompt
    assert "additional_notes" in prompt
    assert "no replacement prompt or anatomical rejection step" in prompt
    assert "Transforms are not part" not in prompt
