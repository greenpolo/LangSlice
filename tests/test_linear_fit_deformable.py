"""`fit_deformable`: the deformable fit as a linear-agent tool (preview, apply, state)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.agent.prompt import build_job_statement
from langslice.core import deformation
from langslice.core.deformable import DeformableRecord
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.core.state import StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.media import package_result
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import load_checkpoint
from langslice.job.job import ingest
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section
from tests.linear_tool_helpers import tool_named as _tool

ID = "s0.png"
FAST = {"engine": "elastix"}


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    """Fits run in this process (no spawn pool) at the coarse detail level, to
    keep the tests quick (the tool itself always runs at standard)."""
    monkeypatch.setattr(deformation, "USE_PROCESS_POOL", False)
    monkeypatch.setattr(deformation, "DETAIL_LEVEL", "coarse")


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _setup(folder: Path, atlas: SyntheticAtlas, *, engine: str = "either",
           tasks: tuple[str, ...] = ("position", "transform", "nonlinear"),
           provider: str = "openai-oauth"):
    image, _ = render_section(atlas, SMOOTH_FIELD())
    image.save(folder / ID)
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none", tasks=list(tasks),
        inputs={"pixel_size_um": 25.0}, transform=TransformSpec(angles=True),
        nonlinear=NonlinearSpec(engine=engine, provider=provider),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    record = state.slices[0]
    record.position_mm = 0.1
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": 25.0, "source": "host"}}
    return state, ctx, spec, build_tools(state, ctx, spec)


def _media(result: dict[str, Any]) -> list[bytes]:
    # What the ADK agent receives: the door's JPEG parts.
    return [part.inline_data.data
            for part in package_result(result).get(TOOL_MEDIA_PARTS_KEY, [])]


def _apply(box: Any, **kwargs: Any) -> dict[str, Any]:
    result = _tool(box, "fit_deformable")([ID], **{**FAST, **kwargs})
    assert result["status"] == "ok", result
    return result


# --- gating and the engine option -------------------------------------------


def test_built_with_the_nonlinear_task_only(tmp_path: Path, atlas):
    *_, box = _setup(tmp_path, atlas, tasks=("position", "transform"))
    assert "fit_deformable" not in box.names
    *_, box = _setup(tmp_path, atlas)
    assert {"trace_borders", "grep_atlas", "fit_deformable"} <= set(box.names)


def test_engine_argument_exists_only_when_the_user_left_it_open(tmp_path: Path, atlas):
    from google.adk.tools import FunctionTool

    def parameters(box: Any) -> set[str]:
        declaration = FunctionTool(_tool(box, "fit_deformable"))._get_declaration().model_dump()
        schema = declaration["parameters"] or declaration["parameters_json_schema"]
        return set(schema["properties"])

    *_, open_box = _setup(tmp_path, atlas)
    assert {"slices", "include", "exclude", "start", "fit_section", "fit_atlas",
            "engine", "stiffness", "candidates", "view"} == parameters(open_box) - {
                "keep_linear"}
    # Detail and line softening are fixed, not arguments.
    assert not {"detail", "line_softening_um"} & parameters(open_box)
    *_, fixed = _setup(tmp_path, atlas, engine="elastix")
    assert "engine" not in parameters(fixed)
    # The traced recommendation names ANTs, so a run fixed to Elastix drops it.
    recommended = "traced_borders with the ANTs engine at medium\n                stiffness"
    assert recommended in (_tool(open_box, "fit_deformable").__doc__ or "")
    assert "recommended" not in (_tool(fixed, "fit_deformable").__doc__ or "")
    result = _tool(fixed, "fit_deformable")([ID])
    assert result["results"][0]["settings"]["engine"] == "elastix"
    refused = _tool(fixed, "fit_deformable")([ID], candidates=[{"engine": "ants"}, {}])
    assert refused["error"] == "UNKNOWN_ARGUMENTS"
    assert refused["problems"][0]["argument"] == "candidates[0]"
    assert "The user fixed the engine for this run." in refused["message"]
    with pytest.raises(ValueError, match="nonlinear.engine"):
        NonlinearSpec(engine="spline")
    assert JobSpec.from_dict(JobSpec(image_folder=".", nonlinear=NonlinearSpec(
        engine="ants")).to_dict()).nonlinear.engine == "ants"


def test_missing_ants_is_said_plainly(tmp_path: Path, atlas, monkeypatch):
    monkeypatch.setattr(deformation, "ants_available", lambda: False)
    *_, box = _setup(tmp_path, atlas)
    refused = _tool(box, "fit_deformable")([ID], engine="ants")
    assert refused["error"] == "UNAVAILABLE"
    assert "ANTs" in refused["message"] and "elastix" in refused["message"]
    # Left open, the default falls back to Elastix.
    result = _tool(box, "fit_deformable")([ID])
    assert result["results"][0]["settings"]["engine"] == "elastix"
    *_, fixed = _setup(tmp_path, atlas, engine="ants")
    assert _tool(fixed, "fit_deformable")([ID])["error"] == "UNAVAILABLE"


# --- preview, apply, cache, undo -------------------------------------------


def test_several_candidates_preview_and_write_nothing(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    before = json.dumps(state.to_dict(), sort_keys=True)
    result = _tool(box, "fit_deformable")(
        [ID], **FAST, candidates=[{"stiffness": "soft"}, {"stiffness": "firm"}],
    )
    assert result["status"] == "ok" and result["applied"] is False
    rows = result["results"]
    assert [row["candidate"] for row in rows] == [1, 2]
    assert [row["settings"]["stiffness"] for row in rows] == ["soft", "firm"]
    for row in rows:
        assert row["displacement_mm"]["max"] > 0 and "fold_fraction" in row
        assert row["engine_settings"]["working_um"] == 40.0
        assert len(row["image_indexes"]) == 1
    assert len(_media(result)) == 2
    assert json.dumps(state.to_dict(), sort_keys=True) == before
    assert not box.job.undo_stack
    assert not list((Path(ctx.job_folder) / "sections").glob("*/deformable"))


def test_one_setting_applies_and_undo_redo_restore_it(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    result = _apply(box, stiffness="soft")
    assert result["applied"] is True
    row = result["results"][0]
    assert row["written"] is True and row["steps"] == 1
    held = state.slices[0].deformation
    assert held is not None and held["steps"][0]["stiffness"] == "soft"
    assert not Path(held["record"]).is_absolute()  # relative to the job folder
    record_dir = Path(ctx.job_folder) / held["record"]
    assert record_dir.parent == Path(ctx.job_folder) / "sections" / Path(ID).stem / "deformable"
    saved = json.loads((record_dir / "record.json").read_text())
    assert saved["provenance"]["section_id"] == ID
    assert saved["provenance"]["linear_handoff"]["section_um_per_px"] > 0
    assert saved["placement"]["position_mm"] == pytest.approx(0.1)
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation == held
    # The state round-trips through JSON like everything else on it.
    assert StackState.from_dict(json.loads(json.dumps(state.to_dict()))).slices[0].deformation \
        == held
    assert _tool(box, "undo")()["status"] == "ok"
    assert state.slices[0].deformation is None
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation is None
    _tool(box, "redo")()
    assert state.slices[0].deformation == held
    # The same setting again only re-draws: no second undo step.
    depth = len(box.job.undo_stack)
    again = _apply(box, stiffness="soft")["results"][0]
    assert again["written"] is False and again["cached"] is True
    assert len(box.job.undo_stack) == depth


def test_applying_a_previewed_candidate_reuses_its_result(tmp_path: Path, atlas, monkeypatch):
    state, _, _, box = _setup(tmp_path, atlas)
    _tool(box, "fit_deformable")([ID], **FAST, candidates=[{"stiffness": "soft"},
                                                         {"stiffness": "firm"}])

    def no_engine(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the engine ran again for identical inputs")

    monkeypatch.setattr(deformation, "run_engine", no_engine)
    row = _apply(box, stiffness="firm")["results"][0]
    assert row["cached"] is True and row["written"] is True
    assert state.slices[0].deformation["steps"][0]["stiffness"] == "firm"


def test_start_current_composes_onto_the_applied_deformation(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    refused = _tool(box, "fit_deformable")([ID], **FAST, start="current")
    assert refused["results"][0]["error"] == "NO_DEFORMATION"
    _apply(box)
    first = state.slices[0].deformation
    result = _apply(box, start="current", include=["STR"], view={"mode": "ab"})
    row = result["results"][0]
    assert row["steps"] == 2 and "step_displacement_mm" in row
    assert len(row["image_indexes"]) == 2  # the step, then what it started from
    held = state.slices[0].deformation
    assert [step["start"] for step in held["steps"]] == ["linear", "current"]
    assert held["steps"][1]["include"] == ["STR"]
    record = DeformableRecord.load(box.job.layout.resolve(held["record"]))
    assert record.step == 1 and record.parent is not None
    assert record.parent.settings.structures == ()
    assert record.settings.structures == ("STR",)
    _tool(box, "undo")()
    assert state.slices[0].deformation == first


def test_include_and_exclude_reach_the_engine(tmp_path: Path, atlas, monkeypatch):
    *_, box = _setup(tmp_path, atlas)
    seen: list[Any] = []
    real = deformation.prepare_fit

    def spy(image: Any, atlas_: Any, placement: Any, settings: Any, **kwargs: Any) -> Any:
        seen.append(settings)
        return real(image, atlas_, placement, settings, **kwargs)

    monkeypatch.setattr(deformation, "prepare_fit", spy)
    result = _tool(box, "fit_deformable")([ID], **FAST, include=["STR"], exclude=["HY:left"],
                                          candidates=[{}, {"stiffness": "firm"}])
    assert result["status"] == "ok", result
    assert [s.structures for s in seen] == [("STR",), ("STR",)]
    assert [s.exclude for s in seen] == [("HY:left",), ("HY:left",)]
    assert [s.atlas_image for s in seen] == ["ara", "ara"]
    assert all(s.detail == "coarse" for s in seen)
    fit = _tool(box, "fit_deformable")
    assert fit([ID], include=["NOPE"])["error"] == "UNKNOWN_REGIONS"
    assert fit([ID], include=["STR:up"])["error"] == "UNKNOWN_REGIONS"
    assert fit([ID], include=["STR"], exclude=["str"])["error"] == "BAD_ARGS"
    assert fit([ID], include=["STR"], exclude=["STR:right"])["error"] == "BAD_ARGS"
    assert fit([ID], include=["STR:left"], exclude=["STR:right"])["status"] == "ok"
    assert fit([ID], fit_atlas="nissl")["error"] in ("FIT_ATLAS_UNAVAILABLE",)
    assert fit([ID], candidates=[{}] * 5)["error"] == "TOO_MANY_CANDIDATES"


def test_excluded_regions_are_drawn_in_their_own_color(tmp_path: Path, atlas):
    *_, box = _setup(tmp_path, atlas)
    plain = _tool(box, "fit_deformable")([ID], **FAST, candidates=[{}, {"stiffness": "soft"}])
    marked = _tool(box, "fit_deformable")([ID], **FAST, exclude=["HY"],
                                          candidates=[{}, {"stiffness": "soft"}])

    def pinkish(data: bytes) -> int:
        import io

        pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)
        return int(((pixels[..., 0] > 200) & (pixels[..., 1] < 140)
                    & (pixels[..., 2] > 100)).sum())

    assert pinkish(_media(marked)[0]) > 20 and pinkish(_media(plain)[0]) == 0


# --- consequences: linear edits, placement pictures -------------------------


@pytest.mark.parametrize("edit", ["adjust", "position", "angles", "orient"])
def test_a_linear_edit_clears_the_warp_and_says_so(tmp_path: Path, atlas, edit):
    state, ctx, _, box = _setup(tmp_path, atlas)
    _apply(box)
    held = state.slices[0].deformation
    transform = dict(state.slices[0].transform)
    if edit == "adjust":
        reply = _tool(box, "adjust_transforms")([{
            "id": ID, "rotation_deg": 2.0, "scale_x": 1.0, "scale_y": 1.0,
            "translate_x_mm": 0.0, "translate_y_mm": 0.0}])
    elif edit == "position":
        reply = _tool(box, "set_positions")([{"id": ID, "position_mm": 0.15}])
    elif edit == "angles":
        reply = _tool(box, "set_cutting_angles")(1.0, 0.0)
    else:
        reply = _tool(box, "orient_slices")([{"id": ID, "flip": True}])
    assert reply["deformation_cleared"] == [ID]
    assert state.slices[0].deformation is None
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation is None
    _tool(box, "undo")()
    assert state.slices[0].deformation == held
    assert state.slices[0].transform == transform


def test_reads_and_unchanged_redraws_keep_the_warp(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    _apply(box)
    held = state.slices[0].deformation
    for reply in (_tool(box, "status")(), _tool(box, "note")("looked"),
                  _tool(box, "view_placement")([{"id": ID}])):
        assert "deformation_cleared" not in reply
    assert state.slices[0].deformation == held


def test_view_placement_draws_the_current_warp(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    view = _tool(box, "view_placement")
    before = view([{"id": ID}], view={"mode": "overlay"})
    assert "deformation_drawn" not in before["compared"][0]
    _apply(box, stiffness="soft")
    after = view([{"id": ID}], view={"mode": "overlay"})
    assert after["compared"][0]["deformation_drawn"] is True
    assert _media(after)[0] != _media(before)[0]
    # Another position than the fitted one shows no warp.
    other = view([{"id": ID, "positions_mm": [0.2]}], view={"mode": "overlay"})
    assert "deformation_drawn" not in other["compared"][0]
    assert state.slices[0].deformation is not None


def test_view_placement_draws_the_warp_in_every_mode_that_draws_the_section(
    tmp_path: Path, atlas,
):
    _, _, _, box = _setup(tmp_path, atlas)
    view = _tool(box, "view_placement")
    _apply(box, stiffness="soft")
    for mode in ("overlay", "checkerboard", "outlines", "section"):
        warped = view([{"id": ID}], view={"mode": mode})
        linear = view([{"id": ID}], view={"mode": mode, "deformation": "none"})
        assert warped["compared"][0]["deformation_drawn"] is True, mode
        assert "deformation_drawn" not in linear["compared"][0], mode
        assert warped["view"]["deformation"] == "applied"
        assert _media(warped)[0] != _media(linear)[0], mode
    for mode in ("template", "stacked", "side_by_side"):
        assert "deformation_drawn" not in view([{"id": ID}], view={"mode": mode})["compared"][0]
    # Built for a run without positioning, too: the read-only registration view.
    (tmp_path / "only").mkdir()
    *_, nonlinear_only = _setup(tmp_path / "only", atlas, tasks=("nonlinear",))
    assert "view_placement" in nonlinear_only.names
    assert "set_positions" not in nonlinear_only.names


def test_view_placement_highlights_one_side_of_a_region(tmp_path: Path, atlas):
    *_, box = _setup(tmp_path, atlas)
    view = _tool(box, "view_placement")
    style = {"mode": "overlay", "atlas_channels": [], "border_color": "#ff00ff",
             "border_thickness": 3.0}
    both = view([{"id": ID}], view={"regions": ["CTX"], **style})
    left = view([{"id": ID}], view={"regions": ["CTX:left"], **style})
    right = view([{"id": ID}], view={"regions": ["CTX:right"], **style})
    assert left["view"]["regions"] == ["CTX:left"]
    images = [np.asarray(Image.open(__import__("io").BytesIO(_media(r)[0])).convert("RGB"))
              for r in (both, left, right)]

    def magenta_columns(pixels: np.ndarray) -> np.ndarray:
        mask = (pixels[..., 0] > 180) & (pixels[..., 1] < 90) & (pixels[..., 2] > 180)
        return np.nonzero(mask[40:].any(axis=0))[0]  # below the caption

    whole, only_left, only_right = (magenta_columns(pixels) for pixels in images)
    middle = (whole.min() + whole.max()) / 2.0
    assert only_left.max() <= middle + 3 and only_left.min() <= whole.min() + 3
    assert only_right.min() >= middle - 3 and only_right.max() >= whole.max() - 3
    refused = view([{"id": ID}], view={"mode": "overlay", "regions": ["CTX:up"]})
    assert refused["error"] == "UNKNOWN_REGIONS"


def test_the_job_statement_names_the_engine_and_the_tool(tmp_path: Path, atlas, monkeypatch):
    state, _, spec, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(deformation, "ants_available", lambda: True)
    text = build_job_statement(spec, state, tool_names=box.names, species="mouse",
                               pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"),
                               atlas_channels=("ara", "borders"))
    assert "`fit_deformable`:" in text
    assert "engine: ants or elastix, your choice per call" in text
    assert "Atlas channels on this host: ara (" in text and "; borders (" in text
    spec.nonlinear.engine = "elastix"
    text = build_job_statement(spec, state, tool_names=box.names, species="mouse",
                               pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"))
    assert "engine: elastix, set by the user" in text


# --- traced section images --------------------------------------------------


def _fake_trace(state: StackState, ctx: Any, folder: Path, *, fingerprint: str) -> None:
    """A completed trace_borders result whose lines are the placed atlas borders."""
    state.slices[0].image_correction = _trace_artifacts(state, ctx, folder,
                                                        fingerprint=fingerprint)


def _trace_artifacts(state: StackState, ctx: Any, folder: Path, *,
                     fingerprint: str) -> dict[str, Any]:
    """Write a trace's artifacts (the placed atlas borders as lines); return its result."""
    grid = deformation.fit_grid(state, ctx, state.slices[0])
    canvas_size = (grid.image.width // 2, grid.image.height // 2)
    scale = np.diag([0.5, 0.5, 1.0])
    atlas_to_canvas = scale @ grid.placement.atlas_to_section
    labels = ctx.atlas.annotation[0].astype(np.float32)
    placed = cv2.warpAffine(labels, atlas_to_canvas[:2], canvas_size, flags=cv2.INTER_NEAREST)
    edges = np.zeros(placed.shape, dtype=np.uint8)
    edges[:, 1:] |= (placed[:, 1:] != placed[:, :-1]).astype(np.uint8)
    edges[1:, :] |= (placed[1:, :] != placed[:-1, :]).astype(np.uint8)
    folder.mkdir(parents=True)
    Image.fromarray(edges * 255).save(folder / "extracted_lines.png")
    (folder / "request.json").write_text(json.dumps({"atlas_to_canvas": atlas_to_canvas.tolist()}))
    return {"status": "ok", "geometry_fingerprint": fingerprint, "artifact_dir": str(folder)}


def test_traced_images_need_a_completed_trace_at_this_placement(tmp_path: Path, atlas,
                                                                monkeypatch):
    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
    fit = _tool(box, "fit_deformable")
    missing = fit([ID], **FAST, fit_section="traced_lines")
    assert missing["results"][0]["error"] == "NO_TRACE"
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    _fake_trace(state, ctx, tmp_path / "trace", fingerprint="earlier")
    assert fit([ID], **FAST, fit_section="traced_lines")["results"][0]["error"] \
        == "TRACE_STALE"
    state.slices[0].image_correction["geometry_fingerprint"] = "now"
    result = fit([ID], **FAST, fit_section="traced_lines")
    assert result["status"] == "ok", result
    row = result["results"][0]
    assert row["settings"]["fit_atlas"] == "borders"  # traced lines fit against borders
    assert row["engine_settings"]["metric"] == "mean_squares"
    assert fit([ID], **FAST, fit_section="traced_borders")["error"] == "LABEL_MAP_ANTS_ONLY"
    assert fit([ID], fit_section="traced_lines", fit_atlas="ara")["error"] == "BAD_ARGS"
    # The stain against atlas borders is refused: borders are for traced lines.
    stain_borders = fit([ID], **FAST, fit_atlas="borders")
    assert stain_borders["error"] == "BAD_ARGS" and "traced" in stain_borders["message"]
    unknown = fit([ID], **FAST, fit_section="nope")
    assert unknown["error"] == "BAD_FIT_SECTION"
    assert unknown["fit_sections"] == ["fit", "traced_borders", "traced_lines"]


def test_a_section_without_a_linear_placement_is_refused(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    state.slices[0].transform = None
    result = _tool(box, "fit_deformable")([ID], **FAST)
    assert result["status"] == "error"
    assert result["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT"
    assert state.slices[0].deformation is None


# --- the nonlinear goal: a deformation (or keep_linear) per section ------------


def _statement(spec: JobSpec, state: StackState, box: Any) -> str:
    return build_job_statement(spec, state, tool_names=box.names, species="mouse",
                               pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"))


def test_the_job_is_a_deformation_per_section_in_both_modes(tmp_path: Path, atlas):
    state, _, spec, box = _setup(tmp_path, atlas)
    text = _statement(spec, state, box)
    assert ("give every section a deformation onto the atlas, on top of its linear "
            "placement, with the section's stain and the borders the image model traces "
            "on it as the evidence.") in text
    assert "use the image model to correct" not in text
    assert "or a `keep_linear` reason saying its linear placement stands" in text
    assert "`trace_borders`:" in text and "completed image correction" in text
    assert "a call waits for a trace that is still running" in text
    assert "Base image-model prompt" in text
    assert ("inspect each returned fit's borders against the section's internal anatomy and its "
            "traced borders") in text

    (tmp_path / "none").mkdir()
    none_state, _, none_spec, none_box = _setup(tmp_path / "none", atlas, provider="none")
    text = _statement(none_spec, none_state, none_box)
    assert ("give every section a deformation onto the atlas, on top of its linear "
            "placement, with the section's stain as the evidence.") in text
    assert "or a `keep_linear` reason saying its linear placement stands" in text
    for absent in ("trace", "image correction", "image model", "Base image-model prompt"):
        assert absent not in text, absent
    assert "reads, `fit_section` (the fit appearance)" in text
    assert "inspect each returned fit's borders against the section's internal anatomy. " in text
    assert "exclude the regions it has lost rather than restricting the fit" in text


def test_without_an_image_model_there_are_no_traces(tmp_path: Path, atlas, monkeypatch):
    from langslice.core import handoff

    _, _, spec, box = _setup(tmp_path, atlas, provider="none")
    assert spec.nonlinear.uses_image_model is False
    assert "trace_borders" not in box.names
    assert {"grep_atlas", "fit_deformable"} <= set(box.names)
    doc = _tool(box, "fit_deformable").__doc__ or ""
    assert "traced" not in doc and "trace_borders" not in doc and "keep_linear" in doc
    refused = _tool(box, "fit_deformable")([ID], fit_section="traced_lines")
    assert refused["error"] == "NO_IMAGE_MODEL"

    def no_trace_check(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("submit looked for an image correction")

    monkeypatch.setattr(handoff, "correction_fingerprint", no_trace_check)
    submit = _tool(box, "submit")
    first = submit("Done", [], [])
    assert first["error"] == "MISSING_DEFORMATIONS"
    assert first["sections"] == [{"id": ID,
                                  "reason": "no deformation and no keep_linear reason"}]
    _apply(box)
    assert submit("Done", [], [])["status"] == "ok"
    with pytest.raises(ValueError, match="nonlinear.provider"):
        NonlinearSpec(provider="mystery")
    assert JobSpec.from_dict(spec.to_dict()).nonlinear.provider == "none"


def test_with_the_image_model_submit_wants_traces_then_deformations(tmp_path: Path, atlas,
                                                                     monkeypatch):
    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    submit = _tool(box, "submit")
    assert submit("Done", [], [])["error"] == "MISSING_IMAGE_CORRECTIONS"
    _fake_trace(state, ctx, tmp_path / "trace", fingerprint="now")
    assert submit("Done", [], [])["error"] == "MISSING_DEFORMATIONS"
    _tool(box, "fit_deformable")([ID], keep_linear="Matches already.")
    assert submit("Done", [], [])["status"] == "ok"


def test_keep_linear_satisfies_submit_until_the_placement_moves(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas, provider="none")
    fit = _tool(box, "fit_deformable")
    reply = fit([ID], keep_linear="The linear placement already matches.")
    assert reply["status"] == "ok" and reply["applied"] is True
    assert reply["changed"][0]["keep_linear"] == "The linear placement already matches."
    held = state.slices[0].deformation
    assert held["keep_linear"] == "The linear placement already matches."
    # Fit settings with keep_linear are refused, not dropped.
    refused = fit([ID], keep_linear="x", exclude=["CTX"], stiffness="firm")
    assert refused["error"] == "BAD_ARGS" and refused["given"] == ["exclude", "stiffness"]
    assert state.slices[0].deformation == held
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation == held
    assert fit([ID], **FAST, start="current")["results"][0]["error"] == "NO_DEFORMATION"
    # A placement change clears it, like an applied fit.
    moved = _tool(box, "adjust_transforms")([{
        "id": ID, "rotation_deg": 2.0, "scale_x": 1.0, "scale_y": 1.0,
        "translate_x_mm": 0.0, "translate_y_mm": 0.0}])
    assert moved["deformation_cleared"] == [ID]
    assert _tool(box, "submit")("Done", [], [])["error"] == "MISSING_DEFORMATIONS"
    _tool(box, "undo")()
    assert state.slices[0].deformation == held
    _tool(box, "undo")()
    assert state.slices[0].deformation is None
    state.slices[0].transform = None
    assert fit([ID], keep_linear="x")["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT"


def test_a_traced_image_waits_for_its_running_trace_and_shows_it(tmp_path: Path, atlas,
                                                                 monkeypatch):
    import io
    import time

    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    folder = tmp_path / "trace"
    state.slices[0].image_correction = {"status": "running", "geometry_fingerprint": "now"}

    def job() -> dict[str, Any]:
        time.sleep(0.3)
        return _trace_artifacts(state, ctx, folder, fingerprint="now")

    box.job.start_image_job(ID, "now", job, workers=1)
    result = _tool(box, "fit_deformable")([ID], **FAST, fit_section="traced_lines")
    assert result["status"] == "ok", result
    assert result["results"][0]["status"] == "ok"
    assert state.slices[0].image_correction["status"] == "ok"
    assert load_checkpoint(ctx.checkpoint_path).slices[0].image_correction["status"] == "ok"
    assert ID not in box.job.image_jobs
    media = _media(result)
    assert len(media) == 2 and result["traces"] == [{"id": ID, "image_indexes": [1]}]
    assert "traced lines" in result["description"]
    # The trace picture carries the lines in the border color on the section.
    pixels = np.asarray(Image.open(io.BytesIO(media[1])).convert("RGB")).astype(int)
    assert int(((pixels[..., 0] > 200) & (pixels[..., 1] > 200)
                & (pixels[..., 2] < 80)).sum()) > 50


def test_a_trace_still_running_after_the_wait_is_reported(tmp_path: Path, atlas, monkeypatch):
    import threading

    from langslice.core import handoff

    state, _, _, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    monkeypatch.setattr(deformation, "TRACE_WAIT_S", 0.2)
    state.slices[0].image_correction = {"status": "running", "geometry_fingerprint": "now"}
    release = threading.Event()

    def job() -> dict[str, Any]:
        # Held until released, even on a slow CI runner.
        release.wait()
        return {"status": "error", "error": "TransportError", "message": "no image",
                "geometry_fingerprint": "now"}

    box.job.start_image_job(ID, "now", job, workers=1)
    fit = _tool(box, "fit_deformable")
    try:
        late = fit([ID], **FAST, fit_section="traced_lines")["results"][0]
        assert late["error"] == "TRACE_TIMEOUT" and "0.2 s" in late["message"]
    finally:
        release.set()
    failed = fit([ID], **FAST, fit_section="traced_lines")["results"][0]
    assert failed["error"] == "TRACE_FAILED" and "no image" in failed["message"]
