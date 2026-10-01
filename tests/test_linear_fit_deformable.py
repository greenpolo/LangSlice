"""`fit_deformable`: the deformable fit as a linear-agent tool (preview, apply, state)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.deformable import DeformableRecord
from langslice.linear import deformation
from langslice.linear.checkpoint import load_checkpoint
from langslice.linear.engine import build_context, ingest
from langslice.linear.prompt import build_job_statement
from langslice.linear.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.linear.state import StackState
from langslice.linear.toolbox import build_tools
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section

ID = "s0.png"
FAST = {"engine": "elastix", "detail": "coarse"}


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    """Fits run in this process (no spawn pool) to keep the tests quick."""
    monkeypatch.setattr(deformation, "USE_PROCESS_POOL", False)


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _setup(folder: Path, atlas: SyntheticAtlas, *, engine: str = "either",
           tasks: tuple[str, ...] = ("position", "transform", "nonlinear")):
    image, _ = render_section(atlas, SMOOTH_FIELD())
    image.save(folder / ID)
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none", tasks=list(tasks),
        inputs={"pixel_size_um": 25.0}, transform=TransformSpec(angles=True),
        nonlinear=NonlinearSpec(engine=engine),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    record = state.slices[0]
    record.position_mm = 0.1
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": 25.0, "source": "host"}}
    return state, ctx, spec, build_tools(state, ctx, spec)


def _tool(box: Any, name: str) -> Any:
    return next(tool for tool in box.tools if tool.__name__ == name)


def _media(result: dict[str, Any]) -> list[bytes]:
    return [part.inline_data.data for part in result.get(TOOL_MEDIA_PARTS_KEY, [])]


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
    assert {"sections", "include", "exclude", "start", "section_image", "atlas_image",
            "engine", "stiffness", "detail", "candidates", "mode", "zoom"} <= parameters(open_box)
    *_, fixed = _setup(tmp_path, atlas, engine="elastix")
    assert "engine" not in parameters(fixed)
    result = _tool(fixed, "fit_deformable")([ID], detail="coarse")
    assert result["results"][0]["settings"]["engine"] == "elastix"
    refused = _tool(fixed, "fit_deformable")([ID], candidates=[{"engine": "ants"}, {}])
    assert refused["error"] == "BAD_CANDIDATE"
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
    result = _tool(box, "fit_deformable")([ID], detail="coarse")
    assert result["results"][0]["settings"]["engine"] == "elastix"
    *_, fixed = _setup(tmp_path, atlas, engine="ants")
    assert _tool(fixed, "fit_deformable")([ID])["error"] == "UNAVAILABLE"


# --- preview, apply, cache, undo -------------------------------------------


def test_several_candidates_preview_and_write_nothing(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    before = json.dumps(state.to_dict(), sort_keys=True)
    result = _tool(box, "fit_deformable")(
        [ID], **FAST, candidates=[{"stiffness": "soft"}, {"stiffness": "stiff"}],
    )
    assert result["status"] == "ok" and result["applied"] is False
    rows = result["results"]
    assert [row["candidate"] for row in rows] == [1, 2]
    assert [row["settings"]["stiffness"] for row in rows] == ["soft", "stiff"]
    for row in rows:
        assert row["displacement_mm"]["max"] > 0 and "fold_fraction" in row
        assert row["engine_settings"]["working_um"] == 40.0
        assert len(row["image_indexes"]) == 1
    assert len(_media(result)) == 2
    assert json.dumps(state.to_dict(), sort_keys=True) == before
    assert not box.undo_stack
    assert not (Path(ctx.results_path).parent / "deformable").exists()


def test_one_setting_applies_and_undo_redo_restore_it(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    result = _apply(box, stiffness="soft")
    assert result["applied"] is True
    row = result["results"][0]
    assert row["written"] is True and row["steps"] == 1
    held = state.slices[0].deformation
    assert held is not None and held["steps"][0]["stiffness"] == "soft"
    record_dir = Path(held["record"])
    assert record_dir.parent.parent == Path(ctx.results_path).parent / "deformable"
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
    depth = len(box.undo_stack)
    again = _apply(box, stiffness="soft")["results"][0]
    assert again["written"] is False and again["cached"] is True
    assert len(box.undo_stack) == depth


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
    result = _apply(box, start="current", include=["STR"], mode="ab")
    row = result["results"][0]
    assert row["steps"] == 2 and "step_displacement_mm" in row
    assert len(row["image_indexes"]) == 2  # the step, then what it started from
    held = state.slices[0].deformation
    assert [step["start"] for step in held["steps"]] == ["linear", "current"]
    assert held["steps"][1]["include"] == ["STR"]
    record = DeformableRecord.load(held["record"])
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
    result = _tool(box, "fit_deformable")([ID], **FAST, include=["STR"], exclude=["HY"],
                                          candidates=[{}, {"atlas_image": "borders"}])
    assert result["status"] == "ok", result
    assert [s.structures for s in seen] == [("STR",), ("STR",)]
    assert [s.exclude for s in seen] == [("HY",), ("HY",)]
    assert [s.atlas_image for s in seen] == ["ara", "borders_merged"]
    fit = _tool(box, "fit_deformable")
    assert fit([ID], include=["NOPE"])["error"] == "UNKNOWN_REGIONS"
    assert fit([ID], include=["STR"], exclude=["str"])["error"] == "BAD_ARGS"
    assert fit([ID], atlas_image="nissl")["error"] in ("ATLAS_IMAGE_UNAVAILABLE",)
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
    before = view([{"id": ID}], mode="overlay")
    assert "deformation_drawn" not in before["compared"][0]
    _apply(box, stiffness="soft")
    after = view([{"id": ID}], mode="overlay")
    assert after["compared"][0]["deformation_drawn"] is True
    assert _media(after)[0] != _media(before)[0]
    # Another position than the fitted one shows no warp.
    other = view([{"id": ID, "positions_mm": [0.2]}], mode="overlay")
    assert "deformation_drawn" not in other["compared"][0]
    assert state.slices[0].deformation is not None


def test_the_job_statement_names_the_engine_and_the_tool(tmp_path: Path, atlas, monkeypatch):
    state, _, spec, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(deformation, "ants_available", lambda: True)
    text = build_job_statement(spec, state, tool_names=box.names, species="mouse",
                               pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"),
                               atlas_images=("ara", "borders"))
    assert "`fit_deformable`:" in text
    assert "engine: ants or elastix, your choice per call" in text
    assert "Atlas images on this host: ara, borders" in text
    spec.nonlinear.engine = "elastix"
    text = build_job_statement(spec, state, tool_names=box.names, species="mouse",
                               pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"))
    assert "engine: elastix, set by the user" in text


# --- traced section images --------------------------------------------------


def _fake_trace(state: StackState, ctx: Any, folder: Path, *, fingerprint: str) -> None:
    """A completed trace_borders result whose lines are the placed atlas borders."""
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
    state.slices[0].image_correction = {
        "status": "ok", "geometry_fingerprint": fingerprint, "artifact_dir": str(folder),
    }


def test_traced_images_need_a_completed_trace_at_this_placement(tmp_path: Path, atlas,
                                                                monkeypatch):
    from langslice import registration_tool

    state, ctx, _, box = _setup(tmp_path, atlas)
    fit = _tool(box, "fit_deformable")
    missing = fit([ID], **FAST, section_image="traced_lines")
    assert missing["results"][0]["error"] == "NO_TRACE"
    monkeypatch.setattr(registration_tool, "correction_fingerprint", lambda *_: "now")
    _fake_trace(state, ctx, tmp_path / "trace", fingerprint="earlier")
    assert fit([ID], **FAST, section_image="traced_lines")["results"][0]["error"] \
        == "TRACE_STALE"
    state.slices[0].image_correction["geometry_fingerprint"] = "now"
    result = fit([ID], **FAST, section_image="traced_lines")
    assert result["status"] == "ok", result
    row = result["results"][0]
    assert row["settings"]["atlas_image"] == "borders"  # traced lines fit against borders
    assert row["engine_settings"]["metric"] == "mean_squares"
    assert fit([ID], **FAST, section_image="traced_borders")["error"] == "LABEL_MAP_ANTS_ONLY"
    assert fit([ID], section_image="traced_lines", atlas_image="ara")["error"] == "BAD_ARGS"
    unknown = fit([ID], **FAST, section_image="nope")
    assert unknown["results"][0]["error"] == "UNKNOWN_CHANNEL"


def test_a_section_without_a_linear_placement_is_refused(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    state.slices[0].transform = None
    result = _tool(box, "fit_deformable")([ID], **FAST)
    assert result["status"] == "error"
    assert result["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT"
    assert state.slices[0].deformation is None
