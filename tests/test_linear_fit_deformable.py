"""The deformation on top of a section's linear placement: the ``ants_syn`` tool,
``submit``'s ``left_linear``, and the fit underneath (``ops.deformable``:
``fit_deformable``, which ``ants_syn`` and the packaged ``trace_borders``
fit with)."""

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
from langslice.core.display import default_options
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.core.state import StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import load_checkpoint
from langslice.job.job import ingest
from langslice.ops import deformable as ops_deformable
from tests.deformable_synthetic import HY, SMOOTH_FIELD, TH, SyntheticAtlas, render_section
from tests.linear_tool_helpers import tool_named as _tool

ID = "s0.png"

needs_ants = pytest.mark.skipif(not ops_deformable.ants_ready(),
                                reason="ants_syn needs antspyx")


def _choice(fit_section: str = deformation.FIT_LOOK, fit_atlas: str = "template",
            stiffness: str = "medium") -> deformation.Choice:
    """One fit setting for ``ops.deformable.fit_deformable`` (ANTs SyN)."""
    return deformation.Choice(fit_section=fit_section, fit_atlas=fit_atlas, engine="ants",
                              stiffness=stiffness)


TRACED = _choice(fit_section=deformation.TRACED_LINES, fit_atlas="borders")


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    """Fits run in this process (no spawn pool) at the coarse detail level, to
    keep the tests quick (the tool itself always runs at standard)."""
    monkeypatch.setattr(deformation, "USE_PROCESS_POOL", False)
    monkeypatch.setattr(deformation, "DETAIL_LEVEL", "coarse")


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _setup(folder: Path, atlas: SyntheticAtlas, *,
           tasks: tuple[str, ...] = ("position", "transform", "nonlinear"),
           provider: str = "openai-oauth", **nonlinear: Any):
    image, _ = render_section(atlas, SMOOTH_FIELD())
    image.save(folder / ID)
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none", tasks=list(tasks),
        inputs={"pixel_size_um": 25.0}, transform=TransformSpec(angles=True),
        nonlinear=NonlinearSpec(provider=provider, **nonlinear),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    record = state.slices[0]
    record.position_mm = 0.1
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": 25.0, "source": "host"}}
    return state, ctx, spec, build_tools(state, ctx, spec)


def _pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(int)


def _ants(box: Any, **kwargs: Any) -> dict[str, Any]:
    result = _tool(box, "ants_syn")([ID], **kwargs)
    assert result["status"] == "ok", result
    return result


def _fit(box: Any, ctx: Any, choice: deformation.Choice | None = None,
         **kwargs: Any) -> ops_deformable.DeformableFit:
    """``ops.deformable.fit_deformable`` on the one section."""
    return ops_deformable.fit_deformable(box.job, ctx, [box.job.state.slices[0]],
                                         choice or _choice(), **kwargs)


# --- gating and the tool's arguments -----------------------------------------


def test_built_with_the_nonlinear_task_only(tmp_path: Path, atlas):
    *_, box = _setup(tmp_path, atlas, tasks=("position", "transform"))
    assert not {"ants_syn", "trace_borders"} & set(box.names)
    *_, box = _setup(tmp_path, atlas)
    assert {"trace_borders", "grep_atlas", "ants_syn"} <= set(box.names)
    assert not {"fit_deformable", "keep_linear"} & set(box.names)


def test_ants_syn_takes_sections_regions_atlas_image_and_stiffness(tmp_path: Path, atlas):
    from google.adk.tools import FunctionTool

    *_, box = _setup(tmp_path, atlas)
    declaration = FunctionTool(_tool(box, "ants_syn"))._get_declaration().model_dump()
    schema = declaration["parameters"] or declaration["parameters_json_schema"]
    assert set(schema["properties"]) == {"sections", "restrict_to", "atlas_image",
                                         "stiffness", "view"}
    # A job saved with the older engine setting still opens: it is ignored.
    saved = JobSpec(image_folder=".").to_dict()
    saved["nonlinear"]["engine"] = "elastix"
    assert JobSpec.from_dict(saved) == JobSpec(image_folder=".")


def test_missing_ants_is_said_plainly(tmp_path: Path, atlas, monkeypatch):
    monkeypatch.setattr(ops_deformable, "ants_ready", lambda: False)
    state, *_, box = _setup(tmp_path, atlas)
    refused = _tool(box, "ants_syn")([ID])
    assert refused["error"] == "ANTS_MISSING"
    assert "antspyx" in refused["message"]
    assert state.slices[0].deformation is None and not box.job.undo_stack


@needs_ants
def test_bad_arguments_are_refused_before_any_fit(tmp_path: Path, atlas, monkeypatch):
    *_, box = _setup(tmp_path, atlas)

    def no_engine(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a refused call ran the engine")

    monkeypatch.setattr(deformation, "run_jobs", no_engine)
    fit = _tool(box, "ants_syn")
    assert fit([])["error"] == "BAD_ARGS"
    assert fit([ID] * 5)["error"] == "BAD_ARGS"
    assert fit([ID], stiffness="jelly")["error"] == "BAD_ARGS"
    assert fit([ID], atlas_image="borders")["error"] == "BAD_ARGS"
    assert fit([ID], atlas_image="nissl")["error"] == "FIT_ATLAS_UNAVAILABLE"
    assert fit([ID], restrict_to=["NOPE"])["error"] == "UNKNOWN_REGIONS"
    assert fit([ID], restrict_to=["STR:up"])["error"] == "BAD_ARGS"  # no such side
    assert fit(["nope.png"])["error"] == "UNKNOWN_SLICE_IDS"
    assert not box.job.undo_stack


# --- apply, compose, undo ------------------------------------------------------


@needs_ants
def test_one_fit_applies_and_undo_redo_restore_it(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    result = _ants(box, stiffness="soft", view=False)
    row = result["results"][0]
    assert row["written"] is True and row["steps"] == 1
    assert row["settings"] == {"engine": "ants", "stiffness": "soft", "fit_section": "fit",
                               "fit_atlas": "template"}
    assert row["displacement_mm"]["max"] > 0 and "fold_fraction" in row
    assert [changed["id"] for changed in result["changed"]] == [ID]
    assert TOOL_MEDIA_PARTS_KEY not in result and "pictures" not in result
    held = state.slices[0].deformation
    assert held is not None and held["steps"][0]["stiffness"] == "soft"
    assert held["steps"][0]["engine"] == "ants"
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


@needs_ants
def test_ants_syn_shows_the_section_under_its_new_registration(tmp_path: Path, atlas):
    *_, box = _setup(tmp_path, atlas)
    result = _ants(box)
    media = result[TOOL_MEDIA_PARTS_KEY]
    assert len(media) == 1 and isinstance(media[0], Image.Image)
    assert len(result["pictures"]) == 1
    saved = box.job.views.lookup(result["pictures"][0]["id"])
    assert saved is not None and saved.tool == "ants_syn"


@needs_ants
def test_a_second_fit_builds_on_the_first_and_undo_goes_back(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    _ants(box, view=False)
    first = state.slices[0].deformation
    row = _ants(box, restrict_to=["STR"], view=False)["results"][0]
    assert row["steps"] == 2 and "step_displacement_mm" in row
    held = state.slices[0].deformation
    assert [step["start"] for step in held["steps"]] == ["linear", "current"]
    assert held["steps"][1]["include"] == ["STR"]
    record = DeformableRecord.load(box.job.layout.resolve(held["record"]))
    assert record.step == 1 and record.parent is not None
    assert record.parent.settings.structures == ()
    assert record.settings.structures == ("STR",)
    _tool(box, "undo")()
    assert state.slices[0].deformation == first


@needs_ants
def test_restrict_to_and_marked_damage_reach_the_engine(tmp_path: Path, atlas, monkeypatch):
    state, *_, box = _setup(tmp_path, atlas)
    state.slices[0].damaged_regions = ["HY:left"]
    seen: list[Any] = []
    real = deformation.prepare_fit

    def spy(image: Any, atlas_: Any, placement: Any, settings: Any, **kwargs: Any) -> Any:
        seen.append(settings)
        return real(image, atlas_, placement, settings, **kwargs)

    monkeypatch.setattr(deformation, "prepare_fit", spy)
    _ants(box, restrict_to=["STR"], view=False)
    (settings,) = seen
    assert settings.structures == ("STR",)
    assert settings.exclude == ("HY:left",)
    assert settings.atlas_image == "template" and settings.engine == "ants"
    assert settings.detail == "coarse"


@needs_ants
def test_a_section_without_a_linear_placement_is_refused(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    state.slices[0].transform = None
    result = _tool(box, "ants_syn")([ID])
    assert result["status"] == "error" and result["error"] == "NOTHING_FITTED"
    assert result["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT"
    assert TOOL_MEDIA_PARTS_KEY not in result
    assert state.slices[0].deformation is None


# --- the fits underneath (ops.deformable.fit_deformable) -----------------------


@needs_ants
def test_one_setting_applies_and_the_same_again_writes_nothing(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    row = _fit(box, ctx, _choice(stiffness="soft")).rows[0]
    assert row["status"] == "ok" and row["written"] is True and row["steps"] == 1
    depth = len(box.job.undo_stack)
    again = _fit(box, ctx, _choice(stiffness="soft")).rows[0]
    assert again["written"] is False and again["cached"] is True
    assert len(box.job.undo_stack) == depth


@needs_ants
def test_start_current_needs_an_applied_deformation(tmp_path: Path, atlas):
    _, ctx, _, box = _setup(tmp_path, atlas)
    assert _fit(box, ctx, start="current").rows[0]["error"] == "NO_DEFORMATION"
    _fit(box, ctx)
    latest = _fit(box, ctx, _choice(stiffness="soft"),
                  start=ops_deformable.START_LATEST).rows[0]
    assert latest["status"] == "ok" and latest["steps"] == 2


@needs_ants
def test_excluded_regions_are_drawn_in_their_own_color(tmp_path: Path, atlas):
    _, ctx, _, box = _setup(tmp_path, atlas)
    options = default_options("overlay")
    plain = _fit(box, ctx, options=options)
    box.job.state.slices[0].damaged_regions = ["HY"]  # marked damage is left out
    marked = _fit(box, ctx, options=options)

    def pinkish(image: Image.Image) -> int:
        pixels = _pixels(image)
        return int(((pixels[..., 0] > 200) & (pixels[..., 1] < 140)
                    & (pixels[..., 2] > 100)).sum())

    assert pinkish(marked.pictures[0]) > 20 and pinkish(plain.pictures[0]) == 0


# --- consequences: linear edits, the overlay -------------------------------------


@needs_ants
@pytest.mark.parametrize("edit", ["transform", "position", "angles", "flip"])
def test_a_linear_edit_clears_the_warp_and_says_so(tmp_path: Path, atlas, edit):
    state, ctx, _, box = _setup(tmp_path, atlas)
    _ants(box, view=False)
    held = state.slices[0].deformation
    transform = dict(state.slices[0].transform)
    if edit == "transform":
        reply = _tool(box, "interactive_transform")([{"id": ID, "rotation_deg": 2.0}],
                                                    view=False)
    elif edit == "position":
        reply = _tool(box, "position_sections")([{"id": ID, "position_mm": 0.15}],
                                                view=False)
    elif edit == "angles":
        reply = _tool(box, "position_sections")(cutting_angles={"pitch_deg": 1.0,
                                                                "yaw_deg": 0.0}, view=False)
    else:
        reply = _tool(box, "interactive_transform")([{"id": ID, "flip": True}], view=False)
    assert reply["status"] == "ok", reply
    assert reply["deformation_cleared"] == [ID]
    assert state.slices[0].deformation is None
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation is None
    _tool(box, "undo")()
    assert state.slices[0].deformation == held
    assert state.slices[0].transform == transform


@needs_ants
def test_reads_keep_the_warp(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    _ants(box, view=False)
    held = state.slices[0].deformation
    for reply in (_tool(box, "status")(), _tool(box, "note")("looked"),
                  _tool(box, "look")("overlay", sections=[ID])):
        assert "deformation_cleared" not in reply
    assert state.slices[0].deformation == held


@needs_ants
def test_look_overlay_draws_the_applied_warp_unless_warp_none(tmp_path: Path, atlas):
    state, _, _, box = _setup(tmp_path, atlas)
    look = _tool(box, "look")

    def drawn(warp: str) -> np.ndarray:
        reply = look("overlay", sections=[ID], warp=warp)
        assert reply["status"] == "ok", reply
        (picture,) = reply[TOOL_MEDIA_PARTS_KEY]
        return _pixels(picture)

    linear = drawn("none")
    assert np.array_equal(drawn("applied"), linear)  # nothing applied yet
    _ants(box, stiffness="soft", view=False)
    assert not np.array_equal(drawn("applied"), linear)
    assert np.array_equal(drawn("none"), linear)
    assert state.slices[0].deformation is not None
    # Built for a run without positioning, too: the overlay of the supplied placement.
    (tmp_path / "only").mkdir()
    *_, nonlinear_only = _setup(tmp_path / "only", atlas, tasks=("nonlinear",))
    assert {"look", "ants_syn"} <= set(nonlinear_only.names)
    assert not {"position_sections", "interactive_transform"} & set(nonlinear_only.names)


def test_the_job_statement_names_the_tools_and_says_when_ants_is_missing(
        tmp_path: Path, atlas, monkeypatch):
    state, _, spec, box = _setup(tmp_path, atlas)

    def statement() -> str:
        text = build_job_statement(spec, state, tool_names=box.names, species="mouse",
                                   pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"),
                                   atlas_channels=("template", "borders"))
        return " ".join(text.split())

    monkeypatch.setattr(deformation, "ants_available", lambda: True)
    text = statement()
    assert "`ants_syn`" in text and "`trace_borders`" in text
    assert "`fit_deformable`" not in text and "`keep_linear`" not in text
    assert "- Atlas layers on this host: template, borders." in text
    assert "ANTs is not installed" not in text
    monkeypatch.setattr(deformation, "ants_available", lambda: False)
    assert ("- ANTs is not installed on this host, so ants_syn and trace_borders cannot "
            "run; a section left without a deformation is named in submit's left_linear."
            ) in statement()


# --- traced section images (the fit a trace gets) ----------------------------------


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


@needs_ants
def test_traced_images_need_a_completed_trace_at_this_placement(tmp_path: Path, atlas,
                                                                monkeypatch):
    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
    assert _fit(box, ctx, TRACED).rows[0]["error"] == "NO_TRACE"
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    _fake_trace(state, ctx, tmp_path / "trace", fingerprint="earlier")
    assert _fit(box, ctx, TRACED).rows[0]["error"] == "TRACE_STALE"
    state.slices[0].image_correction["geometry_fingerprint"] = "now"
    row = _fit(box, ctx, TRACED).rows[0]
    assert row["status"] == "ok", row
    assert row["settings"]["fit_atlas"] == "borders"  # traced lines fit against borders
    assert row["engine_settings"]["metric"] == "mean_squares"
    # The step records the trace it fitted (a packaged trace asked again reuses it).
    assert state.slices[0].deformation["steps"][-1]["trace"] == str(tmp_path / "trace")


@needs_ants
def test_a_traced_fit_drops_the_regions_the_trace_left_out(tmp_path: Path, atlas, monkeypatch):
    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    _fake_trace(state, ctx, tmp_path / "trace", fingerprint="now")
    state.slices[0].image_correction["exclude"] = ["HY"]
    state.slices[0].damaged_regions = ["VS"]  # marked damage is left out too
    row = _fit(box, ctx, TRACED).rows[0]
    assert row["status"] == "ok", row
    assert row["trace_regions"] == {"include": [], "exclude": ["VS", "HY"]}
    steps = state.slices[0].deformation["steps"]
    assert steps[-1]["exclude"] == ["VS", "HY"]
    # A stain fit ignores the trace's regions.
    assert "trace_regions" not in _fit(box, ctx).rows[0]


def test_the_model_is_shown_only_the_kept_regions(tmp_path: Path, atlas):
    from langslice.core.handoff import prepare_linear_registration
    from langslice.core.nonlinear.registration_tool import shown_labels

    state, ctx, _, _ = _setup(tmp_path, atlas)
    prepared = prepare_linear_registration(state, ctx, ID)
    labels = np.asarray(atlas.annotation[0])
    every = shown_labels(atlas, labels, prepared)
    assert (every > 0).sum() == (labels > 0).sum()
    without = shown_labels(atlas, labels, prepared, exclude=("HY",))
    assert not without[labels == HY].any()
    assert (without[labels == TH] > 0).all()
    only = shown_labels(atlas, labels, prepared, include=("TH",))
    assert (only > 0).sum() == (labels == TH).sum()
    with pytest.raises(ValueError, match="no atlas region"):
        shown_labels(atlas, labels, prepared, include=("TH",), exclude=("TH",))


@needs_ants
def test_a_traced_image_waits_for_its_running_trace_and_shows_it(tmp_path: Path, atlas,
                                                                 monkeypatch):
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
    done = _fit(box, ctx, TRACED, options=default_options("overlay"))
    assert done.rows[0]["status"] == "ok", done.rows
    assert state.slices[0].image_correction["status"] == "ok"
    assert load_checkpoint(ctx.checkpoint_path).slices[0].image_correction["status"] == "ok"
    assert ID not in box.job.image_jobs
    assert len(done.pictures) == 2 and done.traces == [{"id": ID, "image_indexes": [1]}]
    # The trace picture carries the lines in the border color on the section.
    pixels = _pixels(done.pictures[1])
    assert int(((pixels[..., 0] > 200) & (pixels[..., 1] > 200)
                & (pixels[..., 2] < 80)).sum()) > 50


@needs_ants
def test_a_trace_still_running_after_the_wait_is_reported(tmp_path: Path, atlas, monkeypatch):
    import threading

    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
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
    real_wait = box.job.wait_image_job
    pending = True

    def wait(section_id: str, timeout: float) -> bool:
        return False if pending else real_wait(section_id, timeout)

    monkeypatch.setattr(box.job, "wait_image_job", wait)
    try:
        late = _fit(box, ctx, TRACED).rows[0]
        assert late["error"] == "TRACE_TIMEOUT" and "0.2 s" in late["message"]
    finally:
        pending = False
        release.set()
    failed = _fit(box, ctx, TRACED).rows[0]
    assert failed["error"] == "TRACE_FAILED" and "no image" in failed["message"]


# --- the nonlinear goal: a deformation (or left_linear) per section ----------------


def _statement(spec: JobSpec, state: StackState, box: Any) -> str:
    text = build_job_statement(spec, state, tool_names=box.names, species="mouse",
                               pos_lo=0, pos_hi=1, axis_ends=("anterior", "posterior"))
    return " ".join(text.split())


def test_the_job_is_a_deformation_per_section_with_or_without_the_image_model(
        tmp_path: Path, atlas):
    state, _, spec, box = _setup(tmp_path, atlas)
    text = _statement(spec, state, box)
    assert ("give every section a deformation onto the atlas, on top of its linear "
            "placement, with the section's stain as the evidence, or, for a section you "
            "choose to trace, the borders the image model traces on it.") in text
    assert ("- `submit` is refused unless every section carries a deformation applied at "
            "its current placement, or is named in its `left_linear` with the reason its "
            "linear placement stands, damaged sections included.") in text
    assert ("- `trace_borders` is optional: you decide which sections, if any, to trace. "
            "A trace requires a position and a linear transform.") in text
    assert ("- Fit each section whole first, then refine regions. Inspect each fit's "
            "borders against the section's internal anatomy and, where traced, its traced "
            "borders; undo a fit that does not improve the alignment, and name in submit's "
            "`left_linear` only a section that no fit improves.") in text

    (tmp_path / "none").mkdir()
    none_state, _, none_spec, none_box = _setup(tmp_path / "none", atlas, provider="none")
    text = _statement(none_spec, none_state, none_box)
    assert ("give every section a deformation onto the atlas, on top of its linear "
            "placement, with the section's stain as the evidence.") in text
    assert "or is named in its `left_linear` with the reason" in text
    for absent in ("trace", "image model"):
        assert absent not in text, absent

    (tmp_path / "required").mkdir()
    req_state, _, req_spec, req_box = _setup(tmp_path / "required", atlas,
                                             require_deformation=True)
    text = _statement(req_spec, req_state, req_box)
    assert ("- `submit` is refused unless every section carries a deformation applied at "
            "its current placement, damaged sections included; the user requires one on "
            "every section.") in text
    assert "left_linear" not in text


@needs_ants
def test_without_an_image_model_there_are_no_traces(tmp_path: Path, atlas, monkeypatch):
    from langslice.core import handoff

    _, _, spec, box = _setup(tmp_path, atlas, provider="none")
    assert spec.nonlinear.uses_image_model is False
    assert "trace_borders" not in box.names
    assert {"grep_atlas", "ants_syn"} <= set(box.names)

    def no_trace_check(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("submit looked for an image correction")

    monkeypatch.setattr(handoff, "correction_fingerprint", no_trace_check)
    submit = _tool(box, "submit")
    first = submit("Done", [], [])
    assert first["error"] == "MISSING_DEFORMATIONS"
    assert first["sections"] == [{"id": ID,
                                  "reason": "no deformation, and not in left_linear"}]
    _ants(box, view=False)
    assert submit("Done", [], [])["status"] == "ok"
    with pytest.raises(ValueError, match="nonlinear.provider"):
        NonlinearSpec(provider="mystery")
    assert JobSpec.from_dict(spec.to_dict()).nonlinear.provider == "none"


def test_submit_left_linear_records_why_the_placement_stands(tmp_path: Path, atlas,
                                                              monkeypatch):
    from langslice.core import handoff

    state, ctx, _, box = _setup(tmp_path, atlas)
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *_: "now")
    submit = _tool(box, "submit")
    assert submit("Done", [], [])["error"] == "MISSING_DEFORMATIONS"
    assert submit("Done", [], [], left_linear=[{"id": ID}])["error"] == "BAD_ARGS"
    assert submit("Done", [], [], left_linear=[{"id": "nope.png", "reason": "x"}])[
        "error"] == "UNKNOWN_SLICE_IDS"
    refused = submit("Done", [], [], left_linear=[{"id": ID, "reason": "x", "why": "y"}])
    assert refused["status"] == "error" and not state.submitted
    reply = submit("Done", [], [], left_linear=[{"id": ID, "reason": "Matches already."}])
    assert reply["status"] == "ok", reply
    assert reply["left_linear"] == {ID: "Matches already."}
    assert state.submitted
    held = state.slices[0].deformation
    assert held["keep_linear"] == "Matches already."
    assert held["linear_key"] == deformation.linear_key(state, state.slices[0])
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation == held


@needs_ants
def test_left_linear_is_refused_for_a_fitted_section(tmp_path: Path, atlas):
    state, ctx, _, box = _setup(tmp_path, atlas)
    _fit(box, ctx)
    refused = _tool(box, "submit")("Done", [], [], left_linear=[{"id": ID, "reason": "x"}])
    assert refused["error"] == "HAS_DEFORMATION" and refused["ids"] == [ID]
    assert not state.submitted


def test_left_linear_is_not_offered_when_the_user_requires_a_deformation(tmp_path: Path,
                                                                         atlas):
    from google.adk.tools import FunctionTool

    state, *_, box = _setup(tmp_path, atlas, require_deformation=True)
    declaration = FunctionTool(_tool(box, "submit"))._get_declaration().model_dump()
    schema = declaration["parameters"] or declaration["parameters_json_schema"]
    assert "left_linear" not in schema["properties"]
    assert "left_linear" not in (_tool(box, "submit").__doc__ or "")
    refused = _tool(box, "submit")("Done", [], [], left_linear=[{"id": ID, "reason": "x"}])
    assert refused["status"] == "error" and not state.submitted


@needs_ants
def test_keep_linear_satisfies_submit_until_the_placement_moves(tmp_path: Path, atlas):
    """The "linear placement stands" record (what submit's left_linear writes)
    is cleared by a placement change like an applied fit."""
    state, ctx, _, box = _setup(tmp_path, atlas, provider="none")
    before = box.job.snapshot()
    state.slices[0].deformation = {
        "keep_linear": "The linear placement already matches.",
        "linear_key": deformation.linear_key(state, state.slices[0])}
    box.job.commit(before)
    held = state.slices[0].deformation
    assert held["keep_linear"] == "The linear placement already matches."
    assert load_checkpoint(ctx.checkpoint_path).slices[0].deformation == held
    assert box.job.submit_errors([]) is None
    assert _fit(box, ctx, start="current").rows[0]["error"] == "NO_DEFORMATION"
    # A placement change clears it, like an applied fit.
    moved = _tool(box, "interactive_transform")([{"id": ID, "rotation_deg": 2.0}], view=False)
    assert moved["deformation_cleared"] == [ID]
    assert box.job.submit_errors([])["error"] == "MISSING_DEFORMATIONS"
    _tool(box, "undo")()
    assert state.slices[0].deformation == held
    _tool(box, "undo")()
    assert state.slices[0].deformation is None
