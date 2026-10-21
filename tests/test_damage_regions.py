"""Damage by atlas region: the mark, its automatic exclusion from every fit
path, and the host's damage notes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langslice.agent.engine import build_context
from langslice.core.damage import exclusions, normalized_entries
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.core.state import SliceState
from langslice.core.status import status_rows, status_text, uniform_rows
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import apply_host_inputs, ingest
from langslice.ops import damage, deformable, traces, transforms
from langslice.ops.refusal import Refused
from tests.test_linear_fit_affine_regions import HalvesAtlas, _left_half_section

ID = "s0.png"


def _box(folder: Path, *, inputs: dict[str, Any] | None = None, plane: str = "coronal"):
    """The halves atlas (regions ``L``, ``R``, ``C``) and its left half as the section."""
    _left_half_section().save(folder / ID)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   plane=plane, inputs={"pixel_size_um": 50.0, **(inputs or {})},
                   tasks=["position", "transform", "nonlinear"],
                   transform=TransformSpec(angles=True), nonlinear=NonlinearSpec())
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: HalvesAtlas())
    state = ingest(spec, ctx)
    apply_host_inputs(state, spec)
    state.slices[0].position_mm = 1.0
    box = build_tools(state, ctx, spec)
    return state, ctx, box


# --- exclusions -----------------------------------------------------------------------


def test_exclusions_add_the_marked_regions_and_an_explicit_request_wins():
    record = SliceState(ID, 0, 0, damaged_regions=["CTX:left", "OB"])
    assert exclusions(record) == ((), ("CTX:left", "OB"))
    # restrict_to naming a marked entry exactly takes it back; another side does not.
    assert exclusions(record, ("ctx:LEFT",)) == (("ctx:LEFT",), ("OB",))
    assert exclusions(record, ("CTX:right",)) == (("CTX:right",), ("CTX:left", "OB"))
    assert exclusions(SliceState(ID, 0, 0)) == ((), ())
    assert normalized_entries([" CTX:Left ", "ctx:left", "", "OB", 672]) == [
        "CTX:left", "OB", "672"]
    with pytest.raises(ValueError):
        normalized_entries(["CTX:up"])


# --- the state ------------------------------------------------------------------------


def test_damaged_is_read_only_and_follows_the_marks():
    record = SliceState(ID, 0, 0)
    assert not record.damaged
    with pytest.raises(AttributeError):
        record.damaged = True  # type: ignore[misc]
    record.damaged_regions = ["R"]
    assert record.damaged
    record.damaged_regions = []
    # A note alone does not make a section damaged.
    record.damage_note = "left hemisphere torn"
    assert not record.damaged


# --- mark_damage ----------------------------------------------------------------------


def test_mark_damage_is_one_undo_step_and_empty_regions_clear(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    job = box.job
    done = damage.mark_damage(job, ctx, ID, ["R", "r"], "right half missing")
    assert done.written and done.regions == ["R"] and done.damaged and not done.by_user
    record = state.by_id(ID)
    assert record.damaged_regions == ["R"] and record.damage_note == "right half missing"
    assert len(job.undo_stack) == 1
    # Status and registration.json carry the regions.
    row = status_rows(state)[0]
    assert row["damaged"] is True and row["damaged_regions"] == ["R"]
    assert "damaged [R]: right half missing" in status_text(state)
    assert uniform_rows([{"id": "x"}])[0]["damaged_regions"] == []
    document = json.loads((job.folder / "registration.json").read_text())
    assert document["sections"][0]["parameters"]["damaged_regions"] == ["R"]
    # The same mark again writes nothing.
    assert not damage.mark_damage(job, ctx, ID, ["R"], "right half missing").written
    assert len(job.undo_stack) == 1
    cleared = damage.mark_damage(job, ctx, ID, [], "ignored")
    assert cleared.written and not cleared.damaged and record.damage_note == ""
    assert job.undo()
    record = state.by_id(ID)  # undo refills the state with new records
    assert record.damaged_regions == ["R"] and record.damaged


def test_mark_damage_refuses_bad_regions(tmp_path: Path):
    _state, ctx, box = _box(tmp_path)
    for regions, code in ((["nope"], "UNKNOWN_REGIONS"), (["R:up"], "BAD_ARGS"),
                          ("R", "BAD_ARGS")):
        with pytest.raises(Refused) as refused:
            damage.mark_damage(box.job, ctx, ID, regions)  # type: ignore[arg-type]
        assert refused.value.code == code
    with pytest.raises(Refused) as unknown:
        damage.mark_damage(box.job, ctx, "x.png", ["R"])
    assert unknown.value.code == "UNKNOWN_SLICE_IDS"
    assert box.job.undo_stack == []


def test_mark_damage_refuses_a_side_the_placement_does_not_have(tmp_path: Path):
    # The midline runs vertically on this placement, so its halves are left
    # and right: "top" is refused before anything is written.
    state, ctx, box = _box(tmp_path)
    transforms.interactive_transform(box.job, ctx, [{"id": ID, "rotation_deg": 0.0}])
    steps = len(box.job.undo_stack)
    with pytest.raises(Refused) as refused:
        damage.mark_damage(box.job, ctx, ID, ["R:top"])
    assert refused.value.code == "WRONG_SIDE_PAIR"
    assert "'left' and 'right'" in refused.value.details["message"]
    assert state.by_id(ID).damaged_regions == [] and len(box.job.undo_stack) == steps
    assert damage.mark_damage(box.job, ctx, ID, ["R:left"]).written


def test_a_users_damage_note_is_a_note_alone_and_stays_first(tmp_path: Path):
    """The host's ``inputs.damaged`` note does not make the section damaged:
    the agent marks the regions, its note after the user's."""
    state, ctx, box = _box(tmp_path, inputs={"damaged": {ID: "torn by the user"}})
    job = box.job
    record = state.by_id(ID)
    assert not record.damaged and record.damaged_regions == []
    assert record.damage_note == "torn by the user"
    # Not damaged: nothing to leave out of a fit.
    assert exclusions(record) == ((), ())
    added = damage.mark_damage(job, ctx, ID, ["R"], "right half gone")
    assert added.by_user and added.regions == ["R"] and added.damaged
    assert record.damage_note == "torn by the user; right half gone"
    # Clearing removes the agent's regions and note; the user's note stays.
    cleared = damage.mark_damage(job, ctx, ID, [])
    assert not cleared.damaged and record.damaged_regions == []
    assert record.damage_note == "torn by the user"
    row = next(t for t in box.tools if t.__name__ == "status")()["rows"][0]
    assert row["damage_note"] == "torn by the user" and not row.get("damaged")


# --- every fit path leaves the marked regions out -------------------------------------
# elastix_affine: tests/test_ops_interactive_transform.py (the regions it is
# handed) and tests/test_linear_fit_affine_elastix.py (a real fit); fireants_syn and
# the trace's SyN fit: tests/test_linear_fit_deformable.py (the engine's
# settings) and tests/test_fireants_syn.py (the recorded step); the image model's
# trace: below.


def test_the_image_model_trace_leaves_the_marked_regions_out(tmp_path: Path, monkeypatch):
    from langslice.core.nonlinear import registration_tool

    seen: list[dict[str, Any]] = []

    def spy(state: Any, ctx: Any, section_id: str, **kwargs: Any):
        seen.append(kwargs)
        return {"status": "ok", "geometry_fingerprint": "fp", "cached": True,
                "include": list(kwargs["include"]), "exclude": list(kwargs["exclude"])}, None

    monkeypatch.setattr(registration_tool, "start_correction", spy)
    monkeypatch.setattr(traces.handoff, "correction_fingerprint", lambda *_a: "fp")
    monkeypatch.setattr(deformable, "refuse_without_engine", lambda: None)
    state, ctx, box = _box(tmp_path)
    # The trace alone: no fit lands in the background here.
    monkeypatch.setattr(box.job.background, "start",
                        lambda *_a, **_k: SimpleNamespace(id="work-1"))
    damage.mark_damage(box.job, ctx, ID, ["R"])
    model = SimpleNamespace(provider="fake", model="fake", call=None)
    traces.trace_borders(box.job, ctx, ID, image_model=model)  # type: ignore[arg-type]
    traces.trace_borders(box.job, ctx, ID, image_model=model,  # type: ignore[arg-type]
                         restrict_to=["L"])
    assert [(call["include"], call["exclude"]) for call in seen] == [
        ((), ("R",)), (("L",), ("R",))]
    # A traced fit then reads the trace's exclusions along with the marks.
    held = state.by_id(ID).image_correction
    assert held["exclude"] == ["R"]
