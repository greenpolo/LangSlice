"""Damage by atlas region: the mark, its automatic exclusion from every fit
path, saved states with the older flag, and the host's damage notes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langslice.agent.engine import build_context
from langslice.core.damage import exclusions, normalized_entries
from langslice.core.display import default_options
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.core.state import SliceState, StackState
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
    # The older per-call exclusions are kept, repeats dropped case-insensitively.
    assert exclusions(record, (), ("HPF", "ob")) == ((), ("HPF", "ob", "CTX:left"))
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


@pytest.mark.parametrize("flag", ["damaged", "damage_marked"])
def test_a_saved_state_with_the_older_flag_loads_as_its_note_alone(flag: str):
    old = {"slices": [
        {"id": "a.png", "index_original": 0, "index_corrected": 0, flag: True,
         "damage_note": "left hemisphere torn"},
        {"id": "b.png", "index_original": 1, "index_corrected": 1, flag: False,
         "damage_note": ""},
    ]}
    state = StackState.from_dict(old)
    first, second = state.slices
    assert not first.damaged and first.damaged_regions == []
    assert first.damage_note == "left hemisphere torn"
    assert not second.damaged and second.damage_note == ""
    # Written back, neither flag is kept; the note and the regions are.
    rows = state.to_dict()["slices"]
    assert "damaged" not in rows[0] and "damage_marked" not in rows[0]
    assert rows[0]["damaged_regions"] == [] and rows[0]["damage_note"] == "left hemisphere torn"
    # Regions make the section damaged, and survive another load.
    state.slices[1].damaged_regions = ["R"]
    again = StackState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert not again.slices[0].damaged
    assert again.slices[1].damaged and again.slices[1].damaged_regions == ["R"]


# --- mark_damage ----------------------------------------------------------------------


def test_mark_damage_is_one_undo_step_and_empty_regions_clear(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    job = box.job
    done = damage.mark_damage(job, ctx, ID, ["R", "r"], "right half missing")
    assert done.written and done.regions == ["R"] and done.damaged and not done.by_user
    record = state.by_id(ID)
    assert record.damaged_regions == ["R"] and record.damage_note == "right half missing"
    assert len(job.undo_stack) == 1
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


def test_a_users_damage_note_is_a_note_alone_and_stays_first(tmp_path: Path):
    """The host's ``inputs.damaged`` note does not make the section damaged:
    the agent marks the regions, its note after the user's."""
    state, ctx, box = _box(tmp_path, inputs={"damaged": {ID: "torn by the user"}})
    job = box.job
    record = state.by_id(ID)
    assert not record.damaged and record.damaged_regions == []
    assert record.damage_note == "torn by the user"
    # Not damaged: a default fit target, with nothing to leave out.
    assert [r.id for r in transforms.fit_targets(job)] == [ID]
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


def test_the_mark_damage_tool_answers_with_the_marks_and_their_picture(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, _ctx, box = _box(tmp_path)
    mark = next(t for t in box.tools if t.__name__ == "mark_damage")
    result = mark(ID, ["R"], "right half missing")
    assert result["status"] == "ok", result
    assert result["regions"] == ["R"] and result["damaged"] is True
    assert result["note"] == "right half missing" and result["written"] is True
    assert [row["id"] for row in result["changed"]] == [ID]
    assert len(result["pictures"]) == 1 and len(result[TOOL_MEDIA_PARTS_KEY]) == 1
    assert state.by_id(ID).damaged_regions == ["R"]
    assert mark(ID, ["nope"])["error"] == "UNKNOWN_REGIONS"
    cleared = mark(ID, [])
    assert cleared["damaged"] is False and "pictures" not in cleared


def test_status_and_registration_carry_the_regions(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    damage.mark_damage(box.job, ctx, ID, ["R"], "right half missing")
    row = status_rows(state)[0]
    assert row["damaged"] is True and row["damaged_regions"] == ["R"]
    assert "damaged [R]: right half missing" in status_text(state)
    assert uniform_rows([{"id": "x"}])[0]["damaged_regions"] == []
    document = json.loads((box.job.folder / "registration.json").read_text())
    assert document["sections"][0]["parameters"]["damaged_regions"] == ["R"]
    status = next(t for t in box.tools if t.__name__ == "status")()
    assert status["rows"][0]["damaged_regions"] == ["R"]


def test_the_reply_picture_shows_the_marks_on_section_and_atlas(tmp_path: Path):
    _state, ctx, box = _box(tmp_path)
    done = damage.mark_damage(box.job, ctx, ID, ["R"], "right half missing",
                              options=default_options("overlay", long_edge=256))
    assert done.render_failed is None and len(done.pictures) == 1
    picture = done.pictures[0]
    # Two panels side by side: wider than tall.
    assert picture.width > picture.height
    # Without a position the write stands and the picture says why not.
    _state.slices[0].position_mm = None
    unplaced = damage.mark_damage(box.job, ctx, ID, ["L"], options=default_options("overlay"))
    assert unplaced.written and unplaced.pictures == []
    assert unplaced.render_failed and unplaced.render_failed["error"] == "NO_POSITION"


# --- every fit path leaves the marked regions out -------------------------------------
# elastix_affine: tests/test_ops_interactive_transform.py (the regions it is
# handed) and tests/test_linear_fit_affine_elastix.py (a real fit); ants_syn and
# the trace's ANTs fit: tests/test_linear_fit_deformable.py (the engine's
# settings) and tests/test_ants_syn.py (the recorded step); the image model's
# trace: below.


def test_a_marked_section_is_still_a_default_fit_target(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    damage.mark_damage(box.job, ctx, ID, ["R"])
    # The marked section needs no regions in the call: its marks are left out.
    assert [r.id for r in transforms.fit_targets(box.job)] == [ID]
    # A note alone has nothing to leave out.
    damage.mark_damage(box.job, ctx, ID, [])
    state.by_id(ID).damage_note = "torn"
    assert [r.id for r in transforms.fit_targets(box.job)] == [ID]


def test_the_image_model_trace_leaves_the_marked_regions_out(tmp_path: Path, monkeypatch):
    from langslice.core.nonlinear import registration_tool

    seen: list[dict[str, Any]] = []

    def spy(state: Any, ctx: Any, section_id: str, **kwargs: Any):
        seen.append(kwargs)
        return {"status": "ok", "geometry_fingerprint": "fp", "cached": True,
                "include": list(kwargs["include"]), "exclude": list(kwargs["exclude"])}, None

    monkeypatch.setattr(registration_tool, "start_correction", spy)
    monkeypatch.setattr(traces.handoff, "correction_fingerprint", lambda *_a: "fp")
    monkeypatch.setattr(deformable, "refuse_without_ants", lambda: None)
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
