"""Damage by atlas region: the mark, its automatic exclusion from every fit
path, and the host's damage notes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langslice.core.refusal import Refused
from langslice.core.sections.damage import exclusions, normalized_entries
from langslice.core.stack.section import Section
from langslice.core.stack.status import status_rows, status_text, uniform_rows
from langslice.doors.agent.engine import build_context
from langslice.doors.tools.toolbox import build_tools
from langslice.job.host_inputs import apply_host_inputs
from langslice.job.job import ingest
from langslice.job.settings.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.tools.changing import damage, transforms
from langslice.tools.fitting import deformable, traces
from tests.shapes import json_of
from tests.test_linear_fit_affine_regions import (
    HalvesAtlas,
    _left_half_section,
)

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
    state.slices[0].set_position(1.0)
    box = build_tools(state, ctx, spec)
    return state, ctx, box



def test_exclusions_add_the_marked_regions_and_an_explicit_request_wins():
    record = Section(ID, 0, 0, damaged_regions=["CTX:left", "OB"])
    assert exclusions(record) == ((), ("CTX:left", "OB"))
    assert exclusions(record, ("ctx:LEFT",)) == (("ctx:LEFT",), ("OB",))
    assert exclusions(record, ("CTX:right",)) == (("CTX:right",), ("CTX:left", "OB"))
    assert exclusions(Section(ID, 0, 0)) == ((), ())
    assert normalized_entries([" CTX:Left ", "ctx:left", "", "OB", 672]) == [
        "CTX:left", "OB", "672"]
    with pytest.raises(ValueError):
        normalized_entries(["CTX:up"])



def test_damaged_is_read_only_and_follows_the_marks():
    record = Section(ID, 0, 0)
    assert not record.damaged
    with pytest.raises(AttributeError):
        record.damaged = True  # type: ignore[misc]
    record.damaged_regions = ["R"]
    assert record.damaged
    record.damaged_regions = []
    record.damage_note = "left hemisphere torn"
    assert not record.damaged



def test_mark_damage_is_one_undo_step_and_empty_regions_clear(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    job = box.job
    done = damage.mark_damage(job, ctx, ID, ["R", "r"], "right half missing")
    assert done.written and done.regions == ["R"] and done.damaged and not done.by_user
    record = state.by_id(ID)
    assert record.damaged_regions == ["R"] and record.damage_note == "right half missing"
    assert len(job.undo_stack) == 1
    row = status_rows(state)[0]
    assert row["damaged"] is True and row["damaged_regions"] == ["R"]
    assert "damaged [R]: right half missing" in status_text(state)
    assert uniform_rows([{"id": "x"}])[0]["damaged_regions"] == []
    document = json.loads((job.folder / "registration.json").read_text(
        encoding="utf-8"))
    assert document["sections"][0]["parameters"]["damaged_regions"] == ["R"]
    assert not damage.mark_damage(job, ctx, ID, ["R"], "right half missing").written
    assert len(job.undo_stack) == 1
    cleared = damage.mark_damage(job, ctx, ID, [], "ignored")
    assert cleared.written and not cleared.damaged and record.damage_note == ""
    assert job.undo()
    record = state.by_id(ID)
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
    state, ctx, box = _box(tmp_path)
    transforms.interactive_transform(box.job, ctx, [{"id": ID, "rotation_deg": 0.0}])
    steps = len(box.job.undo_stack)
    with pytest.raises(Refused) as refused:
        damage.mark_damage(box.job, ctx, ID, ["R:top"])
    assert refused.value.code == "WRONG_SIDE_PAIR"
    assert "'left' and 'right'" in refused.value.facts["message"]
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
    assert exclusions(record) == ((), ())
    added = damage.mark_damage(job, ctx, ID, ["R"], "right half gone")
    assert added.by_user and added.regions == ["R"] and added.damaged
    assert record.damage_note == "torn by the user; right half gone"
    cleared = damage.mark_damage(job, ctx, ID, [])
    assert not cleared.damaged and record.damaged_regions == []
    assert record.damage_note == "torn by the user"
    row = next(t for t in box.tools if t.__name__ == "status")()["rows"][0]
    assert row["damage_note"] == "torn by the user" and not row.get("damaged")



def test_the_image_model_trace_leaves_the_marked_regions_out(tmp_path: Path, monkeypatch):
    from langslice.core.fits.trace import trace as registration_tool

    seen: list[dict[str, Any]] = []

    def spy(state: Any, ctx: Any, section_id: str, **kwargs: Any):
        seen.append(kwargs)
        return {"id": "s0.png", "status": "ok", "geometry_fingerprint": "fp", "cached": True,
                "include": list(kwargs["include"]), "exclude": list(kwargs["exclude"])}, None

    monkeypatch.setattr(registration_tool, "start_correction", spy)
    monkeypatch.setattr(traces.handoff, "correction_fingerprint", lambda *_a: "fp")
    monkeypatch.setattr(deformable, "refuse_without_engine", lambda: None)
    state, ctx, box = _box(tmp_path)
    monkeypatch.setattr(box.job.background, "start",
                        lambda *_a, **_k: SimpleNamespace(id="work-1"))
    damage.mark_damage(box.job, ctx, ID, ["R"])
    model = SimpleNamespace(provider="fake", model="fake", call=None)
    traces.trace_borders(box.job, ctx, ID, image_model=model)  # type: ignore[arg-type]
    traces.trace_borders(box.job, ctx, ID, image_model=model,  # type: ignore[arg-type]
                         restrict_to=["L"])
    assert [(call["include"], call["exclude"]) for call in seen] == [
        ((), ("R",)), (("L",), ("R",))]
    held = json_of(state.by_id(ID).border_trace)
    assert held["exclude"] == ["R"]



def test_a_hosts_damage_regions_are_the_sections_marks(tmp_path: Path):
    """``inputs.damaged_regions`` writes the section's marks at job start,
    normalized as ``mark_damage`` stores them; every fit then leaves them out."""
    from langslice.job.host_inputs import check_damaged_regions

    given = {ID: [" root:Right ", "root:right", "C"]}
    state, ctx, box = _box(tmp_path, inputs={"damaged_regions": given})
    record = state.by_id(ID)
    assert record.damaged_regions == ["root:right", "C"] and record.damaged
    assert record.damage_note == ""
    assert exclusions(record) == ((), ("root:right", "C"))
    check_damaged_regions(state, box.job.spec, ctx)
    assert damage.mark_damage(box.job, ctx, ID, ["R"]).written


def test_a_hosts_damage_regions_are_checked_as_the_agents_are(tmp_path: Path):
    from langslice.job.host_inputs import check_damaged_regions

    state, ctx, box = _box(tmp_path, inputs={"damaged_regions": {ID: ["nope"]}})
    with pytest.raises(ValueError, match="damaged_regions"):
        check_damaged_regions(state, box.job.spec, ctx)
    with pytest.raises(ValueError, match="Unknown side"):
        JobSpec(image_folder=str(tmp_path), inputs={"damaged_regions": {ID: ["R:up"]}})
    with pytest.raises(ValueError, match="sagittal"):
        JobSpec(image_folder=str(tmp_path), plane="sagittal",
                inputs={"damaged_regions": {ID: ["root:left"]}})
    with pytest.raises(ValueError, match="damaged_regions must be"):
        JobSpec(image_folder=str(tmp_path), inputs={"damaged_regions": {ID: "R"}})


def test_frozen_damage_refuses_a_change_of_the_marks(tmp_path: Path):
    """A frozen ``damage`` keeps the host's marks (none when it supplied none):
    ``mark_damage`` refuses other regions with FROZEN and writes nothing."""
    inputs = {"damaged_regions": {ID: ["root:right"]}}
    folder = tmp_path / "given"
    folder.mkdir()
    state, ctx, box = _box(folder, inputs=inputs)
    box.job.spec.frozen = {ID: ["damage"]}
    with pytest.raises(Refused) as refused:
        damage.mark_damage(box.job, ctx, ID, ["R"])
    assert refused.value.code == "FROZEN"
    assert state.by_id(ID).damaged_regions == ["root:right"]
    assert box.job.undo_stack == []
    assert not damage.mark_damage(box.job, ctx, ID, ["root:right"]).written
    empty = tmp_path / "none"
    empty.mkdir()
    state, ctx, box = _box(empty)
    box.job.spec.frozen = {ID: ["damage"]}
    with pytest.raises(Refused) as refused:
        damage.mark_damage(box.job, ctx, ID, ["R"])
    assert refused.value.code == "FROZEN" and state.by_id(ID).damaged_regions == []


def test_damage_frozen_on_every_section_takes_mark_damage_off(tmp_path: Path):
    spec = JobSpec(image_folder=str(tmp_path), frozen={"*": ["damage"]})
    assert spec.off("mark_damage")
    assert not JobSpec(image_folder=str(tmp_path), frozen={ID: ["damage"]}).off("mark_damage")
