"""``JobSpec.frozen``: the setting, its validation, the refusals of every
write, the partial freezes, the tools and tasks it switches off, and the
model-facing text, which names none of them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import langslice
from langslice.core import frozen as freeze
from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec
from langslice.core.toolset import TOOLS
from langslice.doors import jobs
from langslice.job.job import submit_errors
from langslice.ops import deformable as ops_deformable
from langslice.ops import positions, transforms
from langslice.ops.refusal import Refused
from langslice.ops.registry import enabled
from tests.linear_tool_helpers import SLAB, tool_named
from tests.test_disabled_tools import _base_specs, _model_texts, _pattern, _stack_folder
from tests.test_linear_host_controls import _stack
from tests.test_ops_interactive_transform import _open

ALIGNED = ["orientation", "affine"]
POSITIONS = {"s0.png": 1.0, "s1.png": 2.0, "s2.png": 3.0}


def _spec(**fields: Any) -> JobSpec:
    return JobSpec(image_folder="/stack", **fields)


# --- the setting ----------------------------------------------------------------------


def test_the_properties_are_the_five_a_section_has():
    from dataclasses import fields

    from langslice.core.state import SliceState

    assert freeze.PROPERTIES == ("position", "angles", "orientation", "affine", "deformation")
    names = {field.name for field in fields(SliceState)}
    # position, cutting angles, the flip and quarter turn, the transform, the deformation
    assert {"position_mm", "cutting_angles_deg", "flip", "rotation_deg", "transform",
            "deformation"} <= names


def test_a_freeze_is_cleaned_to_properties_in_order_and_round_trips():
    spec = _spec(frozen={"a.tif": ["affine", "orientation", "affine"], "*": ["angles"],
                         "b.tif": []}, inputs={})
    assert spec.frozen == {"a.tif": ["orientation", "affine"], "*": ["angles"]}
    assert JobSpec.from_dict(spec.to_dict()) == spec
    assert "frozen" not in _spec().to_dict()  # the default stays out of saved specs


@pytest.mark.parametrize("bad, message", [
    ("angles", "must map a section filename"),
    ({"a.tif": "angles"}, "must be a list of properties"),
    ({"a.tif": ["spin"]}, r"unknown properties \['spin'\]"),
    ({"a.tif": [3]}, "must be a list of properties"),
    ({"": ["affine"]}, "must map a section filename"),
    ({"a.tif": ["position"]}, "a position has no default"),
])
def test_a_bad_freeze_is_refused_when_the_spec_is_made(bad: Any, message: str):
    with pytest.raises(ValueError, match=message):
        _spec(frozen=bad)


def test_a_frozen_position_must_have_been_supplied(tmp_path: Path):
    # By name, at once; for "*" when the sections are known (at ingest).
    assert _spec(frozen={"a.tif": ["position"]}, inputs={"positions": {"a.tif": 1.0}})
    with pytest.raises(ValueError, match="a position has no default"):
        _stack(tmp_path, n=2, placed=False, frozen={"*": ["position"]})
    state, _spec_made, _tools = _stack(
        tmp_path, n=2, placed=False, frozen={"*": ["position"]}, tasks=["transform"],
        inputs={"positions": {"s0.png": 1.0, "s1.png": 2.0}})
    assert [record.position_mm for record in state.in_order()] == [1.0, 2.0]


def test_the_none_values_are_what_a_section_starts_with(tmp_path: Path):
    state, _spec_made, _tools = _stack(
        tmp_path, n=2, placed=False, frozen={"*": ["angles", "orientation", "deformation"]})
    record = state.by_id("s0.png")
    assert record is not None
    assert (record.flip, record.rotation_deg, record.cutting_angles_deg) == (
        False, 0, {"pitch": 0.0, "yaw": 0.0})
    assert record.deformation is None and record.transform is None
    # A frozen affine with none supplied is the host's identity.
    state, _spec_made, _tools = _stack(tmp_path, n=2, placed=False, frozen={"s1.png": ["affine"]})
    assert state.by_id("s1.png").transform["kind"] == "host"
    assert state.by_id("s0.png").transform is None


def test_what_is_supplied_is_what_a_frozen_property_keeps(tmp_path: Path):
    state, _spec_made, _tools = _stack(
        tmp_path, n=2, placed=False, frozen={"s0.png": ["orientation", "angles"]},
        inputs={"orientation": {"s0.png": {"flip": True, "rotation_deg": 90}},
                "angles": {"s0.png": {"pitch": 2.0, "yaw": -1.0}}})
    record = state.by_id("s0.png")
    assert record is not None
    assert (record.flip, record.rotation_deg, record.angles) == (True, 90, (2.0, -1.0))


def test_a_freeze_naming_no_section_is_refused_at_ingest(tmp_path: Path):
    with pytest.raises(ValueError, match="not here"):
        _stack(tmp_path, n=2, frozen={"nope.png": ["affine"]})


# --- the refusals ---------------------------------------------------------------------


def test_position_sections_refuses_a_frozen_position_or_angles_whole(tmp_path: Path):
    job, ctx = _open(tmp_path, n=3, frozen={"s1.png": ["position"]},
                     inputs={"positions": POSITIONS})
    with pytest.raises(Refused) as refused:
        positions.position_sections(job, ctx, [{"id": "s0.png", "position_mm": 5.0},
                                               {"id": "s1.png", "position_mm": 6.0}])
    assert refused.value.code == "FROZEN"
    assert refused.value.details["ids"] == ["s1.png"]
    assert refused.value.details["properties"] == ["position"]
    assert "s1.png" in refused.value.details["message"]
    assert job.undo_stack == []  # nothing written, the unfrozen section included
    assert positions.position_sections(job, ctx, [{"id": "s0.png", "position_mm": 5.0}]).written
    # The angles are one plane for the stack: one section with them frozen refuses all.
    (tmp_path / "angles").mkdir()
    job, ctx = _open(tmp_path / "angles", n=2, frozen={"s0.png": ["angles"]})
    with pytest.raises(Refused) as angles:
        positions.position_sections(job, ctx, [], {"pitch_deg": 1.0, "yaw_deg": 1.0})
    assert angles.value.code == "FROZEN" and angles.value.details["ids"] == ["s0.png"]


def test_interactive_transform_refuses_per_entry_what_is_frozen(tmp_path: Path):
    job, ctx = _open(tmp_path, n=4, frozen={"s0.png": ["orientation"], "s1.png": ["affine"]})
    done = transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "flip": True},              # orientation frozen
        {"id": "s0.png", "rotation_deg": 2.0},       # the affine is free there
        {"id": "s1.png", "rotation_deg": 2.0},       # affine frozen
        {"id": "s1.png", "rotate_quarter": 90},      # orientation is free, no transform yet
        {"id": "s2.png", "flip": True, "rotation_deg": 1.0},
    ])
    errors = [(entry.error or {}).get("error") for entry in done.entries]
    assert errors == ["FROZEN", None, "FROZEN", None, None]
    assert done.entries[0].error["properties"] == ["orientation"]
    assert done.entries[2].error["properties"] == ["affine"]
    assert sorted(done.written) == ["s0.png", "s1.png", "s2.png"]
    assert job.state.by_id("s0.png").flip is False
    assert job.state.by_id("s1.png").rotation_deg == 90
    assert job.state.by_id("s1.png").transform["kind"] == "host"  # the frozen numbers stand


def test_a_frozen_affine_can_still_be_looked_at_with_its_own_numbers(tmp_path: Path):
    job, ctx = _open(tmp_path, n=2, frozen={"s0.png": ["affine"]})
    done = transforms.interactive_transform(job, ctx, [{"id": "s0.png", "scale_x": 1.0}])
    assert done.entries[0].error is None and done.written == []
    assert job.undo_stack == []


def test_elastix_affine_and_set_transforms_refuse_a_frozen_affine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []
    monkeypatch.setattr("langslice.core.transform.fit_elastix",
                        lambda _s, _w, record, **_k: seen.append(record.id) or _ok_fit(1.0))
    job, ctx = _open(tmp_path, frozen={"s1.png": ["affine"]})
    done = transforms.elastix_affine(job, ctx, ["s0.png", "s1.png"])
    assert [row.get("error") for row in done.rows] == [None, "FROZEN"]
    assert done.rows[1]["properties"] == ["affine"] and seen == ["s0.png"]
    with pytest.raises(Refused) as refused:
        transforms.set_transforms(job, {"s1.png": {"params": [1, 0, 0, 0, 1, 0]}})
    assert refused.value.code == "FROZEN" and refused.value.details["ids"] == ["s1.png"]


def test_the_deformation_tools_refuse_a_frozen_deformation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ops_deformable, "refuse_without_ants", lambda: None)
    job, ctx = _open(tmp_path, tasks=["nonlinear"], frozen={"s0.png": ["deformation"]},
                     nonlinear=NonlinearSpec(provider="none"))
    for record in job.state.slices:
        record.transform = {"kind": "imported", "params": [1, 0, 0, 0, 1, 0]}
    fit = ops_deformable.ants_syn(job, ctx, ["s0.png"])
    assert fit.rows[0]["error"] == "FROZEN" and fit.rows[0]["properties"] == ["deformation"]
    # Even where the host requires a deformation on every section, the gate
    # asks only for the section whose deformation is free.
    required = JobSpec.from_dict({**job.spec.to_dict(), "nonlinear": {
        **job.spec.to_dict()["nonlinear"], "require_deformation": True}})
    refusal = submit_errors(job.state, required, [])
    assert refusal is not None and [row["id"] for row in refusal["sections"]] == ["s1.png"]


# --- the partial freezes ---------------------------------------------------------------


def _ok_fit(determinant: float) -> dict[str, Any]:
    return {"status": "ok", "params": [determinant, 0.0, 0.0, 0.0, 1.0, 0.0],
            "physical": {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
                         "translate_x_mm": 0.0, "translate_y_mm": 0.0, "shear": 0.0,
                         "pivot": [0.5, 0.5]},
            "calibration": {}, "mirrored": determinant < 0}


@pytest.mark.parametrize("determinant, kept", [(1.0, True), (-1.0, False)])
def test_a_fit_under_a_frozen_orientation_keeps_the_handedness(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, determinant: float, kept: bool):
    monkeypatch.setattr("langslice.core.transform.fit_elastix",
                        lambda _s, _w, _r, **_k: _ok_fit(determinant))
    job, ctx = _open(tmp_path, frozen={"s0.png": ["orientation"]})
    done = transforms.elastix_affine(job, ctx, ["s0.png", "s1.png"])
    free = done.rows[1]
    assert free["status"] == "ok"  # s1's orientation is free: a mirroring fit is its to make
    row = done.rows[0]
    if kept:
        assert row["status"] == "ok" and "s0.png" in done.fitted
    else:
        assert row["error"] == "FROZEN" and row["properties"] == ["orientation"]
        assert "mirror" in row["message"] and "s0.png" not in done.fitted
        assert job.state.by_id("s0.png").transform is None
        assert job.state.by_id("s1.png").transform["mirrored"] is True


def test_handedness_is_the_one_the_supplied_transform_carries(tmp_path: Path):
    mirrored = {"kind": "imported", "params": [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}
    job, ctx = _open(tmp_path, n=2, frozen={"s0.png": ["orientation"]},
                     inputs={"transforms": {"s0.png": mirrored}})
    # The host's transform already mirrors: a fit that mirrors the same way is kept,
    # one that turns it back is the orientation changing.
    assert transforms.set_transforms(job, {"s0.png": {**mirrored, "params": [-1, 0, 2, 0, 1, 0]}})
    with pytest.raises(Refused) as refused:
        transforms.set_transforms(job, {"s0.png": {"params": [1, 0, 0, 0, 1, 0]}})
    assert refused.value.code == "FROZEN" and refused.value.details["properties"] == [
        "orientation"]


def test_a_manual_transform_under_a_frozen_orientation_cannot_mirror(tmp_path: Path):
    job, ctx = _open(tmp_path, frozen={"s0.png": ["orientation"]})
    done = transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "scale_x": -1.0}, {"id": "s0.png", "scale_x": 1.2}])
    assert [(entry.error or {}).get("error") for entry in done.entries] == ["FROZEN", None]
    assert done.entries[0].error["properties"] == ["orientation"]
    assert job.state.by_id("s0.png").transform["physical"]["scale_x"] == 1.2


def test_a_frozen_affine_leaves_the_deformation_free_and_a_deformation_fit_writes_no_affine(
        tmp_path: Path):
    job, ctx = _open(tmp_path, tasks=["transform", "nonlinear"],
                     frozen={"s0.png": ["affine"]}, nonlinear=NonlinearSpec(provider="none"))
    assert job.frozen_refusal(["s0.png"], "deformation") is None
    assert job.frozen_refusal(["s0.png"], "affine")["error"] == "FROZEN"
    held = dict(job.state.by_id("s0.png").transform)
    assert held["kind"] == "host"
    # the deformation tools are on, and they only ever write the deformation
    names = set(enabled(job.spec))
    assert {"ants_syn", "elastix_affine", "interactive_transform"} <= names
    assert TOOLS["ants_syn"].writes == ("deformation",)
    assert job.state.by_id("s0.png").transform == held


# --- the Carey configuration ------------------------------------------------------------


CAREY = {"*": ["position", "angles", "orientation"]}


def _carey(folder: Path, **fields: Any) -> JobSpec:
    return JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["transform", "nonlinear"], nonlinear=NonlinearSpec(provider="none"),
        debrief=False, frozen=CAREY,
        inputs={"pixel_size_um": 25.0, "positions": POSITIONS,
                "angles": {"pitch": 1.5, "yaw": -2.0},
                "orientation": {"s0.png": {"flip": True, "rotation_deg": 0},
                                "s1.png": {"flip": False, "rotation_deg": 90}}},
        **fields)


def test_the_carey_configuration_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.linear_tool_helpers import stack as make_stack

    folder = tmp_path / "images"
    folder.mkdir()
    make_stack(folder, n=3)
    spec = _carey(folder)
    # position and angles and orientation are frozen everywhere; the affine and the
    # deformation are the agent's, so Linear and Nonlinear stay on and the tools that
    # write only what is frozen are gone.
    assert spec.has("transform") and spec.has("nonlinear") and not spec.has("position")
    assert set(enabled(spec)) >= {"interactive_transform", "elastix_affine", "ants_syn"}
    assert "position_sections" not in enabled(spec)
    assert spec.off("interactive_transform:flip") and not spec.off("interactive_transform")
    monkeypatch.setattr("langslice.core.transform.fit_elastix",
                        lambda _s, _w, _r, **_k: _ok_fit(1.0))
    created = jobs.create(spec, atlas_loader=lambda _n: SLAB)
    created.close()
    with langslice.open_job(str(folder), atlas_loader=lambda _n: SLAB) as job:
        state = job.state
        assert [(r.position_mm, r.flip, r.rotation_deg, r.angles) for r in state.in_order()] == [
            (1.0, True, 0, (1.5, -2.0)), (2.0, False, 90, (1.5, -2.0)),
            (3.0, False, 0, (1.5, -2.0))]
        assert not hasattr(job, "position_sections")
        # the agent may not turn a section over, and may fit its affine
        # the mirror and the quarter turn are not even offered: the sections take neither
        for key, value in (("rotate_quarter", 90), ("flip", False)):
            assert job.interactive_transform(
                sections=[{"id": "s0.png", key: value}], view=False)["status"] == "error"
        fitted = job.elastix_affine(sections=["s0.png", "s1.png"], view=False)
        assert [row["status"] for row in fitted["results"]] == ["ok", "ok"]
        assert state.by_id("s0.png").flip is True and state.by_id("s0.png").transform
        rows = {row["id"]: row for row in job.status()["rows"]}
        assert rows["s0.png"]["frozen"] == ["position", "angles", "orientation"]
        assert job.job.spec.to_dict()["frozen"] == CAREY


def test_the_agent_is_told_what_is_frozen(tmp_path: Path):
    from tests.linear_tool_helpers import stack as make_stack

    folder = tmp_path / "images"
    folder.mkdir()
    make_stack(folder, n=3)
    spec = _carey(folder)
    opened = jobs.create(spec, atlas_loader=lambda _n: SLAB)
    try:
        from langslice.agent.prompt import build_job_statement, display_facts

        ctx, state = opened.ctx, opened.job.state
        low, high = ctx.position_range
        text = build_job_statement(
            spec, state, tool_names=list(enabled(spec)), species=ctx.species, pos_lo=low,
            pos_hi=high, axis_ends=ctx.axis_ends, **display_facts(ctx, state))
    finally:
        opened.close()
    assert ("- Frozen by the user: position, cutting angles and orientation (flip and quarter "
            "turn) of every section. These keep the values the job gave them; a write that "
            "would change one is refused with FROZEN.") in text
    assert "Your job: give every section one in-plane transform onto its atlas section; " in text
    assert "correcting the orientation" not in text and "`position_sections`" not in text
    assert "- The positions shown are given; this run does not change them." in text
    # Sections sharing a freeze are one clause, and many are counted, not listed.
    many = [f"s{index}.png" for index in range(30)]
    line = freeze.statement({name: ["affine"] for name in many}, many)
    assert line is not None and "of every section" in line
    line = freeze.statement({name: ["affine"] for name in many[:12]}, many)
    assert line is not None and "12 of the 30 sections (status marks them in `frozen`)" in line
    line = freeze.statement({"s1.png": ["affine"], "s2.png": ["affine"], "s3.png": ["deformation"]},
                            many)
    assert line is not None and (
        "affine (in-plane transform) of s1.png, s2.png; deformation of s3.png" in line)
    assert freeze.statement({}, many) is None


# --- tools and tasks ----------------------------------------------------------------------


def test_a_tool_is_off_when_everything_it_writes_is_frozen_everywhere():
    cases = {
        ("affine",): {"elastix_affine"},
        ("orientation",): set(),
        ("deformation",): {"ants_syn", "trace_borders", "set_landmarks", "landmark_triangulation"},
        ("position",): set(),
        ("position", "angles"): {"position_sections"},
        ("orientation", "affine"): {"interactive_transform", "elastix_affine"},
    }
    for held, off in cases.items():
        spec = _spec(tasks=["position", "transform", "nonlinear"],
                     nonlinear=NonlinearSpec(provider="openai-oauth"),
                     frozen={"*": list(held)}, inputs={})
        built = set(enabled(spec, scripting=True))
        gone = {name for name in TOOLS if TOOLS[name].writes and name not in built}
        assert gone == off, held


def test_the_flip_goes_with_a_frozen_orientation_and_a_tool_with_its_other_property_stays():
    spec = _spec(tasks=["transform"], frozen={"*": ["orientation"]})
    assert spec.off("interactive_transform:flip") and not spec.off("interactive_transform")
    assert "interactive_transform" in enabled(spec)


def test_a_task_with_all_its_properties_frozen_everywhere_is_off():
    spec = _spec(tasks=["position", "transform", "nonlinear"],
                 frozen={"*": ["position", "angles", "deformation"]}, inputs={})
    assert spec.active_tasks == ["transform"]
    assert not spec.has("position") and not spec.has("nonlinear") and spec.has("transform")
    assert spec.tasks == ["position", "transform", "nonlinear"]  # what the user switched on
    # A task off supplies its properties: they are frozen on every section.
    off = _spec(tasks=["transform"])
    assert off.frozen_for("a.tif") == {"position", "angles", "deformation"}
    assert _spec(tasks=["transform"], transform=TransformSpec(angles=True)).frozen_for(
        "a.tif") == {"position", "deformation"}
    # With nothing frozen the tasks are as they were.
    plain = _spec(tasks=["position", "transform"])
    assert plain.active_tasks == ["position", "transform"] and plain.frozen_for("a") == {
        "deformation"}


def test_a_disabled_list_cannot_empty_a_task_the_freeze_leaves_on():
    with pytest.raises(ValueError, match="transform task is on and every tool that writes"):
        _spec(tasks=["transform"], frozen={"*": ["affine"]}, disabled=["interactive_transform"])
    # the freeze took the task off: nothing is left to refuse
    assert _spec(tasks=["transform"], frozen={"*": ["orientation", "affine"]},
                 disabled=["interactive_transform"]).active_tasks == []


def test_submit_does_not_ask_a_frozen_section_for_a_deformation(tmp_path: Path):
    state, spec, _tools = _stack(
        tmp_path, n=2, tasks=["nonlinear"],
        nonlinear=NonlinearSpec(provider="none", require_deformation=True),
        frozen={"s0.png": ["deformation"]},
        inputs={"transforms": {"s0.png": {"params": [1, 0, 0, 0, 1, 0]},
                               "s1.png": {"params": [1, 0, 0, 0, 1, 0]}}})
    refusal = submit_errors(state, spec, [])
    assert refusal is not None and [row["id"] for row in refusal["sections"]] == ["s1.png"]


# --- the text -----------------------------------------------------------------------------


FROZEN_CONFIGS: list[dict[str, list[str]]] = [
    CAREY,
    {"*": ["orientation"]},
    {"*": ["affine"]},
    {"*": ["orientation", "affine"]},
    {"*": ["deformation"]},
    {"*": ["position"]},
    {"*": ["angles"]},
    {"*": ["position", "angles"]},
    {"s0.png": ["orientation", "affine"], "s1.png": ["deformation", "angles"]},
]


def test_no_text_names_what_a_freeze_took_away(tmp_path: Path):
    """Every freeze on every base spec: no model-facing text names a tool or
    option the freeze switched off, and the text is rendered for the job that
    has the tools it names."""
    folder = _stack_folder(tmp_path)
    bases = _base_specs(str(folder))
    problems: list[str] = []
    checked = 0
    for frozen in FROZEN_CONFIGS:
        for label, base in bases.items():
            try:
                inputs = {**base.inputs, "pixel_size_um": 25.0, "positions": {
                    f"s{index}.png": 1.0 + index for index in range(3)}}
                spec = JobSpec.from_dict({**base.to_dict(), "frozen": frozen, "resume": False,
                                          "inputs": inputs})
            except ValueError:
                continue
            gone = [name for name in spec.switched_off if name not in spec.disabled]
            if not gone:
                continue
            checked += 1
            for where, text in _model_texts(folder, spec).items():
                for name in gone:
                    if (where.startswith("schema ") and name.startswith("look:")
                            and not where.startswith("schema look ")):
                        continue
                    for hit in _pattern(name).finditer(text):
                        around = text[max(0, hit.start() - 60):hit.end() + 40].replace("\n", " ")
                        problems.append(f"[{label}] {frozen}: {name!r} in {where}: ...{around}...")
                        break
    assert checked >= 20
    assert not problems, "\n".join(sorted(set(problems))[:40]) + f"\n({len(problems)} in all)"


def test_with_nothing_frozen_the_text_is_what_it_was(tmp_path: Path):
    folder = _stack_folder(tmp_path)
    for base in _base_specs(str(folder)).values():
        plain = JobSpec.from_dict({**base.to_dict(), "frozen": {}})
        assert plain == base
        for text in _model_texts(folder, plain).values():
            assert "Frozen by the user" not in text and "FROZEN" not in text


def test_the_status_reply_says_what_the_agent_can_still_change(tmp_path: Path):
    from tests.linear_tool_helpers import box

    folder = tmp_path / "images"
    folder.mkdir()
    _state, _ctx, toolbox = box(folder, placed=True, tasks=["position", "transform"],
                                frozen={"*": ["orientation"]})
    can_change = tool_named(toolbox, "status")()["can_change"]
    assert "in-plane transforms by hand" in can_change
    assert not any("orientation" in item or "mirroring" in item for item in can_change)
    assert "positions (the order follows them)" in can_change and "cutting angles" in can_change
    (folder / "b").mkdir()
    _state, _ctx, toolbox = box(folder / "b", placed=True, tasks=["position", "transform"],
                                frozen={"*": ["position", "angles"]})
    assert "positions (the order follows them)" not in tool_named(toolbox, "status")()[
        "can_change"]


def test_position_and_angles_frozen_leaves_a_run_without_the_positioning_tool():
    spec = _spec(tasks=["position", "transform"], position=PositionSpec(), inputs={},
                 frozen={"*": ["position", "angles"]})
    assert "position_sections" not in enabled(spec) and not spec.has("position")
    angles_only = _spec(tasks=["position"], frozen={"*": ["position"]}, inputs={})
    assert angles_only.has("position") and "position_sections" in enabled(angles_only)
