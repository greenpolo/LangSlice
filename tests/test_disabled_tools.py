"""A job's ``disabled`` list: the names, the refusals, the tools and options it
removes, and the model-facing text, which names none of what it disables."""

from __future__ import annotations

import inspect
import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest

from langslice.core.needs import render
from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec
from langslice.core.toolset import TASK_WRITERS, TOOLS, accepted_names, is_off, valid_names
from langslice.doors import jobs
from langslice.doors.cli.envelope import FIXES, disabled_for
from langslice.doors.declarations import DOORS, Variant, arguments_schema, declaration
from langslice.ops.registry import RETIRED, VERBS, enabled, retired_payload
from tests.linear_tool_helpers import SLAB, tool_named

# --- the names ------------------------------------------------------------------------


def _spec(folder: str = "/stack", **fields: Any) -> JobSpec:
    return JobSpec(image_folder=folder, **fields)


def test_the_valid_names_come_from_the_registry():
    assert set(TOOLS) == set(VERBS)
    names = valid_names()
    # a tool that can be switched off, then its options, in registry order
    assert names[:4] == [f"look:{mode}" for mode in TOOLS["look"].options]
    for name, verb in VERBS.items():
        assert (name in names) is (not verb.tool.essential and not verb.hidden), name
        for option in verb.tool.options:
            assert f"{name}:{option}" in names
    assert {"look:section", "look:atlas", "look:overlay", "look:positioning",
            "interactive_transform:flip"} <= set(names)
    # the hidden scripting verb is accepted and not advertised
    assert "trace_from_atlas" in accepted_names() and "trace_from_atlas" not in names
    assert not {"submit", "status", "look", "undo"} & set(names)


def test_look_modes_are_the_cores_modes():
    from langslice.core.look import MODES

    assert tuple(TOOLS["look"].options) == MODES


@pytest.mark.parametrize("bad, message", [
    (["zoomm"], "unknown name 'zoomm'; the valid names are look:section, .*, zoom,"),
    (["position_sections:view"], "unknown name"),
    (["look:nope"], "unknown name"),
    (["submit"], "submit cannot be disabled"),
    (["look"], "look cannot be disabled"),
    (["undo"], "undo cannot be disabled"),
    (["view_slices"], "unknown name"),  # a retired name is no tool
    ("zoom", "list of tool and option names"),
    ([3], "list of tool and option names"),
])
def test_unknown_and_essential_names_are_refused_at_construction(bad: Any, message: str):
    with pytest.raises(ValueError, match=message):
        _spec(disabled=bad)


def test_names_are_deduplicated_in_order_and_round_trip():
    spec = _spec(disabled=["zoom", " look:positioning", "zoom"])
    assert spec.disabled == ["zoom", "look:positioning"]
    assert JobSpec.from_dict(spec.to_dict()) == spec
    assert "disabled" not in _spec().to_dict()  # the default stays out of saved specs


def test_a_tool_disables_its_options():
    spec = _spec(disabled=["interactive_transform"])
    assert spec.off("interactive_transform:flip") and not spec.off("elastix_affine")
    assert is_off(["look:overlay"], "look:overlay") and not is_off(["look:overlay"], "look")


# --- the older switches ----------------------------------------------------------------


def test_older_specs_read_their_switches_as_disabled_names():
    old = {"image_folder": "/stack", "agent_damage": False,
           "transform": {"interactive": False, "automatic": True, "flip": False}}
    spec = JobSpec.from_dict({**old, "tasks": ["position", "transform"]})
    assert spec.disabled == ["mark_damage", "interactive_transform",
                             "interactive_transform:flip"]
    assert "mark_damage" not in enabled(spec) and "elastix_affine" in enabled(spec)
    assert "interactive_transform" not in enabled(spec)
    written = spec.to_dict()
    assert "agent_damage" not in written
    assert not {"interactive", "automatic", "flip"} & set(written["transform"])
    assert JobSpec.from_dict(written) == spec
    # switches left on say nothing; the explicit list comes first
    kept = JobSpec.from_dict({"image_folder": "/stack", "agent_damage": True,
                              "transform": {"interactive": True, "automatic": False},
                              "disabled": ["zoom"]})
    assert kept.disabled == ["zoom", "elastix_affine"]


def test_an_older_spec_with_every_alignment_tool_off_is_refused_as_a_new_one_is():
    with pytest.raises(ValueError, match="every tool that writes its answer"):
        JobSpec.from_dict({"image_folder": "/stack", "tasks": ["transform"],
                           "transform": {"interactive": False, "automatic": False}})


# --- what cannot work -------------------------------------------------------------------


@pytest.mark.parametrize("fields, message", [
    ({"tasks": ["transform"], "disabled": ["interactive_transform", "elastix_affine"]},
     "transform task is on and every tool that writes its answer"),
    ({"tasks": ["position"], "disabled": ["position_sections"]},
     "position task is on and every tool that writes its answer"),
    ({"tasks": ["nonlinear"], "disabled": ["ants_syn", "landmark_triangulation", "set_landmarks",
                                      "trace_borders"]},
     "nonlinear task is on and every tool that writes its answer"),
    ({"tasks": ["nonlinear"], "nonlinear": NonlinearSpec(provider="none"),
      "disabled": ["ants_syn", "landmark_triangulation", "set_landmarks"]},
     "nonlinear task is on and every tool that writes"),
    ({"tasks": ["nonlinear"], "disabled": ["set_landmarks"]},
     "landmark_triangulation is on and needs set_landmarks"),
    ({"tasks": ["nonlinear"], "disabled": ["landmark_triangulation"]},
     "set_landmarks is on and needs landmark_triangulation"),
    ({"tasks": ["nonlinear"], "disabled": ["look:section", "look:overlay"]},
     "ants_syn is on and needs look:overlay"),
    ({"disabled": ["look:section", "look:atlas", "look:overlay", "look:positioning"]},
     "every look mode is off"),
    ({"tasks": ["position"], "position": PositionSpec(gated=True),
      "disabled": ["look:positioning"]}, "position.gated .* look:positioning"),
    ({"tasks": ["position"], "disabled": ["look:positioning", "look:overlay"]},
     "position_sections is on and needs look:positioning or look:overlay"),
    ({"tasks": ["transform"], "disabled": ["look:overlay"]},
     "interactive_transform is on and needs look:overlay"),
    ({"tasks": ["nonlinear"], "disabled": ["look:overlay"]},
     "ants_syn is on and needs look:overlay"),
])
def test_a_list_that_cannot_work_is_refused_with_one_line(fields: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message) as refused:
        _spec(**fields)
    assert "\n" not in str(refused.value)


def test_what_can_work_is_accepted():
    # a task keeps one writer; a tool that is not on needs nothing
    _spec(tasks=["transform"], disabled=["elastix_affine"])
    _spec(tasks=["position"], disabled=["look:positioning"])  # overlays stand in
    _spec(tasks=["position"], disabled=["look:overlay"])
    _spec(tasks=["nonlinear"], disabled=["ants_syn"])
    _spec(tasks=["position"], disabled=["look:overlay"], position=PositionSpec(gated=True))
    _spec(tasks=["transform"], position=PositionSpec(gated=True),  # no position task: no gate
          disabled=["look:positioning"])


def test_task_writers_name_real_tools():
    for writers in TASK_WRITERS.values():
        assert set(writers) <= set(VERBS)


# --- the tools and options a job gets ----------------------------------------------------


def test_a_disabled_tool_is_not_enabled_anywhere():
    full = _spec(tasks=["position", "transform", "nonlinear"])
    base = enabled(full)
    for name in valid_names():
        if ":" in name:
            continue
        try:
            spec = _spec(tasks=full.tasks, disabled=[name])
        except ValueError:
            continue
        assert enabled(spec) == [tool for tool in base if tool != name], name


def test_the_look_schema_offers_only_the_modes_on():
    spec = _spec(tasks=["position"], disabled=["look:positioning"])
    schema = arguments_schema("look", Variant.of(spec, auto=True))
    assert schema["properties"]["mode"]["enum"] == ["section", "atlas", "overlay"]
    default = arguments_schema("look", Variant.of(_spec(tasks=["position"]), auto=True))
    assert default["properties"]["mode"]["enum"] == list(TOOLS["look"].options)


def test_arguments_only_a_disabled_mode_served_are_dropped():
    spec = _spec(tasks=["position"], disabled=["look:overlay"])
    schema = arguments_schema("look", Variant.of(spec, auto=True))
    assert not {"atlas_opacity", "warp"} & set(schema["properties"])
    assert {"positions_mm", "atlas_layers"} <= set(schema["properties"])
    zoom = arguments_schema("zoom", Variant.of(spec, auto=True))
    assert "region" not in zoom["properties"]
    assert "region" in arguments_schema("zoom", Variant.of(_spec(), auto=True))["properties"]


def test_flip_is_not_in_the_sections_schema_when_it_is_off():
    def keys(spec: JobSpec) -> set[str]:
        schema = arguments_schema("interactive_transform", Variant.of(spec, auto=True))
        item = schema["properties"]["sections"]["items"]
        ref = item["$ref"].split("/")[-1]
        return set(schema["$defs"][ref]["properties"])

    assert "flip" in keys(_spec(tasks=["transform"]))
    assert "flip" not in keys(_spec(tasks=["transform"], disabled=["interactive_transform:flip"]))
    # the quarter turn is an option of its own; a frozen orientation takes both
    assert "rotate_quarter" in keys(_spec(tasks=["transform"]))
    only = keys(_spec(tasks=["transform"], disabled=["interactive_transform:rotate"]))
    assert "rotate_quarter" not in only and "flip" in only
    frozen = keys(_spec(tasks=["transform"], frozen={"*": ["orientation"]}))
    assert not {"flip", "rotate_quarter"} & frozen and "rotation_deg" in frozen


def test_a_call_of_a_disabled_mode_is_refused_by_the_operation(tmp_path: Path):
    from tests.linear_tool_helpers import box

    state, ctx, toolbox = box(tmp_path, placed=True, tasks=["position"],
                              disabled=["look:positioning"])
    look = tool_named(toolbox, "look")
    refused = look(mode="positioning")
    assert refused["error"] == "OPTION_OFF" and refused["modes"] == [
        "section", "atlas", "overlay"]
    assert "switched off" in refused["message"]
    assert look(mode="overlay", sections=["s0.png"])["status"] == "ok"
    assert "positioning" not in inspect.signature(look).parameters["mode"].annotation.__args__


# --- fallbacks -----------------------------------------------------------------------------


def test_position_sections_shows_overlays_where_the_positioning_look_is_off(tmp_path: Path):
    from tests.linear_tool_helpers import box

    for off, mode in (([], "positioning"), (["look:positioning"], "overlay")):
        folder = tmp_path / ("off" if off else "on")
        folder.mkdir()
        state, ctx, toolbox = box(folder, placed=True, tasks=["position"], disabled=off)
        write = tool_named(toolbox, "position_sections")
        reply = write(sections=[{"id": "s0.png", "position_mm": 6.0},
                                {"id": "s1.png", "position_mm": 7.0}])
        assert reply["written"][0]["position_mm"] == 6.0
        shown = reply["pictures"]
        if mode == "positioning":
            assert len(shown) == 1  # one positioning picture for both sections
        else:
            assert len(shown) == 2  # one overlay each
    description = declaration("position_sections", Variant.of(
        _spec(tasks=["position"], disabled=["look:positioning"]), auto=False)).doc
    assert "positioning picture" not in description and "overlay" not in description
    assert "a picture of each written section" in description


def test_the_gates_name_the_modes_that_remain(tmp_path: Path):
    from tests.linear_tool_helpers import box

    for off, words in (([], "overlay or positioning"), (["look:overlay"], "positioning")):
        folder = tmp_path / ("a" if not off else "b")
        folder.mkdir()
        state, ctx, toolbox = box(folder, placed=True, tasks=["position"],
                                  position=PositionSpec(gated=True), disabled=off)
        refused = tool_named(toolbox, "position_sections")(
            sections=[{"id": "s0.png", "position_mm": 6.0}])
        assert refused["error"] == "NOT_COMPARED"
        assert refused["rejected"][0]["reason"].endswith(f"look at it in mode {words} first")


def test_read_file_points_to_zoom_only_when_there_is_one(tmp_path: Path):
    from tests.linear_tool_helpers import box

    texts = {}
    for off in ([], ["zoom"]):
        folder = tmp_path / ("zoom" if not off else "nozoom")
        folder.mkdir()
        state, ctx, toolbox = box(folder, placed=True, tasks=["position"], disabled=off)
        tool_named(toolbox, "look")(mode="overlay", sections=["s0.png"])
        toolbox.job.views.flush()  # the pictures are written in the background
        record = toolbox.job.views.records()[0]
        reply = tool_named(toolbox, "read_file")(f"{record.path}/view.jpg")
        texts[bool(off)] = reply["text"]
    assert "call zoom with picture" in texts[False]
    assert "To see it again" not in texts[True]


# --- model-facing text ------------------------------------------------------------------------


def _pattern(name: str) -> re.Pattern[str]:
    """What a text says when it names the tool or option *name*."""
    tool, _, option = name.partition(":")
    if not option:
        if tool == "note":  # the word is prose; the tool is named in quotes or called
            return re.compile(r"`note`|\bnote\(")
        return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(tool)}(?![A-Za-z0-9_])")
    if tool == "look":
        return re.compile(rf"look:{option}\b|\"{option}\"|`{option}`|mode {option}\b"
                          rf"|\b{option} mode"
                          + (r"|\boverlays?\b" if option == "overlay" else "")
                          + (r"|\bpositioning\b" if option == "positioning" else ""))
    if name == "interactive_transform:flip":  # a key of the sections, or the mirror itself
        return re.compile(r"\"flip\"|\{id, flip|[Mm]irroring|mirrored or turned|mirrors the")
    if name == "interactive_transform:rotate":  # a key of the sections, or the turn itself
        return re.compile(r"rotate_quarter|quarter turn is applied")
    if name == "grep_atlas:points":  # the argument, or what the tool says of its points
        return re.compile(r"\"points\"|region is drawn at|only points are given"
                          r"|look the region up at|its region and containing")
    raise AssertionError(f"no pattern for {name}")


def _model_texts(folder: Path, spec: JobSpec) -> dict[str, str]:
    """Every model-facing text of a run of *spec* that can be rendered
    without a model: the declarations and schemas of each tool on every door,
    the job statement on every door, the reference card, the status reply's
    list of what can change, the fixes and retired-tool answers."""
    from langslice.agent.prompt import build_job_statement, display_facts
    from langslice.doors.card import card_text
    from langslice.doors.statement import (
        image_model_off_text,
        job_statement,
        opening_for_cli,
        opening_for_mcp,
    )

    opened = jobs.create(spec, atlas_loader=lambda _name: SLAB)
    try:
        texts: dict[str, str] = {}
        ctx, state = opened.ctx, opened.job.state
        names = enabled(spec, scripting=True, hidden=True)
        for door, auto, traced in itertools.product(DOORS, (False, True), (True, False)):
            variant = Variant.of(spec, auto=auto, door=door, image_model=traced)
            for tool in names:
                declared = declaration(tool, variant)
                texts[f"declaration {tool} {door} auto={auto} traced={traced}"] = declared.doc
                texts[f"schema {tool} {door} auto={auto} traced={traced}"] = json.dumps(
                    arguments_schema(tool, variant))
        low, high = ctx.position_range
        for door, gates in itertools.product(DOORS, (True, False)):
            for connected in (True, False):
                opening = (opening_for_mcp(3) if door == "mcp"
                           else opening_for_cli("/job", 3, spec.switched_off))
                texts[f"statement {door} gates={gates} connected={connected}"] = job_statement(
                    spec, state, ctx, door=door, tool_names=names, opening=opening,
                    image_model_off=not connected, auto=door == "cli", gates=gates)
        texts["statement agent"] = build_job_statement(
            spec, state, tool_names=names, species=ctx.species, pos_lo=low, pos_hi=high,
            axis_ends=ctx.axis_ends, **display_facts(ctx, state))
        texts["image model off"] = image_model_off_text(spec.switched_off)
        texts["card"] = card_text(opened.job.layout)
        status = tool_named(opened.tools(), "status")()
        texts["status"] = json.dumps(status.get("can_change", status))
        for code in FIXES:
            if code in ("NOT_COMPARED", "NOT_REVIEWED") and not spec.position.gated:
                continue  # the viewing gates' refusals: no gates, no such refusal
            with disabled_for(spec.switched_off):
                from langslice.doors.cli.envelope import Envelope

                texts[f"fix {code}"] = Envelope.failure(code, fix=None).error["fix"]
        for name in RETIRED:
            payload = retired_payload(name, spec.switched_off)
            assert payload is not None
            texts[f"retired {name}"] = payload["message"]
        return texts
    finally:
        opened.close()


def _base_specs(folder: str) -> dict[str, JobSpec]:
    return {
        "all": _spec(folder, tasks=["position", "transform", "nonlinear"],
                     transform=TransformSpec(hemisphere_cue="a notch on the left"),
                     facts=["a fact"], position=PositionSpec(playbook=True),
                     nonlinear=NonlinearSpec(provider="openai-oauth")),
        "position": _spec(folder, tasks=["position"],
                          position=PositionSpec(gated=True, playbook=True)),
        "position-open": _spec(folder, tasks=["position"]),
        "linear": _spec(folder, tasks=["transform"], transform=TransformSpec(angles=True)),
        "nonlinear": _spec(folder, tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none")),
    }


def _with(base: JobSpec, disabled: list[str]) -> JobSpec | None:
    """*base* with *disabled* on top, or None when that cannot work."""
    try:
        return JobSpec.from_dict({**base.to_dict(), "disabled": disabled})
    except ValueError:
        return None


def _stack_folder(tmp_path: Path) -> Path:
    from tests.linear_tool_helpers import stack

    folder = tmp_path / "images"
    folder.mkdir()
    stack(folder, n=3, placed=True)
    return folder


COMBINATIONS = [
    ["look:positioning", "zoom"],
    ["look:positioning", "look:atlas"],
    ["look:overlay", "look:section"],
    ["look:atlas", "look:positioning"],
    ["look:atlas", "look:positioning", "look:overlay"],
    ["interactive_transform", "elastix_affine"],
    ["ants_syn", "trace_borders", "mark_damage"],
    ["grep_atlas", "grep_atlas_view", "mark_damage", "zoom"],
    ["list_files", "search_files", "read_file", "note", "redo"],
    ["position_sections", "set_channel_properties", "set_preprocessed_channel_properties"],
    ["interactive_transform:flip", "interactive_transform"],
    ["elastix_affine", "ants_syn", "look:positioning"],
    ["set_landmarks", "landmark_triangulation"],
    ["set_landmarks", "landmark_triangulation", "ants_syn", "zoom"],
]


def test_no_text_names_what_the_job_disables(tmp_path: Path):
    """Every single name, and a few combinations, on every base spec they can
    be applied to: no model-facing text names a tool or option that is off.
    Every name is checked at least once."""
    folder = _stack_folder(tmp_path)
    bases = _base_specs(str(folder))
    attempts = [[name] for name in sorted(accepted_names())] + COMBINATIONS
    covered: set[str] = set()
    problems: list[str] = []
    for disabled in attempts:
        for label, base in bases.items():
            spec = _with(base, disabled)
            if spec is None:
                continue
            covered |= set(disabled)
            texts = _model_texts(folder, spec)
            for where, text in texts.items():
                for name in disabled:
                    if (where.startswith("schema ") and name.startswith("look:")
                            and not where.startswith("schema look ")):
                        continue  # a schema's quoted keys are not the look's modes
                    for hit in _pattern(name).finditer(text):
                        around = text[max(0, hit.start() - 60):hit.end() + 40].replace("\n", " ")
                        problems.append(
                            f"[{label}] {disabled}: {name!r} in {where}: ...{around}...")
                        break
    assert covered == accepted_names()
    assert not problems, "\n".join(sorted(set(problems))[:40]) + f"\n({len(problems)} in all)"


def test_with_nothing_disabled_the_text_is_what_it_was(tmp_path: Path):
    """Marks leave no trace: a text rendered for a job that disables nothing
    is the text without its marks."""
    folder = _stack_folder(tmp_path)
    for base in _base_specs(str(folder)).values():
        for text in _model_texts(folder, base).values():
            assert not any(opener in text for opener in ("[[?", "[[#", "[[%"))


# --- the marks ------------------------------------------------------------------------------


@pytest.mark.parametrize("text, off, shown", [
    ("a[[?zoom  and zoom]] b", [], "a and zoom b"),
    ("a[[?zoom  and zoom]] b", ["zoom"], "a b"),
    ("[[?look:overlay x||y]]", ["look:overlay"], "y"),
    ("[[?look:overlay x||y]]", ["look"], "y"),
    ("[[?a|zoom z]]", ["zoom"], "z"),
    ("[[?a|zoom,note z]]", ["zoom", "note"], ""),
    ("[[#zoom=z||note=n||redo=r]]", [], "z, n and r"),
    ("[[#zoom=z||note=n||redo=r]]", ["note"], "z and r"),
    ("[[%zoom=z||note=n]]", [], "z or n"),
    ("[[%zoom=z||note=n]]", ["zoom", "note"], ""),
    ("[[?zoom a[[?note  b]]]]", ["note"], "a"),
    ("p\n\n[[?zoom q\n\n]]r", ["zoom"], "p\n\nr"),
    ("fit\n[[?zoom or zoom]].", ["zoom"], "fit."),
    ("fit\n[[?zoom or zoom]].", [], "fit\nor zoom."),
])
def test_marks(text: str, off: list[str], shown: str):
    assert render(text, off) == shown


@pytest.mark.parametrize("text", ["[[?zoom open", "[[?zoom]]", "[[#zoom x]]"])
def test_a_broken_mark_is_an_error(text: str):
    with pytest.raises(ValueError):
        render(text, ["zoom"])


def test_plain_text_is_left_alone():
    text = "no marks || here ]] at all\n\n\n  "
    assert render(text, ["zoom"]) == text



def test_the_setup_status_offers_what_a_dialog_can_switch_off():
    from langslice.doors.api.setup import setup_status

    tools = setup_status()["tools"]
    assert [tool["name"] for tool in tools] == [
        name for name, tool in TOOLS.items() if not tool.hidden]
    by_name = {tool["name"]: tool for tool in tools}
    assert by_name["look"]["disableable"] is False
    assert by_name["look"]["options"] == ["section", "atlas", "overlay", "positioning"]
    assert by_name["zoom"]["disableable"] is True and by_name["zoom"]["options"] == []
    assert by_name["interactive_transform"]["options"] == ["flip", "rotate"]
    assert by_name["position_sections"]["needs"] == [["look:positioning", "look:overlay"]]
    json.dumps(tools)
    names = {tool["name"] for tool in tools if tool["disableable"]} | {
        f"{tool['name']}:{option}" for tool in tools for option in tool["options"]}
    assert names == set(valid_names())


# --- the landmark tools --------------------------------------------------------------------


def test_the_landmark_tools_go_off_together_and_leave_no_mention(tmp_path: Path):
    folder = _stack_folder(tmp_path)
    both = ["set_landmarks", "landmark_triangulation"]
    on = _spec(str(folder), tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"))
    off = _spec(str(folder), tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"),
                disabled=both)
    assert set(both) <= set(enabled(on)) and not set(both) & set(enabled(off))
    assert "ants_syn" in enabled(off)
    for text in _model_texts(folder, off).values():
        assert not re.search(r"set_landmarks|landmark_triangulation|\blandmark pairs?\b", text), (
            text[:200])
    assert any("set_landmarks" in text for text in _model_texts(folder, on).values())


def test_zoom_can_be_switched_off(tmp_path: Path):
    spec = _spec(tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"),
                 disabled=["zoom"])
    assert "zoom" not in enabled(spec)


def test_a_frozen_deformation_refuses_the_triangulation_and_hides_the_tools(tmp_path: Path):
    from tests.linear_tool_helpers import box

    both = {"set_landmarks", "landmark_triangulation"}
    everywhere = _spec(tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"),
                       frozen={"*": ["deformation"]})
    assert not both & set(enabled(everywhere))
    one = _spec(tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none"),
                frozen={"s0.png": ["deformation"]})
    assert both <= set(enabled(one))

    state, ctx, toolbox = box(tmp_path, placed=True, tasks=["nonlinear"],
                              nonlinear=NonlinearSpec(provider="none"),
                              frozen={"s0.png": ["deformation"]})
    refused = tool_named(toolbox, "landmark_triangulation")(section="s0.png", view=False)
    assert refused["error"] == "FROZEN" and refused["properties"] == ["deformation"]
    # a section that is not frozen is not refused for that reason
    other = tool_named(toolbox, "landmark_triangulation")(section="s1.png", view=False)
    assert other["error"] == "INSUFFICIENT_LANDMARKS"
