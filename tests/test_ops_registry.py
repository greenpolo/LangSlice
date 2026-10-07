"""The registry of tools and their operations matches the tools ``build_tools`` makes.

``langslice.ops.registry.VERBS`` is the one list the doors are generated from;
every tool a toolbox can hold must be in it, and every entry must be a tool
some toolbox holds, with an operation from ``langslice.ops``. Each task that
is off removes its tools (the gating table), and every retired tool name
answers with its replacement.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from langslice.ops.registry import GROUPS, RETIRED, VERBS, enabled, retired_payload
from tests.golden.record import atlas_loader, full_spec, write_sections


def _names(spec: Any, *, scripting: bool = False, image_model: bool = True) -> list[str]:
    from langslice.agent.engine import build_context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest

    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    state = ingest(spec, ctx)
    return build_tools(state, ctx, spec, scripting=scripting,
                       image_model_connected=image_model).names


#: The verbs a model is offered (the agent tools, MCP); the scripting verbs
#: are the CLI's and the library's only.
MODEL_VERBS = [name for name, verb in VERBS.items() if not verb.scripting]

#: The tools of every run, whatever its tasks.
EVERY_RUN = ["look", "zoom", "set_channel_properties", "set_preprocessed_channel_properties",
             "grep_atlas", "grep_atlas_view", "status", "list_files", "search_files",
             "read_file", "note", "undo", "redo", "submit"]


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("registry")
    write_sections(path)
    return path


def test_the_seventeen_tools_and_the_file_tools():
    assert MODEL_VERBS == [
        "look", "zoom", "set_channel_properties", "set_preprocessed_channel_properties",
        "grep_atlas", "grep_atlas_view", "status", "list_files", "search_files", "read_file",
        "position_sections", "interactive_transform", "mark_damage",
        "elastix_affine", "ants_syn", "trace_borders",
        "note", "undo", "redo", "submit",
    ]
    files = {"list_files", "search_files", "read_file"}
    assert len([name for name in MODEL_VERBS if name not in files]) == 17


def test_every_tool_of_the_full_toolbox_is_registered_and_back(folder: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(folder / "home"))
    names = _names(full_spec(folder))
    assert names == MODEL_VERBS
    # The scripting door (the CLI, the library) adds the scripting verbs,
    # the hidden one included (callable by name).
    assert _names(full_spec(folder), scripting=True) == list(VERBS)
    assert [name for name, verb in VERBS.items() if verb.scripting] == [
        "trace_from_atlas", "export_maps"]


def test_a_hidden_verb_is_a_scripting_verb_left_out_of_every_listing(folder: Path):
    from langslice.ops.registry import listed

    hidden = [name for name, verb in VERBS.items() if verb.hidden]
    assert hidden == ["trace_from_atlas"]
    assert all(VERBS[name].scripting for name in hidden)
    spec = full_spec(folder)
    assert "trace_from_atlas" not in enabled(spec, scripting=True)
    assert "trace_from_atlas" in enabled(spec, scripting=True, hidden=True)
    assert "trace_from_atlas" not in enabled(spec, hidden=True)  # never a model's
    assert list(listed()) == [name for name in VERBS if name not in hidden]
    # Like trace_borders, it needs the run's image model.
    assert "trace_from_atlas" not in enabled(spec, scripting=True, hidden=True,
                                             image_model=False)


def _spec(folder: Path, **changes: Any) -> Any:
    return dataclasses.replace(full_spec(folder), **changes)


@pytest.mark.parametrize("tasks", [["position"], ["transform"], ["nonlinear"],
                                   ["position", "transform"], []])
def test_every_run_has_the_looking_and_bookkeeping_tools(folder: Path, tasks: list[str]):
    names = enabled(_spec(folder, tasks=tasks))
    assert set(EVERY_RUN) <= set(names)
    assert set(names) <= set(VERBS)


def test_the_gating_table(folder: Path):
    from langslice.core.spec import NonlinearSpec, TransformSpec

    def has(**changes: Any) -> set[str]:
        return set(enabled(_spec(folder, **changes))) - set(EVERY_RUN)

    # Positioning, or the cutting angles left to the agent.
    assert "position_sections" in has(tasks=["position"])
    assert "position_sections" in has(tasks=["transform"],
                                      transform=TransformSpec(angles=True))
    assert "position_sections" not in has(tasks=["transform"],
                                          transform=TransformSpec(angles=False))
    # Manual and automatic alignment are the transform task's, each switchable.
    assert {"interactive_transform", "elastix_affine"} <= has(tasks=["transform"])
    assert "interactive_transform" not in has(
        tasks=["transform"], transform=TransformSpec(interactive=False))
    assert "elastix_affine" not in has(
        tasks=["transform"], transform=TransformSpec(automatic=False))
    assert not {"interactive_transform", "elastix_affine"} & has(tasks=["position"])
    # Nonlinear: ants_syn; trace_borders only with an image model.
    assert {"ants_syn", "trace_borders"} <= has(tasks=["nonlinear"])
    assert "trace_borders" not in has(tasks=["nonlinear"],
                                      nonlinear=NonlinearSpec(provider="none"))
    assert "trace_borders" not in set(enabled(_spec(folder, tasks=["nonlinear"]),
                                              image_model=False))
    assert not {"ants_syn", "trace_borders"} & has(tasks=["position", "transform"])
    # Damage marking is a host switch.
    assert "mark_damage" in has(tasks=["position"])
    assert "mark_damage" not in has(tasks=["position"], agent_damage=False)


def test_every_operation_is_an_op_and_classified():
    for name, verb in VERBS.items():
        assert verb.name == name
        assert verb.function.__module__.startswith("langslice.ops."), name
        assert verb.kind in ("read", "write")
        assert verb.group in GROUPS
    reads = {"look", "zoom", "grep_atlas", "grep_atlas_view", "status", "list_files",
             "search_files", "read_file", "export_maps"}
    assert {name for name, verb in VERBS.items() if verb.kind == "read"} == reads
    assert {name for name, verb in VERBS.items() if verb.long} == {
        "elastix_affine", "ants_syn", "trace_borders", "trace_from_atlas", "export_maps"}


def test_every_retired_name_answers_with_its_replacement():
    assert set(RETIRED) == {
        "view_slices", "view_atlas", "view_placement", "view_stack", "preprocess",
        "set_positions", "set_cutting_angles", "reorder_slices", "orient_slices",
        "adjust_transforms", "mark_damaged", "fit_affine", "fit_deformable",
        "search_position"}
    for name, (replacement, hint) in RETIRED.items():
        assert name not in VERBS, name  # a retired name is never a verb again
        assert hint
        payload = retired_payload(name)
        assert payload is not None
        assert payload["error"] == "RETIRED_TOOL" and payload["tool"] == name
        assert "Nothing was done." in payload["message"]
        if replacement:
            assert replacement in VERBS
            assert payload["use"] == replacement
            assert f"Use {replacement} instead" in payload["message"]
        else:
            assert "use" not in payload
    assert RETIRED["search_position"][0] == ""
    assert retired_payload("look") is None
    assert retired_payload("no_such_tool") is None
