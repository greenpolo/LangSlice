"""The in-tree engines are tools of every job: rows of the tool table.

A row repeats what the engine's own declaration says, so the table answers
without importing an engine; these tests keep the two the same, and pin
which engine tools a job has without asking and which it switches on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import langslice.plugin_api  # noqa: F401 - registers the loader
from langslice.job import plugins
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tool_table import (
    ENGINES,
    TOOLS,
    default_off_names,
    valid_names,
)
from langslice.tools.catalog import enabled, verbs_for

ALL_TASKS = ["position", "transform", "nonlinear"]
ON = ["stalign_lddmm", "stalign_plane_search", "spomialign", "emlddmm_stack",
      "emlddmm_plane"]


def test_each_row_says_what_its_engine_declares() -> None:
    declared = plugins.tool_specs(None)
    rows = {name: tool for name, tool in TOOLS.items() if tool.engine}
    assert set(declared) == set(rows)
    for name, (engine, tool) in declared.items():
        row = rows[name]
        assert row.engine == engine.ref and engine.ref in ENGINES
        assert (row.kind, row.group, row.long) == (tool.kind, tool.group, tool.long)
        assert row.writes == tuple(tool.writes), name
        assert row.needs == tuple(tuple(group) for group in tool.needs), name
        assert row.atlases == tuple(tool.atlases), name
        assert row.planes == tuple(tool.planes), name


def test_a_new_job_has_the_engines_and_two_start_off(tmp_path: Path) -> None:
    spec = JobSpec(image_folder=str(tmp_path), tasks=ALL_TASKS)
    assert spec.plugins == [] and "plugins" not in spec.to_dict()
    names = enabled(spec)
    assert names[-len(ON):] == ON
    assert default_off_names() == ["deepslice_place", "brainways_ap"]
    assert not {"deepslice_place", "brainways_ap", "trace_borders"} & set(names)
    assert set(ON) | {"deepslice_place", "brainways_ap"} <= set(valid_names())
    assert list(verbs_for(spec))[-len(ON):] == ON


def test_enabled_switches_a_tool_on_where_it_fits(tmp_path: Path) -> None:
    folder = str(tmp_path)
    mouse = JobSpec(image_folder=folder, tasks=ALL_TASKS,
                    enabled=["deepslice_place", "brainways_ap"])
    assert "deepslice_place" in enabled(mouse)
    assert "brainways_ap" not in enabled(mouse)  # a rat atlas's tool
    assert mouse.off("brainways_ap") and not mouse.off("deepslice_place")
    rat = JobSpec(image_folder=folder, tasks=ALL_TASKS, atlas="whs_sd_rat_39um",
                  enabled=["deepslice_place", "brainways_ap"])
    assert "brainways_ap" in enabled(rat) and "deepslice_place" not in enabled(rat)
    assert JobSpec.from_dict(rat.to_dict()).enabled == rat.enabled


def test_disabled_takes_an_engine_tool_off_like_any_other(tmp_path: Path) -> None:
    spec = JobSpec(image_folder=str(tmp_path), tasks=ALL_TASKS,
                   disabled=["spomialign", "emlddmm_plane"])
    names = enabled(spec)
    assert "spomialign" not in names and "emlddmm_plane" not in names
    assert {"stalign_lddmm", "emlddmm_stack"} <= set(names)
    # an engine none of whose tools is on is not loaded for the run
    off = JobSpec(image_folder=str(tmp_path), tasks=ALL_TASKS,
                  disabled=["stalign_lddmm", "stalign_plane_search"])
    assert [engine.ref for engine in plugins.engines(off)] == [
        ref for ref in ENGINES if "spomialign" in ref or "emlddmm" in ref]


@pytest.mark.parametrize("names, message", [
    (["zoom"], "is not a tool that starts off"),
    (["trace_borders"], "is not a tool that starts off"),
    (["deepslice"], "is not a tool that starts off"),
])
def test_enabled_takes_only_the_tools_that_start_off(
        tmp_path: Path, names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JobSpec(image_folder=str(tmp_path), enabled=names)
    with pytest.raises(ValueError, match="both enabled and disabled"):
        JobSpec(image_folder=str(tmp_path), enabled=["deepslice_place"],
                disabled=["deepslice_place"])


def test_a_job_that_named_an_engine_as_a_plugin_keeps_its_tool(tmp_path: Path) -> None:
    old = {"image_folder": str(tmp_path), "tasks": ALL_TASKS,
           "plugins": ["langslice.engines.deepslice.plugin:tools",
                       "langslice.engines.stalign.plugin:tools"]}
    spec = JobSpec.from_dict(old)
    assert spec.plugins == [] and spec.enabled == ["deepslice_place"]
    assert {"deepslice_place", "stalign_lddmm"} <= set(enabled(spec))


def test_init_and_linear_run_take_enabled(tmp_path: Path) -> None:
    from langslice.doors.agent_cli.calls import init_parser
    from langslice.doors.commands.linear import spec_from_args
    from langslice.doors.commands.parser import build_parser

    init = spec_from_args(init_parser().parse_args(
        ["--tasks", "position,transform", "--enabled", "deepslice_place"]), str(tmp_path))
    assert init.enabled == ["deepslice_place"] and "deepslice_place" in enabled(init)
    run = spec_from_args(build_parser().parse_args(
        ["linear", "run", str(tmp_path), "--enable", "deepslice_place,brainways_ap"]),
        str(tmp_path))
    assert run.enabled == ["deepslice_place", "brainways_ap"]
