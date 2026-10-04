"""The registry of tools and their operations matches the tools ``build_tools`` makes.

``langslice.ops.registry.VERBS`` is the one list the doors are generated from
in phase 5; every tool a toolbox can hold must be in it, and every entry must
be a tool some toolbox holds, with an operation from ``langslice.ops``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from langslice.ops.registry import GROUPS, VERBS, table
from tests.golden.record import atlas_loader, full_spec, write_sections


def _names(spec: Any) -> list[str]:
    from langslice.linear.engine import build_context
    from langslice.linear.job import ingest
    from langslice.linear.toolbox import build_tools

    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    state = ingest(spec, ctx)
    return build_tools(state, ctx, spec).names


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("registry")
    write_sections(path)
    return path


def test_every_tool_of_the_full_toolbox_is_registered_and_back(folder: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(folder / "home"))
    names = _names(full_spec(folder))
    assert names == list(VERBS)


@pytest.mark.parametrize("tasks", [["reorder"], ["position"], ["transform"], ["nonlinear"],
                                   ["position", "transform"]])
def test_every_tool_of_a_partial_toolbox_is_registered(folder: Path, tasks: list[str],
                                                       monkeypatch):
    monkeypatch.setenv("HOME", str(folder / "home"))
    spec = dataclasses.replace(full_spec(folder), tasks=tasks)
    names = _names(spec)
    assert names and set(names) <= set(VERBS)
    # The always-on tools come with every task.
    common = {name for name, verb in VERBS.items() if verb.group == "Common"}
    assert common <= set(names)


def test_every_operation_is_an_op_and_classified():
    for name, verb in VERBS.items():
        assert verb.name == name
        assert verb.function.__module__.startswith("langslice.ops."), name
        assert verb.kind in ("read", "write")
        assert verb.group in GROUPS
        for alternate in verb.alternates.values():
            assert alternate.__module__.startswith("langslice.ops."), name
    rows = table()
    assert [row["tool"] for row in rows] == list(VERBS)
    assert {row["kind"] for row in rows if row["tool"].startswith("view_")} == {"read"}
