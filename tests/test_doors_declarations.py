"""Every verb has one declaration, and every door is built from it.

``langslice.doors.declarations`` holds each verb's arguments and
description; the agent tools (and through them the MCP tools) carry exactly
those, and the CLI's schema is built from the same signature. The goldens
(``tests/golden/linear_tools/*_declarations_*``) pin the bytes the ADK and
MCP doors send; this checks the wiring.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec
from langslice.doors import declarations
from langslice.doors.declarations import FULL, Variant, declaration, declare
from langslice.doors.tools.arguments import FixedCandidate, View, ViewAuto
from langslice.ops.registry import VERBS, enabled


def test_every_verb_is_declared_once_in_registry_order():
    assert list(declarations.STUBS) == list(VERBS)
    for name in VERBS:
        declared = declaration(name)
        assert declared.name == name and declared.summary
        assert "tool_context" not in declared.signature.parameters


def test_the_variants_change_only_what_they_declare():
    assert declaration("view_slices").annotations["view"] is View
    assert declaration("view_slices", Variant(auto=True)).annotations["view"] is ViewAuto
    fixed = declaration("fit_deformable", Variant(engine="elastix"))
    assert "engine" not in fixed.signature.parameters
    assert fixed.annotations["candidates"] == list[FixedCandidate]
    assert "The elastix engine" in fixed.doc and "recommended pairing" not in fixed.doc
    either = declaration("fit_deformable")
    assert "engine" in either.signature.parameters and "5 minutes" in either.doc
    stain = declaration("fit_deformable", Variant(traces=False))
    assert "traced_borders" not in stain.doc
    assert "preprocess tool" in declaration("fit_deformable", Variant(preprocessing=True)).doc


def test_declare_binds_the_declared_defaults_and_refuses_a_body_without_an_argument():
    seen: dict[str, Any] = {}

    def body(slices: list[str], view: Any, tool_context: Any = None) -> dict[str, Any]:
        seen.update(slices=slices, view=view, tool_context=tool_context)
        return {"status": "ok"}

    tool = declare("view_slices", body, FULL)
    assert tool.__name__ == "view_slices"
    assert tool.__doc__ == declaration("view_slices").doc
    assert list(inspect.signature(tool).parameters) == ["slices", "view", "tool_context"]
    assert tool(["a.png"]) == {"status": "ok"}
    assert seen == {"slices": ["a.png"], "view": {}, "tool_context": None}

    def lacking(slices: list[str]) -> dict[str, Any]:
        return {}

    with pytest.raises(TypeError, match="view"):
        declare("view_slices", lacking, FULL)


def test_the_cli_schema_is_the_declared_signature():
    schema = declarations.arguments_schema("set_positions", Variant(auto=True))
    assert schema["title"] == "set_positions"
    assert schema["required"] == ["entries"]
    assert set(schema["properties"]) == {"entries", "view"}
    assert schema["additionalProperties"] is False
    assert "resolution" in schema["$defs"]["ViewAuto"]["properties"]
    assert declarations.arguments_schema("status")["properties"] == {}


def _spec(**fields: Any) -> JobSpec:
    return JobSpec(image_folder="/stack", **fields)


@pytest.mark.parametrize(("spec", "names"), [
    (_spec(), ["status", "view_slices", "view_atlas", "note", "undo", "redo", "mark_damaged",
               "reorder_slices", "set_positions", "view_placement", "view_stack",
               "orient_slices", "fit_affine", "adjust_transforms", "submit"]),
    (_spec(tasks=["transform"], transform=TransformSpec(automatic=False), agent_damage=False),
     ["status", "view_slices", "view_atlas", "note", "undo", "redo", "view_placement",
      "orient_slices", "adjust_transforms", "submit"]),
    (_spec(tasks=["position"], position=PositionSpec(bayesian=True),
           agent_preprocessing=True),
     ["status", "view_slices", "view_atlas", "note", "undo", "redo", "mark_damaged",
      "preprocess", "set_positions", "view_placement", "view_stack",
      "search_position", "submit"]),
    (_spec(tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none")),
     ["status", "view_slices", "view_atlas", "note", "undo", "redo", "mark_damaged",
      "view_placement", "grep_atlas", "fit_deformable", "submit"]),
    (_spec(tasks=["nonlinear", "transform"], transform=TransformSpec(angles=True)),
     ["status", "view_slices", "view_atlas", "note", "undo", "redo", "mark_damaged",
      "view_placement", "orient_slices", "fit_affine", "adjust_transforms",
      "set_cutting_angles", "trace_borders", "grep_atlas", "fit_deformable", "submit"]),
])
def test_the_registry_switches_verbs_on_per_spec(spec: JobSpec, names: list[str]):
    assert enabled(spec) == names
