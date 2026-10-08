"""Every verb has one declaration, and every door is built from it.

``langslice.doors.declarations`` holds each verb's arguments and
description; the agent tools (and through them the MCP tools) carry exactly
those, and the CLI's schema is built from the same signature. The goldens
(``tests/golden/linear_tools/*_declarations_*``) pin the bytes the ADK and
MCP doors send; this checks the wiring and what each description must say.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from langslice.core.nonlinear.prompts import border_correction_tool_prompt
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.doors import declarations
from langslice.doors.declarations import FULL, VIEW_TOOLS, Variant, declaration, declare
from langslice.doors.tools.arguments import SectionPosition, SectionTransform
from langslice.ops.registry import VERBS, enabled


def test_every_verb_is_declared_once_in_registry_order():
    assert list(declarations.STUBS) == list(VERBS)
    for name in VERBS:
        declared = declaration(name)
        assert declared.name == name and declared.summary
        assert "tool_context" not in declared.signature.parameters


def test_the_variants_change_only_what_they_declare():
    assert "resolution" in declaration("look", Variant(auto=True)).signature.parameters
    fixed = declaration("look", Variant(auto=False))
    assert "resolution" not in fixed.signature.parameters
    assert "resolution:" not in fixed.doc
    for name in VIEW_TOOLS:
        assert "view" in declaration(name).signature.parameters
        forced = declaration(name, Variant(forced_view=True))
        assert "view" not in forced.signature.parameters and "view:" not in forced.doc
    assert declaration("position_sections").annotations["sections"] == list[SectionPosition]
    angles_only = declaration("position_sections", Variant(positions=False))
    assert list(angles_only.signature.parameters) == ["cutting_angles", "view"]
    assert "left_linear" in declaration("submit").signature.parameters
    assert "left_linear" not in declaration(
        "submit", Variant(left_linear=False)).signature.parameters
    assert declaration("interactive_transform").annotations["sections"] == list[
        SectionTransform]


def test_trace_borders_carries_the_base_prompt_and_the_border_rules():
    gpt = border_correction_tool_prompt("coronal", provider="openai-oauth")
    doc = declaration("trace_borders").doc
    # Every line of the base prompt, as the run's image model receives it.
    for line in gpt.split("\n"):
        assert line.strip() in doc
    words = " ".join(doc.split())
    assert "only source of which lines exist" in words
    assert "Never add a sentence that makes a line depend on whether its edge is visible" \
        in words
    assert "BASE_PROMPT" not in doc
    assert "The base prompt" not in declaration("trace_borders", Variant(prompt="")).doc
    sagittal = border_correction_tool_prompt("sagittal", provider="gemini-api")
    other = declaration("trace_borders", Variant(prompt=sagittal)).doc
    assert sagittal.split("\n")[0] in other and gpt.split("\n")[0] not in other
    # The CLI door says how its command answers.
    cli = declaration("trace_borders", Variant(door="cli", prompt=gpt)).doc
    assert "--background" in cli and "returns at once" not in cli


def test_the_descriptions_say_what_the_design_asks():
    assert "grep_atlas_view" in declaration("grep_atlas").doc
    damage = " ".join(declaration("mark_damage").doc.split())
    for words in ("whole hemisphere", "olfactory bulb", "cortex", "bubbles", "stains",
                  "Small tears", "damaged exactly when it has marked regions"):
        assert words in damage, words
    def words(name: str) -> str:
        return " ".join(declaration(name).doc.split())

    assert "Fit the whole section first, then refine regions" in words("ants_syn")
    assert "restrict_to, those regions' borders are drawn thick" in words("elastix_affine")
    assert "order follows the positions" in words("position_sections")


def test_declare_binds_the_declared_defaults_and_refuses_a_body_without_an_argument():
    seen: dict[str, Any] = {}

    def body(box: list[float], picture: int, region: str,
             tool_context: Any = None) -> dict[str, Any]:
        seen.update(box=box, picture=picture, region=region, tool_context=tool_context)
        return {"status": "ok"}

    tool = declare("zoom", body, FULL)
    assert tool.__name__ == "zoom"
    assert tool.__doc__ == declaration("zoom").doc
    assert list(inspect.signature(tool).parameters) == ["box", "picture", "region", "tool_context"]
    assert tool([0, 0, 10, 10]) == {"status": "ok"}
    assert seen == {"box": [0, 0, 10, 10], "picture": 0, "region": "", "tool_context": None}
    assert tool(region="CA1") == {"status": "ok"}
    assert seen == {"box": [], "picture": 0, "region": "CA1", "tool_context": None}

    def lacking(box: list[float]) -> dict[str, Any]:
        return {}

    with pytest.raises(TypeError, match="picture"):
        declare("zoom", lacking, FULL)


def test_the_cli_schema_is_the_declared_signature():
    schema = declarations.arguments_schema("position_sections", Variant(auto=True))
    assert schema["title"] == "position_sections"
    assert "required" not in schema
    assert set(schema["properties"]) == {"sections", "cutting_angles", "view"}
    assert schema["additionalProperties"] is False
    assert set(schema["$defs"]["SectionPosition"]["properties"]) == {"id", "position_mm"}
    assert declarations.arguments_schema("status")["properties"] == {}
    assert declarations.arguments_schema("mark_damage")["required"] == ["section", "regions"]


def _spec(**fields: Any) -> JobSpec:
    return JobSpec(image_folder="/stack", **fields)


_EVERY_RUN = ["look", "zoom", "set_channel_properties", "set_preprocessed_channel_properties",
              "grep_atlas", "grep_atlas_view", "status", "list_files", "search_files",
              "read_file"]
_BOOKKEEPING = ["note", "undo", "redo", "submit"]


@pytest.mark.parametrize(("spec", "names"), [
    (_spec(), [*_EVERY_RUN, "position_sections", "interactive_transform", "mark_damage",
               "elastix_affine", *_BOOKKEEPING]),
    (_spec(tasks=["transform"], transform=TransformSpec(automatic=False), agent_damage=False),
     [*_EVERY_RUN, "interactive_transform", *_BOOKKEEPING]),
    (_spec(tasks=["position"], agent_preprocessing=True),
     [*_EVERY_RUN, "position_sections", "mark_damage", *_BOOKKEEPING]),
    (_spec(tasks=["nonlinear"], nonlinear=NonlinearSpec(provider="none")),
     [*_EVERY_RUN, "mark_damage", "ants_syn", *_BOOKKEEPING]),
    (_spec(tasks=["nonlinear", "transform"], transform=TransformSpec(angles=True)),
     [*_EVERY_RUN, "position_sections", "interactive_transform", "mark_damage",
      "elastix_affine", "ants_syn", "trace_borders", *_BOOKKEEPING]),
])
def test_the_registry_switches_verbs_on_per_spec(spec: JobSpec, names: list[str]):
    assert enabled(spec) == names


def test_a_run_variant_follows_its_spec():
    plain = Variant.of(_spec(tasks=["position"]), auto=False)
    assert plain.positions and not plain.left_linear and not plain.prompt
    angles = Variant.of(_spec(tasks=["transform"], transform=TransformSpec(angles=True)),
                        auto=True)
    assert not angles.positions and angles.auto
    nonlinear = Variant.of(_spec(tasks=["nonlinear"], force_view=True), auto=False)
    assert nonlinear.left_linear and nonlinear.forced_view and nonlinear.prompt
    required = Variant.of(_spec(tasks=["nonlinear"], nonlinear=NonlinearSpec(
        require_deformation=True)), auto=False)
    assert not required.left_linear
    assert not Variant.of(_spec(tasks=["nonlinear"]), auto=False, image_model=False).prompt
