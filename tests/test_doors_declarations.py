"""Every verb has one declaration, and every door is built from it.

Each verb's arguments and description are data in ``langslice.tools.descriptions``
and ``render`` declares them; the agent tools (and through them the MCP tools)
carry exactly those, and the CLI's schema is built from the same signature. This
checks the wiring: the declared signatures, run facts, defaults and the CLI schema.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from langslice.core.fits.trace.prompts import border_correction_tool_prompt
from langslice.job.settings.spec import JobSpec, TransformSpec
from langslice.tools.arguments import SectionPosition, SectionTransform
from langslice.tools.catalog import VERBS
from langslice.tools.descriptions import render as declarations
from langslice.tools.descriptions.description import DescriptionFacts, Offered
from langslice.tools.descriptions.render import (
    DESCRIPTIONS,
    FULL,
    declaration,
    declare,
)

#: The change tools: each returns its picture unless ``view`` is false.
VIEW_TOOLS = [name for name, description in DESCRIPTIONS.items()
              if "view" in description.arguments
              and description.arguments["view"].offered is Offered.VIEW_CHOSEN]


def test_every_verb_is_declared_once_in_registry_order():
    assert list(DESCRIPTIONS) == list(VERBS)
    for name in VERBS:
        declared = declaration(name)
        assert declared.name == name and declared.summary
        assert "tool_context" not in declared.signature.parameters


def test_the_run_facts_change_only_what_they_declare():
    assert len(VIEW_TOOLS) == 9
    assert "resolution" in declaration("look", DescriptionFacts(auto=True)).signature.parameters
    fixed = declaration("look", DescriptionFacts(auto=False))
    assert "resolution" not in fixed.signature.parameters
    assert "resolution:" not in fixed.doc
    for name in VIEW_TOOLS:
        assert "view" in declaration(name).signature.parameters
        forced = declaration(name, DescriptionFacts(forced_view=True))
        assert "view" not in forced.signature.parameters and "view:" not in forced.doc
    assert declaration("position_sections").annotations["sections"] == list[SectionPosition]
    angles_only = declaration("position_sections", DescriptionFacts(positions=False))
    assert list(angles_only.signature.parameters) == ["cutting_angles", "view"]
    assert list(declaration("submit").signature.parameters) == [
        "summary", "notes", "interval_breaks"]
    assert declaration("interactive_transform").annotations["sections"] == list[
        SectionTransform]


def test_trace_borders_carries_the_base_prompt_and_the_border_rules():
    gpt = border_correction_tool_prompt("coronal", provider="openai-oauth")
    doc = declaration("trace_borders").doc
    # Every line of the base prompt, as the run's image model receives it.
    for line in gpt.split("\n"):
        assert line.strip() in doc
    assert "BASE_PROMPT" not in doc
    assert "The base prompt" not in declaration("trace_borders", DescriptionFacts(prompt="")).doc
    sagittal = border_correction_tool_prompt("sagittal", provider="gemini-api")
    other = declaration("trace_borders", DescriptionFacts(prompt=sagittal)).doc
    assert sagittal.split("\n")[0] in other and gpt.split("\n")[0] not in other
    # The CLI door says how its command answers.
    cli = declaration("trace_borders", DescriptionFacts(door="cli", prompt=gpt)).doc
    assert "--background" in cli and "returns at once" not in cli


def test_declare_binds_the_declared_defaults_and_refuses_a_body_without_an_argument():
    seen: dict[str, Any] = {}

    def body(box: list[float], picture: int, region: str,
             tool_context: Any = None) -> dict[str, Any]:
        seen.update(box=box, picture=picture, region=region,
                    tool_context=tool_context)
        return {"status": "ok"}

    tool = declare("zoom", body, FULL)
    assert tool.__name__ == "zoom"
    assert tool.__doc__ == declaration("zoom").doc
    assert list(inspect.signature(tool).parameters) == [
        "box", "picture", "region", "tool_context"]
    assert tool([0, 0, 10, 10]) == {"status": "ok"}
    assert seen == {"box": [0, 0, 10, 10], "picture": 0, "region": "",
                    "tool_context": None}
    assert tool(region="CA1") == {"status": "ok"}
    assert seen == {"box": [], "picture": 0, "region": "CA1", "tool_context": None}

    def lacking(box: list[float]) -> dict[str, Any]:
        return {}

    with pytest.raises(TypeError, match="picture"):
        declare("zoom", lacking, FULL)


def test_the_cli_schema_is_the_declared_signature():
    schema = declarations.arguments_schema("position_sections", DescriptionFacts(auto=True))
    assert schema["title"] == "position_sections"
    assert "required" not in schema
    assert set(schema["properties"]) == {"sections", "cutting_angles", "view"}
    assert schema["additionalProperties"] is False
    assert set(schema["$defs"]["SectionPosition"]["properties"]) == {"id", "position_mm"}
    assert declarations.arguments_schema("status")["properties"] == {}
    assert declarations.arguments_schema("mark_damage")["required"] == ["section", "regions"]


def _spec(**fields: Any) -> JobSpec:
    return JobSpec(image_folder="/stack", **fields)


def test_the_run_facts_follow_the_spec():
    plain = DescriptionFacts.from_spec(_spec(tasks=["position"]), auto=False)
    assert plain.positions and not plain.prompt
    angles = DescriptionFacts.from_spec(
        _spec(tasks=["transform"], transform=TransformSpec(angles=True)), auto=True)
    assert not angles.positions and angles.auto
    nonlinear = DescriptionFacts.from_spec(_spec(tasks=["nonlinear"], force_view=True), auto=False)
    assert nonlinear.forced_view and nonlinear.prompt
    unreached = DescriptionFacts.from_spec(_spec(tasks=["nonlinear"]), auto=False,
                                           image_model=False)
    assert not unreached.prompt
