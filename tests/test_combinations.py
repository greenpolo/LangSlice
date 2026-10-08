"""Every combination of steps, through every door.

The product requirement: LangSlice works in ANY combination of its steps
(Positioning, Linear, Nonlinear with or without an image model, or a
registration made elsewhere with only Nonlinear on top) through every door:
the agent CLI (``langslice-job FOLDER VERB``), the library
(``langslice.open_job``), the agent tools (``build_tools`` /
``ops.registry.enabled``) driven by LangSlice's agent, and MCP.

Each combination runs end to end on the golden recorder's synthetic stack
(``tests/golden/record.py``: three sections of the synthetic atlas, fits in
process at the coarse level). The image model is the golden stub (it answers
with the rough border overlay it was sent), installed where every door
resolves one; the real transport is replaced by a function that fails the
test, and ``HOME`` is private, so no credential is read and no model called.

| # | Combination | tasks | image model |
|---|---|---|---|
| 1 | Positioning only | reorder, position | - |
| 2 | Linear on supplied positions | transform | - |
| 3 | Positioning + Linear | reorder, position, transform | - |
| 4 | Nonlinear verbs called directly, no agent | all four | stub |
| 5 | Nonlinear without an image model | nonlinear | none |
| 6 | Nonlinear on a supplied (external) linear registration | nonlinear | stub |
| 7 | The agent, every task, no image model | all four | none |
| 8 | The placement-free trace (hidden ``trace_from_atlas``), then the fit | nonlinear | stub |
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.doors.cli.jobcli import main
from tests.cli_child import install
from tests.golden.record import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    apply_patches,
    atlas_loader,
    stub_image_model,
    write_sections,
)

IDS = (ID0, ID1, ID2)
STEMS = ("s0", "s1", "s2")
#: Positions inside the synthetic atlas, in stack order.
POSITIONS = {ID0: 0.1, ID1: 0.15, ID2: 0.2}
#: An identity transform per section (the Linear task's cheapest write).
IDENTITY = {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
            "translate_x_mm": 0.0, "translate_y_mm": 0.0}
#: A registration made elsewhere: per section the stored affine (one plain,
#: one mirrored: the flip carried as a negative determinant, as a host's own
#: alignment does), plus stack-wide cutting angles.
EXTERNAL_TRANSFORMS = {
    ID0: {"kind": "interactive", "params": [1.02, 0.01, -0.01, 0.0, 0.98, 0.01],
          "mirrored": False},
    ID1: {"kind": "interactive", "params": [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0], "mirrored": True},
    ID2: {"kind": "interactive", "params": [0.97, -0.02, 0.02, 0.02, 0.97, 0.0],
          "mirrored": False},
}
EXTERNAL_ANGLES = {"pitch": 1.0, "yaw": -0.5}


# --- the stack, the atlas, the image model -----------------------------------------


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("combinations") / "stack"
    write_sections(images)
    return images


def _no_network(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a real image model was called")


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh copy of the stack; the synthetic atlas wherever a door opens a
    job; the stub image model wherever a door resolves one, which counts as
    connected (a test of the opposite says so: :func:`connected`); no network."""
    import langslice.doors.tools.toolbox as toolbox
    import langslice.providers.images as transport
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "image_model_connected", lambda _provider: True)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    install(atlas_loader(), monkeypatch.setattr)
    monkeypatch.setattr(transport, "generate_warped_segmentation_image", _no_network)

    def resolve(provider: str, model: str | None = None) -> Any:
        holder = type("Spec", (), {"nonlinear": NonlinearSpec(provider=provider,
                                                              image_model=model)})()
        return stub_image_model(holder)

    monkeypatch.setattr(toolbox, "resolve_image_model", resolve)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    return folder


def spec_for(images: Path, tasks: list[str], *, provider: str = "none",
             **inputs: Any) -> JobSpec:
    return JobSpec(image_folder=str(images), model="fake-model", preprocess="none",
                   tasks=tasks, nonlinear=NonlinearSpec(provider=provider), debrief=False,
                   inputs={"pixel_size_um": PIXEL_SIZE_UM, **inputs})


def create(spec: JobSpec) -> Path:
    """A job made through the internal door (``doors.jobs.create``; the
    public call is ``langslice.create_job``, ``test_the_library_can_create_a_job``)."""
    from langslice.doors.jobs import create as create_job

    opened = create_job(spec, atlas_loader=atlas_loader(), emit=lambda _m: None)
    opened.close()
    return opened.job.folder


def external_inputs() -> dict[str, Any]:
    return {"positions": dict(POSITIONS), "transforms": json.loads(json.dumps(
        EXTERNAL_TRANSFORMS)), "angles": dict(EXTERNAL_ANGLES)}


# --- the CLI -----------------------------------------------------------------------------


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(list(argv))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1, lines
    return int(code or 0), json.loads(lines[0])


def init(capsys: pytest.CaptureFixture[str], images: Path, tasks: str, *flags: str,
         provider: str = "none") -> Path:
    code, envelope = cli(capsys, str(images), "init", "--tasks", tasks,
                         "--image-provider", provider, "--preprocess", "none",
                         "--pixel-size-um", str(PIXEL_SIZE_UM), "--no-debrief", *flags)
    assert code == 0, envelope
    return Path(envelope["result"]["job_folder"])


def ok(capsys: pytest.CaptureFixture[str], images: Path, verb: str,
       *flags: str) -> dict[str, Any]:
    code, envelope = cli(capsys, str(images), verb, *flags)
    assert code == 0 and envelope["ok"] is True, envelope
    return envelope


SUBMIT_FLAGS = ("--summary", "done", "--notes", "[]", "--interval-breaks", "[]")


def transforms_file(images: Path) -> str:
    """The supplied transforms as a JSON file (inline JSON works as well:
    ``test_init_takes_a_long_inline_json``)."""
    path = images.parent / "transforms.json"
    path.write_text(json.dumps(EXTERNAL_TRANSFORMS))
    return str(path)


def test_init_takes_a_long_inline_json(capsys, images):
    long_inline = json.dumps(EXTERNAL_TRANSFORMS)
    assert len(long_inline) > 255  # longer than any file name
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", long_inline)
    spec = json.loads((job / "job.json").read_text())["spec"]
    assert spec["inputs"]["transforms"] == EXTERNAL_TRANSFORMS


def test_init_names_the_flag_of_a_bad_json_argument(capsys, images):
    code, envelope = cli(capsys, str(images), "init", "--tasks", "nonlinear",
                         "--image-provider", "none", "--positions", "no-such-file.json")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS", envelope
    assert "--positions" in envelope["error"]["message"]


# --- what a finished job must hold ---------------------------------------------------------


def assert_exported(job_folder: Path, *, residual: tuple[str, ...] = ()) -> dict[str, Any]:
    """Submitted, ``registration.json`` and the exports written, every
    section mapped with its maps; the sections in *residual* carry a warp."""
    document = json.loads((job_folder / "registration.json").read_text())
    assert document["submitted"] is True
    assert set(document["exports"]) == {"quicknii", "visualign"}
    for name in ("quicknii.json", "visualign.json"):
        assert (job_folder / "exports" / name).is_file()
    for entry, stem in zip(document["sections"], STEMS, strict=True):
        assert entry["problem"] is None, entry
        assert entry["pixel_to_atlas_um"] is not None
        for name in ("coords.tif", "labels.tif", "maps.json"):
            assert (job_folder / "sections" / stem / name).is_file(), (stem, name)
        warped = entry["parameters"]["deformation"]
        if entry["id"] in residual:
            assert warped is not None and warped["kind"] == "residual"
            assert (job_folder / "sections" / stem / "residual.tif").is_file()
    return document


def assert_external_kept(state: Any) -> None:
    """The supplied registration, verbatim: positions, angles, transforms."""
    assert state.cutting_angles_deg == EXTERNAL_ANGLES
    for name in IDS:
        record = state.by_id(name)
        assert record.position_mm == POSITIONS[name]
        assert record.flip is False and record.rotation_deg == 0
        assert record.transform == EXTERNAL_TRANSFORMS[name]



def tool_names(images: Path, tasks: list[str], provider: str = "none",
               **spec_fields: Any) -> list[str]:
    """The agent tools (ADK and MCP) a run of this combination has."""
    from langslice.agent.engine import build_context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest
    from langslice.ops.registry import enabled

    spec = replace(spec_for(images, tasks, provider=provider), **spec_fields)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    box = build_tools(ingest(spec, ctx), ctx, spec)
    assert box.names == enabled(spec)
    return box.names


# --- the tools each combination has (ADK / MCP tool building) ------------------------------


#: The tools of every run, whatever the tasks (``mark_damage`` behind ``agent_damage``).
COMMON = {"look", "zoom", "set_channel_properties", "set_preprocessed_channel_properties",
          "grep_atlas", "grep_atlas_view", "status", "list_files", "search_files",
          "read_file", "mark_damage", "note", "undo", "redo", "submit"}
LINEAR = {"interactive_transform", "elastix_affine"}


@pytest.mark.parametrize(("tasks", "provider", "extra"), [
    (["reorder", "position"], "none", {"position_sections"}),
    (["transform"], "none", LINEAR),
    (["reorder", "position", "transform"], "none", {"position_sections"} | LINEAR),
    (["reorder", "position", "transform", "nonlinear"], "openai-oauth",
     {"position_sections", "ants_syn", "trace_borders"} | LINEAR),
    (["nonlinear"], "none", {"ants_syn"}),
    (["nonlinear"], "openai-oauth", {"ants_syn", "trace_borders"}),
    (["reorder", "position", "transform", "nonlinear"], "none",
     {"position_sections", "ants_syn"} | LINEAR),
], ids=["1-positioning", "2-linear", "3-positioning+linear", "4-all+image",
        "5-nonlinear-no-image", "6-nonlinear+image", "7-all-no-image"])
def test_each_combination_builds_exactly_its_tools(images, tasks, provider, extra):
    from langslice.ops.registry import RETIRED

    names = tool_names(images, tasks, provider)
    assert set(names) == COMMON | extra
    assert not set(names) & set(RETIRED)  # a retired name never becomes a tool again


@pytest.mark.parametrize(("tasks", "fields", "absent", "present"), [
    # The stack's cutting angles left to a Linear-only run: position_sections
    # comes for the angles alone.
    (["transform"], {"transform": TransformSpec(angles=True)}, set(),
     {"position_sections"} | LINEAR),
    (["transform"], {"transform": TransformSpec(interactive=False)},
     {"interactive_transform"}, {"elastix_affine"}),
    (["transform"], {"transform": TransformSpec(automatic=False)},
     {"elastix_affine"}, {"interactive_transform"}),
    (["reorder", "position"], {"agent_damage": False}, {"mark_damage"},
     {"position_sections"}),
], ids=["angles", "no-interactive", "no-automatic", "no-agent-damage"])
def test_host_switches_add_and_remove_their_tools(images, tasks, fields, absent, present):
    names = set(tool_names(images, tasks, **fields))
    assert present <= names and not absent & names


# --- what the job statement says for each combination --------------------------------------


def _statement(images: Path, tasks: list[str], provider: str = "none",
               **inputs: Any) -> tuple[str, list[str]]:
    """The ADK job statement of a run of this combination, whitespace-normalized,
    on the job as opened (its starting positions included), and its tools."""
    from langslice.agent.engine import build_context
    from langslice.agent.prompt import build_job_statement
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import Job

    spec = spec_for(images, tasks, provider=provider, **inputs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    box = build_tools(job.state, ctx, spec, job=job)
    low, high = ctx.position_range
    text = build_job_statement(spec, job.state, tool_names=box.names, species=ctx.species,
                               pos_lo=low, pos_hi=high, axis_ends=ctx.axis_ends)
    return " ".join(text.split()), box.names


def _said(text: str, *sentences: str) -> list[str]:
    """The *sentences* (whitespace-normalized) that *text* does not hold."""
    return [s for s in sentences if " ".join(s.split()) not in text]


#: Sentences of the statement, verbatim from ``agent/prompt.py``.
POSITION_SUBMIT = ("- `submit` is refused unless every section has a position of its own, "
                   "damaged sections included; a starting position the job gave a section "
                   "does not count.")
STARTING = ("- 3 of 3 sections are at evenly spaced starting positions the job gave them, in "
            "file order; they are not placed yet (status shows position_source \"default\", "
            "the opening labels their atlas \"start\").")
ORDER = ("- The stack's order follows the positions: writing positions renumbers the "
         "sections, so use filenames to name them.")
GIVEN = "- The positions shown are given; this run does not change them."
POSITION_METHOD = (
    "- Place each section on its own evidence: compare it against candidate atlas "
    "positions before writing, and do not let the nominal interval stand in for a look.")
TRANSFORM_SUBMIT = ("- `submit` is refused unless every section carries a transform, "
                    "damaged sections included.")
DAMAGED_TRANSFORM = (
    "- A damaged section (one with marked regions) still needs a transform made for its "
    "surviving anatomy: set it by hand with `interactive_transform`, or fit it with "
    "`elastix_affine`, which leaves the marked regions out of the fit. Inspect each "
    "overlay against the surviving internal anatomy before submitting.")
TRANSFORM_METHOD = (
    "- After each automatic fit or manual adjustment, inspect the returned overlay "
    "against surviving internal anatomy.")
LEFT_LINEAR = (
    "- `submit` is refused unless every section carries a deformation applied at its "
    "current placement, or is named in its `left_linear` with the reason its linear "
    "placement stands, damaged sections included.")
TRACE_OPTIONAL = ("- `trace_borders` is optional: you decide which sections, if any, to "
                  "trace. A trace requires a position and a linear transform.")
MARK_FIRST = ("- Before fitting a damaged section, mark the regions it has lost with "
              "`mark_damage`, so that every region it still has drives its fits.")
FIT_METHOD = "- Fit each section whole first, then refine regions."
FIT_ENDING = ("; undo a fit that does not improve the alignment, and name in submit's "
              "`left_linear` only a section that no fit improves.")
FIXED_LINEAR = ("- Existing linear transforms are supplied and fixed for this run, "
                "orientation included.")


@pytest.mark.parametrize(("tasks", "provider", "inputs", "said", "unsaid"), [
    (["reorder", "position"], "none", {},
     (POSITION_SUBMIT, STARTING, ORDER, POSITION_METHOD),
     (TRANSFORM_SUBMIT, DAMAGED_TRANSFORM, LEFT_LINEAR, GIVEN, MARK_FIRST)),
    (["transform"], "none", {"positions": dict(POSITIONS)},
     (TRANSFORM_SUBMIT, DAMAGED_TRANSFORM, TRANSFORM_METHOD, GIVEN),
     (POSITION_SUBMIT, STARTING, ORDER, POSITION_METHOD, LEFT_LINEAR)),
    (["reorder", "position", "transform"], "none", {},
     (POSITION_SUBMIT, STARTING, ORDER, TRANSFORM_SUBMIT, DAMAGED_TRANSFORM,
      POSITION_METHOD, TRANSFORM_METHOD),
     (LEFT_LINEAR, GIVEN)),
    (["reorder", "position", "transform", "nonlinear"], "openai-oauth", {},
     (POSITION_SUBMIT, TRANSFORM_SUBMIT, LEFT_LINEAR, TRACE_OPTIONAL, MARK_FIRST,
      FIT_METHOD, " and, where traced, its traced borders" + FIT_ENDING),
     (GIVEN,)),
    (["nonlinear"], "none", "external",
     (LEFT_LINEAR, MARK_FIRST, FIT_METHOD, FIXED_LINEAR, GIVEN,
      "Inspect each fit's borders against the section's internal anatomy" + FIT_ENDING),
     (POSITION_SUBMIT, TRANSFORM_SUBMIT, TRACE_OPTIONAL, STARTING, "where traced")),
    (["nonlinear"], "openai-oauth", "external",
     (LEFT_LINEAR, TRACE_OPTIONAL, MARK_FIRST, FIXED_LINEAR, GIVEN),
     (POSITION_SUBMIT, TRANSFORM_SUBMIT)),
], ids=["1-positioning", "2-linear", "3-positioning+linear", "4-all+image",
        "5-nonlinear-no-image", "6-nonlinear+image"])
def test_each_combination_states_its_tools_constraints_and_method(
    images, tasks, provider, inputs, said, unsaid,
):
    from langslice.ops.registry import RETIRED

    text, names = _statement(images, tasks, provider,
                             **(external_inputs() if inputs == "external" else inputs))
    # The tools by name only, in the order they are built.
    tools = ("Tools (each one's own description says what it does and returns): "
             + ", ".join(f"`{name}`" for name in names) + ".")
    assert tools in text
    assert not [name for name in RETIRED if f"`{name}`" in text]
    assert _said(text, *said) == []
    assert [s for s in unsaid if " ".join(s.split()) in text] == []


def test_the_statement_says_a_users_damage_note_marks_no_regions(images):
    text, _names = _statement(images, ["reorder", "position", "transform"],
                              damaged={ID2: "torn"})
    assert _said(text, "- The user noted damage on these sections (damage_note in status; "
                 "no regions are marked from it): s2.png.") == []
    assert "Sections with marked damage regions" not in text


def test_a_required_deformation_drops_left_linear_from_the_statement(images):
    from langslice.agent.engine import build_context
    from langslice.agent.prompt import build_job_statement
    from langslice.job.job import Job

    spec = spec_for(images, ["nonlinear"], **external_inputs())
    spec = replace(spec, nonlinear=replace(spec.nonlinear, require_deformation=True))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    low, high = ctx.position_range
    text = " ".join(build_job_statement(
        spec, job.state, tool_names=enabled_names(spec), species=ctx.species, pos_lo=low,
        pos_hi=high, axis_ends=ctx.axis_ends).split())
    assert _said(text, "- `submit` is refused unless every section carries a deformation "
                 "applied at its current placement, damaged sections included; the user "
                 "requires one on every section.",
                 "; undo a fit that does not improve the alignment.") == []
    assert "left_linear" not in text


def enabled_names(spec: JobSpec) -> list[str]:
    from langslice.ops.registry import enabled

    return enabled(spec)


# --- 1. Positioning only ------------------------------------------------------------------


REVERSED = {ID0: 0.2, ID1: 0.15, ID2: 0.1}


def _sections(positions: dict[str, float]) -> list[dict[str, Any]]:
    return [{"id": name, "position_mm": mm} for name, mm in positions.items()]


def test_1_positioning_only_through_the_library(images):
    import langslice

    create(spec_for(images, ["reorder", "position"]))
    with langslice.open_job(images) as job:
        # The order follows the positions: written back to front, the stack is renumbered.
        backwards = job.position_sections(sections=_sections(REVERSED))
        assert backwards["status"] == "ok" and backwards["order"] == [ID2, ID1, ID0]
        assert backwards["reordered"]
        placed = job.position_sections(sections=_sections(POSITIONS))
        assert placed["status"] == "ok" and placed["order"] == list(IDS)
        assert job.submit(summary="placed", notes=[], interval_breaks=[])["status"] == "ok"
        assert all(record.transform is None for record in job.state.slices)
        assert [record.position_source for record in job.state.in_order()] != ["default"] * 3
    document = assert_exported(images / "langslice")
    assert {entry["mapping"] for entry in document["sections"]} == {
        "linear (identity in-plane: no transform written)"}
    assert [entry["parameters"]["plane"]["starting_position"]
            for entry in document["sections"]] == [False] * 3


def test_1_a_starting_position_is_not_a_position_until_it_is_written(images):
    import langslice

    create(spec_for(images, ["reorder", "position"]))
    with langslice.open_job(images) as job:
        refused = job.submit(summary="placed", notes=[], interval_breaks=[])
        assert refused["error"] == "MISSING_POSITIONS", refused
        start = {name: job.state.by_id(name).position_mm for name in IDS}
        assert all(job.state.by_id(name).position_source == "default" for name in IDS)
        document = json.loads((images / "langslice" / "registration.json").read_text())
        assert [entry["parameters"]["plane"]["starting_position"]
                for entry in document["sections"]] == [True] * 3
        # The same value written again is a position of the section's own.
        assert job.position_sections(sections=_sections(start))["status"] == "ok"
        assert all(job.state.by_id(name).position_source != "default" for name in IDS)
        assert job.submit(summary="placed", notes=[], interval_breaks=[])["status"] == "ok"


def test_1_positioning_only_through_the_cli(capsys, images):
    job = init(capsys, images, "reorder,position")
    backwards = ok(capsys, images, "position-sections", "--sections",
                   json.dumps(_sections(REVERSED)))
    assert backwards["result"]["order"] == [ID2, ID1, ID0]
    ok(capsys, images, "position_sections", "--sections", json.dumps(_sections(POSITIONS)))
    envelope = ok(capsys, images, "submit", *SUBMIT_FLAGS)
    assert {"registration", "quicknii", "visualign"} <= {a["kind"] for a in envelope["artifacts"]}
    assert_exported(job)


def _mcp(server: Any, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    async def body() -> list[dict[str, Any]]:
        replies = []
        async with create_connected_server_and_client_session(server) as client:
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                first = result.content[0]
                assert isinstance(first, TextContent), result
                replies.append(json.loads(first.text))
        return replies

    return asyncio.run(body())


def _mcp_server(spec: JobSpec) -> Any:
    from langslice.doors.mcp.server import build_server

    return build_server(lambda _folder: spec, spec.image_folder, atlas_loader=atlas_loader())


def test_1_positioning_only_through_mcp(images):
    server = _mcp_server(spec_for(images, ["reorder", "position"]))
    replies = _mcp(server, [
        ("position_sections", {"sections": _sections(POSITIONS)}),
        ("submit", {"summary": "placed", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 2. Linear without positioning (positions supplied) -------------------------------------


def test_2_linear_on_supplied_positions_through_the_library(images):
    import langslice

    create(spec_for(images, ["transform"], positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        assert "position_sections" not in job.verbs
        fitted = job.elastix_affine(sections=[ID0])
        assert fitted["status"] == "ok" and fitted["results"][0]["status"] == "ok", fitted
        sections = [{"id": name, **IDENTITY} for name in (ID1, ID2)]
        assert job.interactive_transform(sections=sections)["status"] == "ok"
        assert job.submit(summary="aligned", notes=[], interval_breaks=[])["status"] == "ok"
        assert {r.id: r.position_mm for r in job.state.slices} == POSITIONS
        assert all(record.transform for record in job.state.slices)
    assert_exported(images / "langslice")


def test_2_linear_on_supplied_positions_through_the_cli(capsys, images):
    job = init(capsys, images, "transform", "--positions", json.dumps(POSITIONS))
    code, envelope = cli(capsys, str(images), "position_sections", "--sections", "[]")
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    code, envelope = cli(capsys, str(images), "submit", *SUBMIT_FLAGS)
    assert code == 3 and envelope["error"]["code"] == "MISSING_TRANSFORMS"
    ok(capsys, images, "interactive_transform", "--sections", json.dumps(
        [{"id": name, **IDENTITY} for name in IDS]))
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    document = assert_exported(job)
    assert [e["parameters"]["plane"]["position_mm"] for e in document["sections"]] == list(
        POSITIONS.values())


def test_2_linear_on_supplied_positions_through_mcp(images):
    server = _mcp_server(spec_for(images, ["transform"], positions=dict(POSITIONS)))
    replies = _mcp(server, [
        ("interactive_transform", {"sections": [{"id": name, **IDENTITY}
                                                for name in IDS[:2]]}),
        ("interactive_transform", {"sections": [{"id": ID2, **IDENTITY}]}),
        ("submit", {"summary": "aligned", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 3. Positioning + Linear ---------------------------------------------------------------


def test_3_positioning_and_linear_through_the_library(images):
    import langslice

    create(spec_for(images, ["reorder", "position", "transform"]))
    with langslice.open_job(images) as job:
        assert job.position_sections(sections=_sections(POSITIONS))["status"] == "ok"
        flipped = job.interactive_transform(sections=[{"id": ID1, "flip": True}])
        assert flipped["status"] == "ok", flipped
        fitted = job.elastix_affine(sections=list(IDS))
        assert [row["status"] for row in fitted["results"]] == ["ok"] * 3, fitted
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        assert job.state.by_id(ID1).flip is True
    document = assert_exported(images / "langslice")
    assert document["sections"][1]["parameters"]["orientation"]["flip"] is True


def test_3_positioning_and_linear_through_the_cli(capsys, images):
    job = init(capsys, images, "reorder,position,transform")
    ok(capsys, images, "position_sections", "--sections", json.dumps(_sections(POSITIONS)))
    ok(capsys, images, "elastix_affine", "--sections", ID0, "--sections", ID1,
       "--sections", ID2)
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    assert_exported(job)


def test_3_positioning_and_linear_through_mcp(images):
    server = _mcp_server(spec_for(images, ["reorder", "position", "transform"]))
    replies = _mcp(server, [
        ("position_sections", {"sections": _sections(POSITIONS)}),
        ("elastix_affine", {"sections": list(IDS)}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 4. Nonlinear with an image model, no agent -------------------------------------------


def _left_linear(*names: str) -> list[dict[str, str]]:
    return [{"id": name, "reason": "kept"} for name in names]


def test_4_nonlinear_verbs_called_directly_through_the_library(images):
    import langslice

    create(spec_for(images, ["reorder", "position", "transform", "nonlinear"],
                    provider="openai-oauth"))
    with langslice.open_job(images) as job:
        assert job.position_sections(sections=_sections(POSITIONS))["status"] == "ok"
        assert job.interactive_transform(sections=[{"id": n, **IDENTITY} for n in IDS])[
            "status"] == "ok"
        for name in (ID0, ID1):  # started in the background; submit waits for them
            started = job.trace_borders(section=name)
            assert started["status"] == "started" and started["work"], started
        fit = job.ants_syn(sections=[ID2])
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        submitted = job.submit(summary="done", notes=[], interval_breaks=[])
        assert submitted["status"] == "ok", submitted
        assert job.export_maps()["status"] == "ok"
        assert all((job.state.by_id(name).image_correction or {}).get("status") == "ok"
                   for name in (ID0, ID1))
        assert job.state.by_id(ID2).image_correction is None
    assert_exported(images / "langslice", residual=IDS)


def test_4_nonlinear_verbs_called_directly_through_the_cli(capsys, images):
    job = init(capsys, images, "reorder,position,transform,nonlinear",
               provider="openai-oauth")
    ok(capsys, images, "position_sections", "--sections", json.dumps(_sections(POSITIONS)))
    ok(capsys, images, "interactive_transform", "--sections", json.dumps(
        [{"id": name, **IDENTITY} for name in IDS]))
    code, envelope = cli(capsys, str(images), "submit", *SUBMIT_FLAGS)
    assert code == 3 and envelope["error"]["code"] == "MISSING_DEFORMATIONS"
    # The CLI answers once the trace's work has landed.
    ok(capsys, images, "trace-borders", "--section", ID0)
    ok(capsys, images, "ants-syn", "--sections", ID1)
    ok(capsys, images, "submit", *SUBMIT_FLAGS, "--left-linear", json.dumps(_left_linear(ID2)))
    ok(capsys, images, "export_maps")
    document = assert_exported(job, residual=(ID0, ID1))
    assert document["sections"][2]["parameters"]["deformation"] == {
        "kind": "none", "reason": "kept"}


def test_4_the_library_takes_an_image_model(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    model = stub_image_model(spec_for(images, ["nonlinear"], provider="openai-oauth"))
    with langslice.open_job(images, image_model=model) as job:
        assert job.trace_borders(section=ID0)["status"] == "started"
        job.job.background.wait_all()
        # The model handed in answered (no network): the trace landed and was fitted.
        record = job.state.by_id(ID0)
        assert (record.image_correction or {}).get("status") == "ok", record.image_correction
        assert record.deformation is not None and len(record.deformation["steps"]) == 1


# --- 5. Nonlinear without an image model --------------------------------------------------


def test_5_nonlinear_without_an_image_model_through_the_library(images):
    import langslice

    create(spec_for(images, ["nonlinear"], **external_inputs()))
    with langslice.open_job(images) as job:
        assert "trace_borders" not in job.verbs
        fit = job.ants_syn(sections=[ID0])
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        submitted = job.submit(summary="done", notes=[], interval_breaks=[],
                               left_linear=_left_linear(ID1, ID2))
        assert submitted["status"] == "ok", submitted
        assert submitted["left_linear"] == {ID1: "kept", ID2: "kept"}
        assert_external_kept(job.state)
    assert_exported(images / "langslice", residual=(ID0,))


def test_5_nonlinear_without_an_image_model_through_the_cli(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images),
               "--pitch", str(EXTERNAL_ANGLES["pitch"]), "--yaw", str(EXTERNAL_ANGLES["yaw"]))
    code, envelope = cli(capsys, str(images), "trace_borders", "--section", ID0)
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    ok(capsys, images, "ants_syn", "--sections", ID0)
    ok(capsys, images, "submit", *SUBMIT_FLAGS,
       "--left-linear", json.dumps(_left_linear(ID1, ID2)))
    assert_exported(job, residual=(ID0,))


def test_5_nonlinear_without_an_image_model_through_mcp(images):
    server = _mcp_server(spec_for(images, ["nonlinear"], **external_inputs()))
    replies = _mcp(server, [
        ("ants_syn", {"sections": [ID0]}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": [],
                    "left_linear": _left_linear(ID1, ID2)}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
    assert_exported(images / "langslice", residual=(ID0,))


def test_nonlinear_on_positions_alone_is_refused_until_a_transform_is_written(images):
    """Positions without any in-plane transform: the maps treat the missing
    transform as the identity (``core.maps.placement_problem``), but the
    nonlinear step needs a WRITTEN one (``core.handoff.prepare_linear_registration``)
    and leaving a section linear refuses too, so with Linear off the host must
    supply transforms (or ``locked``, which writes the ``host`` identity)."""
    import langslice

    create(spec_for(images, ["nonlinear"], positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        fit = job.ants_syn(sections=[ID0])
        assert fit["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT", fit
        # The refusal says what to do: supply transforms, or switch Linear on.
        message = fit["results"][0]["message"]
        assert "--transforms" in message and "inputs.transforms" in message, message
        assert "Linear on" in message, message
    fresh = spec_for(images, ["nonlinear"], positions=dict(POSITIONS), locked=list(IDS))
    fresh.resume = False  # a new job on the same images, not the one above
    create(fresh)
    with langslice.open_job(images) as job:
        assert job.submit(summary="done", notes=[], interval_breaks=[],
                          left_linear=_left_linear(*IDS))["status"] == "ok"


def test_leaving_a_section_linear_on_positions_alone_is_refused(images):
    import langslice

    create(spec_for(images, ["nonlinear"], positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        kept = job.submit(summary="done", notes=[], interval_breaks=[],
                          left_linear=_left_linear(*IDS))
        assert kept["status"] != "ok" and not job.state.submitted, kept
        assert "--transforms" in json.dumps(kept) and "Linear on" in json.dumps(kept), kept


def test_tracing_on_positions_alone_says_what_to_do(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        traced = job.trace_borders(section=ID0)
        assert traced["error"] == "INVALID_LINEAR_PLACEMENT", traced
        assert "--transforms" in traced["message"] and "Linear on" in traced["message"]


def test_the_missing_transform_hint_names_the_tool_that_writes_one(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        messages = [job.ants_syn(sections=[ID0])["results"][0]["message"],
                    job.trace_borders(section=ID0)["message"],
                    job.trace_from_atlas(slices=[ID0])["results"][0]["message"]]
    for message in messages:
        assert "elastix_affine" in message and "fit_affine" not in message, message


# --- 6. Nonlinear on a registration made elsewhere ------------------------------------------


def test_6_external_registration_then_nonlinear_through_the_library(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    with langslice.open_job(images) as job:
        assert not {"position_sections", "interactive_transform",
                    "elastix_affine"} & set(job.verbs)
        assert_external_kept(job.state)
        started = job.trace_borders(section=ID0)
        assert started["status"] == "started", started
        status = job.status()
        assert [work["id"] for work in status.get("background_running", [])] in (
            [started["work"]], [])  # it may land before status reads it
        fit = job.ants_syn(sections=[ID1])
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        submitted = job.submit(summary="done", notes=[], interval_breaks=[],
                               left_linear=_left_linear(ID2))
        assert submitted["status"] == "ok", submitted
        assert_external_kept(job.state)  # the linear placement, verbatim, after submit
        # A trace whose fit has landed is not fitted again.
        again = job.trace_borders(section=ID0)
        assert again["status"] == "ok" and again["landed"] is True, again
    document = assert_exported(images / "langslice", residual=(ID0, ID1))
    assert document["cutting_angles_deg"] == EXTERNAL_ANGLES
    assert [e["parameters"]["affine"]["params"] for e in document["sections"]] == [
        EXTERNAL_TRANSFORMS[name]["params"] for name in IDS]
    assert document["sections"][1]["parameters"]["affine"]["mirrored"] is True


def test_6_background_notices_open_the_next_reply_with_their_pictures(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    with langslice.open_job(images) as job:
        started = job.trace_borders(section=ID0)
        assert started["status"] == "started"
        job.job.background.wait_all()
        reply = job.status()
        assert len(reply["background"]) == 1, reply
        notice = reply["background"][0]
        assert notice.startswith(f"{started['work']} trace_borders of {ID0} finished:")
        work = [entry for entry in reply["pictures"] if entry.get("work") == started["work"]]
        assert work and notice.endswith(
            "Pictures " + ", ".join(f"#{entry['id']}" for entry in work) + ".")
        assert "background" not in job.status()  # each notice is handed out once


def test_6_external_registration_then_nonlinear_through_the_cli(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images),
               "--pitch", str(EXTERNAL_ANGLES["pitch"]), "--yaw", str(EXTERNAL_ANGLES["yaw"]),
               provider="openai-oauth")
    for verb in ("position_sections", "interactive_transform", "elastix_affine"):
        code, envelope = cli(capsys, str(images), verb, "--args", "{}")
        assert code == 3 and envelope["error"]["code"] == "VERB_OFF", envelope
    code, envelope = cli(capsys, str(images), "fit_deformable", "--args", "{}")
    assert envelope["error"]["code"] == "RETIRED_TOOL", envelope
    assert envelope["result"]["use"] == "ants_syn"
    for name in (ID0, ID1):
        ok(capsys, images, "trace_borders", "--section", name)
    ok(capsys, images, "submit", *SUBMIT_FLAGS, "--left-linear", json.dumps(_left_linear(ID2)))
    ok(capsys, images, "export_maps")
    from langslice.job.checkpoint import load_checkpoint

    state = load_checkpoint(str(job / "state.json"))
    assert state is not None
    assert_external_kept(state)
    assert_exported(job, residual=(ID0, ID1))


def test_6_an_agent_cannot_move_a_supplied_placement(images, monkeypatch):
    """Linear OFF means the supplied placement stays verbatim: the agent run
    has no verb that writes a position, an orientation, a transform or the
    angles; only ``undo`` past the first step could, and there is none."""
    assert "position_sections" not in tool_names(images, ["nonlinear"])
    install_script(monkeypatch, [
        ("submit", {"summary": "done", "notes": [], "interval_breaks": [],
                    "left_linear": _left_linear(*IDS)}),
    ])
    state = run_agent(spec_for(images, ["nonlinear"], **external_inputs()))
    assert state.submitted is True
    assert_external_kept(state)
    assert_exported(images / "langslice")


def test_6_a_supplied_orientation_is_kept(images):
    import langslice

    inputs = {**external_inputs(), "orientation": {ID1: {"flip": True, "rotation_deg": 90}}}
    create(spec_for(images, ["nonlinear"], **inputs))
    with langslice.open_job(images) as job:
        assert job.state.by_id(ID1).flip is True
        assert job.state.by_id(ID1).rotation_deg == 90
        assert job.state.by_id(ID1).transform == EXTERNAL_TRANSFORMS[ID1]  # kept as supplied
        assert job.state.by_id(ID0).flip is False and job.state.by_id(ID0).rotation_deg == 0
        assert job.submit(summary="done", notes=[], interval_breaks=[],
                          left_linear=_left_linear(*IDS))["status"] == "ok"
    document = assert_exported(images / "langslice")
    oriented = document["sections"][1]["parameters"]["orientation"]
    assert (oriented["flip"], oriented["rotation_deg"]) == (True, 90)


@pytest.mark.parametrize("bad", [
    {"flip": "yes"}, {"rotation_deg": 45}, {"rotate_deg": 90}, {"flip": True, "turn": 1}])
def test_a_bad_supplied_orientation_is_refused(images, bad):
    with pytest.raises(ValueError, match="orientation"):
        create(spec_for(images, ["nonlinear"], **external_inputs(), orientation={ID1: bad}))


def test_cli_init_takes_an_orientation(capsys, images):
    orientation = {ID2: {"flip": True}, ID0: {"rotation_deg": 180}}
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images), "--orientation", json.dumps(orientation))
    from langslice.job.checkpoint import load_checkpoint

    state = load_checkpoint(str(job / "state.json"))
    assert state is not None
    assert (state.by_id(ID2).flip, state.by_id(ID2).rotation_deg) == (True, 0)
    assert (state.by_id(ID0).flip, state.by_id(ID0).rotation_deg) == (False, 180)


def test_6_an_unknown_input_is_refused(images):
    with pytest.raises(ValueError, match="transfroms") as refused:
        create(spec_for(images, ["nonlinear"], positions=dict(POSITIONS),
                        transfroms=EXTERNAL_TRANSFORMS))
    assert "'transforms'" in str(refused.value)  # the allowed keys are named
    with pytest.raises(ValueError, match="transfroms"):  # a saved spec too
        JobSpec.from_dict({"image_folder": str(images), "inputs": {"transfroms": {}}})


def test_6_a_second_init_with_new_inputs_is_not_silently_ignored(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS))
    moved = {name: round(mm + 0.02, 6) for name, mm in POSITIONS.items()}
    flags = ("--tasks", "nonlinear", "--image-provider", "none", "--preprocess", "none",
             "--pixel-size-um", str(PIXEL_SIZE_UM), "--no-debrief")

    def positions() -> dict[str, float]:
        _code, status = cli(capsys, str(images), "status")
        return {row["id"]: row["position_mm"] for row in status["result"]["rows"]}

    # Different inputs: refused, naming --fresh; nothing rewritten.
    code, envelope = cli(capsys, str(images), "init", *flags,
                         "--positions", json.dumps(moved))
    assert code == 3 and envelope["error"]["code"] == "INPUTS_CHANGED", envelope
    assert "positions" in envelope["error"]["message"]
    assert "--fresh" in envelope["error"]["message"] and "--fresh" in envelope["error"]["fix"]
    spec = json.loads((job / "job.json").read_text())["spec"]
    assert spec["inputs"]["positions"] == POSITIONS
    assert positions() == POSITIONS
    # The same inputs: resumed as before.
    code, envelope = cli(capsys, str(images), "init", *flags,
                         "--positions", json.dumps(POSITIONS))
    assert code == 0, envelope
    # --fresh: a new job from the new inputs.
    code, envelope = cli(capsys, str(images), "init", *flags, "--fresh",
                         "--positions", json.dumps(moved))
    assert code == 0, envelope
    assert positions() == moved


def test_6_cli_init_takes_locked_and_damaged_sections(capsys, images):
    damaged = images.parent / "damaged.json"
    damaged.write_text(json.dumps({ID2: "torn"}))
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--locked", json.dumps(list(IDS)), "--damaged", str(damaged))
    spec = json.loads((job / "job.json").read_text())["spec"]
    assert spec["inputs"]["locked"] == list(IDS)
    assert spec["inputs"]["damaged"] == {ID2: "torn"}
    from langslice.job.checkpoint import load_checkpoint

    state = load_checkpoint(str(job / "state.json"))
    assert state is not None
    # The host's damage note is the section's note; it marks no regions, so the
    # section is not damaged.
    record = state.by_id(ID2)
    assert record.damage_note == "torn" and record.damaged is False
    assert record.damaged_regions == []
    rows = {row["id"]: row for row in ok(capsys, images, "status")["result"]["rows"]}
    assert rows[ID2]["damage_note"] == "torn" and rows[ID2]["damaged"] is False
    # Locked sections carry the host identity: they can be left linear and submitted.
    ok(capsys, images, "submit", *SUBMIT_FLAGS,
       "--left-linear", json.dumps(_left_linear(*IDS)))


def test_6_a_quicknii_registration_can_be_imported(capsys, images, tmp_path):
    # A job placed by LangSlice exports quicknii.json; a new job from it must
    # land at the same placement.
    source = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
                  "--transforms", transforms_file(images))
    ok(capsys, images, "export_maps")
    quicknii = tmp_path / "quicknii.json"
    shutil.copy(source / "exports" / "quicknii.json", quicknii)
    code, envelope = cli(capsys, str(images), "init", "--tasks", "nonlinear",
                         "--image-provider", "none", "--job-dir", str(tmp_path / "again"),
                         "--registration", str(quicknii))
    assert code == 0, envelope
    result = envelope["result"]
    assert result["tasks"] == ["nonlinear"]  # the imported placement kept as it is
    assert [row["id"] for row in result["registration"]["sections"]] == list(IDS)
    assert result["registration"]["format"] == "quicknii-json"
    # The new job maps every section's file exactly as the source job did
    # (QuickNII anchorings are rounded to 1e-6 voxels).
    before = json.loads((source / "registration.json").read_text())["sections"]
    after = json.loads((tmp_path / "again" / "registration.json").read_text())["sections"]
    for old, new in zip(before, after, strict=True):
        assert new["problem"] is None, new
        assert np.allclose(new["pixel_to_atlas_um"], old["pixel_to_atlas_um"], atol=1e-3)


def connected(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    """Whether the job's image model counts as connected to LangSlice (its key
    or login present), as the MCP door and ABBA's Claude mode ask it; no credential
    is read."""
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "image_model_connected", lambda _provider: value)


def test_6_external_registration_then_nonlinear_through_mcp(images, monkeypatch):
    connected(monkeypatch, True)
    server = _mcp_server(spec_for(images, ["nonlinear"], provider="openai-oauth",
                                  **external_inputs()))
    replies = _mcp(server, [
        *[("trace_borders", {"section": name}) for name in IDS],
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies[:3]] == ["started"] * 3, replies
    assert replies[3]["status"] == "ok", replies  # submit waits for the three traces
    assert_exported(images / "langslice", residual=IDS)


def _mcp_listing_and_statement(server: Any) -> tuple[dict[str, str], str]:
    """The tools the MCP door lists (name -> description) and its statement."""
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    async def body() -> tuple[dict[str, str], str]:
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            started = await client.call_tool("start_job", {})
            first = started.content[0]
            assert isinstance(first, TextContent), started
            # A host reads every opening page before it writes (the gate).
            import re

            pages = int(re.findall(r"show_stack\(page=(\d+)\)", first.text)[-1])
            for page in range(1, pages + 1):
                await client.call_tool("show_stack", {"page": page})
            return {tool.name: tool.description or "" for tool in listed.tools}, first.text

    return asyncio.run(body())


def _mcp_names_and_statement(server: Any) -> tuple[set[str], str]:
    listed, statement = _mcp_listing_and_statement(server)
    return set(listed), statement


@pytest.mark.parametrize("linked", [True, False], ids=["connected", "not-connected"])
def test_mcp_offers_trace_borders_only_with_a_connected_image_model(images, monkeypatch,
                                                                    linked):
    """Over MCP the fitting verbs come with the nonlinear task; the image-model
    verb only when the job's provider is not none AND its key or login is
    present. Without it, trace_borders is simply not listed, the statement
    says why, and submit does not wait for traces."""
    from langslice.doors.statement import IMAGE_MODEL_OFF

    connected(monkeypatch, linked)
    server = _mcp_server(spec_for(images, ["nonlinear"], provider="openai-oauth",
                                  **external_inputs()))
    listed, statement = _mcp_listing_and_statement(server)
    names = set(listed)
    assert {"grep_atlas", "ants_syn", "look"} <= names
    assert ("trace_borders" in names) is linked
    assert "trace_from_atlas" not in names  # a hidden scripting verb, never a model's
    assert (IMAGE_MODEL_OFF in statement) is not linked
    # The base image prompt is in trace_borders' own description, never the statement.
    if linked:
        assert "The base prompt (its image numbers" in " ".join(listed["trace_borders"].split())
    if not linked:
        replies = _mcp(server, [
            ("ants_syn", {"sections": [ID0]}),
            ("submit", {"summary": "done", "notes": [], "interval_breaks": [],
                        "left_linear": _left_linear(ID1, ID2)}),
        ])
        assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
        spec = json.loads((images / "langslice" / "job.json").read_text())["spec"]
        assert spec["nonlinear"]["provider"] == "openai-oauth"  # the job keeps its provider


def test_mcp_with_provider_none_never_asks_for_an_image_model(images, monkeypatch):
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "image_model_connected", _no_network)
    names, statement = _mcp_names_and_statement(
        _mcp_server(spec_for(images, ["nonlinear"], **external_inputs())))
    assert "trace_borders" not in names and "ants_syn" in names
    assert "image-model tool (trace_borders) is off" not in statement


@pytest.mark.parametrize("linked", [True, False], ids=["connected", "not-connected"])
def test_the_saved_job_prompt_takes_the_nonlinear_task(images, monkeypatch, linked):
    from langslice.doors.api.saved_jobs import copy_prompt

    connected(monkeypatch, linked)
    prompt = copy_prompt("0" * 12, spec_for(images, ["nonlinear"], provider="openai-oauth",
                                            **external_inputs()))
    assert "nonlinear (deformable) alignment" in prompt
    assert ("image-model tool (trace_borders) is off" in prompt) is not linked


def test_the_library_can_create_a_job(images):
    import langslice

    spec = spec_for(images, ["nonlinear"], **external_inputs())
    with langslice.create_job(spec) as job:
        assert_external_kept(job.state)


# --- 7. The agent: every task, no image model -----------------------------------------------


def install_script(monkeypatch: pytest.MonkeyPatch, script: list[tuple[str, dict[str, Any]]],
                   ) -> None:
    """The ADK model is a script: one tool call per turn, in order, whatever
    the replies, then text."""
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.models.registry import LLMRegistry
    from google.genai import types

    class ScriptLlm(BaseLlm):
        calls: list[Any] = []

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False,
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            done = sum(1 for content in llm_request.contents or []
                       for part in content.parts or []
                       if getattr(part, "function_response", None) is not None)
            if done < len(self.calls):
                name, arguments = self.calls[done]
                part = types.Part.from_function_call(name=name, args=arguments)
            else:
                part = types.Part.from_text(text="Done.")
            yield LlmResponse(
                content=types.Content(role="model", parts=[part]), partial=False,
                turn_complete=True,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=0, candidates_token_count=1))

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(
        lambda model: ScriptLlm(model=model, calls=list(script))))


def run_agent(spec: JobSpec) -> Any:
    from langslice.agent.engine import run

    return asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=atlas_loader()))


def test_7_the_agent_with_every_task_and_no_image_model(images, monkeypatch):
    install_script(monkeypatch, [
        ("position_sections", {"sections": _sections(POSITIONS)}),
        ("interactive_transform", {"sections": [{"id": name, **IDENTITY} for name in IDS]}),
        ("ants_syn", {"sections": [ID0]}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": [],
                    "left_linear": _left_linear(ID1, ID2)}),
    ])
    state = run_agent(spec_for(images, ["reorder", "position", "transform", "nonlinear"]))
    assert state.submitted is True
    assert all(record.deformation for record in state.slices)
    assert_exported(images / "langslice", residual=(ID0,))
    results = json.loads((images / "langslice" / "exports" / "linear_results.json").read_text())
    assert results["submitted"] is True


def test_7_every_task_and_no_image_model_through_mcp(images):
    server = _mcp_server(spec_for(images, ["reorder", "position", "transform", "nonlinear"]))
    names, _statement = _mcp_names_and_statement(server)
    assert {"position_sections", "elastix_affine", "ants_syn", "grep_atlas"} <= names
    assert "trace_borders" not in names
    replies = _mcp(server, [
        ("position_sections", {"sections": _sections(POSITIONS)}),
        ("interactive_transform", {"sections": [{"id": name, **IDENTITY} for name in IDS]}),
        ("ants_syn", {"sections": [ID0]}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": [],
                    "left_linear": _left_linear(ID1, ID2)}),
    ])
    assert [reply["status"] for reply in replies] == ["ok"] * 4, replies
    assert_exported(images / "langslice", residual=(ID0,))


# --- 8. The placement-free trace: a hidden scripting verb -----------------------------------


def test_8_trace_from_atlas_through_the_library_then_fit_and_submit(images):
    """Route "atlas" as a job verb: called by name, listed nowhere; its reply
    is recorded as trace_borders' is, so a script's traced fit
    (``ops.deformable.fit_deformable`` with the traced borders), the submit
    gate and the maps read it unchanged."""
    import langslice
    from langslice.ops.deformable import fit_deformable
    from langslice.ops.traces import TRACE_FIT

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    with langslice.open_job(images) as job:
        assert "trace_from_atlas" not in job.verbs and "trace_from_atlas" not in dir(job)
        traced = job.trace_from_atlas(slices=[ID0], passes=2)
        assert traced["status"] == "ok", traced
        assert traced["results"][0]["status"] == "running" and traced["results"][0]["started"]
        rest = job.trace_from_atlas(slices=[ID1, ID2])
        assert [row["status"] for row in rest["results"]] == ["running", "running"], rest
        fit = fit_deformable(job.job, job.workspace, [job.state.by_id(ID0)], TRACE_FIT)
        assert fit.rows[0]["status"] == "ok" and fit.rows[0]["written"] is True, fit.rows
        submitted = job.submit(summary="done", notes=[], interval_breaks=[],
                               left_linear=_left_linear(ID1, ID2))
        assert submitted["status"] == "ok", submitted
        held = job.state.by_id(ID0).image_correction
        assert held["status"] == "ok" and held["trace_route"] == "atlas"
        assert held["passes"] == 2 and held["model_calls"] == 2
        attempt = images / "langslice" / held["artifact_dir"]
        for name in ("input_slice.png", "outlined_atlas.png", "raw_reply.png",
                     "extracted_lines.png", "lines_on_original.png", "prompt.txt",
                     "pass2_prompt.txt", "pass1_raw_correction.png",
                     "pass1_lines_on_tissue.png", "request.json", "result.json"):
            assert (attempt / name).is_file(), name
        request = json.loads((attempt / "request.json").read_text())
        assert request["trace_route"] == "atlas" and len(request["atlas_to_canvas"]) == 3
        # The model is never shown the placement: clean tissue and the atlas only.
        assert [item["role"] for item in request["attachments"]] == [
            "Image 1: clean photograph", "Image 2: outlined atlas"]
        # The same call again reuses the first reply at this geometry.
        again = job.trace_from_atlas(slices=[ID0], passes=2)
        assert again["results"][0]["cached"] is True and not again["results"][0]["started"]
    assert_exported(images / "langslice", residual=(ID0,))
    card = (images / "langslice" / "AGENTS.md").read_text()
    assert "`trace_from_atlas`" not in card and "`trace_borders`" in card


def test_8_trace_from_atlas_through_the_cli_is_callable_and_unlisted(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images), provider="openai-oauth")
    assert "`trace_from_atlas`" not in (job / "CLAUDE.md").read_text()
    dry = ok(capsys, images, "trace-from-atlas", "--slices", ID0, "--dry-run")
    assert dry["result"]["simulated"] is False
    envelope = ok(capsys, images, "trace-from-atlas", "--slices", ID0, "--slices", ID1)
    rows = envelope["result"]["results"]
    # The CLI settles the calls before answering: each landed outcome is shown.
    assert [row["image_correction"]["status"] for row in rows] == ["ok", "ok"], rows
    status = ok(capsys, images, "status")
    assert "trace_from_atlas" not in status["result"]["verbs"]
    assert "trace_borders" in status["result"]["verbs"]
    code, refused = cli(capsys, str(images), "trace_from_atlas", "--passes", "3",
                        "--slices", ID0)
    assert code == 2 and refused["error"]["code"] == "BAD_ARGS", refused


def test_8_trace_from_atlas_needs_the_image_model_in_the_job(capsys, images):
    init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
         "--transforms", transforms_file(images))
    code, envelope = cli(capsys, str(images), "trace_from_atlas", "--slices", ID0)
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF", envelope
    assert "trace_from_atlas" not in envelope["result"]["verbs"]


def test_8_trace_from_atlas_rows_per_section(images, monkeypatch):
    """Per-section problems are rows: no transform, an unknown section, and a
    placement moved while the call was prepared (STALE_INPUT, nothing
    started or written for it)."""
    import langslice
    from langslice.core import handoff

    create(spec_for(images, ["nonlinear"], provider="openai-oauth",
                    positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        reply = job.trace_from_atlas(slices=[ID0, "nope.png"])
        assert reply["status"] == "error" and reply["error"] == "NOTHING_TRACED", reply
        assert [row["error"] for row in reply["results"]] == [
            "INVALID_LINEAR_PLACEMENT", "UNKNOWN_SLICE_IDS"]
        assert "--transforms" in reply["results"][0]["message"]
    fresh = spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs())
    fresh.resume = False
    create(fresh)
    real = handoff.correction_fingerprint
    calls = {"n": 0}

    def moved(state: Any, ctx: Any, section_id: str) -> str:
        calls["n"] += 1  # the second reading (under the lock) sees a moved section
        return real(state, ctx, section_id) + ("-moved" if calls["n"] == 2 else "")

    with langslice.open_job(images) as job:
        monkeypatch.setattr(handoff, "correction_fingerprint", moved)
        reply = job.trace_from_atlas(slices=[ID0])
        assert reply["results"][0]["error"] == "STALE_INPUT", reply
        assert job.state.by_id(ID0).image_correction is None
        assert not job.job.image_jobs


def test_8_trace_from_atlas_draws_each_section_at_its_own_angles(images, monkeypatch):
    """A stack whose sections were supplied with different cutting angles:
    each section's outlined atlas is drawn at its own (pitch, yaw)."""
    import langslice
    from langslice.core.nonlinear import registration_tool

    angles = {ID0: {"pitch": 1.0, "yaw": -0.5}, ID1: {"pitch": -0.75, "yaw": 0.25},
              ID2: {"pitch": 0.0, "yaw": 0.0}}
    inputs = {**external_inputs(), "angles": angles}
    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **inputs))
    real = registration_tool.outlined_atlas_template
    drawn: list[tuple[float, float]] = []

    def outlined(*args: Any, **kwargs: Any) -> Any:
        drawn.append((kwargs["pitch_deg"], kwargs["yaw_deg"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(registration_tool, "outlined_atlas_template", outlined)
    with langslice.open_job(images) as job:
        assert job.state.mixed_angles
        reply = job.trace_from_atlas(slices=list(IDS))
        assert reply["status"] == "ok", reply
        job.job.settle_image_corrections()
        assert all(job.state.by_id(name).image_correction["status"] == "ok"
                   for name in IDS)
    assert drawn == [(angles[name]["pitch"], angles[name]["yaw"]) for name in IDS]
