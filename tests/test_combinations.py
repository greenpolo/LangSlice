"""Every combination of steps, through every door.

The product requirement: LangSlice works in ANY combination of its steps
(Positioning, Linear, Nonlinear with or without an image model, or a
registration made elsewhere with only Nonlinear on top) through every door:
the agent CLI (``langslice-job FOLDER VERB``), the library
(``langslice.open_job``), the agent tools (``build_tools`` /
``ops.registry.enabled``) driven by LangSlice's agent, and MCP.

Each combination runs end to end on the shared synthetic stack
(``tests/synthetic_stack.py``: three sections of the synthetic atlas, fits in
process at the coarse level), through the library, and through the CLI, MCP
and the agent where a door has behaviour of its own. The image model is the stub (it answers
with the rough border overlay it was sent), installed where every door
resolves one; the real transport is replaced by a function that fails the
test, and ``HOME`` is private, so no credential is read and no model called.

| # | Combination | tasks | image model |
|---|---|---|---|
| 1 | Positioning only | position | - |
| 2 | Linear on supplied positions | transform | - |
| 3 | Positioning + Linear | position, transform | - |
| 4 | Nonlinear verbs called directly, no agent | all four | stub |
| 5 | Nonlinear without an image model | nonlinear | none |
| 6 | Nonlinear on a supplied (external) linear registration | nonlinear | stub |
| 7 | The agent, every task, no image model | all four | none |
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
from tests.synthetic_stack import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
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
                         "--pixel-size-um", str(PIXEL_SIZE_UM), *flags)
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


# --- what a finished job must hold ---------------------------------------------------------


def assert_exported(job_folder: Path, *, residual: tuple[str, ...] = (),
                    maps_exported: bool = False) -> dict[str, Any]:
    """Submitted, ``registration.json`` and the exports written, every
    section placed and without maps (submit writes none) unless the test
    exported them (*maps_exported*); then, exported on demand
    (``export_maps``), every section mapped with its maps; the sections in
    *residual* carry a warp."""
    import langslice

    document = json.loads((job_folder / "registration.json").read_text())
    assert document["submitted"] is True
    assert set(document["exports"]) == {"quicknii", "visualign"}
    for name in ("quicknii.json", "visualign.json"):
        assert (job_folder / "exports" / name).is_file()
    for entry, stem in zip(document["sections"], STEMS, strict=True):
        assert entry["problem"] is None, entry
        assert entry["pixel_to_atlas_um"] is not None
        if not maps_exported:
            assert entry["maps"] is None
            assert not (job_folder / "sections" / stem / "coords.tif").exists()
    with langslice.open_job(job_folder) as job:
        assert job.export_maps()["status"] == "ok"
    document = json.loads((job_folder / "registration.json").read_text())
    for entry, stem in zip(document["sections"], STEMS, strict=True):
        assert entry["maps"]["current"] is True
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


#: The tools of every run, whatever the tasks (unless the job disables one).
COMMON = {"look", "zoom", "set_channel",
          "grep_atlas", "grep_atlas_view", "status", "list_files", "search_files",
          "read_file", "mark_damage", "note", "undo", "redo", "submit"}
LINEAR = {"interactive_transform", "elastix_affine", "set_mask", "restore_registration"}
NONLINEAR = {"fireants_syn", "set_landmarks", "landmark_triangulation", "set_mask",
             "restore_registration"}


@pytest.mark.parametrize(("tasks", "provider", "extra"), [
    (["position"], "none", {"position_sections"}),
    (["transform"], "none", LINEAR),
    (["position", "transform"], "none", {"position_sections"} | LINEAR),
    (["position", "transform", "nonlinear"], "openai-oauth",
     {"position_sections", "trace_borders"} | LINEAR | NONLINEAR),
    (["nonlinear"], "none", NONLINEAR),
    (["nonlinear"], "openai-oauth", {"trace_borders"} | NONLINEAR),
    (["position", "transform", "nonlinear"], "none",
     {"position_sections"} | LINEAR | NONLINEAR),
], ids=["1-positioning", "2-linear", "3-positioning+linear", "4-all+image",
        "5-nonlinear-no-image", "6-nonlinear+image", "7-all-no-image"])
def test_each_combination_builds_exactly_its_tools(images, tasks, provider, extra):
    names = tool_names(images, tasks, provider)
    assert set(names) == COMMON | extra


@pytest.mark.parametrize(("tasks", "fields", "absent", "present"), [
    # The stack's cutting angles left to a Linear-only run: position_sections
    # comes for the angles alone.
    (["transform"], {"transform": TransformSpec(angles=True)}, set(),
     {"position_sections"} | LINEAR),
    (["transform"], {"disabled": ["interactive_transform"]},
     {"interactive_transform"}, {"elastix_affine"}),
    (["transform"], {"disabled": ["elastix_affine"]},
     {"elastix_affine"}, {"interactive_transform"}),
    (["position"], {"disabled": ["mark_damage"]}, {"mark_damage"},
     {"position_sections"}),
], ids=["angles", "no-interactive", "no-automatic", "no-mark-damage"])
def test_host_switches_add_and_remove_their_tools(images, tasks, fields, absent, present):
    names = set(tool_names(images, tasks, **fields))
    assert present <= names and not absent & names


# --- 1. Positioning only ------------------------------------------------------------------


REVERSED = {ID0: 0.2, ID1: 0.15, ID2: 0.1}


def _sections(positions: dict[str, float]) -> list[dict[str, Any]]:
    return [{"id": name, "position_mm": mm} for name, mm in positions.items()]


def test_1_positioning_only_through_the_library(images):
    import langslice

    create(spec_for(images, ["position"]))
    with langslice.open_job(images) as job:
        # The order follows the positions: written back to front, the stack is renumbered.
        backwards = job.position_sections(sections=_sections(REVERSED))
        assert backwards["status"] == "ok" and backwards["order"] == [ID2, ID1, ID0]
        assert backwards["reordered"]
        placed = job.position_sections(sections=_sections(POSITIONS))
        assert placed["status"] == "ok" and placed["order"] == list(IDS)
        assert job.submit(summary="placed", notes=[], interval_breaks=[])["status"] == "ok"
        assert all(record.transform is None for record in job.state.slices)
    document = assert_exported(images / "langslice")
    assert {entry["mapping"] for entry in document["sections"]} == {
        "linear (identity in-plane: no transform written)"}


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


# --- 3. Positioning + Linear ---------------------------------------------------------------


def test_3_positioning_and_linear_through_the_library(images):
    import langslice

    create(spec_for(images, ["position", "transform"]))
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
    job = init(capsys, images, "position,transform")
    ok(capsys, images, "position_sections", "--sections", json.dumps(_sections(POSITIONS)))
    ok(capsys, images, "elastix_affine", "--sections", ID0, "--sections", ID1,
       "--sections", ID2)
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    assert_exported(job)


def test_3_positioning_and_linear_through_mcp(images):
    server = _mcp_server(spec_for(images, ["position", "transform"]))
    replies = _mcp(server, [
        ("position_sections", {"sections": _sections(POSITIONS)}),
        ("elastix_affine", {"sections": list(IDS)}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 4. Nonlinear with an image model, no agent -------------------------------------------


def test_4_nonlinear_verbs_called_directly_through_the_library(images):
    import langslice

    create(spec_for(images, ["position", "transform", "nonlinear"],
                    provider="openai-oauth"))
    with langslice.open_job(images) as job:
        assert job.position_sections(sections=_sections(POSITIONS))["status"] == "ok"
        assert job.interactive_transform(sections=[{"id": n, **IDENTITY} for n in IDS])[
            "status"] == "ok"
        for name in (ID0, ID1):  # started in the background; submit waits for them
            started = job.trace_borders(section=name)
            assert started["status"] == "started" and started["work"], started
        fit = job.fireants_syn(sections=[ID2])
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        submitted = job.submit(summary="done", notes=[], interval_breaks=[])
        assert submitted["status"] == "ok", submitted
        assert job.export_maps()["status"] == "ok"
        assert all((job.state.by_id(name).image_correction or {}).get("status") == "ok"
                   for name in (ID0, ID1))
        assert job.state.by_id(ID2).image_correction is None
    assert_exported(images / "langslice", residual=IDS, maps_exported=True)


# --- 5. Nonlinear without an image model --------------------------------------------------


def test_5_nonlinear_without_an_image_model_through_the_library(images):
    import langslice

    create(spec_for(images, ["nonlinear"], **external_inputs()))
    with langslice.open_job(images) as job:
        assert "trace_borders" not in job.verbs
        fit = job.fireants_syn(sections=[ID0])
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        submitted = job.submit(summary="done", notes=[], interval_breaks=[])
        assert submitted["status"] == "ok", submitted
        assert_external_kept(job.state)
    assert_exported(images / "langslice", residual=(ID0,))


def test_nonlinear_on_positions_alone_is_refused_until_a_transform_is_written(images):
    """Positions without any in-plane transform: the maps treat the missing
    transform as the identity (``core.maps.placement_problem``), but the
    nonlinear step needs a WRITTEN one (``core.handoff.prepare_linear_registration``)
    and leaving a section linear refuses too, so with Linear off the host must
    supply transforms (or freeze the affine, which writes the ``host`` identity)."""
    import langslice

    create(spec_for(images, ["nonlinear"], positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        fit = job.fireants_syn(sections=[ID0])
        assert fit["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT", fit
        # The refusal says what to do: supply transforms, or switch Linear on.
        message = fit["results"][0]["message"]
        assert "--transforms" in message and "inputs.transforms" in message, message
        assert "Linear on" in message, message
    fresh = spec_for(images, ["nonlinear"], positions=dict(POSITIONS))
    fresh.frozen = {"*": ["orientation", "affine"]}
    fresh.resume = False  # a new job on the same images, not the one above
    create(fresh)
    with langslice.open_job(images) as job:
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"


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
        fit = job.fireants_syn(sections=[ID1])
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        submitted = job.submit(summary="done", notes=[], interval_breaks=[])
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


def test_6_an_agent_cannot_move_a_supplied_placement(images, monkeypatch):
    """Linear OFF means the supplied placement stays verbatim: the agent run
    has no verb that writes a position, an orientation, a transform or the
    angles; only ``undo`` past the first step could, and there is none."""
    assert "position_sections" not in tool_names(images, ["nonlinear"])
    install_script(monkeypatch, [
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
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
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
    document = assert_exported(images / "langslice")
    oriented = document["sections"][1]["parameters"]["orientation"]
    assert (oriented["flip"], oriented["rotation_deg"]) == (True, 90)


@pytest.mark.parametrize("bad", [
    {"flip": "yes"}, {"rotation_deg": 45}, {"rotate_deg": 90}, {"flip": True, "turn": 1}])
def test_a_bad_supplied_orientation_is_refused(images, bad):
    with pytest.raises(ValueError, match="orientation"):
        create(spec_for(images, ["nonlinear"], **external_inputs(), orientation={ID1: bad}))


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
             "--pixel-size-um", str(PIXEL_SIZE_UM))

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


def test_6_cli_init_takes_frozen_and_damaged_sections(capsys, images):
    damaged = images.parent / "damaged.json"
    damaged.write_text(json.dumps({ID2: "torn"}))
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--frozen", json.dumps({"*": ["orientation", "affine"]}),
               "--damaged", str(damaged))
    spec = json.loads((job / "job.json").read_text())["spec"]
    assert spec["frozen"] == {"*": ["orientation", "affine"]}
    assert spec["inputs"]["damaged"] == {ID2: "torn"}
    from tests.linear_tool_helpers import load_checkpoint

    state = load_checkpoint(str(job / "state.json"))
    assert state is not None
    # The host's damage note is the section's note; it marks no regions, so the
    # section is not damaged.
    record = state.by_id(ID2)
    assert record.damage_note == "torn" and record.damaged is False
    assert record.damaged_regions == []
    rows = {row["id"]: row for row in ok(capsys, images, "status")["result"]["rows"]}
    assert rows[ID2]["damage_note"] == "torn" and rows[ID2]["damaged"] is False
    # Sections with a frozen affine carry the host identity: they can be left linear and submitted.
    ok(capsys, images, "submit", *SUBMIT_FLAGS)


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
    assert {"grep_atlas", "fireants_syn", "look"} <= names
    assert ("trace_borders" in names) is linked
    assert (IMAGE_MODEL_OFF in statement) is not linked
    if not linked:
        replies = _mcp(server, [
            ("fireants_syn", {"sections": [ID0]}),
            ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
        ])
        assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
        spec = json.loads((images / "langslice" / "job.json").read_text())["spec"]
        assert spec["nonlinear"]["provider"] == "openai-oauth"  # the job keeps its provider


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
        ("fireants_syn", {"sections": [ID0]}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    state = run_agent(spec_for(images, ["position", "transform", "nonlinear"]))
    assert state.submitted is True
    assert state.by_id(ID0).deformation
    assert not any(state.by_id(name).deformation for name in (ID1, ID2))
    assert_exported(images / "langslice", residual=(ID0,))
    results = json.loads((images / "langslice" / "exports" / "linear_results.json").read_text())
    assert results["submitted"] is True

