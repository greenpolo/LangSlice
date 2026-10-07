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
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langslice.core.spec import JobSpec, NonlinearSpec
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
#: An identity adjustment per section (the Linear task's cheapest write).
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


def tool_names(images: Path, tasks: list[str], provider: str = "none") -> list[str]:
    """The agent tools (ADK and MCP) a run of this combination has."""
    from langslice.agent.engine import build_context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest
    from langslice.ops.registry import enabled

    spec = spec_for(images, tasks, provider=provider)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    box = build_tools(ingest(spec, ctx), ctx, spec)
    assert box.names == enabled(spec)
    return box.names


# --- the tools each combination has (ADK / MCP tool building) ------------------------------


COMMON = {"status", "view_slices", "view_atlas", "note", "undo", "redo", "mark_damaged",
          "submit"}


@pytest.mark.parametrize(("tasks", "provider", "extra"), [
    (["reorder", "position"], "none",
     {"reorder_slices", "set_positions", "view_placement", "view_stack"}),
    (["transform"], "none",
     {"view_placement", "orient_slices", "fit_affine", "adjust_transforms", "grep_atlas"}),
    (["reorder", "position", "transform"], "none",
     {"reorder_slices", "set_positions", "view_placement", "view_stack", "orient_slices",
      "fit_affine", "adjust_transforms", "grep_atlas"}),
    (["reorder", "position", "transform", "nonlinear"], "openai-oauth",
     {"reorder_slices", "set_positions", "view_placement", "view_stack", "orient_slices",
      "fit_affine", "adjust_transforms", "trace_borders", "grep_atlas", "fit_deformable"}),
    (["nonlinear"], "none", {"view_placement", "grep_atlas", "fit_deformable"}),
    (["nonlinear"], "openai-oauth",
     {"view_placement", "trace_borders", "grep_atlas", "fit_deformable"}),
    (["reorder", "position", "transform", "nonlinear"], "none",
     {"reorder_slices", "set_positions", "view_placement", "view_stack", "orient_slices",
      "fit_affine", "adjust_transforms", "grep_atlas", "fit_deformable"}),
], ids=["1-positioning", "2-linear", "3-positioning+linear", "4-all+image",
        "5-nonlinear-no-image", "6-nonlinear+image", "7-all-no-image"])
def test_each_combination_builds_exactly_its_tools(images, tasks, provider, extra):
    assert set(tool_names(images, tasks, provider)) == COMMON | extra


# --- 1. Positioning only ------------------------------------------------------------------


def test_1_positioning_only_through_the_library(images):
    import langslice

    create(spec_for(images, ["reorder", "position"]))
    with langslice.open_job(images) as job:
        assert job.reorder_slices(slices=[ID2], after="start")["status"] == "ok"
        assert job.reorder_slices(slices=[ID0, ID1, ID2])["status"] == "ok"
        entries = [{"id": name, "position_mm": mm} for name, mm in POSITIONS.items()]
        assert job.set_positions(entries=entries)["status"] == "ok"
        assert job.submit(summary="placed", notes=[], interval_breaks=[])["status"] == "ok"
        assert all(record.transform is None for record in job.state.slices)
    document = assert_exported(images / "langslice")
    assert {entry["mapping"] for entry in document["sections"]} == {
        "linear (identity in-plane: no transform written)"}


def test_1_positioning_only_through_the_cli(capsys, images):
    job = init(capsys, images, "reorder,position")
    ok(capsys, images, "reorder-slices", "--slices", ID2, "--after", "start")
    ok(capsys, images, "set_positions", "--entries", json.dumps(
        [{"id": name, "position_mm": mm} for name, mm in POSITIONS.items()]))
    # s2 was moved first: positions run against the order, which is no
    # longer refused (order follows position).
    ok(capsys, images, "reorder-slices", "--slices", ID0, "--slices", ID1, "--slices", ID2)
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
        ("set_positions", {"entries": [{"id": n, "position_mm": mm}
                                       for n, mm in POSITIONS.items()]}),
        ("submit", {"summary": "placed", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 2. Linear without positioning (positions supplied) -------------------------------------


def test_2_linear_on_supplied_positions_through_the_library(images):
    import langslice

    create(spec_for(images, ["transform"], positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        assert "set_positions" not in job.verbs
        assert job.fit_affine(slices=[ID0], method="silhouette")["status"] == "ok"
        entries = [{"id": name, **IDENTITY} for name in (ID1, ID2)]
        assert job.adjust_transforms(entries=entries)["status"] == "ok"
        assert job.submit(summary="aligned", notes=[], interval_breaks=[])["status"] == "ok"
        assert {r.id: r.position_mm for r in job.state.slices} == POSITIONS
        assert all(record.transform for record in job.state.slices)
    assert_exported(images / "langslice")


def test_2_linear_on_supplied_positions_through_the_cli(capsys, images):
    job = init(capsys, images, "transform", "--positions", json.dumps(POSITIONS))
    code, envelope = cli(capsys, str(images), "set_positions", "--entries", "[]")
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    code, envelope = cli(capsys, str(images), "submit", *SUBMIT_FLAGS)
    assert code == 3 and envelope["error"]["code"] == "MISSING_TRANSFORMS"
    ok(capsys, images, "adjust_transforms", "--entries", json.dumps(
        [{"id": name, **IDENTITY} for name in IDS]))
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    document = assert_exported(job)
    assert [e["parameters"]["plane"]["position_mm"] for e in document["sections"]] == list(
        POSITIONS.values())


def test_2_linear_on_supplied_positions_through_mcp(images):
    server = _mcp_server(spec_for(images, ["transform"], positions=dict(POSITIONS)))
    replies = _mcp(server, [
        ("adjust_transforms", {"entries": [{"id": name, **IDENTITY} for name in IDS[:2]]}),
        ("adjust_transforms", {"entries": [{"id": ID2, **IDENTITY}]}),
        ("submit", {"summary": "aligned", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 3. Positioning + Linear ---------------------------------------------------------------


def test_3_positioning_and_linear_through_the_library(images):
    import langslice

    create(spec_for(images, ["reorder", "position", "transform"]))
    with langslice.open_job(images) as job:
        entries = [{"id": name, "position_mm": mm} for name, mm in POSITIONS.items()]
        assert job.set_positions(entries=entries)["status"] == "ok"
        assert job.orient_slices(entries=[{"id": ID1, "flip": True}])["status"] == "ok"
        assert job.fit_affine(slices=list(IDS), method="silhouette")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        assert job.state.by_id(ID1).flip is True
    document = assert_exported(images / "langslice")
    assert document["sections"][1]["parameters"]["orientation"]["flip"] is True


def test_3_positioning_and_linear_through_the_cli(capsys, images):
    job = init(capsys, images, "reorder,position,transform")
    ok(capsys, images, "set_positions", "--entries", json.dumps(
        [{"id": name, "position_mm": mm} for name, mm in POSITIONS.items()]))
    ok(capsys, images, "fit_affine", "--slices", ID0, "--slices", ID1, "--slices", ID2,
       "--method", "silhouette")
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    assert_exported(job)


def test_3_positioning_and_linear_through_mcp(images):
    server = _mcp_server(spec_for(images, ["reorder", "position", "transform"]))
    replies = _mcp(server, [
        ("set_positions", {"entries": [{"id": n, "position_mm": mm}
                                       for n, mm in POSITIONS.items()]}),
        ("fit_affine", {"slices": list(IDS), "method": "silhouette"}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok", "ok"], replies
    assert_exported(images / "langslice")


# --- 4. Nonlinear with an image model, no agent -------------------------------------------


def test_4_nonlinear_verbs_called_directly_through_the_library(images):
    import langslice

    create(spec_for(images, ["reorder", "position", "transform", "nonlinear"],
                    provider="openai-oauth"))
    with langslice.open_job(images) as job:
        entries = [{"id": name, "position_mm": mm} for name, mm in POSITIONS.items()]
        assert job.set_positions(entries=entries)["status"] == "ok"
        assert job.adjust_transforms(entries=[{"id": n, **IDENTITY} for n in IDS[:3]])[
            "status"] == "ok"
        for name in IDS:  # started in the background; submit settles them
            assert job.trace_borders(id=name)["status"] in ("running", "ok")
        fit = job.fit_deformable(slices=[ID0], fit_section="traced_lines", engine="elastix")
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        assert job.fit_deformable(slices=[ID1, ID2], keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        assert job.export_maps()["status"] == "ok"
        assert all((r.image_correction or {}).get("status") == "ok" for r in job.state.slices)
    assert_exported(images / "langslice", residual=(ID0,))


def test_4_nonlinear_verbs_called_directly_through_the_cli(capsys, images):
    job = init(capsys, images, "reorder,position,transform,nonlinear",
               provider="openai-oauth")
    ok(capsys, images, "set_positions", "--entries", json.dumps(
        [{"id": name, "position_mm": mm} for name, mm in POSITIONS.items()]))
    ok(capsys, images, "adjust_transforms", "--entries", json.dumps(
        [{"id": name, **IDENTITY} for name in IDS]))
    code, envelope = cli(capsys, str(images), "submit", *SUBMIT_FLAGS)
    assert code == 3 and envelope["error"]["code"] == "MISSING_DEFORMATIONS"
    for name in IDS:  # the CLI settles each trace before answering
        assert ok(capsys, images, "trace-borders", "--id", name)["result"]["status"] in (
            "running", "ok")
    ok(capsys, images, "fit-deformable", "--slices", ID0, "--fit-section", "traced_lines",
       "--engine", "elastix")
    ok(capsys, images, "fit_deformable", "--slices", ID1, "--slices", ID2,
       "--keep-linear", "kept")
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    ok(capsys, images, "export_maps")
    assert_exported(job, residual=(ID0,))


def test_4_the_library_takes_an_image_model(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    model = stub_image_model(spec_for(images, ["nonlinear"], provider="openai-oauth"))
    with langslice.open_job(images, image_model=model) as job:
        assert job.trace_borders(id=ID0)["status"] in ("running", "ok")


# --- 5. Nonlinear without an image model --------------------------------------------------


def test_5_nonlinear_without_an_image_model_through_the_library(images):
    import langslice

    create(spec_for(images, ["nonlinear"], **external_inputs()))
    with langslice.open_job(images) as job:
        assert "trace_borders" not in job.verbs
        fit = job.fit_deformable(slices=[ID0], engine="elastix")
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        assert job.fit_deformable(slices=[ID1, ID2], keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        assert_external_kept(job.state)
    assert_exported(images / "langslice", residual=(ID0,))


def test_5_nonlinear_without_an_image_model_through_the_cli(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images),
               "--pitch", str(EXTERNAL_ANGLES["pitch"]), "--yaw", str(EXTERNAL_ANGLES["yaw"]))
    code, envelope = cli(capsys, str(images), "trace_borders", "--id", ID0)
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    ok(capsys, images, "fit_deformable", "--slices", ID0, "--engine", "elastix")
    ok(capsys, images, "fit_deformable", "--slices", ID1, "--slices", ID2,
       "--keep-linear", "kept")
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    assert_exported(job, residual=(ID0,))


def test_5_nonlinear_without_an_image_model_through_mcp(images):
    server = _mcp_server(spec_for(images, ["nonlinear"], **external_inputs()))
    replies = _mcp(server, [
        ("fit_deformable", {"slices": [ID0], "engine": "elastix"}),
        ("fit_deformable", {"slices": [ID1, ID2], "keep_linear": "kept"}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok", "ok", "ok"], replies


def test_nonlinear_on_positions_alone_is_refused_until_a_transform_is_written(images):
    """Positions without any in-plane transform: the maps treat the missing
    transform as the identity (``core.maps.placement_problem``), but the
    nonlinear step needs a WRITTEN one (``core.handoff.prepare_linear_registration``)
    and ``keep_linear`` refuses too, so with Linear off the host must supply
    transforms (or ``locked``, which writes the ``host`` identity)."""
    import langslice

    create(spec_for(images, ["nonlinear"], positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        fit = job.fit_deformable(slices=[ID0], engine="elastix")
        assert fit["results"][0]["error"] == "INVALID_LINEAR_PLACEMENT", fit
        kept = job.fit_deformable(slices=[ID0], keep_linear="kept")
        assert kept["error"] == "NOTHING_WRITTEN", kept
        # Each refusal says what to do: supply transforms, or Linear and fit_affine.
        for message in (fit["results"][0]["message"], kept["results"][0]["message"]):
            assert "--transforms" in message and "inputs.transforms" in message, message
            assert "Linear on" in message and "fit_affine" in message, message
    fresh = spec_for(images, ["nonlinear"], positions=dict(POSITIONS), locked=list(IDS))
    fresh.resume = False  # a new job on the same images, not the one above
    create(fresh)
    with langslice.open_job(images) as job:
        assert job.fit_deformable(slices=list(IDS), keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"


def test_tracing_on_positions_alone_says_what_to_do(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", positions=dict(POSITIONS)))
    with langslice.open_job(images) as job:
        traced = job.trace_borders(id=ID0)
        assert traced["error"] == "INVALID_LINEAR_PLACEMENT", traced
        assert "--transforms" in traced["message"] and "fit_affine" in traced["message"]


# --- 6. Nonlinear on a registration made elsewhere ------------------------------------------


def test_6_external_registration_then_nonlinear_through_the_library(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    with langslice.open_job(images) as job:
        assert not {"set_positions", "reorder_slices", "orient_slices", "fit_affine",
                    "adjust_transforms", "set_cutting_angles"} & set(job.verbs)
        assert_external_kept(job.state)
        for name in IDS:
            assert job.trace_borders(id=name)["status"] in ("running", "ok")
        fit = job.fit_deformable(slices=[ID0], fit_section="traced_lines", engine="elastix")
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        assert job.fit_deformable(slices=[ID1, ID2], keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        assert_external_kept(job.state)  # the linear placement, verbatim, after submit
    document = assert_exported(images / "langslice", residual=(ID0,))
    assert document["cutting_angles_deg"] == EXTERNAL_ANGLES
    assert [e["parameters"]["affine"]["params"] for e in document["sections"]] == [
        EXTERNAL_TRANSFORMS[name]["params"] for name in IDS]
    assert document["sections"][1]["parameters"]["affine"]["mirrored"] is True


def test_6_external_registration_then_nonlinear_through_the_cli(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images),
               "--pitch", str(EXTERNAL_ANGLES["pitch"]), "--yaw", str(EXTERNAL_ANGLES["yaw"]),
               provider="openai-oauth")
    for verb in ("set_positions", "adjust_transforms", "fit_affine"):
        code, envelope = cli(capsys, str(images), verb, "--args", "{}")
        assert code == 3 and envelope["error"]["code"] == "VERB_OFF", envelope
    for name in IDS:
        ok(capsys, images, "trace_borders", "--id", name)
    ok(capsys, images, "fit_deformable", "--slices", ID0, "--fit-section", "traced_lines",
       "--engine", "elastix")
    ok(capsys, images, "fit_deformable", "--slices", ID1, "--slices", ID2,
       "--keep-linear", "kept")
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    ok(capsys, images, "export_maps")
    from langslice.job.checkpoint import load_checkpoint

    state = load_checkpoint(str(job / "state.json"))
    assert state is not None
    assert_external_kept(state)
    assert_exported(job, residual=(ID0,))


def test_6_an_agent_cannot_move_a_supplied_placement(images, monkeypatch):
    """Linear OFF means the supplied placement stays verbatim: the agent run
    has no verb that writes a position, an orientation, a transform or the
    angles; only ``undo`` past the first step could, and there is none."""
    install_script(monkeypatch, [
        ("fit_deformable", {"slices": list(IDS), "keep_linear": "kept"}),
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
        assert job.fit_deformable(slices=list(IDS), keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
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
    assert state is not None and state.by_id(ID2).damaged is True
    # Locked sections carry the host identity: the nonlinear step can start.
    ok(capsys, images, "fit_deformable", "--slices", ID0, "--slices", ID1, "--slices", ID2,
       "--keep-linear", "kept")


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
        *[("trace_borders", {"id": name}) for name in IDS],
        ("fit_deformable", {"slices": list(IDS), "keep_linear": "kept"}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert all(reply["status"] in ("running", "ok") for reply in replies[:3]), replies
    assert [reply["status"] for reply in replies[3:]] == ["ok", "ok"], replies
    assert_exported(images / "langslice")


def _mcp_names_and_statement(server: Any) -> tuple[set[str], str]:
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    async def body() -> tuple[set[str], str]:
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
            return {tool.name for tool in listed.tools}, first.text

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
    names, statement = _mcp_names_and_statement(server)
    assert {"grep_atlas", "fit_deformable", "view_placement"} <= names
    assert ("trace_borders" in names) is linked
    assert "trace_from_atlas" not in names  # a hidden scripting verb, never a model's
    assert (IMAGE_MODEL_OFF in statement) is not linked
    assert ("Base image-model prompt" in statement) is linked
    if not linked:
        replies = _mcp(server, [
            ("fit_deformable", {"slices": list(IDS), "keep_linear": "kept"}),
            ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
        ])
        assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
        spec = json.loads((images / "langslice" / "job.json").read_text())["spec"]
        assert spec["nonlinear"]["provider"] == "openai-oauth"  # the job keeps its provider


def test_mcp_with_provider_none_never_asks_for_an_image_model(images, monkeypatch):
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "image_model_connected", _no_network)
    names, statement = _mcp_names_and_statement(
        _mcp_server(spec_for(images, ["nonlinear"], **external_inputs())))
    assert "trace_borders" not in names and "fit_deformable" in names
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
        ("set_positions", {"entries": [{"id": n, "position_mm": mm}
                                       for n, mm in POSITIONS.items()]}),
        ("adjust_transforms", {"entries": [{"id": name, **IDENTITY} for name in IDS]}),
        ("fit_deformable", {"slices": [ID0], "engine": "elastix"}),
        ("fit_deformable", {"slices": [ID1, ID2], "keep_linear": "kept"}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
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
    assert {"set_positions", "fit_affine", "fit_deformable", "grep_atlas"} <= names
    assert "trace_borders" not in names
    replies = _mcp(server, [
        ("set_positions", {"entries": [{"id": n, "position_mm": mm}
                                       for n, mm in POSITIONS.items()]}),
        ("adjust_transforms", {"entries": [{"id": name, **IDENTITY} for name in IDS]}),
        ("fit_deformable", {"slices": [ID0], "engine": "elastix"}),
        ("fit_deformable", {"slices": [ID1, ID2], "keep_linear": "kept"}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ])
    assert [reply["status"] for reply in replies] == ["ok"] * 5, replies
    assert_exported(images / "langslice", residual=(ID0,))


# --- 8. The placement-free trace: a hidden scripting verb -----------------------------------


def test_8_trace_from_atlas_through_the_library_then_fit_and_submit(images):
    """Route "atlas" as a job verb: called by name, listed nowhere; its reply
    is recorded as trace_borders' is, so the traced fit, the submit gate and
    the maps read it unchanged."""
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    with langslice.open_job(images) as job:
        assert "trace_from_atlas" not in job.verbs and "trace_from_atlas" not in dir(job)
        traced = job.trace_from_atlas(slices=[ID0], passes=2)
        assert traced["status"] == "ok", traced
        assert traced["results"][0]["status"] == "running" and traced["results"][0]["started"]
        rest = job.trace_from_atlas(slices=[ID1, ID2])
        assert [row["status"] for row in rest["results"]] == ["running", "running"], rest
        fit = job.fit_deformable(slices=[ID0], fit_section="traced_lines", engine="elastix")
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        assert job.fit_deformable(slices=[ID1, ID2], keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
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
        assert "fit_affine" in reply["results"][0]["message"]
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
