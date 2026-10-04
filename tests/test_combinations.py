"""Every combination of steps, through every door (audit, 2026-10-04).

The product requirement: LangSlice works in ANY combination of its steps
(Positioning, Linear, Nonlinear with or without an image model, or a
registration made elsewhere with only Nonlinear on top) through every door:
the agent CLI (``langslice job FOLDER VERB``), the library
(``langslice.open_job``), the agent tools (``build_tools`` /
``ops.registry.enabled``) driven by LangSlice's agent, and MCP.

Each combination runs end to end on the golden recorder's synthetic stack
(``tests/golden/record.py``: three sections of the synthetic atlas, fits in
process at the coarse level). The image model is the golden stub (it answers
with the rough border overlay it was sent), installed where every door
resolves one; the real transport is replaced by a function that fails the
test, and ``HOME`` is private, so no credential is read and no model called.

A test marked ``xfail(strict=True)`` is a gap: what the requirement asks
for and the code does not do yet. Its reason says where it is blocked.

| # | Combination | tasks | image model |
|---|---|---|---|
| 1 | Positioning only | reorder, position | - |
| 2 | Linear on supplied positions | transform | - |
| 3 | Positioning + Linear | reorder, position, transform | - |
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
from pathlib import Path
from typing import Any

import pytest

from langslice.cli import main
from langslice.core.spec import JobSpec, NonlinearSpec
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
    job; the stub image model wherever a door resolves one; no network."""
    import langslice.doors.tools.toolbox as toolbox
    import langslice.providers.images as transport

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
    """A job as a script makes one today (``doors.jobs.create``: there is no
    public library call for it, see ``test_the_library_can_create_a_job``)."""
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
    code, envelope = cli(capsys, "job", str(images), "init", "--tasks", tasks,
                         "--image-provider", provider, "--preprocess", "none",
                         "--pixel-size-um", str(PIXEL_SIZE_UM), "--no-debrief", *flags)
    assert code == 0, envelope
    return Path(envelope["result"]["job_folder"])


def ok(capsys: pytest.CaptureFixture[str], images: Path, verb: str,
       *flags: str) -> dict[str, Any]:
    code, envelope = cli(capsys, "job", str(images), verb, *flags)
    assert code == 0 and envelope["ok"] is True, envelope
    return envelope


SUBMIT_FLAGS = ("--summary", "done", "--notes", "[]", "--interval-breaks", "[]")


def transforms_file(images: Path) -> str:
    """The supplied transforms as a JSON file (inline JSON this long breaks
    ``init``: ``test_init_takes_a_long_inline_json``)."""
    path = images.parent / "transforms.json"
    path.write_text(json.dumps(EXTERNAL_TRANSFORMS))
    return str(path)


@pytest.mark.xfail(strict=True, reason=(
    "load_json_arg (doors/cli/linear.py:229-231) tries the value as a path first; "
    "inline JSON longer than a file name (255 bytes, any real --transforms map) "
    "raises OSError ENAMETOOLONG, and init exits 4 (INTERNAL). Catch OSError there, "
    "or parse JSON first."))
def test_init_takes_a_long_inline_json(capsys, images):
    init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
         "--transforms", json.dumps(EXTERNAL_TRANSFORMS))


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
     {"view_placement", "orient_slices", "fit_affine", "adjust_transforms"}),
    (["reorder", "position", "transform"], "none",
     {"reorder_slices", "set_positions", "view_placement", "view_stack", "orient_slices",
      "fit_affine", "adjust_transforms"}),
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
    code, envelope = cli(capsys, "job", str(images), "submit", *SUBMIT_FLAGS)
    # s2 was moved first: positions now run against the order.
    assert code == 3 and envelope["error"]["code"] == "ORDER_POSITION_MISMATCH"
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
    code, envelope = cli(capsys, "job", str(images), "set_positions", "--entries", "[]")
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    code, envelope = cli(capsys, "job", str(images), "submit", *SUBMIT_FLAGS)
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
    code, envelope = cli(capsys, "job", str(images), "submit", *SUBMIT_FLAGS)
    assert code == 3 and envelope["error"]["code"] == "MISSING_IMAGE_CORRECTIONS"
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


@pytest.mark.xfail(strict=True, reason=(
    "The library cannot be handed an image model: open_job takes no image_model, "
    "Opened.tools() (doors/jobs.py:142) builds the toolbox without one, so "
    "build_tools resolves the spec's provider (doors/tools/toolbox.py:532-533). A "
    "script's own or replayed model needs a monkeypatch."))
def test_4_the_library_takes_an_image_model(images):
    import langslice

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **external_inputs()))
    model = stub_image_model(spec_for(images, ["nonlinear"], provider="openai-oauth"))
    with langslice.open_job(images, image_model=model) as job:  # type: ignore[call-arg]
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
    code, envelope = cli(capsys, "job", str(images), "trace_borders", "--id", ID0)
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    ok(capsys, images, "fit_deformable", "--slices", ID0, "--engine", "elastix")
    ok(capsys, images, "fit_deformable", "--slices", ID1, "--slices", ID2,
       "--keep-linear", "kept")
    ok(capsys, images, "submit", *SUBMIT_FLAGS)
    assert_exported(job, residual=(ID0,))


@pytest.mark.xfail(strict=True, reason=(
    "MCP refuses the nonlinear task outright, even with provider none, where no "
    "image model is involved: doors/mcp/server.py:191 (open_job) and :281 "
    "(open_saved_job), and Claude mode in doors/api/claude_jobs.py:97-100,119. "
    "The refusal should apply only when spec.nonlinear.uses_image_model."))
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
    fresh = spec_for(images, ["nonlinear"], positions=dict(POSITIONS), locked=list(IDS))
    fresh.resume = False  # a new job on the same images, not the one above
    create(fresh)
    with langslice.open_job(images) as job:
        assert job.fit_deformable(slices=list(IDS), keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"


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
        code, envelope = cli(capsys, "job", str(images), verb, "--args", "{}")
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


@pytest.mark.xfail(strict=True, reason=(
    "No orientation in the host inputs: apply_host_inputs (job/job.py:163-250) reads "
    "order, positions, angles, transforms, damaged and locked only, so a supplied flip "
    "or quarter turn can only be folded into the affine; unknown keys are ignored "
    "silently. JobSpec.inputs (core/spec.py) needs an 'orientation' entry "
    "({filename: {flip, rotation_deg}})."))
def test_6_a_supplied_orientation_is_kept(images):
    import langslice

    inputs = {**external_inputs(), "orientation": {ID1: {"flip": True, "rotation_deg": 90}}}
    create(spec_for(images, ["nonlinear"], **inputs))
    with langslice.open_job(images) as job:
        assert job.state.by_id(ID1).flip is True
        assert job.state.by_id(ID1).rotation_deg == 90


@pytest.mark.xfail(strict=True, reason=(
    "Unknown keys in JobSpec.inputs are ignored without a word (job/job.py:163-250): "
    "a misspelled 'transfroms' leaves every section without its supplied placement."))
def test_6_an_unknown_input_is_refused(images):
    with pytest.raises(ValueError, match="transfroms"):
        create(spec_for(images, ["nonlinear"], positions=dict(POSITIONS),
                        transfroms=EXTERNAL_TRANSFORMS))


@pytest.mark.xfail(strict=True, reason=(
    "Re-running `init` with a different supplied registration on a folder that holds "
    "a job resumes the old checkpoint and drops the new inputs without a word "
    "(Job.open, job/job.py:682-696: apply_host_inputs runs on a fresh ingest only), "
    "while job.json is rewritten with the new inputs (job/job.py:678). It should "
    "refuse (and name --fresh) or apply them."))
def test_6_a_second_init_with_new_inputs_is_not_silently_ignored(capsys, images):
    init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS))
    moved = {name: mm + 0.02 for name, mm in POSITIONS.items()}
    code, envelope = cli(capsys, "job", str(images), "init", "--tasks", "nonlinear",
                         "--image-provider", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         "--positions", json.dumps(moved))
    if code == 0:
        code, status = cli(capsys, "job", str(images), "status")
        assert {row["id"]: row["position_mm"] for row in status["result"]["rows"]} == moved


@pytest.mark.xfail(strict=True, reason=(
    "`langslice job FOLDER init` has --positions/--order/--transforms/--pitch/--yaw "
    "but no --damaged, --locked or orientation flag (doors/cli/linear.py "
    "add_linear_arguments / build_linear_spec), though JobSpec.inputs takes damaged "
    "and locked."))
def test_6_cli_init_takes_locked_and_damaged_sections(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--locked", json.dumps(list(IDS)), "--damaged", json.dumps({ID2: "torn"}))
    spec = json.loads((job / "job.json").read_text())["spec"]
    assert spec["inputs"]["locked"] == list(IDS)


@pytest.mark.xfail(strict=True, reason=(
    "No importer for a registration made elsewhere: LangSlice writes QuickNII/"
    "VisuAlign JSON (job/quint.py job_export) but reads none, nor DeepSlice output "
    "(core/deepslice.py is a stub), nor an ABBA state file. A supplied affine must "
    "already be LangSlice's normalized six numbers on the oriented section render."))
def test_6_a_quicknii_registration_can_be_imported(capsys, images, tmp_path):
    # A job placed by LangSlice exports quicknii.json; a new job from it must
    # land at the same placement.
    source = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
                  "--transforms", transforms_file(images))
    ok(capsys, images, "export_maps")
    quicknii = tmp_path / "quicknii.json"
    shutil.copy(source / "exports" / "quicknii.json", quicknii)
    code, envelope = cli(capsys, "job", str(images), "init", "--tasks", "nonlinear",
                         "--image-provider", "none", "--job-dir", str(tmp_path / "again"),
                         "--registration", str(quicknii))
    assert code == 0, envelope


@pytest.mark.xfail(strict=True, reason=(
    "MCP refuses the nonlinear task (doors/mcp/server.py:191), so a registration "
    "supplied to `langslice mcp --positions ... --transforms ...` cannot get its "
    "nonlinear step through Claude Desktop / Claude Code, with or without an "
    "image model."))
def test_6_external_registration_then_nonlinear_through_mcp(images):
    server = _mcp_server(spec_for(images, ["nonlinear"], provider="openai-oauth",
                                  **external_inputs()))
    replies = _mcp(server, [("trace_borders", {"id": ID0})])
    assert replies[0]["status"] in ("running", "ok")


@pytest.mark.xfail(strict=True, reason=(
    "The library cannot create a job: langslice exposes open_job only "
    "(langslice/__init__.py), and open_job needs an existing job folder "
    "(doors/library.py:99-107); creating one (with host inputs) takes the CLI's "
    "`init` or the internal doors.jobs.create."))
def test_the_library_can_create_a_job(images):
    import langslice

    spec = spec_for(images, ["nonlinear"], **external_inputs())
    with langslice.create_job(spec) as job:  # type: ignore[attr-defined]
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


@pytest.mark.xfail(strict=True, reason=(
    "MCP refuses the nonlinear task even without an image model "
    "(doors/mcp/server.py:191), so Claude cannot run every task."))
def test_7_every_task_and_no_image_model_through_mcp(images):
    server = _mcp_server(spec_for(images, ["reorder", "position", "transform", "nonlinear"]))
    replies = _mcp(server, [("status", {})])
    assert "fit_deformable" in json.dumps(replies)
