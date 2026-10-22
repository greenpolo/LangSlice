"""What the agent is told, as text: tool declarations, job statements, refusals.

One parametrized test; one human-readable ``.md`` per artifact in
``tests/snapshots/agent_text/``, so a change to model-facing wording shows in
a pull request as a prose diff. No fits, no pictures, no real atlas (the
synthetic one), no network.

An intended change: ``LANGSLICE_UPDATE_SNAPSHOT=1 python -m pytest
tests/test_agent_text_snapshot.py`` rewrites the files and fails once
("regenerated: review git diff"), so a regeneration can never pass CI.
The job statements and declarations are also checked once for the rule
that no text names a tool the job disables.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.synthetic_stack import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    Normaliser,
    atlas_loader,
    full_spec,
    write_sections,
)

pytest.importorskip("google.adk")

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "agent_text"
UPDATE_ENV = "LANGSLICE_UPDATE_SNAPSHOT"
MAX_DIFF_LINES = 40

_BASE: dict[str, Any] = {"model": "fake-model", "preprocess": "none",
                         "inputs": {"pixel_size_um": PIXEL_SIZE_UM}}


# --- builders ---------------------------------------------------------------


def _specs(folder: Path) -> dict[str, Any]:
    from langslice.core.spec import JobSpec, NonlinearSpec

    image_folder = str(folder)
    all_tasks = ["position", "transform", "nonlinear"]
    return {
        "full": full_spec(folder),
        "auto": JobSpec(image_folder=image_folder, tasks=all_tasks, image_resolution="auto",
                        nonlinear=NonlinearSpec(provider="none"), **_BASE),
        "position": JobSpec(image_folder=image_folder, tasks=["position"], **_BASE),
        "host": JobSpec(image_folder=image_folder, tasks=["transform", "nonlinear"],
                        force_view=True, disabled=["mark_damage"],
                        nonlinear=NonlinearSpec(provider="none", require_deformation=True),
                        **_BASE),
        "transform_known_position": JobSpec(
            image_folder=image_folder, tasks=["transform"], model="fake-model",
            preprocess="none",
            inputs={"pixel_size_um": PIXEL_SIZE_UM,
                    "positions": {ID0: 0.1, ID1: 0.15, ID2: 0.2}}),
        "frozen_one": JobSpec(
            image_folder=image_folder, tasks=["position", "transform"],
            frozen={ID0: ["position"], "*": ["angles", "affine"]},
            **{**_BASE, "inputs": {"pixel_size_um": PIXEL_SIZE_UM, "positions": {ID0: 0.1}}}),
        "frozen": JobSpec(image_folder=image_folder, tasks=all_tasks,
                          frozen={"*": ["position", "angles"]},
                          disabled=["elastix_affine", "mark_damage"],
                          nonlinear=NonlinearSpec(provider="none"), **_BASE),
    }


def _context(spec: Any) -> Any:
    from langslice.agent.engine import build_context

    return build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())


def _toolbox(spec: Any, image_model: Any = None) -> Any:
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest

    ctx = _context(spec)
    return build_tools(ingest(spec, ctx), ctx, spec, image_model=image_model)


def _schema_lines(schema: dict[str, Any], indent: str = "") -> list[str]:
    lines: list[str] = []
    required = set(schema.get("required") or [])
    for name, prop in (schema.get("properties") or {}).items():
        kind = str(prop.get("type", "any")).lower()
        if kind == "array":
            kind += f" of {str((prop.get('items') or {}).get('type', 'any')).lower()}"
        flags = ", required" if name in required else ""
        if prop.get("enum"):
            flags += f", one of {prop['enum']}"
        lines.append(f"{indent}- `{name}` ({kind}{flags}): {prop.get('description', '')}".rstrip())
        nested = prop.get("items") if isinstance(prop.get("items"), dict) else prop
        if nested.get("properties"):
            lines += _schema_lines(nested, indent + "  ")
    return lines


def _declarations_md(title: str, spec: Any, image_model: Any = None) -> str:
    from google.adk.tools import FunctionTool

    from langslice.doors.tools.media import packaged_tools

    box = _toolbox(spec, image_model)
    out = [f"# Tool declarations: {title}", ""]
    for tool in packaged_tools(box.tools):
        decl = FunctionTool(tool)._get_declaration().model_dump(  # noqa: SLF001
            exclude_none=True, mode="json")
        out += [f"## {decl['name']}", "", decl.get("description", ""), ""]
        out += _schema_lines(decl.get("parameters") or {}) + [""]
    return "\n".join(out)


def _mcp_md(label: str, tasks: list[str], folder: Path) -> str:
    import dataclasses

    from mcp.shared.memory import create_connected_server_and_client_session

    from langslice.core.spec import JobSpec
    from langslice.doors.mcp.server import build_server

    # Each listing gets its own images folder: the server adopts a job folder
    # an earlier listing left, which would carry that listing's tasks.
    own = folder.parent / f"mcp_{label}" / folder.name
    if not own.exists():
        shutil.copytree(folder, own, ignore=shutil.ignore_patterns("langslice"))
    folder = own
    spec = JobSpec(image_folder=str(folder), tasks=tasks, **_BASE)
    server = build_server(lambda image_folder: dataclasses.replace(spec, image_folder=image_folder),
                          str(folder), atlas_loader=atlas_loader())

    async def listed() -> list[Any]:
        async with create_connected_server_and_client_session(server) as client:
            return list((await client.list_tools()).tools)

    out = [f"# MCP tool list: {label} job", ""]
    for tool in asyncio.run(listed()):
        out += [f"## {tool.name}", "", tool.description or "", ""]
        out += _schema_lines(tool.inputSchema) + [""]
    return "\n".join(out)


def _statement_md(label: str, spec: Any, with_notes: bool = False) -> str:
    from langslice.agent.prompt import build_job_statement, display_facts
    from langslice.doors.statement import user_notes_lines

    ctx = _context(spec)
    from langslice.job.job import ingest

    state = ingest(spec, ctx)
    box = _toolbox(spec)
    lo, hi = ctx.position_range
    text = build_job_statement(spec, state, tool_names=box.names, species=ctx.species,
                               pos_lo=lo, pos_hi=hi, axis_ends=ctx.axis_ends,
                               max_resolution=box.max_view_edge, **display_facts(ctx, state))
    if with_notes:
        notes = user_notes_lines("Sections were cut on a cryostat; ignore the stain.")
        text += "\n" + "\n".join(notes)
    return f"# Job statement: {label}\n\n{text}\n"


def _cli_md(folder: Path) -> str:
    """What the agent CLI gives a coding agent of a job: the job folder's
    reference card (``AGENTS.md``) and the statement ``brief`` writes."""
    import dataclasses

    from langslice.doors.card import card_text
    from langslice.doors.cli.brief import build
    from langslice.doors.jobs import create

    images = folder.parent / "cli_images"
    if not images.exists():
        write_sections(images)
    spec = dataclasses.replace(_specs(images)["auto"], resume=False)
    opened = create(spec, atlas_loader=atlas_loader(), emit=lambda _m: None, door="cli")
    try:
        card = card_text(opened.job.layout)
        statement = build(opened, pictures=False).statement
    finally:
        opened.close()
    norm = Normaliser(images)
    return "\n".join(["# Agent CLI: reference card", "", norm.text(card), "",
                      "# Agent CLI: brief statement", "", norm.text(statement), ""])


#: Scripted refusals on the synthetic three-section job (no fits). Each is
#: (label, spec key, calls); a call is (tool, args, kwargs).
def _refusal_script() -> list[tuple[str, str, list[tuple[str, tuple[Any, ...], dict[str, Any]]]]]:
    return [
        ("exploring", "full", [
            ("look", ("section",), {"sections": [0, "nope.png"]}),
            ("look", ("section",), {"sections": [ID0], "view": {"mode": "section"}}),
            ("look", ("section",), {"sections": [ID0], "images": ["purple"]}),
            ("zoom", ([0, 0, 50, 50],), {"picture": 999}),
            ("read_file", ("../outside.txt",), {}),
            ("undo", (), {}),
            ("submit", ("Nothing placed.", [], []), {}),
        ]),
        ("placing", "full", [
            ("position_sections", ([{"id": ID0, "position_mm": 0.1}],), {}),
            ("position_sections", ([{"id": ID0, "position_mm": 9.0}],), {}),
            ("position_sections", ([{"id": ID0, "position_mm": 0.1, "mm": 1}],), {}),
            ("position_sections", (), {"cutting_angles": {"pitch_deg": 1.0, "wobble": 2}}),
            ("mark_damage", (ID1, ["NOPE"]), {}),
            ("interactive_transform", ([{"id": ID0, "rotate_quarter": 45}],), {}),
            ("submit", ("Not yet.", [], []), {}),
        ]),
        ("frozen", "frozen_one", [
            ("position_sections", ([{"id": ID0, "position_mm": 0.3}],), {}),
            ("position_sections", ([{"id": ID1, "position_mm": 0.3}],), {}),
            ("position_sections", (), {"cutting_angles": {"pitch_deg": 1.0, "yaw_deg": 0.0}}),
            ("interactive_transform", ([{"id": ID1, "rotation_deg": 3.0}],), {}),
        ]),
    ]


def _refusals_md(folder: Path) -> str:
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.doors.tools.media import package_result

    norm = Normaliser(folder)
    specs = _specs(folder)
    out = ["# Refusals and gates", ""]
    for label, key, calls in _refusal_script():
        box = _toolbox(specs[key])
        tools = {tool.__name__: tool for tool in box.tools}
        out += [f"## {label}", ""]
        for name, args, kwargs in calls:
            if name not in tools:
                out += [f"### `{name}`", "", "(no such tool in this job)", ""]
                continue
            shown = ", ".join([*(repr(a) for a in args),
                               *(f"{k}={v!r}" for k, v in kwargs.items())])
            packaged = package_result(tools[name](*args, **kwargs))
            body = dict(packaged) if isinstance(packaged, dict) else {"<result>": packaged}
            body.pop(TOOL_MEDIA_PARTS_KEY, None)
            reply = json.dumps(norm.data(body), indent=1, sort_keys=True, ensure_ascii=False)
            out += [f"### `{name}({norm.text(shown)})`", "", "```json", reply, "```", ""]
    return "\n".join(out)


# --- the artifacts -----------------------------------------------------------


def _artifacts(folder: Path) -> dict[str, Callable[[], str]]:
    from tests.synthetic_stack import stub_image_model

    specs = _specs(folder)
    return {
        "declarations_full": lambda: _declarations_md(
            "full (all tasks, image model on)", specs["full"], stub_image_model(specs["full"])),
        "declarations_auto": lambda: _declarations_md(
            "auto picture size, no image model", specs["auto"]),
        "declarations_position": lambda: _declarations_md("positioning only", specs["position"]),
        "declarations_host": lambda: _declarations_md(
            "host (forced pictures, one tool off, deformation required)", specs["host"]),
        "mcp_transform": lambda: _mcp_md("transform", ["transform"], folder),
        "mcp_position": lambda: _mcp_md("position", ["position"], folder),
        "job_statement_default": lambda: _statement_md("all tasks", specs["full"], with_notes=True),
        "job_statement_position": lambda: _statement_md("positioning only", specs["position"]),
        "job_statement_known_position": lambda: _statement_md(
            "transform with positions given", specs["transform_known_position"]),
        "job_statement_frozen": lambda: _statement_md(
            "position and angles frozen, tools disabled", specs["frozen"]),
        "job_statement_auto": lambda: _statement_md("auto picture size", specs["auto"]),
        "refusals": lambda: _refusals_md(folder),
        "cli_card_and_brief": lambda: _cli_md(folder),
    }


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Callable[[], str]]:
    pytest.importorskip("mcp")
    folder = tmp_path_factory.mktemp("agent_text") / "images"
    write_sections(folder)
    return _artifacts(folder)


def _normalised(text: str, folder_hint: Path | None = None) -> str:
    return Normaliser(folder_hint or Path("/nonexistent")).text(text).rstrip() + "\n"


@pytest.mark.parametrize("name", [
    "declarations_full", "declarations_auto", "declarations_position", "declarations_host",
    "mcp_transform", "mcp_position", "job_statement_default", "job_statement_position",
    "job_statement_known_position", "job_statement_frozen", "job_statement_auto", "refusals",
    "cli_card_and_brief",
])
def test_agent_text_matches_snapshot(name: str, artifacts: dict[str, Callable[[], str]]) -> None:
    got = _normalised(artifacts[name]())
    path = SNAPSHOT_DIR / f"{name}.md"
    if os.environ.get(UPDATE_ENV):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(got, encoding="utf-8")
        pytest.fail(f"regenerated {path.name}: review git diff")
    assert path.exists(), f"missing snapshot {path.name}; run with {UPDATE_ENV}=1"
    want = path.read_text(encoding="utf-8")
    if got != want:
        diff = list(difflib.unified_diff(want.splitlines(), got.splitlines(),
                                         f"snapshot/{path.name}", "now", lineterm="", n=1))
        more = len(diff) - MAX_DIFF_LINES
        shown = "\n".join(diff[:MAX_DIFF_LINES])
        if more > 0:
            shown += f"\n... {more} more diff lines"
        pytest.fail(f"{path.name} changed (re-record with {UPDATE_ENV}=1 if intended):\n{shown}")


def test_no_text_names_what_the_job_disables(artifacts: dict[str, Callable[[], str]]) -> None:
    for name in ("declarations_host", "job_statement_frozen"):
        text = artifacts[name]()
        assert "mark_damage" not in text, name
    assert "elastix_affine" not in artifacts["job_statement_frozen"](), "frozen statement"
