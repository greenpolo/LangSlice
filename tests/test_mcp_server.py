"""The MCP server: the linear toolbox as a host sees it, driven by a test client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ImageContent, TextContent
from PIL import Image

from langslice.linear.spec import JobSpec
from langslice.mcp_server.server import build_server
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _folder(root: Path, name: str = "stack", n: int = 3) -> Path:
    folder = root / name
    folder.mkdir()
    for index in range(n):
        Image.fromarray(
            np.full((30, 40, 3), (40 + 10 * index) % 256, dtype=np.uint8)
        ).save(folder / f"s{index}.png")
    return folder


def _spec_for(folder: str) -> JobSpec:
    return JobSpec(image_folder=folder, preprocess="none")


def _session(server: Any, body: Any) -> Any:
    async def main() -> Any:
        async with create_connected_server_and_client_session(server) as client:
            return await body(client)

    return asyncio.run(main())


def _text(result: Any) -> dict[str, Any]:
    first = result.content[0]
    assert isinstance(first, TextContent)
    return json.loads(first.text)


def test_a_folder_given_at_startup_lists_the_toolbox_without_adk_context(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in _session(server, body)}
    assert {"start_job", "status", "view_slices", "view_atlas", "set_positions",
            "submit"} <= set(tools)
    for tool in tools.values():
        assert "tool_context" not in tool.inputSchema.get("properties", {})
    assert tools["status"].annotations.readOnlyHint is True
    assert tools["set_positions"].annotations.readOnlyHint is False
    assert "End the run" in (tools["submit"].description or "")


def test_start_job_is_text_only_and_show_stack_has_every_section(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        result = await client.call_tool("start_job", {})
        pages = []
        for page in range(1, 3):
            pages.extend((await client.call_tool("show_stack", {"page": page})).content)
        return result, pages

    result, pages = _session(server, body)
    assert isinstance(result.content[0], TextContent)
    assert "submit" in result.content[0].text
    assert not any(isinstance(block, ImageContent) for block in result.content)
    images = [block for block in pages if isinstance(block, ImageContent)]
    # One per section, plus the atlas strip.
    assert len(images) >= 3
    assert all(image.mimeType.startswith("image/") and image.data for image in images)


def test_tool_results_carry_json_text_and_pictures(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (
            await client.call_tool("status", {}),
            await client.call_tool("view_slices", {"slices": ["s0.png", "s1.png"]}),
            await client.call_tool(
                "submit", {"summary": "done", "notes": [], "interval_breaks": []}
            ),
        )

    status, view, submit = _session(server, body)
    assert len(_text(status)["rows"]) == 3
    assert "images" not in _text(view)
    assert sum(isinstance(block, ImageContent) for block in view.content) == 2
    # The submit gates hold: nothing is positioned yet.
    assert _text(submit)["error"] == "MISSING_POSITIONS"


def test_unknown_and_misplaced_arguments_are_refused_over_mcp(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (
            await client.call_tool("view_slices", {"slices": ["s0.png"], "mode": "section"}),
            await client.call_tool("view_slices", {"slices": ["s0.png"],
                                                   "view": {"mode": "section", "glow": 1}}),
            # Claude Desktop may send a nested object as a JSON string.
            await client.call_tool("view_slices", {"slices": ["s0.png"],
                                                   "view": '{"mode": "channels"}'}),
        )

    top, nested, encoded = _session(server, body)
    refused = _text(top)
    assert refused["error"] == "UNKNOWN_ARGUMENTS"
    assert refused["problems"][0]["unknown"] == ["mode"]
    assert "`mode` belongs inside `view`." in refused["message"]
    assert not any(isinstance(block, ImageContent) for block in top.content)
    assert _text(nested)["problems"][0] == {
        "argument": "view", "unknown": ["glow"],
        "accepted": ["mode", "channels", "atlas_channels", "atlas_opacity", "regions",
                     "outlines", "border_color", "border_thickness", "zoom", "deformation"],
    }
    assert _text(encoded)["view"]["mode"] == "channels"
    assert sum(isinstance(block, ImageContent) for block in encoded.content) == 1


def test_without_a_folder_start_job_opens_one_and_the_tools_appear(tmp_path: Path):
    first = _folder(tmp_path, "first")
    second = _folder(tmp_path, "second", n=2)
    server = build_server(_spec_for, atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        before = {tool.name for tool in (await client.list_tools()).tools}
        refused = await client.call_tool("start_job", {})
        await client.call_tool("start_job", {"image_folder": str(first)})
        during = {tool.name for tool in (await client.list_tools()).tools}
        await client.call_tool("start_job", {"image_folder": str(second)})
        status = await client.call_tool("status", {})
        return before, refused, during, status

    before, refused, during, status = _session(server, body)
    assert before == {"start_job"}
    assert _text(refused)["error"] == "NO_FOLDER"
    assert {"status", "submit"} <= during
    # The second folder replaced the first job.
    assert len(_text(status)["rows"]) == 2


def test_show_stack_page_budget_and_corrected_order(tmp_path: Path):
    from langslice.mcp_server.server import PAGE_BYTES, briefing, open_job, page_size

    folder = _folder(tmp_path, n=36)
    rng = np.random.default_rng(17)
    for path in folder.glob("*.png"):
        Image.fromarray(rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)).save(path)
    job = open_job(_spec_for(str(folder)), atlas_loader=lambda _n: _ATLAS)
    for index, record in enumerate(reversed(job.state.in_order())):
        record.index_corrected = index
    briefing(job)
    sizes = [page_size(page) for page in job.pages]
    print(f"show_stack serialized page bytes: {sizes}")
    assert len(sizes) > 2
    assert all(size <= PAGE_BYTES for size in sizes)
    labels = [block.text for page in job.pages[:-1] for block in page
              if isinstance(block, TextContent) and ": s" in block.text]
    assert labels == [f"{record.index_corrected}: {record.id}" for record in job.state.in_order()]
    assert sum(isinstance(block, ImageContent) for page in job.pages[:-1] for block in page) == 36
    assert isinstance(job.pages[-1][0], TextContent)
    assert "Atlas reference" in job.pages[-1][0].text


def test_saved_job_settings_and_offline_submission(tmp_path: Path, monkeypatch: Any):
    from langslice.api import claude_jobs
    from langslice.mcp_server.server import host_tool, open_saved_job

    folder = _folder(tmp_path)
    monkeypatch.setattr(claude_jobs, "jobs_root", lambda: tmp_path / "jobs")
    params = {"image_folder": str(folder), "pixel_size_um": 25,
              "positions_mm": {f"s{i}.png": i + 1 for i in range(3)},
              "spec": {"tasks": ["transform"], "transform": {"angles": False}},
              "locked": [f"s{i}.png" for i in range(3)], "damaged": {"s0.png": "torn"},
              "preprocessing": {"mode": "auto"}, "notes": "Keep the supplied positions."}
    prepared = claude_jobs.prepare_claude(params)
    path = Path(prepared["job_dir"])
    record = json.loads((path / "job.json").read_text())
    assert record["params"] == {key: value for key, value in params.items() if key != "notes"}
    assert record["format_version"] == 1
    assert prepared["job_id"] in prepared["prompt"]
    assert "Keep the supplied positions." in prepared["prompt"]
    job = open_saved_job(prepared["job_id"], lambda _n: _ATLAS)
    assert job.spec.tasks == ["transform"]
    assert job.state.in_order()[0].damaged
    assert Path(job.ctx.image_folder).name != "agent_view"  # snapshots are read as they are
    assert job.spec.host_preprocessing["mode"] == "auto"
    submit = next(tool for tool in job.box.tools if tool.__name__ == "submit")
    result = asyncio.run(host_tool(job, submit)(summary="done", notes=[], interval_breaks=[]))
    assert result and job.state.submitted
    saved = json.loads((path / "result.json").read_text())
    assert saved["state"]["submitted"] is True
    assert saved["final_updates"] == []
    assert (path / "linear_state.json").exists()
    assert (path / "linear_results.json").exists()


def test_host_channel_loopback_envelopes_and_disconnect():
    import socket

    from langslice.mcp_server.host_channel import HostChannel

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        settings = {"address": "127.0.0.1", "port": listener.getsockname()[1], "token": "a" * 32}
        channel = HostChannel("abc", settings)
        conn, _address = listener.accept()
        with conn, conn.makefile("r") as stream:
            assert json.loads(stream.readline()) == {"token": "a" * 32}
            channel.event({"kind": "checkpoint", "host_updates": [{"id": "s0", "position_mm": 1}]})
            event = json.loads(stream.readline())
            assert event["id"] == "abc" and event["type"] == "event"
            assert event["event"]["payload"]["host_updates"][0]["position_mm"] == 1
            channel.send({"type": "result", "result": {"final_updates": []}})
            assert json.loads(stream.readline())["type"] == "result"
        channel.close()
        channel.event({"kind": "checkpoint"})
    # A refused localhost connection is a non-fatal offline job.
    channel = HostChannel("abc", settings)
    channel.event({"kind": "checkpoint"})
    assert channel.socket is None


def test_job_validation_and_no_image_provider(tmp_path: Path, monkeypatch: Any):
    import pytest

    from langslice.api import claude_jobs
    from langslice.mcp_server.server import open_job

    monkeypatch.setattr(claude_jobs, "jobs_root", lambda: tmp_path / "jobs")
    with pytest.raises(ValueError, match="Invalid LangSlice job id"):
        claude_jobs.load_job("../../outside")
    with pytest.raises(ValueError, match="loopback"):
        claude_jobs.validate_channel({"address": "example.com", "port": 10, "token": "a" * 32})
    with pytest.raises(ValueError, match="Image generation"):
        open_job(JobSpec(image_folder=str(tmp_path), tasks=["nonlinear"]))


def test_saved_job_start_over_mcp_ignores_development_defaults(tmp_path: Path, monkeypatch: Any):
    from langslice.api import claude_jobs, setup
    from langslice.api.models import EngineRequest
    from langslice.api.service import handle_request

    monkeypatch.setattr(claude_jobs, "jobs_root", lambda: tmp_path / "jobs")
    monkeypatch.setattr(setup, "apply_saved_credentials", lambda: (_ for _ in ()).throw(
        AssertionError("Claude preparation must not load model credentials")))
    folder = _folder(tmp_path)
    params = {"image_folder": str(folder), "pixel_size_um": 25,
              "positions_mm": {f"s{i}.png": i + 1 for i in range(3)},
              "spec": {"tasks": ["transform"], "transform": {"angles": False}},
              "locked": [f"s{i}.png" for i in range(3)]}
    result = handle_request(EngineRequest(id="copy", method="claude.prepare", params=params),
                            lambda _event: None)
    job_id = result.result["job_id"]
    server = build_server(lambda _folder: (_ for _ in ()).throw(
        AssertionError("Saved jobs must not use development CLI settings")),
        atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        first = await client.call_tool("start_job", {"job_id": job_id})
        tools = {tool.name for tool in (await client.list_tools()).tools}
        repeat = await client.call_tool("start_job", {"job_id": job_id})
        return first, tools, repeat

    first, tools, repeat = _session(server, body)
    assert not first.isError and not repeat.isError
    assert all(isinstance(block, TextContent) for block in first.content)
    assert "set_positions" not in tools
    assert {"adjust_transforms", "show_stack", "submit"} <= tools


def test_cli_prepared_folder_job_keeps_the_folder_clean_and_resumes(
    tmp_path: Path, monkeypatch: Any, capsys: Any
):
    from langslice.api import claude_jobs
    from langslice.cli import main

    monkeypatch.setattr(claude_jobs, "jobs_root", lambda: tmp_path / "jobs")
    folder = _folder(tmp_path)
    main(["claude", "prepare", str(folder), "--tasks", "position", "--interval", "150",
          "--preprocess", "none", "--notes", "Section 2 is torn."])
    prompt = capsys.readouterr().out
    job_id = next(iter((tmp_path / "jobs").iterdir())).name
    assert f'start_job(job_id="{job_id}")' in prompt
    assert "interval 150 µm" in prompt and "Section 2 is torn." in prompt

    def server() -> Any:
        return build_server(lambda _folder: (_ for _ in ()).throw(
            AssertionError("Saved jobs must not use development CLI settings")),
            atlas_loader=lambda _n: _ATLAS)

    async def first(client: Any) -> Any:
        briefing = await client.call_tool("start_job", {"job_id": job_id})
        tools = {tool.name for tool in (await client.list_tools()).tools}
        note = await client.call_tool("note", {"text": "checked s0"})
        return briefing, tools, note

    briefing, tools, note = _session(server(), first)
    assert not briefing.isError and not note.isError
    assert "0.150 mm" in briefing.content[0].text  # the saved interval, not a default
    assert "set_positions" in tools and "adjust_transforms" not in tools
    job_dir = tmp_path / "jobs" / job_id
    assert (job_dir / "linear_state.json").exists()
    assert sorted(path.name for path in folder.iterdir()) == ["s0.png", "s1.png", "s2.png"]

    # A new server (a restarted Claude Desktop) resumes the saved checkpoint.
    async def again(client: Any) -> Any:
        await client.call_tool("start_job", {"job_id": job_id})
        return await client.call_tool("status", {})

    _session(server(), again)
    saved = json.loads((job_dir / "linear_state.json").read_text())
    assert any("checked s0" in line for line in saved["notes"])


def test_a_saved_job_opened_at_startup_lists_its_tools_from_the_first_request(
    tmp_path: Path, monkeypatch: Any
):
    from langslice.api import claude_jobs

    monkeypatch.setattr(claude_jobs, "jobs_root", lambda: tmp_path / "jobs")
    job = claude_jobs.prepare_folder(
        JobSpec(image_folder=str(_folder(tmp_path)), tasks=["reorder", "position"],
                preprocess="none")
    )
    server = build_server(_spec_for, job_id=job["job_id"], atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return {tool.name for tool in (await client.list_tools()).tools}

    assert {"start_job", "show_stack", "reorder_slices", "set_positions", "submit"} <= _session(
        server, body
    )
