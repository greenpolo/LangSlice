"""The MCP server: the linear toolbox as a host sees it, driven by a test client."""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ImageContent, TextContent
from PIL import Image

from langslice.core.pictures.opening import (
    CLAUDE_MAX_IMAGE_EDGE,
    CLAUDE_MAX_IMAGE_PATCHES,
    patches,
)
from langslice.doors.mcp.server import build_server, initialization_options
from langslice.job.settings.spec import JobSpec
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def page_size(blocks: list[Any]) -> int:
    """A reply's serialized bytes, as the host receives it."""
    return len(json.dumps([block.model_dump(exclude_none=True) for block in blocks]).encode())


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
    assert {"start_job", "status", "look", "zoom", "position_sections", "elastix_affine",
            "submit"} <= set(tools)
    for tool in tools.values():
        assert "tool_context" not in tool.inputSchema.get("properties", {})
    assert tools["status"].annotations.readOnlyHint is True
    assert tools["look"].annotations.readOnlyHint is True
    assert tools["position_sections"].annotations.readOnlyHint is False
    assert "End the run" in (tools["submit"].description or "")


def test_tool_results_carry_json_text_and_pictures(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (
            await client.call_tool("status", {}),
            await client.call_tool("look", {"mode": "section",
                                            "sections": ["s0.png", "s1.png"]}),
            await client.call_tool(
                "submit", {"summary": "done", "notes": [], "interval_breaks": []}
            ),
        )

    status, view, submit = _session(server, body)
    assert len(_text(status)["rows"]) == 3
    looked = _text(view)
    assert "images" not in looked
    # Every picture is numbered, in the order the pictures are attached.
    assert [entry["id"] for entry in looked["pictures"]] == [1, 2]
    assert [entry["caption"].split(" ", 1)[0] for entry in looked["pictures"]] == [
        "s0.png", "s1.png"]
    assert sum(isinstance(block, ImageContent) for block in view.content) == 2
    # The submit gates hold: nothing is positioned yet.
    assert _text(submit)["error"] == "MISSING_POSITIONS"


def test_every_mcp_tool_call_is_an_event_in_the_job_folder(tmp_path: Path):
    """Without any ABBA channel, each tool call over MCP leaves a `tool_start`
    and a `tool_end` line in the job folder's logs/events.jsonl."""
    folder = _folder(tmp_path)
    server = build_server(_spec_for, str(folder), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return await client.call_tool("look", {"mode": "section", "sections": ["s1.png"]})

    _session(server, body)
    lines = [json.loads(line) for line in
             (folder / "langslice" / "logs" / "events.jsonl").read_text().splitlines()]
    events = [line for line in lines if line["kind"] in ("tool_start", "tool_end")]
    assert [(e["kind"], e["name"], e["door"]) for e in events] == [
        ("tool_start", "look", "mcp"), ("tool_end", "look", "mcp")]
    start, end = events
    assert start["execution_id"] == end["execution_id"]
    assert start["target_ids"] == ["s1.png"] and start["args"]["sections"] == ["s1.png"]
    assert end["failed"] is False and end["status"] == "ok" and end["step"] == 0
    assert len(end["views"]) == 1 and not Path(end["views"][0]).is_absolute()
    assert "response" not in end


def test_a_tool_that_crashes_is_answered_with_its_exceptions_name(tmp_path: Path, monkeypatch):
    """The pipeline answers an MCP tool's crash (the box answers crashes); the
    event log's ``tool_end`` carries the same reply."""
    from langslice.tools import status

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk gone")

    monkeypatch.setattr(status, "raw_channels", broken)
    folder = _folder(tmp_path)
    server = build_server(_spec_for, str(folder), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return await client.call_tool("status", {})

    reply = _text(_session(server, body))
    assert reply == {"status": "error", "error": "OSError", "message": "disk gone"}
    lines = [json.loads(line) for line in
             (folder / "langslice" / "logs" / "events.jsonl").read_text().splitlines()]
    (end,) = [line for line in lines if line["kind"] == "tool_end"]
    assert end["failed"] is True

def test_unknown_and_misplaced_arguments_are_refused_over_mcp(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (
            await client.call_tool("position_sections", {"pitch_deg": 2.0, "yaw_deg": 0.0}),
            await client.call_tool("position_sections", {
                "cutting_angles": {"pitch_deg": 1.0, "yaw_deg": 0.0, "roll_deg": 2.0}}),
            await client.call_tool("look", {"mode": "section", "sections": ["s0.png"],
                                            "glow": 1}),
            # Claude Desktop may send a nested object as a JSON string.
            await client.call_tool("position_sections", {
                "cutting_angles": '{"pitch_deg": 1.5, "yaw_deg": -2.0}'}),
        )

    top, nested, stray, encoded = _session(server, body)
    refused = _text(top)
    assert refused["error"] == "UNKNOWN_ARGUMENTS"
    assert refused["problems"][0]["unknown"] == ["pitch_deg", "yaw_deg"]
    assert "`pitch_deg` belongs inside `cutting_angles`." in refused["message"]
    assert not any(isinstance(block, ImageContent) for block in top.content)
    assert _text(nested)["problems"][0] == {
        "argument": "cutting_angles", "unknown": ["roll_deg"],
        "accepted": ["pitch_deg", "yaw_deg"],
    }
    assert _text(stray)["problems"][0]["unknown"] == ["glow"]
    assert not any(isinstance(block, ImageContent) for block in stray.content)
    applied = _text(encoded)
    assert applied["status"] == "ok"
    assert applied["cutting_angles_deg"] == {"pitch": 1.5, "yaw": -2.0}
    # The change shows its positioning picture.
    assert len(applied["pictures"]) == 1
    assert sum(isinstance(block, ImageContent) for block in encoded.content) == len(
        applied["pictures"])


def test_mcp_takes_what_the_adk_agent_may_send(tmp_path: Path):
    """Handed-over bug 2: numbers in a list of sections and nulls inside an
    object argument passed ADK but failed FastMCP's schema check. A number
    now reaches the tool, which refuses it by name: sections are named by
    filename."""
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        await client.call_tool("position_sections", {
            "sections": [{"id": "s1.png", "position_mm": 2.0}], "view": False})
        return (
            await client.call_tool("look", {"mode": "section", "sections": [0, "s2.png"]}),
            await client.call_tool("interactive_transform", {"sections": [
                {"id": "s1.png", "flip": None, "rotation_deg": 5.0}], "view": False}),
            await client.call_tool("position_sections", {
                "sections": [{"id": "s2", "position_mm": 9.8}], "cutting_angles": None,
                "view": False}),
            await client.call_tool("position_sections", {
                "cutting_angles": {"pitch_deg": None, "glow": None}}),
        )

    numbers, nulls, no_angles, stray = _session(server, body)
    refused = _text(numbers)
    assert refused["error"] == "UNKNOWN_SLICE_IDS" and refused["unknown"] == ["0"]
    assert refused["filenames"] == ["s0.png", "s1.png", "s2.png"]
    assert "named by filename" in refused["message"]
    assert not any(isinstance(block, ImageContent) for block in numbers.content)
    # A null inside an entry is "not given": the flip stays as it was.
    written = _text(nulls)
    assert written["status"] == "ok"
    entry, = written["results"]
    assert entry["id"] == "s1.png" and entry["orientation"]["flip"] is False
    assert entry["transform"]["rotation_deg"] == 5.0
    row, = written["changed"]
    assert row["id"] == "s1.png" and row["flip"] is False
    # A null object argument is its default: the angles are left as they are.
    placed = _text(no_angles)
    assert placed["written"] == [{"id": "s2.png", "position_mm": 9.8}]
    assert placed["cutting_angles_deg"] == {"pitch": 0.0, "yaw": 0.0}
    # A null under an unknown key is still an unknown key.
    assert _text(stray)["error"] == "UNKNOWN_ARGUMENTS"


def test_a_job_that_disables_tools_and_options_offers_neither_over_mcp(tmp_path: Path):
    def spec_for(folder: str) -> JobSpec:
        return JobSpec(image_folder=folder, preprocess="none", tasks=["position"],
                       disabled=["zoom", "look:positioning", "mark_damage"])

    server = build_server(spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return ((await client.list_tools()).tools,
                await client.call_tool("look", {"mode": "positioning"}))

    tools, looked = _session(server, body)
    by_name = {tool.name: tool for tool in tools}
    assert "zoom" not in by_name and "mark_damage" not in by_name
    assert by_name["look"].inputSchema["properties"]["mode"]["enum"] == [
        "section", "atlas", "overlay"]
    for tool in tools:
        text = json.dumps(tool.inputSchema) + (tool.description or "")
        assert "zoom takes" not in text and "positioning mode" not in text
    # A mode the schema does not offer is refused before the tool runs.
    assert looked.isError or _text(looked).get("error") in ("OPTION_OFF", "UNKNOWN_ARGUMENTS")


def test_open_job_refuses_a_nonlinear_task_with_no_tool_left_to_write_it(tmp_path: Path):
    from langslice.doors.mcp.server import open_job
    from langslice.job.settings.spec import NonlinearSpec

    spec = JobSpec(image_folder=str(_folder(tmp_path)), preprocess="none",
                   tasks=["nonlinear"],
                   disabled=["fireants_syn", "set_landmarks", "landmark_triangulation"],
                   nonlinear=NonlinearSpec(provider="openai-api"))
    # Valid with the image model; the door that cannot reach it has no writer.
    with pytest.raises(ValueError, match="image model is not connected"):
        open_job(spec, atlas_loader=lambda _n: _ATLAS)


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


def test_the_server_advertises_that_its_tool_list_changes():
    """start_job installs the job's tools and says so; a host acts on that
    notice only from a server whose capabilities say tools.listChanged."""
    tools = initialization_options(build_server(_spec_for)).capabilities.tools
    assert tools is not None and tools.listChanged is True


def test_show_stack_page_budget_and_corrected_order(tmp_path: Path):
    from langslice.doors.mcp.pages import PAGE_BYTES
    from langslice.doors.mcp.server import arm_opening, open_job

    folder = _folder(tmp_path, n=36)
    rng = np.random.default_rng(17)
    for path in folder.glob("*.png"):
        Image.fromarray(rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)).save(path)
    job = open_job(_spec_for(str(folder)), atlas_loader=lambda _n: _ATLAS)
    for index, record in enumerate(reversed(job.state.in_order())):
        record.index_corrected = index
    arm_opening(job)
    sizes = [page_size(page) for page in job.pages]
    print(f"show_stack serialized page bytes: {sizes}")
    assert all(size <= PAGE_BYTES for size in sizes)
    blocks = [block for page in job.pages for block in page]
    strips = [block.text for block in blocks
              if isinstance(block, TextContent) and block.text.startswith("Strip ")]
    named = [name for text in strips for name in text.split(": ", 1)[1].split(", ")]
    assert named == [record.id for record in job.state.in_order()]
    assert len(strips) > 1
    # Every strip text is followed by its picture, at Claude's size and area.
    for index, block in enumerate(blocks):
        if isinstance(block, TextContent) and block.text.startswith(("Strip ", "Atlas strip ")):
            picture = blocks[index + 1]
            assert isinstance(picture, ImageContent)
            with Image.open(BytesIO(base64.b64decode(picture.data))) as image:
                assert max(image.size) <= CLAUDE_MAX_IMAGE_EDGE
                assert patches(image.size) <= CLAUDE_MAX_IMAGE_PATCHES
    assert any(isinstance(block, TextContent) and block.text.startswith("Atlas reference")
                   for page in job.pages for block in page)


def test_start_job_takes_a_folder_only_and_names_one_when_none_is_open():
    """``start_job`` has one argument, ``image_folder``; called with no job open
    and no folder, it answers NO_FOLDER naming that argument."""
    server = build_server(_spec_for, atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        return tools["start_job"].inputSchema, await client.call_tool("start_job", {})

    schema, reply = _session(server, body)
    assert set(schema["properties"]) == {"image_folder"}
    refusal = json.loads(reply.content[0].text)
    assert refusal["error"] == "NO_FOLDER"
    assert refusal["message"] == "Name a job: start_job(image_folder=...)."


def _init(folder: Path, *flags: str) -> Any:
    """``langslice-job FOLDER init`` with *flags*, on the test atlas."""
    from langslice.doors.agent_cli.calls import init

    envelope = init(str(folder), list(flags), atlas_loader=lambda _n: _ATLAS)
    assert envelope.ok, envelope
    return envelope


def test_a_folder_job_opens_as_saved_next_to_the_sections_and_resumes(tmp_path: Path):
    """A job made with ``langslice-job FOLDER init`` opens through
    ``start_job(image_folder=...)`` as it stands: its saved settings and
    notes, never the server's job flags, resuming its checkpoint."""
    folder = _folder(tmp_path)
    _init(folder, "--tasks", "position", "--position-interval-um", "150", "--preprocess", "none",
          "--notes", "Section 2 is torn.")

    def server() -> Any:
        return build_server(lambda _folder: (_ for _ in ()).throw(
            AssertionError("A folder's own job must not use the server's job flags")),
            atlas_loader=lambda _n: _ATLAS)

    async def first(client: Any) -> Any:
        briefing = await client.call_tool("start_job", {"image_folder": str(folder)})
        tools = {tool.name for tool in (await client.list_tools()).tools}
        # Every write waits until the opening pages were read.
        early = await client.call_tool("note", {"text": "too early"})
        await client.call_tool("show_stack", {"page": 1})
        note = await client.call_tool("note", {"text": "checked s0"})
        return briefing, tools, early, note

    briefing, tools, early, note = _session(server(), first)
    assert not briefing.isError and not note.isError
    refused = json.loads(early.content[0].text)
    assert refused["error"] == "OPENING_NOT_READ" and refused["pages"] == [1]
    assert "show_stack(page=1)" in refused["message"]
    # The job comes in the launching message, not in start_job's reply.
    assert briefing.content[0].text == (
        f"Job opened: {folder / 'langslice'}. 3 sections. The opening is show_stack "
        "page 1: read every page before any write.")
    assert "position_sections" in tools and "interactive_transform" not in tools
    job_dir = folder / "langslice"
    # The sections are untouched; the job folder sits beside them.
    assert sorted(path.name for path in folder.iterdir()) == [
        "langslice", "s0.png", "s1.png", "s2.png"]

    # A new server (a restarted Claude Desktop) resumes the checkpoint, named
    # by its job folder this time.
    async def again(client: Any) -> Any:
        await client.call_tool("start_job", {"image_folder": str(job_dir)})
        return await client.call_tool("status", {})

    assert not _session(server(), again).isError
    saved = json.loads((job_dir / "state.json").read_text())
    assert any("checked s0" in line for line in saved["notes"])


# --- door parity: reply budget, parallel calls, closing, the opening at its size ---------


def test_every_reply_stays_within_the_hosts_budget_and_says_when_shrunk():
    from langslice.doors.mcp.pages import result_blocks
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.tools.reply import REPLY_BYTES

    rng = np.random.default_rng(3)
    noise = [Image.fromarray(rng.integers(0, 256, (1500, 2000, 3), dtype=np.uint8))
             for _ in range(4)]
    blocks = result_blocks({"status": "ok", TOOL_MEDIA_PARTS_KEY: ["four pictures", *noise]})
    assert page_size(blocks) <= REPLY_BYTES
    body = json.loads(blocks[0].text)
    assert body["images_attached"] == 5
    images = [block for block in blocks if isinstance(block, ImageContent)]
    assert len(images) == 4  # nothing dropped
    note = blocks[-1]
    assert isinstance(note, TextContent) and "shrunk together" in note.text
    assert "from 2000 to" in note.text and "fewer sections" in note.text
    # A reply that fits is untouched and says nothing.
    small = result_blocks({"status": "ok", TOOL_MEDIA_PARTS_KEY: [noise[0].resize((200, 150))]})
    assert len(small) == 2 and isinstance(small[1], ImageContent)


def test_a_replaced_or_stopped_session_settles_its_image_calls(tmp_path: Path, monkeypatch):
    from concurrent.futures import Future

    from langslice.doors.mcp import server as door

    sessions: dict[str, Any] = {}
    first, second = _folder(tmp_path, "one"), _folder(tmp_path, "two")
    built = build_server(_spec_for, str(first), atlas_loader=lambda _n: _ATLAS,
                         sessions=sessions)
    job = sessions["job"].job
    settled: list[str] = []
    done: Future[dict[str, Any]] = Future()
    done.set_result({})
    job.image_jobs["s0.png"] = ("fingerprint", done)
    monkeypatch.setattr(job, "settle_border_traces", lambda: settled.append("one") or True)

    async def body(client: Any) -> Any:
        return await client.call_tool("start_job", {"image_folder": str(second)})

    _session(built, body)
    assert settled == ["one"]  # replaced: its calls landed before the new job opened
    assert sessions["job"].ctx.image_folder == str(second)
    sessions["job"].close()  # what serve() does when the host goes away
    assert door.Session.close is not None


def test_a_refused_argument_is_traced(tmp_path: Path, monkeypatch):
    traces = tmp_path / "traces"
    monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(traces))
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return await client.call_tool("look", {"mode": "section", "sections": ["s0.png"],
                                               "glow": 1})

    _session(server, body)
    records = [json.loads(line) for file in traces.glob("mcp_*.jsonl")
               for line in file.read_text().splitlines()]
    refused = [record for record in records if record["kind"] == "tool_result"]
    assert refused and refused[-1]["name"] == "look"
    assert "UNKNOWN_ARGUMENTS" in refused[-1]["content"][0]["text"]
