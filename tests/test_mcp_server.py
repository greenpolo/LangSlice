"""The MCP server: the linear toolbox as a host sees it, driven by a test client."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ImageContent, TextContent
from PIL import Image

from langslice.core.opening import CLAUDE_MAX_IMAGE_EDGE, CLAUDE_MAX_IMAGE_PATCHES, patches
from langslice.core.spec import JobSpec
from langslice.doors.mcp.server import build_server
from langslice.doors.tools.arguments import CuttingAngles, SectionPosition, normalize_arguments
from langslice.ops.registry import RETIRED, retired_payload
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
    # A retired name is never listed; it only answers when called.
    assert not set(tools) & set(RETIRED)
    for tool in tools.values():
        assert "tool_context" not in tool.inputSchema.get("properties", {})
    assert tools["status"].annotations.readOnlyHint is True
    assert tools["look"].annotations.readOnlyHint is True
    assert tools["position_sections"].annotations.readOnlyHint is False
    assert "End the run" in (tools["submit"].description or "")


def test_start_job_is_text_only_and_show_stack_has_every_section(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        result = await client.call_tool("start_job", {})
        first = result.content[0]
        assert isinstance(first, TextContent)
        count = int(re.search(r"show_stack\(page=(\d+)\) before", first.text).group(1))
        pages = []
        for page in range(1, count + 1):
            pages.extend((await client.call_tool("show_stack", {"page": page})).content)
        return result, pages

    result, pages = _session(server, body)
    assert isinstance(result.content[0], TextContent)
    assert "submit" in result.content[0].text
    assert not any(isinstance(block, ImageContent) for block in result.content)
    images = [block for block in pages if isinstance(block, ImageContent)]
    texts = [block.text for block in pages if isinstance(block, TextContent)]
    # One strip of the three sections, each over the atlas at its starting
    # position (every section has one), so no atlas reference strip.
    assert "Strip 1 of 1: 0: s0.png, 1: s1.png, 2: s2.png" in texts
    assert not any(text.startswith("Atlas reference strip") for text in texts)
    assert len(images) >= 1
    assert all(image.mimeType.startswith("image/") and image.data for image in images)
    for image in images:
        with Image.open(BytesIO(base64.b64decode(image.data))) as picture:
            assert max(picture.size) <= CLAUDE_MAX_IMAGE_EDGE


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
    assert [entry["caption"].split(" ", 2)[:2] for entry in looked["pictures"]] == [
        ["0:", "s0.png"], ["1:", "s1.png"]]
    assert sum(isinstance(block, ImageContent) for block in view.content) == 2
    # The submit gates hold: nothing is positioned yet.
    assert _text(submit)["error"] == "MISSING_POSITIONS"


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
    assert len(applied["pictures"]) >= 1
    assert sum(isinstance(block, ImageContent) for block in encoded.content) == len(
        applied["pictures"])


def test_mcp_takes_what_the_adk_agent_may_send(tmp_path: Path):
    """Handed-over bug 2: corrected indices as numbers in a list of sections
    and nulls inside an object argument passed ADK but failed FastMCP's
    schema check."""
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (
            await client.call_tool("look", {"mode": "section", "sections": [0, "s2.png"]}),
            await client.call_tool("interactive_transform", {"sections": [
                {"id": 1, "flip": None, "rotation_deg": 5.0}], "view": False}),
            await client.call_tool("position_sections", {
                "sections": [{"id": 2, "position_mm": 9.8}], "cutting_angles": None,
                "view": False}),
            await client.call_tool("position_sections", {
                "cutting_angles": {"pitch_deg": None, "glow": None}}),
        )

    numbers, nulls, no_angles, stray = _session(server, body)
    captions = [entry["caption"] for entry in _text(numbers)["pictures"]]
    assert [caption.split(" ", 2)[1] for caption in captions] == ["s0.png", "s2.png"]
    assert sum(isinstance(block, ImageContent) for block in numbers.content) == 2
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


def test_a_retired_tool_answers_with_the_tool_to_use_over_mcp(tmp_path: Path):
    server = build_server(_spec_for, str(_folder(tmp_path)), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return (
            await client.call_tool("fit_affine", {"slices": ["s0.png"]}),
            await client.call_tool("fit_affine", {}),
            await client.call_tool("view_slices", {"slices": ["s0.png"]}),
            await client.call_tool("search_position", {"slice": "s0.png"}),
            await client.call_tool("no_such_tool", {}),
        )

    first, again, view, search, unknown = _session(server, body)
    for reply in (first, again):
        assert not reply.isError
        assert _text(reply) == retired_payload("fit_affine")
        assert _text(reply)["error"] == "RETIRED_TOOL" and _text(reply)["use"] == "elastix_affine"
    assert _text(view)["use"] == "look"
    assert not any(isinstance(block, ImageContent) for block in view.content)
    assert _text(search)["error"] == "RETIRED_TOOL" and "use" not in _text(search)
    # A name that never was a tool is still FastMCP's unknown tool.
    assert unknown.isError
    assert "RETIRED_TOOL" not in unknown.content[0].text


def test_normalize_arguments_only_touches_what_the_schema_would_refuse():
    def tool(sections: list[str], section: str = "",
             entries: list[SectionPosition] = [],  # noqa: B006
             cutting_angles: CuttingAngles = {},  # noqa: B006
             positions_mm: list[float] = []) -> None:  # noqa: B006
        del sections, section, entries, cutting_angles, positions_mm

    assert normalize_arguments(tool, {
        "sections": [3, "a.png", True], "section": 2,
        "entries": [{"id": 1, "position_mm": None}],
        "cutting_angles": {"pitch_deg": None, "yaw_deg": 2.0}, "positions_mm": [1, 2],
    }) == {"sections": ["3", "a.png", True], "section": "2",
           "entries": [{"id": 1}], "cutting_angles": {"yaw_deg": 2.0},
           "positions_mm": [1, 2]}
    assert normalize_arguments(tool, {"sections": [], "section": "x",
                                      "cutting_angles": None}) == {
        "sections": [], "section": "x"}


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
    from langslice.doors.mcp.server import PAGE_BYTES, briefing, open_job

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
    assert all(size <= PAGE_BYTES for size in sizes)
    blocks = [block for page in job.pages for block in page]
    strips = [block.text for block in blocks
              if isinstance(block, TextContent) and block.text.startswith("Strip ")]
    named = [name for text in strips for name in text.split(": ", 1)[1].split(", ")]
    assert named == [f"{record.index_corrected}: {record.id}" for record in job.state.in_order()]
    assert len(strips) > 1
    # Every strip text is followed by its picture, at Claude's size and area.
    for index, block in enumerate(blocks):
        if isinstance(block, TextContent) and block.text.startswith(("Strip ", "Atlas strip ")):
            picture = blocks[index + 1]
            assert isinstance(picture, ImageContent)
            with Image.open(BytesIO(base64.b64decode(picture.data))) as image:
                assert max(image.size) <= CLAUDE_MAX_IMAGE_EDGE
                assert patches(image.size) <= CLAUDE_MAX_IMAGE_PATCHES
    assert not any(isinstance(block, TextContent) and block.text.startswith("Atlas reference")
                   for page in job.pages for block in page)


def test_saved_job_settings_and_offline_submission(tmp_path: Path, monkeypatch: Any):
    from langslice.doors.api import saved_jobs
    from langslice.doors.mcp.server import host_tool, open_saved_job

    folder = _folder(tmp_path)
    monkeypatch.setattr(saved_jobs, "jobs_root", lambda: tmp_path / "jobs")
    params = {"image_folder": str(folder), "pixel_size_um": 25,
              "positions_mm": {f"s{i}.png": i + 1 for i in range(3)},
              "spec": {"tasks": ["transform"], "transform": {"angles": False}},
              "locked": [f"s{i}.png" for i in range(3)], "damaged": {"s0.png": "torn"},
              "preprocessing": {"mode": "auto"}, "notes": "Keep the supplied positions."}
    prepared = saved_jobs.prepare_saved_job(params)
    path = Path(prepared["job_dir"])
    assert path == folder / "langslice"  # the job folder next to the snapshots
    record = json.loads((path / "job.json").read_text())
    assert record["host"]["params"] == {key: value for key, value in params.items()
                                        if key != "notes"}
    assert record["format_version"] == 1 and record["host"]["format"] == 2
    assert record["job_id"] == prepared["job_id"]
    entry = json.loads((tmp_path / "jobs" / f"{prepared['job_id']}.json").read_text())
    assert entry["job_folder"] == str(path)
    assert prepared["job_id"] in prepared["prompt"]
    # The user's notes live in job.json, read by every door; the statement
    # carries them, the copy prompt does not repeat them.
    assert record["notes"] == "Keep the supplied positions."
    assert "Keep the supplied positions." not in prepared["prompt"]
    job = open_saved_job(prepared["job_id"], lambda _n: _ATLAS)
    assert job.notes == "Keep the supplied positions."
    assert job.spec.tasks == ["transform"]
    # The host's damage note is kept with the section; it does not mark it
    # damaged (damaged regions do).
    first = job.state.in_order()[0]
    assert first.damage_note == "torn" and not first.damaged
    assert Path(job.ctx.image_folder).name != "agent_view"  # snapshots are read as they are
    assert job.spec.host_preprocessing["mode"] == "auto"
    submit = next(tool for tool in job.box.tools if tool.__name__ == "submit")
    result = asyncio.run(host_tool(job, submit)(summary="done", notes=[], interval_breaks=[]))
    assert result and job.state.submitted
    saved = json.loads((path / "exports" / "result.json").read_text())
    assert saved["state"]["submitted"] is True
    assert saved["final_updates"] == []
    assert (path / "state.json").exists()
    assert (path / "exports" / "linear_results.json").exists()


def test_host_channel_loopback_envelopes_and_disconnect():
    import socket

    from langslice.doors.mcp.host_channel import HostChannel

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


def test_job_validation(tmp_path: Path, monkeypatch: Any):
    import pytest

    from langslice.doors.api import saved_jobs

    monkeypatch.setattr(saved_jobs, "jobs_root", lambda: tmp_path / "jobs")
    with pytest.raises(ValueError, match="Invalid LangSlice job id"):
        saved_jobs.load_job("../../outside")
    with pytest.raises(ValueError, match="loopback"):
        saved_jobs.validate_channel({"address": "example.com", "port": 10, "token": "a" * 32})


def test_image_model_connected_checks_presence_only(tmp_path: Path, monkeypatch: Any):
    """The MCP door offers trace_borders only when the job's image provider is
    not none and its key or login is present (a private HOME here: no real
    credential is read)."""
    import json

    from langslice.doors.api import setup
    from langslice.doors.api.setup import image_model_connected

    monkeypatch.setenv("HOME", str(tmp_path))
    credentials = tmp_path / ".langslice" / "provider_credentials.json"
    monkeypatch.setattr(setup, "_credentials_path", lambda: credentials)
    for name in ("LANGSLICE_OPENAI_AUTH", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                 "OPENAI_IMAGE_API_KEY", "OPENAI_IMAGE_BASE_URL", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSLICE_OPENAI_AUTH", str(tmp_path / ".langslice" / "openai_auth.json"))
    assert not image_model_connected("none")
    assert not image_model_connected("openai-oauth")
    assert not image_model_connected("openai-api")
    assert not image_model_connected("gemini-api")
    login = tmp_path / ".langslice" / "openai_auth.json"
    login.parent.mkdir()
    login.write_text(json.dumps({"tokens": {"access_token": "fake"}}))
    assert image_model_connected("openai-oauth")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    assert image_model_connected("gemini-api")
    (tmp_path / ".langslice" / "provider_credentials.json").write_text(
        json.dumps({"openai-api": "fake"}))
    assert image_model_connected("openai-api")
    other = tmp_path / "other.json"
    monkeypatch.setenv("LANGSLICE_OPENAI_AUTH", str(other))
    assert not image_model_connected("openai-oauth")  # the override's file, which is absent


def test_saved_job_start_over_mcp_ignores_development_defaults(tmp_path: Path, monkeypatch: Any):
    from langslice.doors.api import saved_jobs, setup
    from langslice.doors.api.models import EngineRequest
    from langslice.hosts.api.service import handle_request

    monkeypatch.setattr(saved_jobs, "jobs_root", lambda: tmp_path / "jobs")
    monkeypatch.setattr(setup, "apply_saved_credentials", lambda: (_ for _ in ()).throw(
        AssertionError("Saving a job must not load model credentials")))
    folder = _folder(tmp_path)
    params = {"image_folder": str(folder), "pixel_size_um": 25,
              "positions_mm": {f"s{i}.png": i + 1 for i in range(3)},
              "spec": {"tasks": ["transform"], "transform": {"angles": False}},
              "locked": [f"s{i}.png" for i in range(3)]}
    result = handle_request(EngineRequest(id="copy", method="mcp.prepare", params=params),
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
    assert "position_sections" not in tools
    assert {"interactive_transform", "elastix_affine", "show_stack", "submit"} <= tools


def _init(folder: Path, *flags: str) -> Any:
    """``langslice-job FOLDER init`` with *flags*, on the test atlas."""
    from langslice.doors.cli.job import init

    envelope = init(str(folder), list(flags), atlas_loader=lambda _n: _ATLAS)
    assert envelope.ok, envelope
    return envelope


def test_a_folder_job_opens_as_saved_next_to_the_sections_and_resumes(tmp_path: Path):
    """A job made with ``langslice-job FOLDER init`` opens through
    ``start_job(image_folder=...)`` as it stands: its saved settings and
    notes, never the server's job flags, resuming its checkpoint."""
    folder = _folder(tmp_path)
    _init(folder, "--tasks", "position", "--interval", "150", "--preprocess", "none",
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
    assert "show_stack(page=1)" in refused["detail"]
    assert "0.150 mm" in briefing.content[0].text  # the saved interval, not a default
    assert "User notes:\nSection 2 is torn." in briefing.content[0].text
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


def test_a_fresh_server_starts_a_folder_job_over_from_its_own_flags(tmp_path: Path):
    """``langslice mcp --fresh``: the server's job flags make the job even
    where one was saved."""
    folder = _folder(tmp_path)
    _init(folder, "--tasks", "position", "--preprocess", "none")
    server = build_server(lambda image_folder: JobSpec(
        image_folder=image_folder, tasks=["transform"], preprocess="none", resume=False),
        str(folder), atlas_loader=lambda _n: _ATLAS, fresh=True)

    async def body(client: Any) -> Any:
        return {tool.name for tool in (await client.list_tools()).tools}

    tools = _session(server, body)
    assert "interactive_transform" in tools and "position_sections" not in tools
    saved = json.loads((folder / "langslice" / "job.json").read_text())
    assert saved["spec"]["tasks"] == ["transform"]


def test_a_folder_job_opened_at_startup_lists_its_tools_from_the_first_request(
    tmp_path: Path,
):
    folder = _folder(tmp_path)
    _init(folder, "--tasks", "reorder,position", "--preprocess", "none")
    server = build_server(lambda _folder: (_ for _ in ()).throw(AssertionError()),
                          str(folder), atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return {tool.name for tool in (await client.list_tools()).tools}

    assert {"start_job", "show_stack", "look", "position_sections", "submit"} <= _session(
        server, body
    )


def test_a_saved_abba_job_forwards_tool_events_with_view_paths(tmp_path: Path, monkeypatch: Any):
    """ABBA's Claude mode: the tools' start/end events and the opening pages reach the
    ABBA channel as agent events, with the saved pictures' paths, never bytes."""
    from langslice.doors.api import saved_jobs
    from langslice.doors.mcp.server import host_tool, open_saved_job

    folder = _folder(tmp_path)
    monkeypatch.setattr(saved_jobs, "jobs_root", lambda: tmp_path / "jobs")
    params = {"image_folder": str(folder), "pixel_size_um": 25,
              "positions_mm": {f"s{i}.png": i + 1 for i in range(3)},
              "spec": {"tasks": ["position", "transform"]}, "z_offset_mm": 5.7}
    prepared = saved_jobs.prepare_saved_job(params)
    session = open_saved_job(prepared["job_id"], lambda _n: _ATLAS)
    sent: list[dict[str, Any]] = []
    assert session.channel is not None
    session.channel.event = sent.append  # type: ignore[method-assign]
    view = next(tool for tool in session.box.tools if tool.__name__ == "look")
    asyncio.run(host_tool(session, view)(mode="section", sections=["s1.png"]))
    events = [payload["event"] for payload in sent if payload["kind"] == "agent_event"]
    kinds = [event["kind"] for event in events]
    assert kinds == ["tool_start", "tool_end"]
    start, end = events
    assert start["name"] == "look" and start["target_ids"] == ["s1.png"]
    assert end["execution_id"] == start["execution_id"]
    session.job.views.flush()
    assert end["views"] and all(Path(path).is_file() for path in end["views"])
    assert all(Path(path).is_relative_to(session.job.layout.folder) for path in end["views"])
    assert "data" not in json.dumps(sent)

    server = build_server(_spec_for, job_id=prepared["job_id"], atlas_loader=lambda _n: _ATLAS)

    async def body(client: Any) -> Any:
        return await client.call_tool("show_stack", {"page": 1})

    from langslice.doors.mcp import host_channel

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(host_channel.HostChannel, "event",
                        lambda self, payload: seen.append(payload))
    _session(server, body)
    seeds = [p["event"] for p in seen if p.get("kind") == "agent_event"
             and p["event"]["kind"] == "seed"]
    assert len(seeds) == 1 and seeds[0]["page"] == 1 and seeds[0]["views"]
    # The ABBA session's facts are kept with the job.
    record = json.loads((Path(prepared["job_dir"]) / "job.json").read_text())
    assert record["host"]["abba"]["z_offset_mm"] == 5.7


# --- door parity: reply budget, parallel calls, closing, the opening at its size ---------


def test_every_reply_stays_within_the_hosts_budget_and_says_when_shrunk():
    from langslice.doors.mcp.server import result_blocks
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.doors.tools.reply import REPLY_BYTES

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
    monkeypatch.setattr(job, "settle_image_corrections", lambda: settled.append("one") or True)

    async def body(client: Any) -> Any:
        return await client.call_tool("start_job", {"image_folder": str(second)})

    _session(built, body)
    assert settled == ["one"]  # replaced: its calls landed before the new job opened
    assert sessions["job"].ctx.image_folder == str(second)
    sessions["job"].close()  # what serve() does when the host goes away
    assert door.Session.close is not None


def test_opening_strips_are_composed_within_the_page_budget(tmp_path: Path):
    from langslice.core.jpeg import encode_jpeg
    from langslice.core.opening import CLAUDE_IMAGE_LIMIT, opening_items
    from langslice.doors.mcp.server import open_job, opening_pages
    from langslice.doors.tools.reply import strip_bytes

    folder = _folder(tmp_path, n=12)
    rng = np.random.default_rng(5)
    for path in folder.glob("*.png"):
        Image.fromarray(rng.integers(0, 256, (900, 1200, 3), dtype=np.uint8)).save(path)
    positions = {f"s{index}.png": 0.1 + 0.05 * index for index in range(12)}
    # 13.2 mm of noise at 11 um/px: about the atlas plane's size, so each
    # section and its atlas (drawn at one scale) both nearly fill a tile.
    session = open_job(JobSpec(image_folder=str(folder), preprocess="none",
                               image_resolution="high",
                               inputs={"positions": positions, "pixel_size_um": 11.0}),
                       atlas_loader=lambda _n: _ATLAS)
    pages = opening_pages(session)
    sent = [base64.b64decode(block.data) for page in pages
            for block in page if isinstance(block, ImageContent)]
    # Noise does not compress: three high tiles a strip, each over its atlas,
    # would pass the budget, so the strips hold fewer sections (more than 4).
    strips = [block.text for page in pages for block in page
              if isinstance(block, TextContent) and block.text.startswith("Strip ")]
    assert len(strips) > 4
    composed = [encode_jpeg(item) for item in opening_items(
        session.state, session.ctx, limit=CLAUDE_IMAGE_LIMIT, max_bytes=strip_bytes())
        if not isinstance(item, str)]
    # The strips the host gets are the composed ones, never shrunk after.
    assert sent == composed
    assert all(len(data) <= strip_bytes() for data in sent)


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
