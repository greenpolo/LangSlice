"""Review comments: boxes a person draws in the LangSlice window
(``langslice.job.review``), kept by the window's server
(``POST jobs/<n>/comments``), shown in the snapshot, and read by the agent's
``review_comments`` tool, open until the section changes."""

from __future__ import annotations

import http.client
import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

import langslice
from langslice.hosts.gui.render import RenderPool
from langslice.hosts.gui.server import GuiServer
from langslice.hosts.gui.watch import JobList
from langslice.job.layout import REVIEW_FILE, JobLayout
from langslice.job.review import read_comments
from langslice.tools.descriptions.door_text import review_lines
from tests.synthetic_stack import ID0, ID1, ID2
from tests.test_gui_render import LOADER, synthetic_job


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthetic_job(tmp_path_factory.mktemp("review"))


@pytest.fixture
def served(made: Path, tmp_path: Path) -> Iterator[tuple[GuiServer, Path]]:
    """A copy of the synthetic job, served."""
    shutil.copytree(made.parent, tmp_path / "stack")
    job = tmp_path / "stack" / "langslice"
    gui = GuiServer(JobList([job], recent=None), renders=RenderPool(atlas_loader=LOADER))
    gui.start()
    yield gui, job
    gui.close()


def _post(gui: GuiServer, body: object, *, origin: bool = True) -> tuple[int, dict[str, object]]:
    data = json.dumps(body).encode()
    connection = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=10)
    connection.putrequest("POST", f"/{gui.token}/jobs/0/comments", skip_host=True,
                          skip_accept_encoding=True)
    headers = {"Host": f"127.0.0.1:{gui.port}", "Content-Type": "application/json",
               "Content-Length": str(len(data))}
    if origin:
        headers["Origin"] = f"http://127.0.0.1:{gui.port}"
    for key, value in headers.items():
        connection.putheader(key, value)
    connection.endheaders(data)
    response = connection.getresponse()
    out = response.read()
    connection.close()
    try:
        return response.status, json.loads(out or b"{}")
    except ValueError:
        return response.status, {}


def _snapshot(gui: GuiServer) -> dict[str, object]:
    connection = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=10)
    connection.putrequest("GET", f"/{gui.token}/jobs/0/snapshot", skip_host=True)
    connection.putheader("Host", f"127.0.0.1:{gui.port}")
    connection.endheaders()
    out = connection.getresponse().read()
    connection.close()
    return json.loads(out)


def _files(folder: Path) -> dict[str, int]:
    return {str(p.relative_to(folder)): p.stat().st_mtime_ns
            for p in folder.rglob("*") if p.is_file()}


def test_the_window_keeps_comments_in_the_job_and_nothing_else(
        served: tuple[GuiServer, Path]) -> None:
    gui, job = served
    _snapshot(gui)
    before = _files(job)
    status, answer = _post(gui, {"section": ID1, "box": [0.8, 0.6, 0.2, 0.3],
                                 "text": "  the  border is off  "})
    assert status == 200
    [row] = answer["comments"]
    assert row["section"] == ID1 and row["box"] == [0.2, 0.3, 0.8, 0.6]
    assert row["text"] == "the border is off" and row["revised"] is False
    after = _files(job)
    changed = {name for name in after if before.get(name) != after[name]}
    assert changed == {REVIEW_FILE}
    assert [c["id"] for c in _snapshot(gui)["comments"]] == ["1"]

    refusals = [
        {"section": ID1, "box": [0.1, 0.1, 0.101, 0.5], "text": "x"},
        {"section": ID1, "box": [0.1, 0.1, 0.5], "text": "x"},
        {"section": ID1, "box": ["a", 0, 1, 1], "text": "x"},
        {"section": ID1, "box": [0.1, 0.1, 0.5, 0.5], "text": "   "},
        {"section": "nope.png", "box": [0.1, 0.1, 0.5, 0.5], "text": "x"},
    ]
    for body in refusals:
        status, answer = _post(gui, body)
        assert status == 400 and answer["error"], body
    assert _post(gui, {"section": ID1, "box": [0, 0, 1, 1], "text": "x"},
                 origin=False)[0] == 403

    status, answer = _post(gui, {"remove": "1"})
    assert status == 200 and answer["comments"] == []
    assert read_comments(JobLayout(job)) == []


def test_the_agent_reads_the_open_comments_with_their_boxes(
        served: tuple[GuiServer, Path]) -> None:
    gui, job = served
    _snapshot(gui)
    assert _post(gui, {"section": ID0, "box": [0.25, 0.25, 0.75, 0.5],
                       "text": "warp pulls the edge"})[0] == 200
    assert _post(gui, {"section": ID2, "box": [0.1, 0.1, 0.4, 0.4],
                       "text": "not placed yet"})[0] == 200
    handle = langslice.open_job(str(job), atlas_loader=LOADER)
    try:
        state, spec = handle.job.state, handle.job.spec
        [_blank, line] = review_lines(handle.job.layout, state, spec.switched_off)
        assert "2 comments" in line and ID0 in line and ID2 in line
        reply = handle.review_comments()
        assert reply["status"] == "ok" and reply["open"] == 2
        rows = {row["section"]: row for row in reply["comments"]}
        assert {row["state"] for row in rows.values()} == {"open"}
        for row in rows.values():
            picture = Image.open(next(
                a["path"] for a in reply["artifacts"]
                if a["kind"] == "view" and str(row["picture"]).zfill(6) in a["path"]))
            x0, y0, x1, y1 = row["box_px"]
            assert 0 <= x0 < x1 < picture.width and 0 <= y0 < y1 < picture.height
            red, green, blue = picture.convert("RGB").getpixel((x0 + 1, (y0 + y1) // 2))
            assert red > 190 and green > 160 and red - blue > 80

        handle.interactive_transform(sections=[{"id": ID0, "rotation_deg": 4.0}], view=False)
        reply = handle.review_comments()
        states = {row["section"]: row["state"] for row in reply["comments"]}
        assert states == {ID0: "revised", ID2: "open"}
        assert "picture" not in next(r for r in reply["comments"] if r["section"] == ID0)
        [_blank, line] = review_lines(handle.job.layout, handle.job.state, spec.switched_off)
        assert "a comment on " + ID2 in line
        assert review_lines(handle.job.layout, handle.job.state,
                            [*spec.switched_off, "review_comments"]) == []
    finally:
        handle.close()
    deadline = time.monotonic() + 5
    while True:
        revised = {c["section"]: c["revised"] for c in _snapshot(gui)["comments"]}
        if revised[ID0] or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    assert revised == {ID0: True, ID2: False}


def test_the_command_line_runs_it_for_the_main_session(
        served: tuple[GuiServer, Path], capsys: pytest.CaptureFixture[str]) -> None:
    from langslice.doors.agent_cli.main import main

    gui, job = served
    _snapshot(gui)
    assert _post(gui, {"section": ID1, "box": [0.2, 0.2, 0.6, 0.6],
                       "text": "shift it left"})[0] == 200
    capsys.readouterr()
    assert main([str(job), "review_comments"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    [row] = envelope["result"]["comments"]
    assert row["comment"] == "shift it left" and row["state"] == "open"
    [picture] = envelope["picture_files"]
    assert Path(picture).is_file()


def test_a_job_without_comments_says_so(made: Path, tmp_path: Path) -> None:
    shutil.copytree(made.parent, tmp_path / "stack")
    handle = langslice.open_job(str(tmp_path / "stack" / "langslice"), atlas_loader=LOADER)
    try:
        reply = handle.review_comments()
        assert reply["comments"] == [] and reply["open"] == 0 and reply["message"]
    finally:
        handle.close()
