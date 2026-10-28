"""The GUI's loopback server (``langslice.hosts.gui.server``): token, Host
check, GET only, pictures by number, only the listed jobs."""

from __future__ import annotations

import http.client
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from langslice.hosts.gui.server import HEADERS, GuiServer
from langslice.hosts.gui.watch import JobList


@pytest.fixture
def server(tmp_path: Path) -> Iterator[GuiServer]:
    folder = tmp_path / "images" / "langslice"
    view = folder / "views" / "000001_look"
    view.mkdir(parents=True)
    (view / "view.jpg").write_bytes(b"\xff\xd8 picture one")
    (folder / "views.jsonl").write_text(json.dumps(
        {"seq": 1, "path": "views/000001_look", "tool": "look", "call": 1,
         "sections": ["a.png"]}) + "\n")
    (tmp_path / "secret.txt").write_text("not served")
    gui = GuiServer(JobList([folder], recent=None))
    gui.start()
    yield gui
    gui.close()


def _get(gui: GuiServer, path: str, *, method: str = "GET", host: str | None = None
         ) -> tuple[int, bytes, dict[str, str]]:
    connection = http.client.HTTPConnection("127.0.0.1", gui.port, timeout=5)
    connection.putrequest(method, path, skip_host=True)
    connection.putheader("Host", host if host is not None else f"127.0.0.1:{gui.port}")
    connection.endheaders()
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body, dict(response.getheaders())


def test_the_page_needs_the_token_and_the_host(server: GuiServer) -> None:
    prefix = f"/{server.token}/"
    status, body, headers = _get(server, prefix)
    assert status == 200 and b"<title>LangSlice</title>" in body
    for key, value in HEADERS.items():
        assert headers[key] == value
    assert _get(server, "/")[0] == 404
    assert _get(server, "/wrong-token/")[0] == 404
    assert _get(server, prefix, host="evil.example:80")[0] == 404
    assert _get(server, prefix, host=f"localhost:{server.port}")[0] == 404
    for name in ("app.js", "app.css"):
        assert _get(server, prefix + name)[0] == 200


def test_only_get(server: GuiServer) -> None:
    prefix = f"/{server.token}/"
    for method in ("POST", "PUT", "DELETE", "PATCH", "HEAD"):
        assert _get(server, prefix + "jobs", method=method)[0] == 405


def test_jobs_snapshot_and_pictures(server: GuiServer) -> None:
    prefix = f"/{server.token}/"
    status, body, _ = _get(server, prefix + "jobs")
    jobs = json.loads(body)["jobs"]
    assert status == 200 and [j["index"] for j in jobs] == [0]
    status, body, _ = _get(server, prefix + "jobs/0/snapshot?since=0&gen=0&sv=-1")
    snap = json.loads(body)
    assert status == 200 and snap["reset"] is True
    assert [i["number"] for i in snap["items"] if i["kind"] == "picture"] == [1]
    status, body, headers = _get(server, prefix + "jobs/0/picture/1")
    assert (status, body, headers["Content-Type"]) == (200, b"\xff\xd8 picture one",
                                                       "image/jpeg")


def test_unknown_jobs_pictures_and_paths_are_refused(server: GuiServer) -> None:
    prefix = f"/{server.token}/"
    assert _get(server, prefix + "jobs/1/snapshot")[0] == 404
    assert _get(server, prefix + "jobs/0/picture/2")[0] == 404
    assert _get(server, prefix + "jobs/0/picture/x")[0] == 404
    assert _get(server, prefix + "jobs/0/snapshot?since=x")[0] == 400
    for path in ("../secret.txt", "%2e%2e/secret.txt", "jobs/0/../../secret.txt",
                 "jobs/0/views/000001_look/view.jpg", "index.html"):
        assert _get(server, prefix + path)[0] == 404
