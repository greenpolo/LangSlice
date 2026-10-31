"""The GUI's loopback server (``langslice.hosts.gui.server``): token, Host
check, GET only, pictures by number, only the listed jobs, plain static names,
and the viewer's strip and renders (404 for a section the job does not hold,
409 with the reason for a refusal, 503 when a picture takes too long)."""

from __future__ import annotations

import http.client
import io
import json
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
from PIL import Image

from langslice.hosts.gui import render
from langslice.hosts.gui.render import RenderPool
from langslice.hosts.gui.server import HEADERS, GuiServer
from langslice.hosts.gui.watch import JobList
from tests.synthetic_stack import ID0, ID1, ID2
from tests.test_gui_render import LOADER, synthetic_job


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
        assert _get(server, prefix + "generator?job=0", method=method)[0] == 405
    status, body, _ = _get(server, prefix + "generator?job=0")
    assert status == 200 and json.loads(body)["prefill"] is None  # no job.json


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


def test_static_files_are_plain_names_with_a_known_extension(server: GuiServer) -> None:
    prefix = f"/{server.token}/"
    assert _get(server, prefix + "tokens.css")[0] == 200
    for path in ("static/app.js", "fonts/x.woff2", "app.exe", "tokens.json", "APP.JS",
                 ".app.js", "app.js/", "%2e%2e%2fsecret.txt", "app.js%2f"):
        assert _get(server, prefix + path)[0] == 404, path


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthetic_job(tmp_path_factory.mktemp("gui_server"))


@pytest.fixture
def served(made: Path, tmp_path: Path) -> Iterator[tuple[GuiServer, Path]]:
    """The synthetic job (a copy) served, drawn on the synthetic atlas."""
    shutil.copytree(made.parent, tmp_path / "stack")
    job = tmp_path / "stack" / "langslice"
    gui = GuiServer(JobList([job], recent=None), renders=RenderPool(atlas_loader=LOADER))
    gui.start()
    yield gui, job
    gui.close()


def _jpeg(body: bytes) -> Image.Image:
    picture = Image.open(io.BytesIO(body))
    assert picture.format == "JPEG"
    return picture


def test_the_strip_and_the_pictures_of_a_job(served: tuple[GuiServer, Path]) -> None:
    gui, _ = served
    prefix = f"/{gui.token}/jobs/0/"
    status, body, _ = _get(gui, prefix + "strip")
    strip = json.loads(body)
    assert status == 200 and strip["axis"]["low"] == "anterior"
    assert [s["id"] for s in strip["sections"]] == [ID0, ID1, ID2]
    assert [a["k"] for a in strip["atlas"]] == [0, 1]
    for kind in ("overlay", "thumb"):
        for section in (ID0, ID2):
            status, body, headers = _get(gui, f"{prefix}render/{kind}/{quote(section)}"
                                         "?edge=300&key=whatever")
            assert (status, headers["Content-Type"]) == (200, "image/jpeg")
            assert max(_jpeg(body).size) <= 320
    # The name is URL-decoded, never read as a path.
    assert _get(gui, prefix + "render/thumb/%73%30.png?edge=64")[0] == 200
    status, body, _ = _get(gui, prefix + "render/atlas/1?edge=100")
    assert status == 200 and max(_jpeg(body).size) <= 128
    snap = json.loads(_get(gui, prefix + "snapshot?since=0&gen=0&sv=-1")[1])
    assert [r["render_key"] for r in snap["sections"]] == [
        s["render_key"] for s in strip["sections"]]


def test_what_a_job_does_not_hold_is_404(served: tuple[GuiServer, Path]) -> None:
    gui, _ = served
    prefix = f"/{gui.token}/jobs/0/"
    for path in ("render/overlay/nope.png", "render/overlay/..%2Fjob.json",
                 "render/thumb/%2Fetc%2Fpasswd", "render/overlay/", "render/other/s0.png",
                 "render/overlay/s0.png/x", "render/atlas/2", "render/atlas/-1",
                 "strip/x"):
        assert _get(gui, prefix + path)[0] == 404, path
    assert _get(gui, f"/{gui.token}/jobs/1/strip")[0] == 404
    assert _get(gui, prefix + "render/overlay/s0.png?edge=big")[0] == 400


def test_a_refused_render_is_409_with_its_reason(served: tuple[GuiServer, Path]) -> None:
    gui, job = served
    record = json.loads((job / "job.json").read_text())
    record["image_folder"] = str(job.parent.parent / "gone")
    (job / "job.json").write_text(json.dumps(record))
    for path in ("strip", f"render/overlay/{ID1}", f"render/thumb/{ID1}", "render/atlas/0"):
        status, body, headers = _get(gui, f"/{gui.token}/jobs/0/{path}")
        assert (status, headers["Content-Type"]) == (409, "application/json"), path
        assert "are not in" in json.loads(body)["error"]


def test_a_picture_that_takes_too_long_is_503(
        served: tuple[GuiServer, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    gui, _ = served
    release = threading.Event()

    def overlay(*_args: object) -> Image.Image:
        release.wait(5)
        return Image.new("RGB", (64, 64))

    monkeypatch.setattr(render, "WAIT_S", 0.1)
    monkeypatch.setattr(render, "_overlay_picture", overlay)
    status, body, _ = _get(gui, f"/{gui.token}/jobs/0/render/overlay/{ID0}")
    release.set()
    assert status == 503 and "still being drawn" in json.loads(body)["error"]
