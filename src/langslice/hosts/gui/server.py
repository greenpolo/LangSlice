"""The GUI's loopback server: the page, the job list, snapshots and pictures.

One ``ThreadingHTTPServer`` on 127.0.0.1, GET only, every path under a random
token (``/<token>/``), the ``Host`` header checked against the port (a page on
another site cannot reach it through a rebound name), a strict
Content-Security-Policy and ``no-store``. Endpoints:

- ``""``, ``app.css``, ``app.js``: the page (``static/``);
- ``jobs``: the job list (:meth:`JobList.listing`);
- ``jobs/<n>/snapshot?since=&gen=&sv=``: :meth:`JobWatcher.snapshot`;
- ``jobs/<n>/picture/<number>``: a saved ``view.jpg`` by its picture number,
  read from disk, never by path.

Only the folders the :class:`~langslice.hosts.gui.watch.JobList` lists are
read. There are no command endpoints: the viewer is read-only.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from langslice.hosts.gui.watch import JobList

STATIC = Path(__file__).with_name("static")
_FILES = {"": ("index.html", "text/html; charset=utf-8"),
          "app.css": ("app.css", "text/css"),
          "app.js": ("app.js", "text/javascript")}
_SNAPSHOT = re.compile(r"jobs/(\d+)/snapshot")
_PICTURE = re.compile(r"jobs/(\d+)/picture/(\d+)")
#: The headers every answer carries.
HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
}


def _int(query: dict[str, list[str]], key: str, default: int) -> int:
    return int(query.get(key, [str(default)])[0])


class GuiServer:
    """The server of one ``langslice gui`` process (see the module text)."""

    def __init__(self, jobs: JobList, *, port: int = 0) -> None:
        self.jobs = jobs
        self.token = secrets.token_urlsafe(24)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _refuse(self) -> None:
                self.send_error(405)

            do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

            def do_HEAD(self) -> None:
                self.send_error(405)

            def do_GET(self) -> None:
                answer = owner.answer(self.path, self.headers.get("Host", ""))
                if isinstance(answer, int):
                    self.send_error(answer)
                    return
                body, mime = answer
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                for key, value in HEADERS.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.http.daemon_threads = True
        self.port = int(self.http.server_port)
        self.url = f"http://127.0.0.1:{self.port}/{self.token}/"
        self.thread: threading.Thread | None = None

    def answer(self, raw_path: str, host: str) -> int | tuple[bytes, str]:
        """``(body, mime)`` for a GET, or the HTTP error code."""
        parsed = urlsplit(raw_path)
        prefix = f"/{self.token}/"
        if host != f"127.0.0.1:{self.port}" or not parsed.path.startswith(prefix):
            return 404
        resource = parsed.path[len(prefix):]
        try:
            query = parse_qs(parsed.query)
            if resource in _FILES:
                name, mime = _FILES[resource]
                return (STATIC / name).read_bytes(), mime
            if resource == "jobs":
                return self._json({"jobs": self.jobs.listing()})
            if match := _SNAPSHOT.fullmatch(resource):
                watcher = self.jobs.get(int(match[1]))
                if watcher is None:
                    return 404
                return self._json(watcher.snapshot(
                    since=_int(query, "since", 0), generation=_int(query, "gen", 0),
                    state_version=_int(query, "sv", -1)))
            if match := _PICTURE.fullmatch(resource):
                watcher = self.jobs.get(int(match[1]))
                path = watcher.picture(int(match[2])) if watcher is not None else None
                if path is None:
                    return 404
                return path.read_bytes(), "image/jpeg"
        except (ValueError, TypeError):
            return 400
        except OSError:
            return 404
        return 404

    @staticmethod
    def _json(value: Any) -> tuple[bytes, str]:
        return json.dumps(value, default=str).encode(), "application/json"

    def start(self) -> None:
        self.thread = threading.Thread(
            target=lambda: self.http.serve_forever(poll_interval=0.1), daemon=True,
            name="langslice-gui")
        self.thread.start()

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)
