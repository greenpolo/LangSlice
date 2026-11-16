"""The GUI's loopback server: the page, the job list, snapshots and pictures.

One ``ThreadingHTTPServer`` on 127.0.0.1, every path under a random token
(``/<token>/``), the ``Host`` header checked against the port (a page on
another site cannot reach it through a rebound name), a strict
Content-Security-Policy and ``no-store``. GET endpoints:

- ``""`` (``index.html``, served at that one address) and every other file
  directly in ``static/`` whose name :data:`STATIC_NAME` matches (the page's
  scripts, styles, fonts and logo), its type from its extension
  (:data:`STATIC_TYPES`);
- ``jobs``: the job list (``JobList.listing``);
- ``generator``: the prompt generator's choices (``generator.payload``);
  ``generator?job=<n>`` adds ``prefill``, the form as that job's
  ``job.json`` holds it (``generator.prefill``);
- ``jobs/<n>/snapshot?since=&gen=&sv=``: ``JobWatcher.snapshot``;
- ``jobs/<n>/picture/<number>``: a saved ``view.jpg`` by its picture number,
  read from disk, never by path;
- ``jobs/<n>/strip``: the positioning strip's facts (``JobRenderer.strip``);
- ``jobs/<n>/render/overlay/<section>?edge=&key=`` and
  ``jobs/<n>/render/thumb/<section>?edge=&key=``: a section under its
  current registration, or its thumbnail, drawn now (``render.py``); the
  section is its filename, URL-encoded, and must be one of the job's;
  ``key`` is the page's own (the current state is drawn whatever it says);
- ``jobs/<n>/render/atlas/<k>?edge=``: the strip's *k*-th atlas thumbnail;
- ``settings``: the Settings tab's state (``Settings.state``);
- ``jobs/<n>/export``: the job's last export (``export.Exports.state``).

The snapshot carries the job's review comments (``review.json``,
:mod:`langslice.job.review`).

A render that is refused answers 409 with ``{"error"}``, one not drawn
within ``render.WAIT_S`` 503. Only the folders the job list holds are
read, and the viewer writes nothing of the registration: every job endpoint
is GET only but ``jobs/<n>/comments`` and the export's, and a render writes
nothing to the job folder. Three kinds of POST, all through
:meth:`GuiServer.post`, which also needs this server's ``Origin``: the
Settings tab's (``settings.POSTS``); the review comments'
(``jobs/<n>/comments``: ``{"section", "box", "text"}`` keeps a comment,
``{"remove": id}`` takes one away; the job's ``review.json``); and the
Export menu's, ``jobs/<n>/export`` (``{"format"}``: start writing the job's
export, under its ``exports/`` folder only) and ``jobs/<n>/export/reveal``
(open that folder in the system's file browser).
"""

from __future__ import annotations

import errno
import json
import re
import secrets
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

from langslice.hosts.gui import generator
from langslice.hosts.gui.comments import CommentRow, CommentWatch
from langslice.hosts.gui.export import ExportError, Exports, reveal
from langslice.hosts.gui.render import (
    DEFAULT_EDGE_PX,
    JobRenderer,
    RenderBusy,
    RenderError,
    RenderPool,
    UnknownSection,
    edge_px,
)
from langslice.hosts.gui.settings import POSTS, Settings, SettingsError
from langslice.hosts.gui.watch import JobList, JobWatcher
from langslice.job.layout import JobLayout
from langslice.job.review import ReviewError, add_comment, remove_comment

STATIC = Path(__file__).with_name("static")
#: A file of ``static/`` the page may load: a plain name, no folder.
STATIC_NAME = re.compile(
    r"[a-z0-9][a-z0-9.-]*\.(html|css|js|woff2|svg|txt|ico)")
#: Each static file's type, by extension.
STATIC_TYPES = {"html": "text/html; charset=utf-8", "css": "text/css",
                "js": "text/javascript", "woff2": "font/woff2",
                "svg": "image/svg+xml", "txt": "text/plain; charset=utf-8",
                "ico": "image/x-icon"}
# --- Settings tab (hosts/gui/settings.py) ---
#: The largest POST body read.
MAX_BODY = 16 * 1024
# --- end Settings tab ---
_SNAPSHOT = re.compile(r"jobs/(\d+)/snapshot")
#: The review comments' POST (:meth:`GuiServer._comment`).
_COMMENTS = re.compile(r"jobs/(\d+)/comments")
#: What a comment on a job without sections is told.
NO_SECTIONS = "This job has no sections to comment on yet."
#: What a comment that could not be written is told.
NOT_SAVED = "The comment could not be saved in the job folder."
_PICTURE = re.compile(r"jobs/(\d+)/picture/(\d+)")
_STRIP = re.compile(r"jobs/(\d+)/strip")
_RENDER = re.compile(r"jobs/(\d+)/render/(overlay|thumb)/([^/]+)")
_ATLAS = re.compile(r"jobs/(\d+)/render/atlas/(\d+)")
_EXPORT = re.compile(r"jobs/(\d+)/export(/reveal)?")
#: An answer: an HTTP error code, or the body, its type and the code.
Answer = int | tuple[bytes, str] | tuple[bytes, str, int]
#: A job route's handler: the job, the path's match and the query.
_Route = Callable[[JobWatcher, re.Match[str], dict[str, list[str]]], Answer]
#: The pictures of one section a ``render/<kind>/<section>`` path draws.
_SECTION_PICTURES = {"overlay": JobRenderer.overlay,
                     "thumb": JobRenderer.thumb}
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


def _exports(exports: Exports | None) -> Exports:
    """Return the Export menu's exports, a new one for None."""
    return Exports() if exports is None else exports


def port_in_use(exc: OSError) -> bool:
    """Tell whether a server failed to bind because the port is taken.

    Windows refuses a port another program listens on as forbidden
    (WSAEACCES), not as in use.

    Parameters
    ----------
    exc
        The error binding the server's socket raised.

    Returns
    -------
    bool
        True when another socket holds the port.
    """
    if sys.platform == "win32":
        return exc.winerror in (errno.WSAEADDRINUSE, errno.WSAEACCES)
    return exc.errno == errno.EADDRINUSE


class GuiServer:
    """The server of one ``langslice gui`` process (see the module text).

    *renders* draws the viewer's pictures (one pool for the server; None
    makes one on BrainGlobe's atlases).
    """

    def __init__(self, jobs: JobList, *, port: int = 0,
                 settings: Settings | None = None,
                 renders: RenderPool | None = None,
                 exports: Exports | None = None) -> None:
        self.jobs = jobs
        self.exports = _exports(exports)
        self.settings = settings if settings is not None else Settings()
        self.renders = renders if renders is not None else RenderPool()
        self.token = secrets.token_urlsafe(24)
        self._comments: dict[Path, CommentWatch] = {}
        self._comments_lock = threading.Lock()
        self._routes: tuple[tuple[re.Pattern[str], _Route], ...] = (
            (_SNAPSHOT, self._snapshot), (_PICTURE, self._picture),
            (_STRIP, self._strip), (_RENDER, self._render),
            (_ATLAS, self._atlas), (_EXPORT, self._export_state))
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _refuse(self) -> None:
                self.send_error(405)

            do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

            def do_HEAD(self) -> None:
                self.send_error(405)

            # --- Settings tab: the only POSTs (owner.post) ---
            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                body = (self.rfile.read(length) if 0 <= length <= MAX_BODY else None)
                self._send(owner.post(self.path, self.headers, body))
            # --- end Settings tab ---

            def do_GET(self) -> None:
                self._send(owner.answer(self.path, self.headers.get("Host", "")))

            def _send(self, answer: Answer) -> None:
                if isinstance(answer, int):
                    self.send_error(answer)
                    return
                body, mime, code = (answer if len(answer) == 3
                                    else (answer[0], answer[1], 200))
                self.send_response(code)
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

    def answer(self, raw_path: str, host: str) -> Answer:
        """Answer a GET: ``(body, mime[, code])``, or an HTTP error code."""
        parsed = urlsplit(raw_path)
        prefix = f"/{self.token}/"
        if host != f"127.0.0.1:{self.port}" or not parsed.path.startswith(prefix):
            return 404
        resource = parsed.path[len(prefix):]
        try:
            query = parse_qs(parsed.query)
            if resource == "" or (STATIC_NAME.fullmatch(resource)
                                  and resource != "index.html"):
                name = resource or "index.html"
                kind = STATIC_TYPES[name.rsplit(".", 1)[1]]
                return (STATIC / name).read_bytes(), kind
            if resource == "jobs":
                return self._json({"jobs": self.jobs.listing(),
                                   "home": str(Path.home())})
            if resource == "generator":  # prompt generator
                return self._generator(query)
            if resource == "settings":  # the Settings tab (hosts/gui/settings.py)
                return self._json(self.settings.state())
            return self._job_route(resource, query)
        except (ValueError, TypeError):
            return 400
        except OSError:
            return 404

    def _job_route(self, resource: str, query: dict[str, list[str]]) -> Answer:
        """Answer a ``jobs/<n>/...`` route.

        404 for another path, or an index the job list does not hold.
        """
        for pattern, route in self._routes:
            match = pattern.fullmatch(resource)
            if match is None:
                continue
            watcher = self.jobs.get(int(match[1]))
            if watcher is None:
                return 404
            return route(watcher, match, query)
        return 404

    def _snapshot(self, watcher: JobWatcher, _match: re.Match[str],
                  query: dict[str, list[str]]) -> Answer:
        data = watcher.snapshot(
            since=_int(query, "since", 0), generation=_int(query, "gen", 0),
            state_version=_int(query, "sv", -1))
        rows = self._comment_rows(watcher)
        data["comments"] = [asdict(row) for row in rows]
        return self._json(data)

    def _picture(self, watcher: JobWatcher, match: re.Match[str],
                 _query: dict[str, list[str]]) -> Answer:
        path = watcher.picture(int(match[2]))
        if path is None:
            return 404
        return path.read_bytes(), "image/jpeg"

    def _strip(self, watcher: JobWatcher, _match: re.Match[str],
               _query: dict[str, list[str]]) -> Answer:
        try:
            return self._json(asdict(self._renderer(watcher).strip()))
        except RenderError as exc:
            return self._error(exc, 409)

    def _render(self, watcher: JobWatcher, match: re.Match[str],
                query: dict[str, list[str]]) -> Answer:
        """Draw a section's overlay or thumbnail, named by its filename."""
        draw = _SECTION_PICTURES[match[2]]
        edge = edge_px(_int(query, "edge", DEFAULT_EDGE_PX))
        return self._jpeg(partial(
            draw, self._renderer(watcher), unquote(match[3]), edge))

    def _atlas(self, watcher: JobWatcher, match: re.Match[str],
               query: dict[str, list[str]]) -> Answer:
        renderer = self._renderer(watcher)
        edge = edge_px(_int(query, "edge", DEFAULT_EDGE_PX))
        return self._jpeg(partial(renderer.atlas_thumb, int(match[2]), edge))

    def _export_state(self, watcher: JobWatcher, match: re.Match[str],
                      _query: dict[str, list[str]]) -> Answer:
        if match[2]:
            return 405
        return self._json(asdict(self.exports.state(watcher.folder)))

    def _export_post(self, resource: str, request: object) -> Answer:
        """Start a listed job's export, or open its exports folder."""
        match = _EXPORT.fullmatch(resource)
        watcher = None if match is None else self.jobs.get(int(match[1]))
        if match is None or watcher is None:
            return 404
        try:
            if match[2]:
                reveal(watcher.folder)
                return self._json({"opened": True})
            fields = cast("dict[str, object]", request)
            kind = fields.get("format") if isinstance(request, dict) else None
            started = self.exports.start(watcher.folder, str(kind))
            return self._json(asdict(started))
        except (ExportError, OSError) as exc:
            return (*self._json({"error": str(exc)}), 400)

    def _renderer(self, watcher: JobWatcher) -> JobRenderer:
        return self.renders.renderer(watcher.folder, watcher.current_stack)

    def _jpeg(self, draw: Callable[[], bytes]) -> Answer:
        """Answer with a drawn picture, or why there is none.

        404 for a section or atlas position the job does not have, 409 with
        ``{"error"}`` for a refusal, 503 when the picture took too long.
        """
        try:
            return draw(), "image/jpeg"
        except UnknownSection:
            return 404
        except RenderError as exc:
            return self._error(exc, 409)
        except RenderBusy as exc:
            return self._error(exc, 503)

    def _error(self, error: Exception, code: int) -> Answer:
        body, mime = self._json({"error": str(error)})
        return body, mime, code

    def _generator(
        self, query: dict[str, list[str]]
    ) -> int | tuple[bytes, str]:
        """Answer ``generator``, with a listed job's form for ``job=<n>``.

        The prompt generator's choices; with ``job=<n>``, also ``prefill``:
        the form as that job's ``job.json`` holds it (null without one). An
        index the list does not hold is 404.
        """
        data = generator.payload()
        if "job" in query:
            watcher = self.jobs.get(_int(query, "job", -1))
            if watcher is None:
                return 404
            form = generator.prefill(watcher.folder)
            data["prefill"] = asdict(form) if form is not None else None
        return self._json(data)

    # --- Settings tab (hosts/gui/settings.py) ---
    def post(self, raw_path: str, headers: Any,
             body: bytes | None) -> int | tuple[bytes, str] | tuple[bytes, str, int]:
        """A POST to a Settings or Export resource: the token, the ``Host``, an
        ``Origin`` of this server (a page of another site cannot send one),
        ``Sec-Fetch-Site`` same-origin when sent, a JSON body of at most
        :data:`MAX_BODY` bytes. Every other path is GET only (405). A refused
        request is answered ``{"error"}`` with 400."""
        parsed = urlsplit(raw_path)
        prefix = f"/{self.token}/"
        origin = f"http://127.0.0.1:{self.port}"
        if (headers.get("Host", "") != f"127.0.0.1:{self.port}"
                or not parsed.path.startswith(prefix)):
            return 404
        resource = parsed.path[len(prefix):]
        comments = _COMMENTS.fullmatch(resource)
        if (resource not in POSTS and comments is None
                and _EXPORT.fullmatch(resource) is None):
            return 405
        if (headers.get("Origin") != origin
                or headers.get("Sec-Fetch-Site", "same-origin") != "same-origin"):
            return 403
        if not (headers.get("Content-Type") or "").startswith("application/json"):
            return 415
        if body is None:
            return 413
        try:
            request = json.loads(body.decode("utf-8") or "null")
        except (UnicodeDecodeError, ValueError):
            return 400
        if comments is not None:
            index = int(comments.group(1))
            return self._comment(index, cast("object", request))
        if resource not in POSTS:
            return self._export_post(resource, cast("object", request))
        try:
            return self._json(self.settings.post(resource, request))
        except SettingsError as exc:
            return (*self._json({"error": str(exc)}), 400)
        except OSError:
            return (*self._json({"error": "The settings could not be saved in ~/.langslice/."}),
                    500)
    # --- end Settings tab ---

    def _comment(
        self, index: int, request: object
    ) -> int | tuple[bytes, str] | tuple[bytes, str, int]:
        """Keep or remove a review comment of job *index*.

        A refused one is ``{"error"}`` with 400; else the answer is the
        comments as they now stand.
        """
        watcher = self.jobs.get(index)
        if watcher is None:
            return 404
        if not isinstance(request, dict):
            return 400
        fields = cast("dict[str, object]", request)
        layout = JobLayout(watcher.folder)
        stack = watcher.current_stack()
        try:
            if "remove" in fields:
                remove_comment(layout, str(fields["remove"]))
            elif stack is None:
                raise ReviewError(NO_SECTIONS)
            else:
                section = str(fields.get("section") or "")
                add_comment(layout, stack, section, fields.get("box"),
                            fields.get("text"))
        except ReviewError as exc:
            return (*self._json({"error": str(exc)}), 400)
        except OSError:
            return (*self._json({"error": NOT_SAVED}), 500)
        rows = self._comment_rows(watcher)
        return self._json({"comments": [asdict(row) for row in rows]})

    def _comment_rows(self, watcher: JobWatcher) -> list[CommentRow]:
        """Return a job's review comments as the page gets them.

        The file is read again only when it or the job's state changed.
        """
        with self._comments_lock:
            watch = self._comments.setdefault(
                watcher.folder, CommentWatch(watcher.folder))
            return watch.rows(watcher.current_stack(), watcher.state_version)

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
        self.renders.close()
        if self.thread is not None:
            self.thread.join(timeout=2)
