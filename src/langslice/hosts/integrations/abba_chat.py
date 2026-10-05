"""Read-only, local browser companion for the existing sanitized agent stream."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

MAX_EVENT_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 1500
_STATIC = Path(__file__).with_name("static")


def _public(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:16000]
    if isinstance(value, dict):
        return {
            k: _public(v, depth + 1)
            for k, v in list(value.items())[:100]
            if isinstance(k, str)
            and k not in {"images", "thought_signature", "encrypted_content", "data"}
        }
    if isinstance(value, (list, tuple)):
        return [_public(v, depth + 1) for v in value[:100]]
    return None


class ChatHistory:
    """A bounded replay buffer. Media endpoints contain exact incoming image bytes."""

    def __init__(self) -> None:
        self.status = "Starting"
        self.lock = threading.Lock()
        self.events: deque[tuple[dict[str, Any], int]] = deque()
        self.images: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self.sequence = 0
        self.event_bytes = 0
        self.image_bytes = 0

    def add(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind not in {
            "text",
            "reasoning",
            "seed",
            "tool_start",
            "tool_end",
            "tool_result",
            "complete",
            "error",
            "status",
        }:
            return
        clean = {
            k: _public(event[k])
            for k in (
                "kind",
                "name",
                "id",
                "execution_id",
                "args",
                "target_ids",
                "response",
                "revised",
                "submitted",
            )
            if k in event
        }
        # Seed text is the instruction, not a conversational message.
        if kind in {"text", "reasoning", "error", "status"}:
            clean["text"] = str(event.get("text") or "")[:180000]
        with self.lock:
            media = []
            for entry in event.get("images") or []:
                data = entry.get("data")
                mime = entry.get("mime_type", "image/png")
                if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
                    continue
                if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                    continue
                key = secrets.token_hex(12)
                self.images[key] = (data, mime)
                self.image_bytes += len(data)
                media.append({"id": key, "label": str(entry.get("label") or "Agent image")[:500]})
                while self.image_bytes > MAX_IMAGE_BYTES or len(self.images) > 160:
                    _, (old, _) = self.images.popitem(last=False)
                    self.image_bytes -= len(old)
            if media:
                clean["images"] = media
            if kind == "seed" and not media:
                return
            if kind == "status":
                self.status = str(clean.get("text", ""))
            elif kind == "complete":
                self.status = "Submitted" if event.get("submitted") else "Stopped"
            elif kind == "error":
                self.status = "Run failed"
            elif kind in {"text", "reasoning", "tool_start"}:
                self.status = "Running"
            self.sequence += 1
            clean["seq"] = self.sequence
            size = len(json.dumps(clean).encode())
            self.events.append((clean, size))
            self.event_bytes += size
            while self.event_bytes > MAX_EVENT_BYTES or len(self.events) > MAX_EVENTS:
                _, old_size = self.events.popleft()
                self.event_bytes -= old_size

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            first = self.events[0][0]["seq"] if self.events else self.sequence + 1
            return {
                "status": self.status,
                "sequence": self.sequence,
                "reset": since > self.sequence or (since > 0 and since < first - 1),
                "trimmed": first > 1,
                "events": [e for e, _ in self.events if e["seq"] > since],
            }


class ChatServer:
    """Token-addressed loopback GET surface; no commands or registration endpoints."""

    def __init__(self, history: ChatHistory) -> None:
        self.history = history
        self.token = secrets.token_urlsafe(32)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                prefix = f"/{owner.token}/"
                host = self.headers.get("Host", "")
                if host != f"127.0.0.1:{owner.http.server_port}" or not parsed.path.startswith(
                    prefix
                ):
                    self.send_error(404)
                    return
                resource = parsed.path[len(prefix) :]
                if resource == "events":
                    try:
                        since = max(0, int(parse_qs(parsed.query).get("since", ["0"])[0]))
                    except (ValueError, TypeError):
                        self.send_error(400)
                        return
                    body = json.dumps(owner.history.snapshot(since)).encode()
                    mime = "application/json"
                elif resource.startswith("image/"):
                    with owner.history.lock:
                        item = owner.history.images.get(resource[6:])
                    if item is None:
                        self.send_error(404)
                        return
                    body, mime = item
                elif resource in {"", "chat.css", "chat.js"}:
                    path = _STATIC / (resource or "chat.html")
                    body = path.read_bytes()
                    mime = {
                        "": "text/html; charset=utf-8",
                        "chat.css": "text/css",
                        "chat.js": "text/javascript",
                    }[resource]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; "
                    "style-src 'self'; img-src 'self'; connect-src 'self'; "
                    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
                )
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        self.thread = threading.Thread(
            target=lambda: self.http.serve_forever(poll_interval=0.1),
            daemon=True,
            name="langslice-chat",
        )
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_port}/{self.token}/"

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)


def _browser(*, windows: bool | None = None) -> str:
    for name in ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable",
                 "chrome.exe", "msedge.exe"):
        found = shutil.which(name)
        if found:
            return found
    if windows is None:
        windows = os.name == "nt"
    if windows:
        for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                     os.environ.get("LOCALAPPDATA")):
            if root:
                for name in ("Google/Chrome/Application/chrome.exe",
                             "Microsoft/Edge/Application/msedge.exe"):
                    browser = Path(root) / name
                    if browser.is_file():
                        return str(browser)
    raise RuntimeError("A Chromium browser is unavailable; use the Swing activity viewer")


def _arrange(parent: Any) -> tuple[int, int, int, int]:
    if parent is None:
        return (1280, 0, 640, 1000)
    from jpype import JProxy  # pyright: ignore[reportMissingImports]
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    bounds = []

    def place() -> None:
        config = parent.getGraphicsConfiguration()
        rect = config.getBounds()
        inset = jimport("java.awt.Toolkit").getDefaultToolkit().getScreenInsets(config)
        x, y = int(rect.x + inset.left), int(rect.y + inset.top)
        width = int(rect.width - inset.left - inset.right)
        height = int(rect.height - inset.top - inset.bottom)
        side = min(660, max(430, int(width * 0.30)))
        parent.setBounds(x, y, width - side, height)
        bounds.extend((x + width - side, y, side, height))

    swing = jimport("javax.swing.SwingUtilities")
    if swing.isEventDispatchThread():
        place()
    else:
        swing.invokeAndWait(JProxy("java.lang.Runnable", dict(run=place)))
    return tuple(bounds)  # type: ignore[return-value]


class ChatWindow:
    """Same host lifecycle as ActivityWindow, with a private browser app process."""

    def __init__(self, *, parent: Any = None) -> None:
        self.browser = _browser()
        self.history = ChatHistory()
        self.server = ChatServer(self.history)
        self.url = self.server.url
        self.profile = tempfile.mkdtemp(prefix="langslice-chat-")
        self.process: subprocess.Popen[bytes] | None = None
        self.closed = False
        try:
            self.bounds = _arrange(parent)
            self.show()
        except Exception:
            self.dispose()
            raise

    def on_event(self, event: dict[str, Any]) -> None:
        if not self.closed:
            self.history.add(event)

    def show(self) -> None:
        """Open the app window unless it is already open (a closed window's
        browser process has exited; the server and history live on)."""
        if self.closed or (self.process is not None and self.process.poll() is None):
            return
        x, y, width, height = self.bounds
        args = [
            self.browser,
            f"--app={self.url}",
            f"--user-data-dir={self.profile}",
            f"--window-position={x},{y}",
            f"--window-size={width},{height}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
            "--disable-background-mode",
            "--disable-extensions",
        ]
        if os.name == "posix" and os.environ.get("DISPLAY"):
            args.append("--ozone-platform=x11")
        # A private profile keeps this process independent of the user's browser.
        self.process = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy()
        )
        try:
            code = self.process.wait(timeout=0.15)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise RuntimeError(f"Browser app exited during startup (status {code})")

    def dispose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.server.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        shutil.rmtree(self.profile, ignore_errors=True)


def create_activity_window(*, parent: Any = None) -> Any:
    """The browser log, or the Swing window when no Chromium browser starts."""
    import logging

    try:
        return ChatWindow(parent=parent)
    except Exception:
        logging.getLogger(__name__).warning(
            "Browser activity window unavailable; using Swing viewer", exc_info=True
        )
        from langslice.hosts.integrations.abba_activity import ActivityWindow

        return ActivityWindow(parent=parent)
