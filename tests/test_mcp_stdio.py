"""`langslice mcp` as a host runs it: a process of its own on stdin and stdout
pipes, spoken to in newline-delimited JSON-RPC.

The in-memory tests (`tests/test_mcp_server.py`) cannot see what only a real
pipe shows: a program the server starts inheriting the host's stdin hung the
server on Windows (Git for Windows blocks at start-up querying a pipe with a
read pending). Every exchange here has a deadline, so a hang fails the test.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

from mcp.types import LATEST_PROTOCOL_VERSION

import langslice
from tests.test_mcp_server import _folder

REPO = Path(__file__).resolve().parents[1]
DEADLINE_S = 120.0

FAKE_ATLAS_SERVER = """
import functools, sys
from langslice.doors.mcp import server
from tests.fakes import SlabAtlas
atlas = SlabAtlas()
server.build_server = functools.partial(server.build_server, atlas_loader=lambda _name: atlas)
from langslice.cli import main
sys.argv = ["langslice", "mcp", "--preprocess", "none"]
raise SystemExit(main())
"""


class Server:
    """A ``langslice mcp`` process; its stdout read line by line on a thread."""

    def __init__(self, argv: list[str], stderr: Path) -> None:
        env = dict(os.environ)
        source = str(Path(langslice.__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(
            [source, str(REPO), *filter(None, [env.get("PYTHONPATH")])])
        self.stderr_path = stderr
        self._stderr = stderr.open("wb")
        self.process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            cwd=REPO, env=env)
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True).start()
        self.next_id = 0

    def _read(self, stream: IO[bytes] | None) -> None:
        assert stream is not None
        for line in iter(stream.readline, b""):
            self.lines.put(line)
        self.lines.put(None)

    def fail(self, why: str) -> None:
        self.close()
        raise AssertionError(
            f"{why}\n--- server stderr ---\n"
            + self.stderr_path.read_text(encoding="utf-8", errors="replace")[-6000:])

    def send(self, message: dict[str, Any]) -> None:
        stdin = self.process.stdin
        assert stdin is not None
        stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return its reply, skipping notifications."""
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        deadline = time.monotonic() + DEADLINE_S
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self.fail(f"no reply to {method} within {DEADLINE_S:.0f} s")
                raise
            if line is None:
                self.fail(f"the server closed stdout before answering {method}")
                raise AssertionError
            body = line.rstrip(b"\r\n")
            assert line.endswith(b"\n") and b"\n" not in body and b"\r" not in body, line[-40:]
            message = json.loads(body)
            assert message.get("jsonrpc") == "2.0", message
            if message.get("id") == self.next_id:
                return message

    def initialize(self) -> dict[str, Any]:
        reply = self.request("initialize", {
            "protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "langslice-test", "version": "0"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return reply

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        reply = self.request("tools/call", {"name": tool, "arguments": arguments})
        assert "result" in reply, reply
        return reply["result"]

    def stop(self) -> int:
        """Close stdin as a host does on quitting; the server must exit."""
        stdin = self.process.stdin
        assert stdin is not None
        stdin.close()
        try:
            return self.process.wait(timeout=DEADLINE_S)
        except subprocess.TimeoutExpired:
            self.fail("the server did not exit after its stdin closed")
            raise

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self._stderr.close()


def _text(result: dict[str, Any]) -> str:
    first = result["content"][0]
    assert first["type"] == "text", first
    return first["text"]


def test_langslice_mcp_answers_start_job_without_a_folder_over_pipes(tmp_path: Path):
    server = Server([sys.executable, "-m", "langslice", "mcp"], tmp_path / "stderr.log")
    try:
        hello = server.initialize()
        assert hello["result"]["serverInfo"]["name"], hello
        assert hello["result"]["capabilities"]["tools"]["listChanged"] is True
        refusal = json.loads(_text(server.call("start_job", {})))
        assert refusal["error"] == "NO_FOLDER", refusal
        assert server.stop() == 0
    finally:
        server.close()


def _checkout_commit() -> str | None:
    """The commit of the git checkout the langslice under test comes from, or
    None (an installed package, or no git)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(Path(langslice.__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def test_start_job_opens_an_image_folder_over_pipes(tmp_path: Path):
    """Opening a job writes its files and starts git for the registration's
    provenance while the host's pipe is the server's stdin. Given that pipe,
    Git for Windows waits until its timeout, and the provenance is lost."""
    folder = _folder(tmp_path)
    server = Server([sys.executable, "-c", FAKE_ATLAS_SERVER], tmp_path / "stderr.log")
    try:
        server.initialize()
        opened = server.call("start_job", {"image_folder": str(folder)})
        assert not opened.get("isError"), opened
        assert _text(opened).startswith(f"Job opened: {folder / 'langslice'}. 3 sections."), opened
        rows = json.loads(_text(server.call("status", {})))["rows"]
        assert len(rows) == 3
        assert server.stop() == 0
    finally:
        server.close()
    registration = json.loads(
        (folder / "langslice" / "registration.json").read_text(encoding="utf-8"))
    assert registration["source"]["commit"] == _checkout_commit()
