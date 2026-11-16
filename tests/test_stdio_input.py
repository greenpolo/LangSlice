"""A module first loaded during a tool call loads while the host's pipe waits.

On Windows a thread waiting on a synchronous pipe read blocks every other
call on that pipe, and the runtime of compiled modules (numpy's, scipy's
Fortran ``_odepack``) asks about the standard input when it loads: an
engine's first import in a tool call hung the MCP server for good. The
servers read a duplicate of their input (``langslice.doors.stdio``), so
the process's own stdin is the null device.

Each test runs a server in a fresh interpreter whose stdin is a pipe the
test holds open, as a host does, and has it import a compiled module while
it waits for the next message.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: Loaded by scipy.stats on emlddmm's import: the module that hung.
MODULE = "scipy.integrate"
#: Seconds the import may take; a hang never ends.
TIMEOUT_S = 60

MCP_SERVER = f"""
import importlib, sys
from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from langslice.doors.mcp import server as door

assert {MODULE!r} not in sys.modules

def build_server(spec_for, folder=None, *, sessions=None):
    server = FastMCP("probe")

    async def load() -> str:
        # in a worker thread, as host_tool runs every tool
        await to_thread.run_sync(importlib.import_module, {MODULE!r})
        return "loaded"

    server.add_tool(load)
    return server

door.build_server = build_server
door.serve(lambda folder: None)
"""

SERVICE = f"""
import importlib, sys
from langslice.hosts import service

assert {MODULE!r} not in sys.modules

def follow(request, emit, stop):
    # abba.follow runs while a thread waits on the rest of the input
    import time
    time.sleep(1)
    importlib.import_module({MODULE!r})

service.follow = follow
service.AbbaFollowRequest.model_validate = staticmethod(lambda params: params)
service.run_stdio()
"""


def _start(script: str) -> tuple[subprocess.Popen[bytes], queue.Queue[bytes]]:
    """Start *script* with its stdin an open pipe; its stdout lines on a queue."""
    process = subprocess.Popen(
        [sys.executable, "-c", script], cwd=REPO, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines: queue.Queue[bytes] = queue.Queue()

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)

    threading.Thread(target=pump, daemon=True).start()
    return process, lines


def _send(process: subprocess.Popen[bytes], message: dict[str, object]) -> None:
    assert process.stdin is not None
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()


def _reply(lines: queue.Queue[bytes], process: subprocess.Popen[bytes]) -> dict[str, object]:
    try:
        return json.loads(lines.get(timeout=TIMEOUT_S))
    except queue.Empty:
        process.kill()
        raise AssertionError(f"no reply in {TIMEOUT_S} s: the import hung") from None


def test_mcp_tool_imports_while_the_host_pipe_waits():
    process, lines = _start(MCP_SERVER)
    try:
        _send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}})
        assert _reply(lines, process)["id"] == 1
        _send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "load", "arguments": {}}})
        reply = _reply(lines, process)
        assert reply["id"] == 2
        assert "loaded" in json.dumps(reply["result"])
    finally:
        process.kill()
        process.wait()


def test_service_follow_imports_while_the_host_pipe_waits():
    process, lines = _start(SERVICE)
    try:
        _send(process, {"id": "follow", "method": "abba.follow", "params": {}})
        assert _reply(lines, process) == {"id": "follow", "type": "result", "result": {}}
    finally:
        process.kill()
        process.wait()
