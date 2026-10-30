"""The warm process's client: the first thing ``langslice-job`` does.

Standard library only, so a warm call starts in milliseconds: it hands the
call to the job's warm process and waits for its exit status, or returns
None so the call runs cold here and starts a warm process for the next.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
from typing import Any

from langslice.doors.agent_cli.warm.protocol import (
    HANDSHAKE_S,
    PROTOCOL,
    STREAMS,
    Place,
    envkey,
    identity,
    job_folder,
    read_message,
    send_message,
)

#: ``0`` (or ``false``, ``no``, ``off``) runs every call cold.
WARM_ENV = "LANGSLICE_WARM"
#: Signals the client passes on to its call.
FORWARDED = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")
#: How the server is started: ``-c`` puts the working folder first on the
#: path, which is taken off again.
BOOT = ("import sys\nif sys.path and sys.path[0] == '': del sys.path[0]\n"
        "from langslice.doors.agent_cli.warm.server import serve_main\n"
        "raise SystemExit(serve_main(sys.argv[1:]))")
#: How long a server that could not serve (another import path, a job it
#: could not load) keeps clients from starting another one.
REFUSED_S = 3600.0


def _flag_options() -> list[str]:
    """Command-line options that give a new interpreter this one's :func:`_flags`."""
    flags = sys.flags
    options = ["-O"] * int(flags.optimize)
    for on, option in ((flags.dont_write_bytecode, "-B"), (flags.no_user_site, "-s"),
                       (flags.ignore_environment, "-E"), (flags.isolated, "-I")):
        if on:
            options.append(option)
    if flags.dev_mode:
        options += ["-X", "dev"]
    if flags.utf8_mode:
        options += ["-X", "utf8"]
    return options


def available() -> bool:
    """Whether a call may go to a warm process here (Linux, not switched off)."""
    if os.environ.get(WARM_ENV, "").strip().lower() in ("0", "false", "no", "off"):
        return False
    return (sys.platform.startswith("linux") and hasattr(os, "fork")
            and hasattr(socket, "send_fds") and hasattr(socket, "AF_UNIX"))


def _stream(stream: Any) -> dict[str, Any] | None:
    if stream is None:
        return None
    try:
        return {"encoding": stream.encoding, "errors": stream.errors,
                "line_buffering": bool(stream.line_buffering),
                "write_through": bool(getattr(stream, "write_through", False))}
    except (AttributeError, ValueError):
        return None


def request(argv: list[str], kind: str = "job") -> dict[str, Any] | None:
    """What a client hands over (None when this process cannot be served:
    a closed standard stream, a working folder that is gone)."""
    streams = {name: _stream(getattr(sys, name)) for name in STREAMS}
    if any(value is None for value in streams.values()):
        return None
    try:
        for fd in (0, 1, 2):
            os.fstat(fd)
        cwd = os.getcwd()
    except OSError:
        return None
    return {"protocol": PROTOCOL, "kind": kind, "argv": list(argv), "argv0": sys.argv[0],
            "executable": sys.executable,
            "cwd": cwd, "env": dict(os.environ), "path": list(sys.path),
            "identity": identity(sys.path[1:]), "envkey": envkey(os.environ),
            "streams": streams}


def run(argv: list[str]) -> int | None:
    """Answer ``langslice-job`` *argv* through the job's warm process: the
    call's exit code, or None when it is to run cold here (then a warm
    process is started for the next calls when there is none)."""
    if not available():
        return None
    folder = job_folder(argv)
    if folder is None:
        return None
    message = request(argv)
    if message is None:
        return None
    place = Place.of(folder, message["identity"])
    if place is None:
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(HANDSHAKE_S)
        sock.connect(place.socket)
    except OSError:
        sock.close()
        start_server(place, message["identity"])
        return None
    with sock:
        return _converse(sock, message, place)


def _converse(sock: socket.socket, message: dict[str, Any], place: Place) -> int | None:
    try:
        socket.send_fds(sock, [b"\0"], [0, 1, 2])
        send_message(sock, message)
        reply = read_message(sock)
        if reply is None or not reply.get("ready"):
            if reply is not None and reply.get("stale"):
                start_server(place, message["identity"], force=True)
            return None
        send_message(sock, {"go": True})
        sock.settimeout(None)
        started = read_message(sock)
    except (OSError, ValueError):
        return None
    if started is None or "started" not in started:
        return None  # nothing ran: the server went away before the call started
    held = _forward_signals(sock)
    try:
        while True:
            try:
                answer = read_message(sock)
            except InterruptedError:
                continue
            if answer is None:
                print("langslice-job: the warm process ended before the call did",
                      file=sys.stderr, flush=True)
                return 4
            if "exit" in answer:
                return int(answer["exit"])
            if "signal" in answer:
                return _die_by(int(answer["signal"]))
    finally:
        for signum, handler in held.items():
            signal.signal(signum, handler)


def _forward_signals(sock: socket.socket) -> dict[int, Any]:
    """Pass :data:`FORWARDED` signals to the call; the handlers they replaced."""
    def forward(signum: int, _frame: Any) -> None:
        try:
            send_message(sock, {"signal": signum})
        except OSError:
            pass

    held: dict[int, Any] = {}
    for name in FORWARDED:
        signum = getattr(signal, name, None)
        if signum is not None:
            held[signum] = signal.signal(signum, forward)
    return held


def _die_by(signum: int) -> int:
    """End this process as one killed by *signum* (else its shell code)."""
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except (OSError, ValueError):
        pass
    return 128 + signum


def _lock_free(path: str) -> bool:
    """Whether no warm process holds *path* (it is starting or serving)."""
    import fcntl

    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


def start_server(place: Place, ident: str, *, force: bool = False) -> None:
    """Start the job's warm process, detached, unless one is starting or
    serving already, or one could not serve lately. With
    *force* it is started anyway (it waits for the one before it)."""
    try:
        if not force:
            try:
                if time.time() - os.stat(place.refused).st_mtime < REFUSED_S:
                    return
            except FileNotFoundError:
                pass
            if not _lock_free(place.lock):
                return
        import subprocess

        with open(place.log, "ab") as log:
            subprocess.Popen(  # noqa: S603 - this interpreter, this module
                [sys.executable, *_flag_options(), "-c", BOOT, place.folder, place.key, ident],
                cwd="/", stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
    except OSError:
        pass
