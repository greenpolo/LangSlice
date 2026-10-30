"""The warm process's server: warm up once, then fork a child per call.

:func:`serve_main` holds the place's lock for its life, loads what a call
would load (:func:`warm_up`) and serves (:class:`Server`) until it is idle,
its code changed or the job folder is gone.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

from langslice.doors.agent_cli.warm.child import run_child
from langslice.doors.agent_cli.warm.pools import quiet_pools
from langslice.doors.agent_cli.warm.protocol import (
    HANDSHAKE_S,
    PROTOCOL,
    Place,
    envkey,
    identity,
    langslice_dir,
    read_message,
    runtime_dir,
    send_message,
)

#: Seconds without a call after which the warm process exits (an agent's
#: pauses between calls are far shorter; a long verb's call is no pause).
IDLE_S = 10 * 60
#: Seconds without a call after which the warm process of a submitted job exits.
SUBMITTED_IDLE_S = 60.0
#: Seconds between the idle warm process's checks for changed code and state.
CHECK_S = 5.0
#: How long a new server waits for the one it replaces to finish its calls.
LOCK_WAIT_S = 15 * 60


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp} {os.getpid()}] {message}", file=sys.stderr, flush=True)


def _stat(path: str) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_size, info.st_mtime_ns)


def _library_roots() -> tuple[str, ...]:
    import site
    import sysconfig

    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
             *(sysconfig.get_paths().get(key, "") for key in ("stdlib", "platstdlib",
                                                               "purelib", "platlib"))}
    try:
        roots.update(site.getsitepackages())
    except AttributeError:
        pass
    return tuple(os.path.join(os.path.realpath(root), "") for root in roots if root)


def tree_stats(folder: str) -> dict[str, tuple[int, int] | None]:
    """Every ``.py`` file under *folder* (its real path) with its size and
    modification time."""
    found: dict[str, tuple[int, int] | None] = {}
    for root, dirs, files in os.walk(os.path.realpath(folder)):
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                found[path] = _stat(path)
    return found


class Sources:
    """The files the server's code came from: every loaded first-party module
    (outside the standard library and site-packages), as it was when the
    server started (*before*: LangSlice's files, read before they were
    imported) or when it was loaded, the folders of the import path, and the
    atlas's own folder; :meth:`changed` names the first that differs."""

    def __init__(self, before: dict[str, tuple[int, int] | None]) -> None:
        self.roots = _library_roots()
        self.before = before
        self.files: dict[str, tuple[int, int] | None] = {}
        self.folders = {path: _stat(path) for path in sys.path if path}
        self.add_loaded()

    def add_loaded(self) -> None:
        for module in list(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if not isinstance(file, str) or not file.endswith(".py"):
                continue
            real = os.path.realpath(file)
            if real not in self.files and not real.startswith(self.roots):
                self.files[real] = self.before[real] if real in self.before else _stat(real)

    def watch(self, path: str) -> None:
        self.folders.setdefault(path, _stat(path))

    def changed(self) -> str | None:
        for path, held in self.folders.items():
            if _stat(path) != held:
                return path
        for path, held in self.files.items():
            if _stat(path) != held:
                return path
        return None


def warm_up(folder: str) -> dict[str, Any]:
    """Load what a call would load (the CLI's modules, both engines'
    libraries, the job's tools and atlas) and share the frames of its placed
    sections (:func:`langslice.core.maps.section_maps.share_frames`); what was
    loaded."""
    import gc
    import logging

    started = time.perf_counter()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    import langslice.doors.agent_cli.background
    import langslice.doors.agent_cli.calls
    import langslice.doors.agent_cli.catalog
    import langslice.doors.agent_cli.environment
    import langslice.doors.agent_cli.launch
    import langslice.doors.library.handle  # noqa: F401
    from langslice.core.fits.deformable.engines import preload
    from langslice.core.maps.section_maps import share_frames
    from langslice.doors.jobs import open_folder

    preload()
    opened = open_folder(folder, persist=False, door="cli", emit=_log)
    try:
        opened.tools()
        atlas = opened.ctx.atlas
        for name in ("annotation", "template"):
            try:
                getattr(atlas, name)
            except Exception:  # noqa: BLE001 - an atlas without one
                pass
        frames = share_frames(opened.job.state, opened.ctx)
        _oblique_volumes(atlas, opened.job.state)
        atlas_folder = getattr(atlas, "root_dir", None)
    finally:
        opened.close()
    quiet_pools()
    import threading

    deadline = time.monotonic() + 2.0  # the stopped pools' threads finishing
    while True:
        others = [thread.name for thread in threading.enumerate()
                  if thread is not threading.main_thread()]
        if not others or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    if others:
        _log(f"threads still running before the first fork: {others}")
    root.handlers[:] = handlers  # what the imports configured, a cold call has not yet
    root.setLevel(level)
    gc.collect()
    gc.freeze()
    return {"seconds": round(time.perf_counter() - started, 2), "frames": frames,
            "atlas": opened.spec.atlas,
            "atlas_folder": str(Path(atlas_folder).parent) if atlas_folder else None}


def _oblique_volumes(atlas: Any, state: Any) -> None:
    """When the job's sections are cut at an angle: the template volumes its
    oblique planes are sampled from (a fit's linear one, a picture's spline
    one), each built once per process
    (:mod:`langslice.core.geometry.oblique`)."""
    from langslice.core.geometry.oblique import sample_oblique_plane

    tilted = [record for record in state.slices
              if record.position_mm is not None and (record.pitch_deg or record.yaw_deg)]
    if not tilted:
        return
    record = tilted[0]
    for order in (1, 3):
        sample_oblique_plane(atlas, float(record.position_mm), state.plane, record.pitch_deg,
                             record.yaw_deg, volume="template", order=order)


def _submitted(folder: str) -> bool:
    """Whether the job's saved state says it was submitted."""
    try:
        with open(os.path.join(folder, "state.json"), encoding="utf-8") as handle:
            return bool(json.load(handle).get("submitted"))
    except (OSError, ValueError, AttributeError):
        return False


def _reseed(folder: str) -> int:
    """Share the frames of the job's sections as they stand now."""
    from langslice.core.maps.section_maps import share_frames
    from langslice.doors.jobs import open_folder

    opened = open_folder(folder, persist=False, door="cli", emit=_log)
    try:
        return share_frames(opened.job.state, opened.ctx)
    finally:
        opened.close()
        quiet_pools()


def _acquire(place: Place) -> int | None:
    """Hold the place's lock for this process's life (waiting, up to
    :data:`LOCK_WAIT_S`, for a server finishing its calls); None when
    another server is serving."""
    import fcntl

    fd = os.open(place.lock, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + LOCK_WAIT_S
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            pass
        if os.path.exists(place.socket) or time.monotonic() > deadline:
            os.close(fd)
            return None
        time.sleep(0.5)


def serve_main(args: list[str]) -> int:
    """The warm process of the job folder ``args[0]`` (see the module text);
    ``args[1]`` is its key, ``args[2]`` the identity of the client that
    started it."""
    folder, key, ident = args[:3]
    home = runtime_dir()
    if home is None:
        return 1
    place = Place(folder, key, home)
    lock = _acquire(place)
    if lock is None:
        return 0
    before = tree_stats(langslice_dir())
    if identity(sys.path) != ident:
        Path(place.refused).touch()
        _log(f"not serving {folder}: this interpreter's import path is not its client's")
        return 1
    try:
        os.unlink(place.refused)
    except FileNotFoundError:
        pass
    boot_env = envkey(os.environ)
    _log(f"warming up for {folder}")
    try:
        loaded = warm_up(folder)
    except Exception as exc:  # noqa: BLE001 - logged; the calls run cold
        Path(place.refused).touch()
        _log(f"could not warm up: {type(exc).__name__}: {exc}")
        return 1
    sources = Sources(before)
    if loaded["atlas_folder"]:
        sources.watch(loaded["atlas_folder"])
    _log(f"warm in {loaded['seconds']} s ({loaded['frames']} frames, atlas {loaded['atlas']})")
    server = Server(place, folder, ident, boot_env, sources, lock)
    try:
        server.serve()
    finally:
        server.close()
    return 0


class _Call:
    def __init__(self, pid: int, conn: socket.socket) -> None:
        self.pid = pid
        self.conn: socket.socket | None = conn


class Server:
    """The serving loop: one thread, a fork per call (see the module text)."""

    def __init__(self, place: Place, folder: str, ident: str, env: str, sources: Sources,
                 lock: int) -> None:
        import selectors

        self.place, self.folder, self.ident, self.env = place, folder, ident, env
        self.sources = sources
        self.lock = lock
        self.calls: dict[int, _Call] = {}
        self.selector = selectors.DefaultSelector()
        self.wake_r, self.wake_w = os.pipe()
        os.set_blocking(self.wake_r, False)
        os.set_blocking(self.wake_w, False)
        signal.set_wakeup_fd(self.wake_w)
        signal.signal(signal.SIGCHLD, lambda *_: None)
        self.stopping = False
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)
        self.listener: socket.socket | None = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(place.socket)
        except FileNotFoundError:
            pass
        self.listener.bind(place.socket)
        os.chmod(place.socket, 0o600)
        self.listener.listen(64)
        self.inode = os.stat(place.socket).st_ino
        self.selector.register(self.listener, selectors.EVENT_READ, "listen")
        self.selector.register(self.wake_r, selectors.EVENT_READ, "wake")
        self.last_call = time.monotonic()
        self.state_stamp = _stat(os.path.join(folder, "state.json"))
        self.submitted = _submitted(folder)

    def _stop(self, *_: Any) -> None:
        self.stopping = True

    def serve(self) -> None:
        last_check = time.monotonic()
        while self.listener is not None or self.calls:
            for key, _events in self.selector.select(timeout=CHECK_S):
                if key.data == "listen":
                    if self.listener is not None and not self.stopping:
                        self.accept()
                elif key.data == "wake":
                    try:
                        while os.read(self.wake_r, 512):
                            pass
                    except BlockingIOError:
                        pass
                else:
                    self.from_client(key.data)
            self.reap()
            if self.stopping:
                self.stop_listening("signalled")
            now = time.monotonic()
            if self.listener is None or self.calls:
                continue
            if now - self.last_call > (SUBMITTED_IDLE_S if self.submitted else IDLE_S):
                self.stop_listening("idle, the job submitted" if self.submitted else "idle")
            elif now - last_check >= CHECK_S:
                last_check = now
                self.check()

    def check(self) -> None:
        """While idle: stop on changed code or a job folder that is gone;
        share the frames again when the state changed (not once the job is
        submitted: then it exits after :data:`SUBMITTED_IDLE_S` idle)."""
        changed = self.sources.changed()
        if changed is not None:
            self.stop_listening(f"{changed} changed")
            return
        stamp = _stat(os.path.join(self.folder, "state.json"))
        if stamp is None or not os.path.isfile(os.path.join(self.folder, "job.json")):
            self.stop_listening("the job folder is gone")
            return
        if stamp != self.state_stamp:
            self.state_stamp = stamp
            self.submitted = _submitted(self.folder)
            if self.submitted:
                return
            try:
                frames = _reseed(self.folder)
                self.sources.add_loaded()
            except Exception as exc:  # noqa: BLE001 - the children compute their own
                _log(f"could not share the frames again: {type(exc).__name__}: {exc}")
            else:
                _log(f"state changed: {frames} frames shared")

    def stop_listening(self, why: str) -> None:
        if self.listener is None:
            return
        _log(f"stopping ({why})")
        self.selector.unregister(self.listener)
        self.listener.close()
        self.listener = None
        try:
            if os.stat(self.place.socket).st_ino == self.inode:
                os.unlink(self.place.socket)
        except OSError:
            pass

    def refusal(self, message: dict[str, Any], fds: list[int]) -> dict[str, Any] | None:
        """Why the call cannot be served here (``stale``: never again)."""
        if message.get("protocol") != PROTOCOL or len(fds) != 3:
            return {"cold": "protocol"}
        if message.get("identity") != self.ident:
            return {"cold": "identity"}
        changed = self.sources.changed()
        if changed is not None:
            return {"cold": f"{changed} changed", "stale": True}
        if not os.path.isfile(os.path.join(self.folder, "job.json")):
            return {"cold": "the job folder is gone", "stale": True}
        if message.get("envkey") != self.env:
            return {"cold": "environment"}
        return None

    def accept(self) -> None:
        assert self.listener is not None
        try:
            conn, _address = self.listener.accept()
        except OSError:
            return
        fds: list[int] = []
        try:
            conn.settimeout(HANDSHAKE_S)
            creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                    struct.calcsize("3i"))
            if struct.unpack("3i", creds)[1] != os.getuid():
                raise OSError("another user")
            _byte, fds, _flags, _address = socket.recv_fds(conn, 1, 3)
            message = read_message(conn)
            if message is None:
                raise OSError("no request")
            refused = self.refusal(message, fds)
            if refused is not None:
                _log(f"a call runs cold: {refused['cold']}")
                send_message(conn, refused)
                if refused.get("stale"):
                    self.stop_listening(str(refused["cold"]))
                raise OSError(str(refused["cold"]))
            send_message(conn, {"ready": True})
            go = read_message(conn)
            if not go or not go.get("go"):
                raise OSError("the client left")
            conn.settimeout(None)
        except (OSError, ValueError):
            for fd in fds:
                os.close(fd)
            conn.close()
            return
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            inherited = [self.wake_r, self.wake_w, self.lock, conn.fileno(),
                         self.selector.fileno(),  # type: ignore[attr-defined]
                         *(call.conn.fileno() for call in self.calls.values()
                           if call.conn is not None)]
            if self.listener is not None:
                inherited.append(self.listener.fileno())
            run_child(message, fds, inherited)
        for fd in fds:
            os.close(fd)
        try:
            os.setpgid(pid, pid)
        except OSError:
            pass
        import selectors

        self.calls[pid] = _Call(pid, conn)
        self.selector.register(conn, selectors.EVENT_READ, pid)
        self.last_call = time.monotonic()
        try:
            send_message(conn, {"started": pid})
        except OSError:
            self.client_gone(self.calls[pid])

    def from_client(self, pid: int) -> None:
        call = self.calls.get(pid)
        if call is None or call.conn is None:
            return
        try:
            message = read_message(call.conn)
        except (OSError, ValueError):
            message = None
        if message is None:
            self.client_gone(call)
        elif isinstance(message.get("signal"), int):
            _kill(call.pid, int(message["signal"]))

    def client_gone(self, call: _Call) -> None:
        """The caller died or was killed: so is its call."""
        _kill(call.pid, signal.SIGKILL)
        self._drop(call)

    def _drop(self, call: _Call) -> None:
        if call.conn is not None:
            try:
                self.selector.unregister(call.conn)
            except (KeyError, ValueError):
                pass
            call.conn.close()
            call.conn = None

    def reap(self) -> None:
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            call = self.calls.pop(pid, None)
            if call is None:
                continue
            if call.conn is not None:
                answer = ({"signal": os.WTERMSIG(status)} if os.WIFSIGNALED(status)
                          else {"exit": os.WEXITSTATUS(status)})
                try:
                    send_message(call.conn, answer)
                except OSError:
                    pass
            self._drop(call)
            self.last_call = time.monotonic()

    def close(self) -> None:
        self.stop_listening("done")
        for fd in (self.wake_r, self.wake_w, self.lock):
            try:
                os.close(fd)
            except OSError:
                pass


def _kill(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except OSError:
        try:
            os.kill(pid, signum)
        except OSError:
            pass
