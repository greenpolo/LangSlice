"""The agent CLI's warm process: one fork server per job folder.

A ``langslice-job FOLDER ...`` call is a process of its own, and most of what
a call costs is starting one: importing ITK's Elastix classes, torch and
FireANTs and LangSlice's toolbox, and loading the job's atlas. The warm
process pays that once per job folder and then forks a child for every call.
The child runs what a cold call runs (:func:`langslice.doors.cli.jobcli.run`)
with the caller's arguments, working folder, environment variables and
standard streams, so a call still behaves as a process of its own: its own
memory, its own holds of the job folder's lock, ``Job.load`` and
``Job.sync`` as before, and calls run in parallel when they are made in
parallel. What it writes on stdout and stderr and its exit code are a cold
call's.

The client (:func:`run`, the first thing ``langslice-job`` does; standard
library only, so a warm call starts in milliseconds) works out the job
folder of ``FOLDER VERB`` (``init``, ``ops`` and ``schema`` always run cold),
connects to that job's socket and hands over its argv, working folder,
environment and the file descriptors of its stdin, stdout and stderr
(``SCM_RIGHTS``); then it waits for the child's exit status and ends with it
(a child killed by a signal: the client kills itself with that signal).
Signals the client gets (Ctrl-C, ``SIGTERM``, ``SIGHUP``, ``SIGQUIT``) are
passed to the child's process group; a client that dies or is killed takes
the child with it (``SIGKILL`` to its process group). With no warm process
up, the call runs cold and starts one, detached, for the next calls.
``LANGSLICE_WARM=0`` turns it off; it is Linux only (fork with these
libraries is not safe on macOS, and Windows has none).

The server (:func:`serve_main`) holds ``<key>.lock`` for its life (one per
job folder and Python environment), loads the libraries and the job's atlas
(:func:`warm_up`; nothing touches the GPU, so a child can initialise CUDA),
computes every placed section's frame for the children to reuse
(:func:`langslice.core.maps.share_frames`), then listens on ``<key>.sock`` in
``$XDG_RUNTIME_DIR/langslice-warm/`` (else ``/tmp/langslice-warm-<uid>/``,
mode 0700; only the same user's processes are served). It never runs stale
code: before each call it checks the files of every first-party module it
has loaded (LangSlice's own, a plugin's: anything outside the standard
library and site-packages), the site-packages folders and the atlas's
folder; one changed, it answers that call ``stale`` (the call runs cold and
starts a new server), stops listening and exits once its running calls end.
A caller whose environment differs where the loaded libraries read it
(:data:`ENV_PREFIXES`, ``HOME``) runs cold. It exits after
:data:`IDLE_S` without a call. A section written since the last call has its
frame computed again while the server is idle.

A child: its own process group, killed with the server, the caller's
streams on 0, 1 and 2 (Python's ``sys.stdin``/``stdout``/``stderr`` made
again with the caller's encodings and buffering), ``sys.argv``, ``sys.path``,
``os.environ`` and the working folder the caller's, then the cold call's
code, then Python's own ending (threads joined, ``atexit``, streams
flushed). ``langslice-job FOLDER python SCRIPT`` runs the agent's script in
the child (:func:`run_script`) as the job environment's python would.
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

#: ``0`` (or ``false``, ``no``, ``off``) runs every call cold.
WARM_ENV = "LANGSLICE_WARM"
#: Seconds without a call after which the warm process exits.
IDLE_S = 30 * 60
#: Seconds between the idle warm process's checks for changed code and state.
CHECK_S = 5.0
#: How long a new server waits for the one it replaces to finish its calls.
LOCK_WAIT_S = 15 * 60
#: How long either side waits for the other's next message before a call starts.
HANDSHAKE_S = 10.0
#: Bumped when the messages change.
PROTOCOL = 1
#: Environment variables read when the libraries the server has loaded start
#: (thread pools, devices, allocators, Python's own): a caller whose values
#: differ runs cold. ``HOME`` too (the atlases' folder).
ENV_PREFIXES = ("PYTHON", "OMP_", "MKL_", "OPENBLAS_", "GOTO", "KMP_", "CUDA_", "NVIDIA_",
                "PYTORCH_", "TORCH", "NPY_", "LD_", "ITK_", "SITK_", "MALLOC_", "MPL",
                "LANGSLICE_ATLAS", "BRAINGLOBE")
#: The streams handed to a child, in fd order.
STREAMS = ("stdin", "stdout", "stderr")
#: Signals the client passes on to its call.
FORWARDED = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")
#: How the server is started: ``-c`` puts the working folder first on the
#: path, which is taken off again.
BOOT = ("import sys\nif sys.path and sys.path[0] == '': del sys.path[0]\n"
        "from langslice.doors.cli.warm import serve_main\n"
        "raise SystemExit(serve_main(sys.argv[1:]))")
#: How long a server that could not serve (another import path, a job it
#: could not load) keeps clients from starting another one.
REFUSED_S = 3600.0

#: Set in a warm process's child (:func:`in_child`).
_IN_CHILD = [False]
#: Streams replaced in a child, kept so they never close the fds they wrapped.
_KEEP: list[Any] = []


def in_child() -> bool:
    """Whether this process is a warm process's child serving a call."""
    return _IN_CHILD[0]


class ExitBySignal(BaseException):
    """End the call as a process killed by *signum* (an uncaught Ctrl-C)."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


# --- where: one socket per job folder and environment ------------------------------


def _langslice_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _flags() -> list[Any]:
    """The interpreter flags that change what a process imports."""
    flags = sys.flags
    return [flags.optimize, flags.dont_write_bytecode, flags.no_user_site, flags.no_site,
            flags.ignore_environment, flags.isolated, flags.dev_mode, flags.utf8_mode]


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


def identity(path: list[str]) -> str:
    """The Python environment a process runs in: its interpreter, flags,
    import path (*path*: a client's without its first entry, the script's
    folder) and LangSlice's folder."""
    import hashlib

    data = [PROTOCOL, os.path.realpath(sys.executable), sys.version, _flags(), list(path),
            _langslice_dir()]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def envkey(environ: dict[str, str] | os._Environ[str]) -> str:
    """The environment variables a loaded library may have read on start."""
    import hashlib

    kept = sorted((name, value) for name, value in environ.items()
                  if name == "HOME" or name.startswith(ENV_PREFIXES))
    return hashlib.sha256(json.dumps(kept).encode()).hexdigest()


def runtime_dir() -> Path | None:
    """The private folder of the sockets (see the module text), or None."""
    uid = os.getuid()
    places = []
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        places.append(Path(runtime) / "langslice-warm")
    places.append(Path(os.environ.get("TMPDIR") or "/tmp") / f"langslice-warm-{uid}")
    places.append(Path("/tmp") / f"langslice-warm-{uid}")
    for folder in places:
        if len(str(folder)) > 60:  # room for the socket's name within AF_UNIX's 108 bytes
            continue
        try:
            folder.mkdir(mode=0o700, exist_ok=True)
            info = folder.lstat()
        except OSError:
            continue
        import stat

        if stat.S_ISDIR(info.st_mode) and info.st_uid == uid and not info.st_mode & 0o077:
            return folder
    return None


class Place:
    """The files of one job folder's warm process: ``<key>.sock``,
    ``<key>.lock``, ``<key>.log`` and ``<key>.refused`` (a server that could
    not serve: none is started again for :data:`REFUSED_S`)."""

    def __init__(self, folder: str, key: str, home: Path) -> None:
        self.folder = folder
        self.key = key
        self.socket = str(home / f"{key}.sock")
        self.lock = str(home / f"{key}.lock")
        self.log = str(home / f"{key}.log")
        self.refused = str(home / f"{key}.refused")

    @classmethod
    def of(cls, folder: str, ident: str) -> Place | None:
        import hashlib

        home = runtime_dir()
        if home is None:
            return None
        key = hashlib.sha256(json.dumps([folder, ident]).encode()).hexdigest()[:24]
        return cls(folder, key, home)


def job_folder(argv: list[str]) -> str | None:
    """The job folder ``FOLDER VERB ...`` works on, when the call can be warm:
    a job folder (or the image folder beside one) that holds a job, any
    verb or command but ``init``; None otherwise."""
    if len(argv) < 2 or argv[0] in ("ops", "schema") or argv[0].startswith("-"):
        return None
    if argv[1].replace("-", "_") == "init":
        return None
    given = os.path.abspath(os.path.expanduser(argv[0]))
    for folder in (given, os.path.join(given, "langslice")):
        if (os.path.isfile(os.path.join(folder, "job.json"))
                and os.path.isfile(os.path.join(folder, "state.json"))):
            return os.path.realpath(folder)
    return None


# --- messages ------------------------------------------------------------------------


def _send(sock: socket.socket, message: dict[str, Any]) -> None:
    data = json.dumps(message).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def _exactly(sock: socket.socket, size: int) -> bytes | None:
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            return None
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _read(sock: socket.socket) -> dict[str, Any] | None:
    """The next message, or None at the end of the connection."""
    head = _exactly(sock, 4)
    if head is None:
        return None
    body = _exactly(sock, struct.unpack("!I", head)[0])
    if body is None:
        return None
    message = json.loads(body)
    return message if isinstance(message, dict) else None


# --- the client ----------------------------------------------------------------------


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
        _send(sock, message)
        reply = _read(sock)
        if reply is None or not reply.get("ready"):
            if reply is not None and reply.get("stale"):
                start_server(place, message["identity"], force=True)
            return None
        _send(sock, {"go": True})
        sock.settimeout(None)
        started = _read(sock)
    except (OSError, ValueError):
        return None
    if started is None or "started" not in started:
        return None  # nothing ran: the server went away before the call started
    held = _forward_signals(sock)
    try:
        while True:
            try:
                answer = _read(sock)
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
            _send(sock, {"signal": signum})
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


# --- the server ----------------------------------------------------------------------


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
    sections (:func:`langslice.core.maps.share_frames`); what was loaded."""
    import gc
    import logging

    started = time.perf_counter()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    import langslice.doors.cli.background
    import langslice.doors.cli.brief
    import langslice.doors.cli.catalog
    import langslice.doors.cli.job
    import langslice.doors.cli.jobenv
    import langslice.doors.library  # noqa: F401
    from langslice.core.deformable.engines import preload
    from langslice.core.maps import share_frames
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
    _quiet_pools()
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
    one), each built once per process (:mod:`langslice.core.oblique`)."""
    from langslice.core.oblique import sample_oblique_plane

    tilted = [record for record in state.slices
              if record.position_mm is not None and (record.pitch_deg or record.yaw_deg)]
    if not tilted:
        return
    record = tilted[0]
    for order in (1, 3):
        sample_oblique_plane(atlas, float(record.position_mm), state.plane, record.pitch_deg,
                             record.yaw_deg, volume="template", order=order)


def _quiet_pools() -> None:
    """Stop the threads loading left behind, which a forked child would find
    gone: dask's pool (it reads BrainGlobe's arrays), zarr's event loop and
    LangSlice's sampling threads (each makes them again on its next use),
    and fsspec's event loop with the S3 sessions on it (BrainGlobe's version
    check), whose closing at a child's exit would wait for that loop."""
    import weakref

    oblique = sys.modules.get("langslice.core.oblique")
    if oblique is not None:
        for pool in list(oblique._sample_pools.values()):
            pool.shutdown(wait=True)
        oblique._sample_pools.clear()
    zarr_sync = sys.modules.get("zarr.core.sync")
    if zarr_sync is not None and zarr_sync.loop[0] is not None:
        zarr_sync.cleanup_resources()

    threaded: Any = sys.modules.get("dask.threaded")
    if threaded is not None:
        pool = getattr(threaded, "default_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        threaded.default_pool = None
        getattr(threaded, "pools", {}).clear()
    asyn = sys.modules.get("fsspec.asyn")
    if asyn is None:
        return
    registry = getattr(weakref.finalize, "_registry", {})
    for finalizer, info in list(registry.items()):
        if getattr(info.func, "__name__", "") == "close_session":
            finalizer()
    loop, thread = asyn.loop[0], asyn.iothread[0]
    if loop is not None and thread is not None and thread.is_alive():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        executor = getattr(loop, "_default_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        asyn.reset_lock()


def _reseed(folder: str) -> int:
    """Share the frames of the job's sections as they stand now."""
    from langslice.core.maps import share_frames
    from langslice.doors.jobs import open_folder

    opened = open_folder(folder, persist=False, door="cli", emit=_log)
    try:
        return share_frames(opened.job.state, opened.ctx)
    finally:
        opened.close()
        _quiet_pools()


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
    before = tree_stats(_langslice_dir())
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
            if now - self.last_call > IDLE_S:
                self.stop_listening("idle")
            elif now - last_check >= CHECK_S:
                last_check = now
                self.check()

    def check(self) -> None:
        """While idle: stop on changed code or a job folder that is gone;
        share the frames again when the state changed."""
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
            message = _read(conn)
            if message is None:
                raise OSError("no request")
            refused = self.refusal(message, fds)
            if refused is not None:
                _log(f"a call runs cold: {refused['cold']}")
                _send(conn, refused)
                if refused.get("stale"):
                    self.stop_listening(str(refused["cold"]))
                raise OSError(str(refused["cold"]))
            _send(conn, {"ready": True})
            go = _read(conn)
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
            _child(message, fds, inherited)
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
            _send(conn, {"started": pid})
        except OSError:
            self.client_gone(self.calls[pid])

    def from_client(self, pid: int) -> None:
        call = self.calls.get(pid)
        if call is None or call.conn is None:
            return
        try:
            message = _read(call.conn)
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
                    _send(call.conn, answer)
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


# --- the child -----------------------------------------------------------------------


def _die_with_parent() -> None:
    """``PR_SET_PDEATHSIG``: killed when the server dies."""
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _streams(attributes: dict[str, dict[str, Any]]) -> None:
    """``sys.stdin``, ``stdout`` and ``stderr`` made again on fds 0, 1 and 2
    as the caller's interpreter made them."""
    import io

    for fd, name in enumerate(STREAMS):
        held = attributes[name]
        _KEEP.extend([getattr(sys, name), getattr(sys, f"__{name}__")])
        raw = io.FileIO(fd, "rb" if fd == 0 else "wb", closefd=False)
        buffer: Any = io.BufferedReader(raw) if fd == 0 else io.BufferedWriter(raw)
        stream = io.TextIOWrapper(buffer, encoding=held["encoding"], errors=held["errors"],
                                  newline=None if fd == 0 else "\n",
                                  line_buffering=held["line_buffering"],
                                  write_through=held["write_through"])
        setattr(sys, name, stream)
        setattr(sys, f"__{name}__", stream)


def _fresh_process() -> None:
    """What a new process would not share with the server: caches of the
    environment it started in, the code's commit, the global random state,
    the logging the imports set up."""
    import logging

    tempfile: Any = sys.modules.get("tempfile")
    if tempfile is not None:
        tempfile.tempdir = None
    time.tzset()
    logging.getLogger().handlers[:] = []
    logging.getLogger().setLevel(logging.WARNING)
    formats = sys.modules.get("langslice.job.formats")
    if formats is not None:
        formats.source.cache_clear()
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed()
    _quiet_pools()


def _executable(path: str) -> None:
    """``sys.executable`` (and multiprocessing's) the caller's."""
    sys.executable = path
    spawn = sys.modules.get("multiprocessing.spawn")
    if spawn is not None:
        spawn.set_executable(path)


def _child(message: dict[str, Any], fds: list[int], inherited: list[int]) -> None:
    """A forked child: become the caller's process, run its call, end as
    Python ends. Never returns."""
    code, signum = 70, 0
    try:
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        _die_with_parent()
        signal.set_wakeup_fd(-1)
        for name in ("SIGCHLD", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2"):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        for fd in inherited:
            try:
                os.close(fd)
            except OSError:
                pass
        for target, fd in enumerate(fds):
            os.dup2(fd, target)
        for fd in fds:
            if fd > 2:
                os.close(fd)
        os.chdir(message["cwd"])
        os.environ.clear()
        os.environ.update(message["env"])
        sys.argv = [message["argv0"], *message["argv"]]
        sys.path[:] = message["path"]
        _executable(message["executable"])
        _streams(message["streams"])
        _fresh_process()
        _IN_CHILD[0] = True
        from langslice.doors.cli.jobcli import run as run_cold

        code, signum = as_python(lambda: run_cold(list(message["argv"])))
    except BaseException:  # noqa: BLE001 - the child must reach os._exit
        try:
            import traceback

            traceback.print_exc()
        except BaseException:  # noqa: BLE001
            pass
    finally:
        _end(code, signum)


def as_python(call: Any, script: str | None = None) -> tuple[int, int]:
    """Run *call* as Python runs a program: ``(exit code, signal)``. A
    ``SystemExit`` gives its code (a text is printed, code 1), an uncaught
    exception its traceback (from *script*'s first frame when given) and 1,
    an uncaught Ctrl-C its traceback and ``SIGINT``."""
    try:
        result = call()
        return (int(result) if isinstance(result, int) else 0), 0
    except SystemExit as exc:
        return _exit_code(exc.code), 0
    except ExitBySignal as exc:
        return 0, exc.signum
    except BaseException as exc:  # noqa: BLE001 - reported as Python reports it
        tb = exc.__traceback__
        if script is not None:
            while tb is not None and tb.tb_frame.f_code.co_filename != script:
                tb = tb.tb_next
        else:
            tb = tb.tb_next if tb is not None else None  # this function's own frame
        try:
            sys.excepthook(type(exc), exc.with_traceback(tb), tb)
        except BaseException:  # noqa: BLE001
            pass
        if isinstance(exc, KeyboardInterrupt):
            return 1, int(signal.SIGINT)
        return 1, 0


def _exit_code(code: Any) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    try:
        print(code, file=sys.stderr)
    except BaseException:  # noqa: BLE001
        pass
    return 1


def _end(code: int, signum: int) -> None:
    """Python's own ending: threads joined, ``atexit``, streams flushed; then
    the exit code, or death by *signum*."""
    import atexit
    import threading

    try:
        threading._shutdown()  # type: ignore[attr-defined]  # what Py_Finalize calls
    except BaseException:  # noqa: BLE001
        pass
    try:
        atexit._run_exitfuncs()
    except BaseException:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:  # noqa: BLE001
            code = code or 120
    if signum:
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except BaseException:  # noqa: BLE001
            pass
    os._exit(code & 0xFF)


# --- an agent's script ---------------------------------------------------------------


#: What the job environment's python is asked: its import path and prefixes.
PROBE = ("import json, site, sys\n"
         "print(json.dumps({'path': sys.path[1:] if sys.path[:1] == [''] else sys.path,"
         " 'executable': sys.executable,"
         " 'prefix': sys.prefix, 'exec_prefix': sys.exec_prefix,"
         " 'base_prefix': sys.base_prefix, 'version': list(sys.version_info),"
         " 'safe_path': bool(getattr(sys.flags, 'safe_path', False)),"
         " 'site': site.getsitepackages()}))")
#: Packages every environment brings itself, never taken for one of the agent's.
SEEDED = frozenset({"pip", "setuptools", "wheel", "_distutils_hack", "pkg_resources",
                    "distutils-precedence", "_virtualenv"})


def _own_modules(folders: list[str]) -> set[str]:
    """The top-level names the environment's own site-packages provide."""
    names: set[str] = set()
    for folder in folders:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name == "__pycache__" or name.endswith((".dist-info", ".egg-info", ".data")):
                    continue
                names.add(name)
            elif name.endswith((".py", ".so", ".pyd")):
                names.add(name.split(".", 1)[0])
    return names - SEEDED


def _cannot_run_here(info: dict[str, Any], env: str) -> str | None:
    """Why the environment's python must run the script itself: another
    Python, or a package of its own that this process loaded from LangSlice's
    environment already."""
    if list(info.get("version") or []) != list(sys.version_info):
        return "another Python version"
    if os.path.realpath(str(info.get("base_prefix"))) != os.path.realpath(sys.base_prefix):
        return "another Python"
    own = [path for path in info.get("site") or []
           if os.path.realpath(path).startswith(os.path.join(os.path.realpath(env), ""))]
    loaded = {name.split(".", 1)[0] for name in sys.modules}
    clash = _own_modules(own) & loaded
    if clash:
        return f"its own {', '.join(sorted(clash))}"
    return None


def run_script(interpreter: Path, rest: list[str]) -> int:
    """In a warm child: run the script ``rest[0]`` with arguments ``rest[1:]``
    as *interpreter* (the job environment's python) would: its import path
    (the script's folder first, then that python's own), its ``.pth`` files,
    ``sys.executable``, ``sys.argv``, ``__main__``; the exit code, a
    traceback, ``SystemExit`` and Ctrl-C as Python gives them. When that
    python must run it itself (:func:`_cannot_run_here`, a script that is
    not a file) this process becomes it."""
    import site
    import subprocess

    script = rest[0]
    env = str(Path(interpreter).parent.parent)
    try:
        probe = subprocess.run([str(interpreter), "-c", PROBE], capture_output=True,
                               text=True, timeout=120, check=True)
        info = json.loads(probe.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        info = None
    if (info is None or not os.path.isfile(script) or script.endswith(".pyc")
            or _cannot_run_here(info, env) is not None):
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(interpreter, [str(interpreter), *rest])
    first = [] if info.get("safe_path") else [os.path.dirname(os.path.realpath(script))]
    sys.path[:] = [*first, *info["path"]]
    known = site._init_pathinfo()  # type: ignore[attr-defined]
    from langslice.doors.cli.jobenv import PARENT_PTH

    for folder in info["site"]:
        if not os.path.realpath(folder).startswith(os.path.join(os.path.realpath(env), "")):
            continue
        try:
            names = sorted(name for name in os.listdir(folder)
                           if name.endswith(".pth") and not name.startswith("."))
        except OSError:
            continue
        for name in names:
            if name != PARENT_PTH:
                site.addpackage(folder, name, known)  # type: ignore[attr-defined]
    _executable(info["executable"])
    sys.prefix, sys.exec_prefix = info["prefix"], info["exec_prefix"]
    sys.argv = list(rest)
    path = os.path.abspath(script)
    code, signum = as_python(lambda: _run_main(path), script=path)
    if signum:
        raise ExitBySignal(signum)
    return code


def _run_main(path: str) -> None:
    """Run the file *path* as ``python FILE`` runs it: compiled with its
    absolute path, in a fresh ``__main__`` module (``__file__``, a source
    loader, no spec or package) that stands in ``sys.modules`` while it runs."""
    import builtins
    import importlib.machinery
    import types

    with open(path, "rb") as handle:
        source = handle.read()
    code = compile(source, path, "exec", dont_inherit=True)
    main = types.ModuleType("__main__")
    main.__dict__.update({
        "__file__": path, "__cached__": None, "__package__": None, "__spec__": None,
        "__loader__": importlib.machinery.SourceFileLoader("__main__", path),
        "__builtins__": builtins, "__annotations__": {},
    })
    held = sys.modules.get("__main__")
    sys.modules["__main__"] = main
    try:
        exec(code, main.__dict__)  # noqa: S102 - the agent's own script, as python runs it
    finally:
        if held is not None:
            sys.modules["__main__"] = held
