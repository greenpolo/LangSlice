"""A warm process's child: become the caller's process and run its call.

A forked child takes the caller's streams, working folder, environment,
``sys.argv`` and ``sys.path``, runs the cold call's code
(:func:`langslice.doors.agent_cli.main.run`) and ends as Python ends.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Any

from langslice.doors.agent_cli.warm.pools import quiet_pools
from langslice.doors.agent_cli.warm.protocol import STREAMS

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
    formats = sys.modules.get("langslice.job.public_files")
    if formats is not None:
        formats.source.cache_clear()
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed()
    quiet_pools()


def set_executable(path: str) -> None:
    """Make ``sys.executable``, and multiprocessing's, the caller's.

    Parameters
    ----------
    path
        The caller's interpreter.
    """
    sys.executable = path
    spawn = sys.modules.get("multiprocessing.spawn")
    if spawn is not None:
        spawn.set_executable(path)


def run_child(message: dict[str, Any], fds: list[int], inherited: list[int]) -> None:
    """Become the caller's process, run its call, end as Python ends.

    Runs in a forked child and never returns.

    Parameters
    ----------
    message
        The caller's request: argv, working folder, environment, streams.
    fds
        The caller's stdin, stdout and stderr, received over the socket.
    inherited
        The server's own file descriptors, closed in the child.
    """
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
        set_executable(message["executable"])
        _streams(message["streams"])
        _fresh_process()
        _IN_CHILD[0] = True
        from langslice.doors.agent_cli.main import run as run_cold

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
