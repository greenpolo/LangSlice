"""Run an engine's process; on a timeout, end it with every process it started.

An engine runs as a subprocess under its own interpreter, and its libraries
may start processes of their own (compilers, worker pools). When the time
runs out, killing only the direct child leaves those running, and reading
its output then waits for as long as one of them holds the pipes. So the
engine starts in a process group of its own (``CREATE_NEW_PROCESS_GROUP``
on Windows, a new session on POSIX), and a timeout kills the whole tree:
``taskkill /T /F`` on Windows, ``killpg`` on POSIX.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

#: How long the output of a killed tree is waited for before it is dropped, s.
KILLED_OUTPUT_WAIT_S = 10.0

#: The engine's own process group on Windows, so its tree can be ended whole.
CREATION_FLAGS = (
    subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
)
#: The engine leads its own session on POSIX, whose group is its tree.
NEW_SESSION = sys.platform != "win32"

if sys.platform == "win32":

    def kill_tree(pid: int) -> None:
        """End a process and every process it started.

        Parameters
        ----------
        pid
            The process at the root of the tree.
        """
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )

else:

    def kill_tree(pid: int) -> None:
        """End a process and every process it started.

        Parameters
        ----------
        pid
            The process at the root of the tree, which leads its group.
        """
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def run_engine(
    command: list[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with its output captured; on a timeout, kill its tree.

    Parameters
    ----------
    command
        The program and its arguments.
    timeout_s
        Seconds the command may run before its tree is killed.
    env
        The command's environment; None means this process's.
    cwd
        The folder the command runs in; None means this process's.

    Returns
    -------
    subprocess.CompletedProcess
        The return code and the text of standard output and error; standard
        input is the null device.

    Raises
    ------
    subprocess.TimeoutExpired
        The command ran past ``timeout_s``; its tree is dead, and what it
        wrote before is on the exception.
    """
    # no ``with``: closing a pipe a reader thread still blocks on would hang
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=env,
        cwd=cwd,
        creationflags=CREATION_FLAGS,
        start_new_session=NEW_SESSION,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        kill_tree(process.pid)
        stdout, stderr = _killed_output(process)
        raise subprocess.TimeoutExpired(
            command, timeout_s, output=stdout, stderr=stderr
        ) from None
    except BaseException:  # interrupted: leave nothing of the engine running
        kill_tree(process.pid)
        raise
    return subprocess.CompletedProcess(
        command, process.returncode, stdout, stderr
    )


def _killed_output(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Read what a killed tree wrote.

    Nothing when a survivor holds the pipes past
    :data:`KILLED_OUTPUT_WAIT_S`.
    """
    try:
        return process.communicate(timeout=KILLED_OUTPUT_WAIT_S)
    except subprocess.TimeoutExpired:  # a process outside the tree holds them
        process.kill()
        process.wait()
        return "", ""
