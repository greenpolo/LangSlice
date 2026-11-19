"""Files another process may hold open: replaced, locked and removed again.

On Windows a file another process has open (a reader that takes no lock,
the LangSlice window's watcher, a memory-mapped array, a virus scanner)
cannot be replaced or deleted while that handle lives, and a lock file
being deleted cannot be opened: the call fails with ``PermissionError``
(``WinError 5`` or ``32``). Such handles live for milliseconds, so the
functions here try again for up to :data:`RETRY_S` before the error
stands. Elsewhere the first try is the answer and they cost nothing.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

Result = TypeVar("Result")

RETRY_S = 1.0
RETRY_POLL_S = 0.02


def retried(operation: Callable[[], Result]) -> Result:
    """Run *operation*, again after a ``PermissionError`` for a while.

    The tries go on for up to :data:`RETRY_S` seconds; the last error is
    raised.

    Parameters
    ----------
    operation
        The call to make, without arguments (a ``functools.partial``).

    Returns
    -------
    Result
        What *operation* returns.
    """
    deadline = time.monotonic() + RETRY_S
    while True:
        try:
            return operation()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(RETRY_POLL_S)


def replace_file(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Rename *source* over *target*, retried while another process has it.

    Parameters
    ----------
    source
        The finished temporary file.
    target
        The file it replaces, atomically.
    """
    retried(functools.partial(os.replace, source, target))


def write_json_atomic(
    path: str | os.PathLike[str],
    data: Any,
    *,
    indent: int | None = 2,
    allow_nan: bool = True,
) -> None:
    """Write *data* as JSON to *path* through a uniquely named temporary file
    in the same folder and a rename, so two writers never share a temporary
    and a reader never sees half a file.

    Parameters
    ----------
    path
        The file to write; its folder is made when missing.
    data
        The JSON-serialisable value.
    indent
        The indent of the text; None writes it compact.
    allow_nan
        False raises ``ValueError`` for a NaN or infinity.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            if indent is None:
                json.dump(data, handle, separators=(",", ":"), allow_nan=allow_nan)
            else:
                json.dump(data, handle, indent=indent, allow_nan=allow_nan)
        os.chmod(tmp_path, 0o644)
        replace_file(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def remove_tree(path: str | os.PathLike[str]) -> None:
    """Delete the folder *path* and everything in it.

    A read-only file is made writable, and a file another process holds
    is tried again.

    Parameters
    ----------
    path
        The folder to delete.
    """
    shutil.rmtree(path, onerror=_remove_again)


def _remove_again(
    function: Callable[[str], object],
    path: str,
    error: tuple[type[BaseException], BaseException, object],
) -> None:
    """Make *path* writable and call *function* on it again.

    ``rmtree``'s error handler; any other failure is raised as it was.
    """
    if not isinstance(error[1], PermissionError):
        raise error[1]
    target = Path(path)
    if not target.is_symlink():
        target.chmod(target.stat().st_mode | stat.S_IWRITE)
    retried(functools.partial(function, path))
