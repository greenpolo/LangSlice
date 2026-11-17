"""The host's input, read by a door's protocol away from the process's stdin.

A door that serves a host over stdio (the MCP server, ``langslice serve
--stdio``) waits on its input in a thread for as long as it runs. On
Windows a synchronous read of a pipe blocks every other call on the same
pipe, and the runtime of many compiled modules asks about the standard
handles when it loads (numpy's, scipy's Fortran ones such as ``_odepack``):
importing one in any thread then never returns while the read waits. The
engines import such modules on their first call, so the first fit of a
server would hang.

:func:`protocol_input` gives the protocol a duplicate of the input of its
own and puts the null device on the standard input, so nothing else in the
process reads, or asks about, the pipe the host writes to.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import BinaryIO

STDIN = 0


@contextmanager
def protocol_input() -> Generator[BinaryIO]:
    """Hold the host's input for the protocol alone while the block runs.

    Yields
    ------
    BinaryIO
        The host's input, unbuffered. Meanwhile the process's standard input
        (file descriptor 0, the handle Windows gives a loading module and
        ``sys.stdin``) reads the null device; both are put back after.
    """
    saved = os.dup(STDIN)
    wire = os.fdopen(os.dup(saved), "rb", buffering=0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, STDIN)
    os.close(null)
    try:
        yield wire
    finally:
        os.dup2(saved, STDIN)
        os.close(saved)
