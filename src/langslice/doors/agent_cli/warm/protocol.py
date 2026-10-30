"""Where a job's warm process listens, and the messages on its socket.

One warm process per job folder and Python environment (:func:`identity`),
its files in a private folder (:func:`runtime_dir`, :class:`Place`); a
message is a JSON object after its length.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any

#: Bumped when the messages change.
PROTOCOL = 1
#: How long either side waits for the other's next message before a call starts.
HANDSHAKE_S = 10.0
#: Environment variables read when the libraries the server has loaded start
#: (thread pools, devices, allocators, Python's own): a caller whose values
#: differ runs cold. ``HOME`` too (the atlases' folder).
ENV_PREFIXES = ("PYTHON", "OMP_", "MKL_", "OPENBLAS_", "GOTO", "KMP_", "CUDA_", "NVIDIA_",
                "PYTORCH_", "TORCH", "NPY_", "LD_", "ITK_", "SITK_", "MALLOC_", "MPL",
                "LANGSLICE_ATLAS", "BRAINGLOBE")
#: The streams handed to a child, in fd order.
STREAMS = ("stdin", "stdout", "stderr")


def langslice_dir() -> str:
    """Return LangSlice's own package folder.

    Returns
    -------
    str
        The folder of the ``langslice`` package this process imported.
    """
    here = Path(os.path.abspath(__file__))  # langslice/doors/agent_cli/warm/
    return str(here.parents[3])


def _flags() -> list[Any]:
    """The interpreter flags that change what a process imports."""
    flags = sys.flags
    return [flags.optimize, flags.dont_write_bytecode, flags.no_user_site, flags.no_site,
            flags.ignore_environment, flags.isolated, flags.dev_mode, flags.utf8_mode]


def identity(path: list[str]) -> str:
    """The Python environment a process runs in: its interpreter, flags,
    import path (*path*: a client's without its first entry, the script's
    folder) and LangSlice's folder."""
    import hashlib

    data = [PROTOCOL, os.path.realpath(sys.executable), sys.version, _flags(), list(path),
            langslice_dir()]
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


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    """Send one message: its length, then its JSON.

    Parameters
    ----------
    sock
        The connection between a client and the server.
    message
        The JSON object to send.
    """
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


def read_message(sock: socket.socket) -> dict[str, Any] | None:
    """Read the next message.

    Parameters
    ----------
    sock
        The connection between a client and the server.

    Returns
    -------
    dict or None
        The message, or None at the end of the connection.
    """
    head = _exactly(sock, 4)
    if head is None:
        return None
    body = _exactly(sock, struct.unpack("!I", head)[0])
    if body is None:
        return None
    message = json.loads(body)
    return message if isinstance(message, dict) else None
