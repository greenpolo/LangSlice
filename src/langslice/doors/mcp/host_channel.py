"""Saved ABBA jobs opened by id, and their loopback-only live ABBA channel.

A saved ABBA job is a job folder whose ``job.json`` carries ``job_id`` and,
under ``host``, ``kind`` ``"host"``; the id leads there through the index
(:mod:`langslice.job.index`, ``~/.langslice/jobs/<id>.json``), which also
holds the host's channel. :func:`load_job` finds one for
``start_job(job_id=...)``. The channel is best-effort; persistence is
independent of it.
"""

from __future__ import annotations

import json
import logging
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langslice.job import index
from langslice.job.layout import JobLayout, read_job_file

logger = logging.getLogger(__name__)

#: The newest saved job's own format inside ``job.json`` (``host``); a newer
#: one is refused.
FORMAT_VERSION = 2
#: A saved job's ``host.kind``.
SAVED_KIND = "host"
#: The addresses a host channel may use: loopback only.
LOOPBACK = ("127.0.0.1", "::1")
#: The length a channel token may have, in characters.
TOKEN_CHARS = (32, 256)
#: Why a channel's port is refused.
PORT_REFUSAL = "The host channel port must be an integer from 1 to 65535"


@dataclass(frozen=True)
class Channel:
    """Where a saved job's ABBA listens, and the token it expects."""

    #: A literal loopback address (:data:`LOOPBACK`).
    address: str
    port: int
    token: str


@dataclass(frozen=True)
class SavedJob:
    """A saved ABBA job as its id finds it."""

    #: Its job folder.
    folder: Path
    #: Its ABBA's channel, or None.
    channel: Channel | None


def validate_channel(value: object) -> Channel | None:
    """Return a host channel's settings, checked.

    Parameters
    ----------
    value
        The index entry's ``host_channel``: ``{"address", "port",
        "token"}`` or None.

    Returns
    -------
    Channel or None
        The settings, or None without a channel.

    Raises
    ------
    ValueError
        When the address is not loopback, the port is not 1 to 65535 or
        the token is not 32 to 256 characters long.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("host_channel requires address, port and token")
    fields = cast("dict[str, object]", value)
    if set(fields) != {"address", "port", "token"}:
        raise ValueError("host_channel requires address, port and token")
    address, port, token = fields["address"], fields["port"], fields["token"]
    if not isinstance(address, str) or address not in LOOPBACK:
        raise ValueError(
            "The host channel must use a literal loopback address"
        )
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError(PORT_REFUSAL)
    if not 1 <= port <= 65535:
        raise ValueError(PORT_REFUSAL)
    shortest, longest = TOKEN_CHARS
    if not isinstance(token, str) or not shortest <= len(token) <= longest:
        raise ValueError(
            "The host channel token must contain 32 to 256 characters"
        )
    return Channel(address, port, token)


def load_job(job_id: str) -> SavedJob:
    """Return a saved job, by id.

    Parameters
    ----------
    job_id
        The saved job's id, twelve hexadecimal digits.

    Returns
    -------
    SavedJob
        Its job folder and its ABBA's channel.

    Raises
    ------
    ValueError
        When the id is malformed or unknown, or its folder holds another
        job, a newer format or no saved job.
    """
    if re.fullmatch(r"[0-9a-f]{12}", job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    entry = index.lookup(index.default_root(), job_id)
    if entry is None:
        raise ValueError(f"No saved LangSlice job {job_id}")
    layout = JobLayout(Path(cast("str", entry["job_folder"])))
    data = read_job_file(layout)
    host = None if data is None else data.get("host")
    if data is None or not isinstance(host, dict):
        raise ValueError(
            f"Saved job {job_id}: {layout.folder} holds no saved job"
        )
    if data.get("job_id") != job_id:
        raise ValueError(
            f"Saved job {job_id} was replaced by job {data.get('job_id')} "
            f"in {layout.folder} (one job folder per image folder)"
        )
    record = cast("dict[str, object]", host)
    version = record.get("format", FORMAT_VERSION)
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise ValueError(
            "This saved job was written by a newer LangSlice. Update "
            "LangSlice to open it."
        )
    if record.get("kind", SAVED_KIND) != SAVED_KIND:
        raise ValueError(
            f"Saved job {job_id} is no ABBA job; open its image folder "
            "instead: start_job(image_folder=...)"
        )
    channel = validate_channel(entry.get("host_channel"))
    return SavedJob(layout.folder, channel)


class HostChannel:
    def __init__(self, job_id: str, channel: Channel | None) -> None:
        self.job_id = job_id
        self.socket: socket.socket | None = None
        if channel is None:
            return
        try:
            self.socket = socket.create_connection(
                (channel.address, channel.port),
                timeout=0.5,
            )
            self.socket.sendall(
                (json.dumps({"token": channel.token}) + "\n").encode()
            )
        except OSError:
            self.close()
            logger.warning(
                "ABBA is unreachable; results will still be saved in the job directory"
            )

    def send(self, envelope: dict[str, Any]) -> None:
        if self.socket is None:
            return
        try:
            self.socket.sendall(
                (json.dumps({"id": self.job_id, **envelope}) + "\n").encode()
            )
        except OSError:
            self.close()
            logger.warning(
                "ABBA disconnected; results will still be saved in the job directory"
            )

    def event(self, payload: dict[str, Any]) -> None:
        self.send(
            {"type": "event", "event": {"kind": "data", "payload": payload}}
        )

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
