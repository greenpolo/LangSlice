"""Best-effort, loopback-only live ABBA channel; persistence is independent."""
from __future__ import annotations

import json
import logging
import socket
from typing import Any

from langslice.api.claude_jobs import validate_channel

logger = logging.getLogger(__name__)


class HostChannel:
    def __init__(self, job_id: str, settings: dict[str, Any] | None) -> None:
        self.job_id = job_id
        self.socket: socket.socket | None = None
        settings = validate_channel(settings)
        if settings is None:
            return
        try:
            self.socket = socket.create_connection(
                (settings["address"], settings["port"]), timeout=0.5,
            )
            self.socket.sendall((json.dumps({"token": settings["token"]}) + "\n").encode())
        except OSError:
            self.close()
            logger.warning("ABBA is unreachable; results will still be saved in the job directory")

    def send(self, envelope: dict[str, Any]) -> None:
        if self.socket is None:
            return
        try:
            self.socket.sendall((json.dumps({"id": self.job_id, **envelope}) + "\n").encode())
        except OSError:
            self.close()
            logger.warning("ABBA disconnected; results will still be saved in the job directory")

    def event(self, payload: dict[str, Any]) -> None:
        self.send({"type": "event", "event": {"kind": "data", "payload": payload}})

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
