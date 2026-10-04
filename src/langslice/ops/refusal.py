"""What an operation answers when it cannot do what it was asked."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Refused(ValueError):
    """An operation refused, with nothing written.

    *code* is the machine-readable reason (``UNKNOWN_SLICE_IDS``,
    ``BAD_ARGS``...); *details* are the plain facts behind it (the ids, the
    allowed values). *status* is ``error`` (the call could not run) or
    ``refused`` (a gate: the call is valid but the job is not ready, as
    ``submit``'s). :meth:`payload` is the shape every door answers with.
    """

    def __init__(self, code: str, *, status: str = "error", **details: Any) -> None:
        super().__init__(str(details.get("message") or code))
        self.code = code
        self.status = status
        self.details = details

    @classmethod
    def of(cls, payload: Mapping[str, Any]) -> Refused:
        """The refusal a ``{"status", "error", ...}`` payload states (a job gate's)."""
        rest = {key: value for key, value in payload.items() if key not in ("status", "error")}
        return cls(str(payload.get("error")), status=str(payload.get("status") or "error"),
                   **rest)

    def payload(self) -> dict[str, Any]:
        return {"status": self.status, "error": self.code, **self.details}
