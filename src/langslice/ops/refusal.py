"""What an operation answers when it cannot do what it was asked."""

from __future__ import annotations

from typing import Any


class Refused(ValueError):
    """An operation refused, with nothing written.

    *code* is the machine-readable reason (``UNKNOWN_SLICE_IDS``,
    ``BAD_ARGS``...); *details* are the plain facts behind it (the ids, the
    allowed values). :meth:`payload` is the shape every door answers with.
    """

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(str(details.get("message") or code))
        self.code = code
        self.details = details

    def payload(self) -> dict[str, Any]:
        return {"status": "error", "error": self.code, **self.details}
