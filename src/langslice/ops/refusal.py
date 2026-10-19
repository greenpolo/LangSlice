"""What an operation answers when it cannot do what it was asked."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langslice.core.atlas.sides import SideError
from langslice.core.state import StackState
from langslice.core.state import unknown_sections as state_unknown_sections

if TYPE_CHECKING:
    from langslice.job.job import Job

#: The code of a call naming a section that is not in the stack.
UNKNOWN_SLICE_IDS = "UNKNOWN_SLICE_IDS"


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


@contextmanager
def sides_refused() -> Iterator[None]:
    """Inside the block a :class:`~langslice.core.atlas.sides.SideError` (a side
    the placement cannot define) becomes the :class:`Refused` of its own code."""
    try:
        yield
    except SideError as exc:
        raise Refused(exc.code, message=str(exc)) from exc


def unknown_sections(state: StackState, refs: Any) -> Refused:
    """``UNKNOWN_SLICE_IDS`` for *refs* that name no section: the refs, the
    valid filenames and the message saying sections are named by filename
    (:func:`langslice.core.state.unknown_sections`)."""
    return Refused(UNKNOWN_SLICE_IDS, **state_unknown_sections(state, refs))


def check_frozen(job: Job, section_ids: Collection[str], *properties: str) -> None:
    """Raise :class:`Refused` ``SECTION_BUSY`` when background work is running
    on any of *section_ids* (:meth:`~langslice.job.job.Job.busy_refusal`), else
    ``FROZEN`` when a write to *properties* of any of them would change what
    the job froze: the whole call refuses, nothing written."""
    found = job.busy_refusal(section_ids) or job.frozen_refusal(section_ids, *properties)
    if found is not None:
        raise Refused(found.pop("error"), **found)


def frozen_row(job: Job, section_id: str, *properties: str) -> dict[str, Any] | None:
    """One section's ``SECTION_BUSY`` or ``FROZEN`` row (``id``, ``status``
    ``error``, ``error``, ``message``; ``work`` or ``properties``), as a
    per-section answer carries it, as :func:`check_frozen` decides, else None."""
    found = job.busy_refusal([section_id]) or job.frozen_refusal([section_id], *properties)
    if found is None:
        return None
    extra = ({"work": found["work"]} if "work" in found
             else {"properties": found["properties"]})
    return {"id": section_id, "status": "error", "error": found["error"], **extra,
            "message": found["message"]}
