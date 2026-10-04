"""Run notes: free text saved with the results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.linear.job import Job


def add_note(job: Job, text: str) -> list[str]:
    """Append one stripped line to the run notes; the notes as they now stand.

    One undo step. An empty note is refused (``BAD_ARGS``).
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        raise Refused("BAD_ARGS")
    before = job.snapshot()
    job.state.notes.append(cleaned)
    job.commit(before)
    return list(job.state.notes)
