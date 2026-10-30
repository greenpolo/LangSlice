"""The refusals every tool shares: an unknown section, a frozen or busy one.

The refusal itself is the core's (:class:`langslice.core.refusal.Refused`,
its codes :class:`langslice.core.refusal.RefusalCode`); these are the ones
the job's rules decide, raised for a whole call or given as one section's
row of a per-section answer.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.stack import Stack
from langslice.core.stack.stack import (
    unknown_sections as state_unknown_sections,
)

if TYPE_CHECKING:
    from langslice.job.job import Job


def unknown_sections(state: Stack, refs: object) -> Refused:
    """``UNKNOWN_SLICE_IDS`` for *refs* that name no section: the refs, the
    valid filenames and the message saying sections are named by filename
    (:func:`langslice.core.stack.stack.unknown_sections`)."""
    return Refused(RefusalCode.UNKNOWN_SLICE_IDS,
                   **state_unknown_sections(state, refs))


def check_frozen(job: Job, section_ids: Collection[str], *properties: str) -> None:
    """Raise :class:`Refused` ``SECTION_BUSY`` when background work is running
    on any of *section_ids* (:meth:`~langslice.job.job.Job.busy_refusal`), else
    ``FROZEN`` when a write to *properties* of any of them would change what
    the job froze: the whole call refuses, nothing written."""
    found = job.busy_refusal(section_ids) or job.frozen_refusal(section_ids, *properties)
    if found is not None:
        raise found


def section_refusal(
    job: Job, section_id: str, *properties: str,
) -> Refused | None:
    """Return one section's busy or frozen refusal (:func:`check_frozen`).

    Parameters
    ----------
    job
        The job whose background work and frozen properties decide.
    section_id
        The section's filename.
    *properties
        The properties the call would write.

    Returns
    -------
    Refused or None
        ``SECTION_BUSY`` (``work``, ``message``) or ``FROZEN``
        (``properties``, ``message``); None when the section may be written.
    """
    refusal = (job.busy_refusal([section_id])
               or job.frozen_refusal([section_id], *properties))
    if refusal is None:
        return None
    named = "work" if "work" in refusal.facts else "properties"
    return Refused(refusal.code, **{named: refusal.facts[named]},
                   message=refusal.message)


def frozen_row(job: Job, section_id: str, *properties: str) -> dict[str, Any] | None:
    """One section's :func:`section_refusal` as a per-section answer's row
    (``id``, ``status`` ``error``, ``error``, then its facts), else None."""
    refusal = section_refusal(job, section_id, *properties)
    return None if refusal is None else refusal.payload(section_id=section_id)
