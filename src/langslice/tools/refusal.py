"""The refusals every tool shares: unknown, frozen or busy sections, caps.

The refusal itself is the core's (:class:`langslice.core.refusal.Refused`,
its codes :class:`langslice.core.refusal.RefusalCode`); these are the ones
the job's rules decide, raised for a whole call or given as one section's
row of a per-section answer. A call naming more sections than its tool
takes is refused whole (:func:`check_section_count`).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, cast

from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.codec import Written
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
    return Refused(
        RefusalCode.UNKNOWN_SLICE_IDS, **state_unknown_sections(state, refs)
    )


def check_frozen(
    job: Job, section_ids: Collection[str], *properties: str
) -> None:
    """Raise :class:`Refused` ``SECTION_BUSY`` when background work is running
    on any of *section_ids* (:meth:`~langslice.job.job.Job.busy_refusal`), else
    ``FROZEN`` when a write to *properties* of any of them would change what
    the job froze: the whole call refuses, nothing written."""
    found = job.busy_refusal(section_ids) or job.frozen_refusal(
        section_ids, *properties
    )
    if found is not None:
        raise found


def section_refusal(
    job: Job,
    section_id: str,
    *properties: str,
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
    refusal = job.busy_refusal([section_id]) or job.frozen_refusal(
        [section_id], *properties
    )
    if refusal is None:
        return None
    named = "work" if "work" in refusal.facts else "properties"
    return Refused(
        refusal.code, **{named: refusal.facts[named]}, message=refusal.message
    )


def refused_row(section_id: str, refusal: Refused) -> Written:
    """Return a refused section's row in a per-section answer.

    Parameters
    ----------
    section_id
        The section's filename.
    refusal
        Why it was refused.

    Returns
    -------
    dict
        ``id``, ``status``, ``error``, then the facts; a refusal that names
        the section among its facts (the core's fit refusals) keeps its
        ``id`` where it stands.
    """
    if "id" in refusal.facts:
        return cast("Written", refusal.payload())
    return cast("Written", refusal.payload(section_id=section_id))


def reworded(refusal: Refused, message: str) -> Refused:
    """Return *refusal* with another message, its other facts in place.

    Parameters
    ----------
    refusal
        The refusal whose message changes.
    message
        Its new message.

    Returns
    -------
    Refused
        The same code, status and facts, the message replaced where it
        stood (after the facts when it had none).
    """
    facts = {**refusal.facts, "message": message, "status": refusal.status}
    return Refused(refusal.code, **facts)


#: The tools the job's own ``transform.max_parallel`` also caps.
TRANSFORM_TOOLS = ("interactive_transform", "elastix_affine")


def check_section_count(
    job: Job,
    name: str,
    count: int,
    *,
    cap: int | None = None,
    reason: str = "",
) -> None:
    """Refuse a call of tool *name* naming too many sections.

    ``TOO_MANY_SECTIONS``, with ``max_sections`` and ``requested``, when
    *count* sections is more than *cap*, the tool's own limit (its ``limits["sections"]`` in
    :data:`langslice.tools.catalog.VERBS`), held for *reason*; or, for the
    transform tools, more than the door lets one call take
    (``Job.picture_cap``) or the job lets one transform call take
    (``Job.transform_cap``). A cap that is None is no cap. The message says
    why the cap holds."""
    why = f" ({reason})"
    if name in TRANSFORM_TOOLS:
        caps = (job.picture_cap, job.transform_cap)
        held = [value for value in caps if value is not None]
        cap = min(held) if held else None
        why = " here"
    if cap is not None and count > cap:
        raise Refused(
            RefusalCode.TOO_MANY_SECTIONS,
            max_sections=cap,
            requested=count,
            message=f"{name} takes at most {cap} sections per call{why}; "
            f"{count} were named. Nothing was done: split them over calls.",
        )
