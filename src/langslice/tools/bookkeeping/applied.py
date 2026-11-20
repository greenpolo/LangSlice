"""A section's applied deformation record, read from the job's store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langslice.job.exports.public_files import applied_deformation

if TYPE_CHECKING:
    from langslice.core.fits.deformable.record import DeformableRecord
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.job.job import Job


def applied_warp(
    job: Job,
    state: Stack,
    record: Section,
) -> tuple[DeformableRecord | None, str | None]:
    """The section's applied deformation record (None: linear) and its
    stored folder; ``ValueError`` when one is applied but cannot be read."""
    held = applied_deformation(state, record)
    if held is None:
        return None, None
    loaded = job.deformations.get(record.id, held.key or "")
    if loaded is None:
        raise ValueError("its applied deformation record cannot be read")
    return loaded, held.record or None


def applied_record(
    job: Job,
    state: Stack,
    record: Section,
) -> DeformableRecord | None:
    """The section's applied deformation record (None: linear);
    ``ValueError`` when one is applied but cannot be read."""
    return applied_warp(job, state, record)[0]
