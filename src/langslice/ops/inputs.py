"""What a long operation's result depends on, per section: checked before it applies.

A fit (``fit_affine``, ``fit_deformable``) computes outside the job's write
lock, from the state it read, so a running agent and CLI calls keep working
meanwhile. Before it applies, under the lock and after the job has reloaded
what others wrote (:meth:`langslice.job.job.Job.writing`), it compares
each section's :func:`section_inputs` with the value it computed from: an
unchanged section is applied, a changed one is refused as that section's
row (:data:`STALE_INPUT`) and the others still apply.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langslice.core import appearance as looks
from langslice.core.deformation import linear_key
from langslice.core.state import SliceState, StackState

#: A section whose inputs changed while its result was computed.
STALE_INPUT = "STALE_INPUT"


def section_inputs(
    state: StackState, record: SliceState, *, deformation: bool = False, trace: bool = False,
) -> str:
    """A digest of what a fit of *record* reads: its linear placement
    (position, plane, cutting angles, flip, rotation, transform), its fit
    appearance, its damage (the marked regions, which every fit leaves out,
    and a mark naming none), and with *deformation* its applied
    deformation (``start="current"``), with *trace* its image correction."""
    held: dict[str, Any] = {
        "linear": linear_key(state, record),
        "fit_look": looks.section_settings(state, "fit", record.id),
        "damage": {"regions": list(record.damaged_regions),
                   "marked": bool(record.damage_marked)},
    }
    if deformation:
        held["deformation"] = (record.deformation or {}).get("key")
    if trace:
        correction = record.image_correction or {}
        held["trace"] = {key: correction.get(key) for key in (
            "status", "geometry_fingerprint", "artifact_dir", "attempt")}
    text = json.dumps(held, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def stale_row(section_id: str) -> dict[str, Any]:
    """The row of a section refused because its inputs changed."""
    return {"id": section_id, "status": "error", "error": STALE_INPUT,
            "message": "This section's placement, appearance or deformation changed while "
            "the fit ran; nothing was written for it. Run the fit again."}
