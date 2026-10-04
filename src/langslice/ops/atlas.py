"""Atlas lookups: the region hierarchy, searched like text. Reads, never writes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langslice.core.atlas_grep import GREP_ATLAS_LIMIT, grep_structures, plane_structure_ids
from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


def grep_atlas(job: Job, workspace: Workspace, query: str, section: object = "") -> dict[str, Any]:
    """Regions whose acronym or name matches *query*
    (:func:`langslice.core.atlas_grep.grep_structures`).

    With *section* (a filename or corrected index of a section with a
    position), each row says whether the region or a descendant appears in
    the atlas plane at that placement; a section without a position gets a
    ``note`` instead. Returns ``query``, ``matches`` (the total), ``rows``
    (at most :data:`~langslice.core.atlas_grep.GREP_ATLAS_LIMIT`), ``more``
    when some were left out. Refused: ``BAD_ARGS`` (empty query),
    ``NO_STRUCTURES``, ``UNKNOWN_SLICE_IDS``.
    """
    text = str(query).strip()
    if not text:
        raise Refused("BAD_ARGS", message="Empty query.")
    structures = getattr(workspace.atlas, "structures", None)
    entries = list(structures.values()) if structures else []
    if not entries:
        raise Refused("NO_STRUCTURES", message="This atlas has no region hierarchy.")
    state = job.state
    present: set[int] | None = None
    note = ""
    if section != "":
        record = state.resolve(section)
        if record is None:
            raise Refused("UNKNOWN_SLICE_IDS", unknown=[section])
        if record.position_mm is None:
            note = f"{record.id} has no position yet, so in_section is omitted."
        else:
            present = plane_structure_ids(state, workspace, record.position_mm,
                                          record.angles)
    rows, total = grep_structures(entries, text, present, limit=GREP_ATLAS_LIMIT)
    result: dict[str, Any] = {"query": text, "matches": total, "rows": rows}
    if total > len(rows):
        result["more"] = total - len(rows)
    if note:
        result["note"] = note
    return result
