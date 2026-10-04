"""Text lookup in the atlas region hierarchy, for the agent's `grep_atlas` tool."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np

from langslice.core.atlas.core import _resolve_idx_axis
from langslice.core.space import Plane

GREP_ATLAS_LIMIT = 40


def plane_structure_ids(
    state: Any, ctx: Any, position_mm: float, angles: tuple[float, float],
) -> set[int]:
    """Annotation ids present in the atlas plane at a placement and *angles*
    (``(pitch, yaw)``: the section's own)."""
    plane = cast(Plane, state.plane)
    pitch, yaw = angles
    if pitch or yaw:
        from langslice.core.oblique import sample_oblique_annotation

        labels = sample_oblique_annotation(ctx.atlas, position_mm, plane, pitch, yaw)
    else:
        idx, axis = _resolve_idx_axis(ctx.atlas, position_mm, plane)
        labels = np.take(np.asarray(ctx.atlas.annotation), idx, axis=axis)
    return {int(v) for v in np.unique(labels)}


def grep_structures(
    entries: Iterable[Mapping[str, Any]],
    query: str,
    present: set[int] | None = None,
    *,
    limit: int = GREP_ATLAS_LIMIT,
) -> tuple[list[dict[str, Any]], int]:
    """Rows for the regions matching *query*, and the total match count.

    Matches are a numeric id, an exact acronym, an acronym substring, a name
    starting with the query, then any name substring (case-insensitive), in
    that order and by ontology depth within each. *present* is the set of
    annotation ids in a section's atlas plane; a region counts as in the
    section when it or any descendant is in it.
    """
    structures = {int(e["id"]): e for e in entries}
    acronym = {sid: str(e["acronym"]) for sid, e in structures.items()}
    descendants: dict[int, list[int]] = {sid: [] for sid in structures}
    for sid, e in structures.items():
        for ancestor in e["structure_id_path"][:-1]:
            if int(ancestor) in descendants:
                descendants[int(ancestor)].append(sid)

    needle = query.strip().lower()
    ranked: list[tuple[int, int, int]] = []
    for sid, e in structures.items():
        name = str(e["name"]).lower()
        short = acronym[sid].lower()
        if needle.isdigit() and int(needle) == sid:
            tier = 0
        elif short == needle:
            tier = 1
        elif needle in short:
            tier = 2
        elif name.startswith(needle):
            tier = 3
        elif needle in name:
            tier = 4
        else:
            continue
        ranked.append((tier, len(e["structure_id_path"]), sid))
    ranked.sort()

    rows: list[dict[str, Any]] = []
    for _, _, sid in ranked[:limit]:
        e = structures[sid]
        row: dict[str, Any] = {
            "acronym": acronym[sid],
            "id": sid,
            "name": str(e["name"]),
            "ancestry": [acronym.get(int(a), str(a)) for a in e["structure_id_path"][:-1]],
            "n_descendants": len(descendants[sid]),
        }
        if present is not None:
            row["in_section"] = sid in present or any(d in present for d in descendants[sid])
        rows.append(row)
    return rows, len(ranked)
