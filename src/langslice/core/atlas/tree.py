"""The atlas structure table and walks over its hierarchy.

BrainGlobe's ``atlas.structures`` holds one record per region: ``id``,
``acronym``, ``name`` and ``structure_id_path`` (root first, the region
itself last). Records are read in place and never copied: a copy of a
BrainGlobe record reads every key, which loads its lazy ``mesh``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def structure_table(atlas: Any) -> dict[int, Any]:
    """``{id: record}`` for every region the atlas lists; empty when it has none."""
    structures = getattr(atlas, "structures", None)
    if not structures:
        return {}
    records = structures.values() if hasattr(structures, "values") else structures
    return {int(record["id"]): record for record in records}


def path(record: Any) -> list[int]:
    """The record's ``structure_id_path``: root first, the region itself last."""
    return [int(v) for v in record.get("structure_id_path") or ()]


def ancestry(record: Any) -> list[int]:
    """The ids containing the record, root first, its parent last."""
    return path(record)[:-1]


def ancestor_at(sid: int, record: Any, depth: int) -> int:
    """The record's ancestor at *depth* (root = 0); *sid* when its path is shorter."""
    ids = path(record)
    return ids[depth] if len(ids) > depth else sid


def descendants(table: dict[int, Any]) -> dict[int, list[int]]:
    """``{id: ids below it}`` for every id that contains another region."""
    below: dict[int, list[int]] = {}
    for sid, record in table.items():
        for ancestor in ancestry(record):
            below.setdefault(ancestor, []).append(sid)
    return below


def with_descendants(atlas: Any, ids: Iterable[int]) -> frozenset[int]:
    """*ids* plus every structure whose ``structure_id_path`` passes through one."""
    roots = {int(i) for i in ids}
    if not roots:
        return frozenset()
    result = set(roots)
    for sid, record in structure_table(atlas).items():
        if roots.intersection(path(record)):
            result.add(sid)
    return frozenset(result)


__all__ = ["ancestor_at", "ancestry", "descendants", "path", "structure_table",
           "with_descendants"]
