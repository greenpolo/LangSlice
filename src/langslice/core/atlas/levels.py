"""Region levels: every annotation id at one depth of the atlas ontology.

The depth of a structure is its place in BrainGlobe's ``structure_id_path``
(the root is depth 0). :func:`labels_at` maps each id of a label map to its
ancestor at *depth*; an id whose path is shorter than that stays itself, as
does an id the atlas does not list, and 0 (background) stays 0. One rule for
every atlas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from langslice.core.atlas.tree import ancestor_at, structure_table

#: The depth agent overlays draw region contours at unless a call asks for
#: another (``look``'s ``depth``).
OVERLAY_DEPTH = 6
#: The depth of the regions the image model is shown and traces, and of the
#: atlas regions fits against its lines read. Fixed.
MODEL_DEPTH = 6

#: ``id(structures) -> (structures, {depth: (ids, ancestors)})``; the table is
#: held so its id is never reused while cached.
_LOOKUPS: dict[int, tuple[Any, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}


def _lookup(atlas: Any, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, ancestors)``: the atlas's structure ids, sorted, and the
    ancestor at *depth* of each; cached per structure table and depth."""
    structures = getattr(atlas, "structures", None)
    if not structures:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    held = _LOOKUPS.get(id(structures))
    if held is None or held[0] is not structures:
        held = _LOOKUPS[id(structures)] = (structures, {})
    per_atlas = held[1]
    if depth not in per_atlas:
        pairs = sorted((sid, ancestor_at(sid, record, depth))
                       for sid, record in structure_table(atlas).items() if sid != 0)
        per_atlas[depth] = (np.array([p[0] for p in pairs], dtype=np.int64),
                            np.array([p[1] for p in pairs], dtype=np.int64))
    return per_atlas[depth]


def labels_at(labels: np.ndarray, atlas: Any, depth: int) -> np.ndarray:
    """*labels* with every id replaced by its ancestor at *depth*.

    Same shape and dtype as *labels*. *depth* is a whole number, 0 or more;
    0 = root, as in BrainGlobe (the index into ``structure_id_path``).
    """
    if isinstance(depth, bool) or int(depth) != depth or depth < 0:
        raise ValueError(f"depth is a whole number, 0 or more; got {depth!r}")
    labels = np.asarray(labels)
    ids, ancestors = _lookup(atlas, int(depth))
    if ids.size == 0:
        return labels.copy()
    index = np.clip(np.searchsorted(ids, labels), 0, ids.size - 1)
    hit = ids[index] == labels
    return np.where(hit, ancestors[index], labels).astype(labels.dtype)


__all__ = ["MODEL_DEPTH", "OVERLAY_DEPTH", "labels_at"]
