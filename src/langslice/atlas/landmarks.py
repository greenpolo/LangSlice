"""What the atlas annotation says exists where, along the slicing axis.

Two questions, both answered from the annotation volume and the structure
tree rather than from a model's opinion:

* :func:`structures_at` — which structures does the atlas carry at this level?
* :func:`axis_range_of` — over what span of the slicing axis does a structure
  (including its descendants) exist at all?

Together they are an EXTERNAL check on a placement: a section placed where a
structure it plainly shows cannot exist is placed wrong, whatever the spacing
arithmetic says.

Everything here is generic over the BrainGlobe structure tree — no atlas, no
plane and no acronym is special-cased. The presence scan behind
:func:`axis_range_of` walks the annotation volume once per ``(atlas, plane)``
and is cached; every range after that is a dict lookup.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np

from langslice.atlas.core import (
    BrainGlobeAtlas,
    get_slice_region_metadata,
    index_to_position_mm,
)
from langslice.space import Plane, atlas_space_context, slice_axis_index

logger = logging.getLogger(__name__)

#: Structures holding less than this share of the section's tissue area are
#: dropped from :func:`structures_at` — the long tail of slivers is noise at
#: the level of "what is on this section".
MIN_AREA_SHARE = 0.005

#: Most structures one level report lists.
MAX_STRUCTURES = 15

# ponytail: caches keyed by atlas NAME, not object identity. Two different
# atlas objects sharing a name would collide; load_atlas is lru_cached, so in
# practice one name is one object. Swap in a WeakKeyDictionary if that changes.
_PRESENCE_CACHE: dict[tuple[str, str], dict[int, tuple[int, int]]] = {}
_STRUCTURE_CACHE: dict[str, dict[int, dict[str, Any]]] = {}


def _atlas_key(atlas: BrainGlobeAtlas) -> str:
    return str(getattr(atlas, "atlas_name", "unknown"))


# --- the structure tree --------------------------------------------------


def _structure_records(atlas: BrainGlobeAtlas) -> dict[int, dict[str, Any]]:
    """``{id: {"id", "acronym", "name", "path"}}`` for every structure.

    ``path`` is the structure's ``structure_id_path`` as a tuple, which is how
    descendants are found without depending on any per-atlas helper method.
    """
    key = _atlas_key(atlas)
    cached = _STRUCTURE_CACHE.get(key)
    if cached is not None:
        return cached

    structures = getattr(atlas, "structures", None)
    records: dict[int, dict[str, Any]] = {}
    values = getattr(structures, "values", None)
    entries: Iterable[Any] = (
        cast(Callable[[], Iterable[Any]], values)() if callable(values) else []
    )
    for entry in entries:
        try:
            structure_id = int(entry["id"])
            records[structure_id] = {
                "id": structure_id,
                "acronym": str(entry["acronym"]),
                "name": str(entry["name"]),
                "path": tuple(int(p) for p in entry.get("structure_id_path", ())),
            }
        except Exception:  # a malformed row must not cost us the whole tree
            continue
    _STRUCTURE_CACHE[key] = records
    return records


def structure_count(atlas: BrainGlobeAtlas) -> int:
    """How many structures the atlas's tree carries; 0 means "cannot answer"."""
    return len(_structure_records(atlas))


def find_structure(atlas: BrainGlobeAtlas, query: str) -> dict[str, Any] | None:
    """Resolve *query* to one structure record, or ``None``.

    Accepted, in order: an exact acronym, an exact name, or a name substring
    that matches exactly one structure. All case-insensitive.
    """
    wanted = str(query or "").strip().lower()
    if not wanted:
        return None
    records = _structure_records(atlas)
    for record in records.values():
        if record["acronym"].lower() == wanted:
            return record
    for record in records.values():
        if record["name"].lower() == wanted:
            return record
    hits = [r for r in records.values() if wanted in r["name"].lower()]
    return hits[0] if len(hits) == 1 else None


def near_misses(atlas: BrainGlobeAtlas, query: str, limit: int = 5) -> list[str]:
    """Acronyms a :func:`find_structure` lookup was probably reaching for.

    Ranked: acronyms that START with the query first (shortest first, so the
    query itself leads), then structures whose NAME contains it, then fuzzy
    matches. Acronyms lead because that is how anatomists type: a query of
    "SC" means the superior colliculus family, not every structure whose name
    happens to contain the letters s-c.
    """
    wanted = str(query or "").strip().lower()
    records = list(_structure_records(atlas).values())
    if not wanted:
        return []
    suggestions = sorted(
        (r["acronym"] for r in records if r["acronym"].lower().startswith(wanted)),
        key=lambda acronym: (len(acronym), acronym),
    )
    suggestions += [r["acronym"] for r in records if wanted in r["name"].lower()]
    suggestions += difflib.get_close_matches(
        wanted, [r["acronym"].lower() for r in records], n=limit, cutoff=0.6
    )
    by_lower = {r["acronym"].lower(): r["acronym"] for r in records}
    out: list[str] = []
    for candidate in suggestions:
        acronym = by_lower.get(candidate.lower(), candidate)
        if acronym not in out:
            out.append(acronym)
    return out[:limit]


def descendant_ids(atlas: BrainGlobeAtlas, structure_id: int) -> set[int]:
    """*structure_id* and every structure below it in the tree."""
    return {
        sid
        for sid, record in _structure_records(atlas).items()
        if structure_id in record["path"] or sid == structure_id
    }


# --- what is at a level --------------------------------------------------


def structures_at(
    atlas: BrainGlobeAtlas,
    position_mm: float,
    plane: Plane = "coronal",
    *,
    limit: int = MAX_STRUCTURES,
    min_area_share: float = MIN_AREA_SHARE,
) -> list[dict[str, Any]]:
    """Structures the atlas carries at *position_mm*, largest in-plane first.

    ``area_share`` is the structure's share of the section's TISSUE area (not
    of the canvas), so it reads the same whatever the atlas's framing. ``root``
    and unnamed ids are dropped, as is the sliver tail below *min_area_share*.
    """
    # ponytail: reuses get_slice_region_metadata, which also computes centroids
    # we throw away (~0.5 s per level on a 25 µm atlas). Inline a
    # np.unique(..., return_counts=True) pass if the tool ever feels slow.
    regions = get_slice_region_metadata(atlas, position_mm, plane=plane)
    tissue = sum(cast(float, r["area_fraction"]) for r in regions) or 1.0
    out: list[dict[str, Any]] = []
    for region in regions:
        acronym = str(region["acronym"])
        if not acronym or acronym.lower() == "root":
            continue
        share = cast(float, region["area_fraction"]) / tissue
        if share < min_area_share:
            continue
        out.append(
            {
                "acronym": acronym,
                "name": str(region["name"]),
                "area_share": round(share, 4),
            }
        )
        if len(out) >= limit:
            break
    return out


# --- where a structure can be -------------------------------------------


def _presence_by_id(
    atlas: BrainGlobeAtlas, plane: Plane
) -> dict[int, tuple[int, int]]:
    """``{annotation id: (first index, last index)}`` along the slicing axis.

    One pass over the annotation volume per ``(atlas, plane)``, cached — the
    scan costs about a second on a 25 µm atlas, every range afterwards is free.
    """
    key = (_atlas_key(atlas), plane)
    cached = _PRESENCE_CACHE.get(key)
    if cached is not None:
        return cached

    annotation = np.asarray(atlas.annotation)
    axis = slice_axis_index(atlas_space_context(atlas), plane)
    spans: dict[int, tuple[int, int]] = {}
    for index in range(annotation.shape[axis]):
        for raw in np.unique(np.take(annotation, index, axis=axis)):
            structure_id = int(raw)
            if structure_id == 0:
                continue
            first = spans.get(structure_id, (index, index))[0]
            spans[structure_id] = (first, index)
    _PRESENCE_CACHE[key] = spans
    return spans


def axis_range_of(
    atlas: BrainGlobeAtlas, acronym: str, plane: Plane = "coronal"
) -> tuple[float, float] | None:
    """``(lo_mm, hi_mm)`` over which *acronym* (with descendants) exists.

    Positions are atlas-native millimetres along the plane's slicing axis, the
    same convention as every position in the pipeline. Returns ``None`` when
    the structure resolves but has no voxels in the annotation volume.

    Raises:
        LookupError: *acronym* matches no structure. :func:`near_misses`
            turns that into a suggestion list.
    """
    record = find_structure(atlas, acronym)
    if record is None:
        raise LookupError(
            f"no structure in {_atlas_key(atlas)} matches {acronym!r}"
        )
    spans = _presence_by_id(atlas, plane)
    hits = [spans[sid] for sid in descendant_ids(atlas, record["id"]) if sid in spans]
    if not hits:
        return None
    lo = min(first for first, _ in hits)
    hi = max(last for _, last in hits)
    return (
        index_to_position_mm(atlas, lo, plane=plane),
        index_to_position_mm(atlas, hi, plane=plane),
    )
