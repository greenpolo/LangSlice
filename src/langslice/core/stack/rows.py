"""A section as one JSON row, both ways.

A row holds every field of a :class:`~langslice.core.stack.section.Section`
under its own key, its parts in their JSON form
(:mod:`langslice.core.stack.codec`); a section's cutting angles are under
``cutting_angles_deg``. The state file writes one row per section around
the stack's own fields, and a picture records the rows' fields it was
drawn from, so it can be drawn again from them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from langslice.core.jsonable import JsonObject, JsonValue
from langslice.core.stack.codec import (
    affine_from_json,
    affine_to_json,
    angles_from_json,
    angles_to_json,
    deformation_from_json,
    deformation_to_json,
    image_from_json,
    image_to_json,
    layer_from_json,
    layer_to_json,
    mask_from_json,
    mask_to_json,
    trace_from_json,
    trace_to_json,
)
from langslice.core.stack.images import ImageRef, LayerRef, MaskRef
from langslice.core.stack.json_reads import (
    as_object,
    flag_at,
    integer_at,
    number_at,
    object_at,
    objects_at,
    text_at,
    texts_at,
)
from langslice.core.stack.landmarks import Landmarks
from langslice.core.stack.landmarks_json import (
    landmarks_from_json,
    landmarks_to_json,
)
from langslice.core.stack.section import FLAT, Angles, Section

_T = TypeVar("_T")

#: A section row as this module writes it.
Row = dict[str, JsonValue]

#: The keys of a section row, in the order a row writes them.
ROW_KEYS = (
    "id", "index_original", "index_corrected", "flip", "rotation_deg",
    "damaged_regions", "damage_note", "position_mm", "transform",
    "border_trace", "deformation", "landmarks", "layers", "masks",
    "images", "caveats", "cutting_angles_deg",
)  # fmt: skip
#: The key a row holds a section's cutting angles under.
ANGLES_KEY = "cutting_angles_deg"


def section_to_json(section: Section) -> Row:
    """Return a section as its JSON row.

    Parameters
    ----------
    section
        The section written.

    Returns
    -------
    dict
        Every key of :data:`ROW_KEYS`, in order.
    """
    transform = trace = deformation = None
    if section.transform is not None:
        transform = affine_to_json(section.transform)
    if section.border_trace is not None:
        trace = trace_to_json(section.border_trace)
    if section.deformation is not None:
        deformation = deformation_to_json(section.deformation)
    images = {name: image_to_json(ref) for name, ref in section.images.items()}
    masks = {name: mask_to_json(ref) for name, ref in section.masks.items()}
    row: Row = {
        "id": section.id,
        "index_original": section.index_original,
        "index_corrected": section.index_corrected,
        "flip": section.flip,
        "rotation_deg": section.rotation_deg,
        "damaged_regions": list[JsonValue](section.damaged_regions),
        "damage_note": section.damage_note,
        "position_mm": section.position_mm,
        "transform": transform,
        "border_trace": trace,
        "deformation": deformation,
        "landmarks": landmarks_to_json(section.landmarks),
        "layers": [layer_to_json(ref) for ref in section.layers],
        "masks": dict[str, JsonValue](masks),
        "images": dict[str, JsonValue](images),
        "caveats": list[JsonValue](section.caveats),
        ANGLES_KEY: angles_to_json(section.angles),
    }
    return row


def fields_to_json(section: Section, keys: Sequence[str]) -> Row:
    """Return some of a section row's fields.

    Parameters
    ----------
    section
        The section written.
    keys
        Keys of :data:`ROW_KEYS`, in the order wanted.

    Returns
    -------
    dict
        Each key and its value in the row.
    """
    row = section_to_json(section)
    return {key: row[key] for key in keys}


def _or_default(
    row: JsonObject,
    key: str,
    read: Callable[[JsonObject, str], _T],
    default: _T,
) -> _T:
    """Return what a row holds under *key*, or *default* without it."""
    if key not in row:
        return default
    return read(row, key)


def _named(row: JsonObject, key: str, read: Callable[[JsonObject], _T]) -> _T:
    """Return what *read* makes of a row.

    Its error names the section and the part *key*.
    """
    try:
        return read(row)
    except ValueError as exc:
        raise ValueError(f"section {row.get('id')}, {key}: {exc}") from exc


def _part(
    row: JsonObject, key: str, read: Callable[[JsonObject], _T]
) -> _T | None:
    """Return the part a row holds under *key* as *read* makes it.

    None for null; its error names the section and the part.
    """

    def whole(given: JsonObject) -> _T | None:
        data = _part_at(given, key)
        return None if data is None else read(data)

    return _named(row, key, whole)


def _layers(row: JsonObject) -> list[LayerRef]:
    entries = objects_at(row, "layers") if row.get("layers") else ()
    return [layer_from_json(item) for item in entries]


def _part_at(row: JsonObject, key: str) -> JsonObject | None:
    """Return the part a row holds under *key*, or None for null."""
    if row.get(key) is None:
        return None
    return object_at(row, key)


def _texts(row: JsonObject, key: str) -> list[str]:
    """Return a row's list of texts; an absent or null one is empty."""
    if row.get(key) is None:
        return []
    return list(texts_at(row, key))


def _position(row: JsonObject) -> float | None:
    """Return a row's position, or None for an unplaced section."""
    if row.get("position_mm") is None:
        return None
    return number_at(row, "position_mm")


def _images(row: JsonObject) -> dict[str, ImageRef]:
    entries = _part_at(row, "images") or {}
    return {
        name: image_from_json(as_object("images", entry))
        for name, entry in entries.items()
    }


def _masks(row: JsonObject) -> dict[str, MaskRef]:
    entries = _part_at(row, "masks") or {}
    return {
        name: mask_from_json(as_object("masks", entry))
        for name, entry in entries.items()
    }


def _landmarks(row: JsonObject) -> Landmarks | None:
    data = _part_at(row, "landmarks")
    if data is None:
        return None
    return landmarks_from_json(data)


def _angles(row: JsonObject) -> Angles:
    data = _part_at(row, ANGLES_KEY)
    if data is None:
        return FLAT
    return angles_from_json(data)


def section_from_json(row: JsonObject) -> Section:
    """Return the section a JSON row holds.

    A row without a field has the section's default for it: no flip, no
    turn, no damage, no parts, the flat plane.

    Parameters
    ----------
    row
        A row :func:`section_to_json` writes.

    Returns
    -------
    Section

    Raises
    ------
    ValueError
        When a key holds a value of the wrong type; a part's error names
        the section and the part (``section <id>, deformation: ...``).
    """
    transform = _part(row, "transform", affine_from_json)
    trace = _part(row, "border_trace", trace_from_json)
    deformation = _part(row, "deformation", deformation_from_json)
    return Section(
        id=text_at(row, "id"),
        index_original=integer_at(row, "index_original"),
        index_corrected=integer_at(row, "index_corrected"),
        flip=_or_default(row, "flip", flag_at, False),
        rotation_deg=_or_default(row, "rotation_deg", integer_at, 0),
        damaged_regions=_texts(row, "damaged_regions"),
        damage_note=_or_default(row, "damage_note", text_at, ""),
        position_mm=_position(row),
        transform=transform,
        border_trace=trace,
        deformation=deformation,
        landmarks=_named(row, "landmarks", _landmarks),
        layers=_named(row, "layers", _layers),
        masks=_named(row, "masks", _masks),
        images=_named(row, "images", _images),
        caveats=_texts(row, "caveats"),
        angles=_angles(row),
    )


def with_fields(section: Section, fields: JsonObject) -> Section:
    """Return a copy of a section with some row fields put back.

    Parameters
    ----------
    section
        The section the other fields are kept from; not changed.
    fields
        Keys of :data:`ROW_KEYS` and their values in a row.

    Returns
    -------
    Section
    """
    return section_from_json({**section_to_json(section), **fields})
