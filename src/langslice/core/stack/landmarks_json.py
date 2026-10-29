"""The JSON form of a section's landmarks, both ways.

A section's pairs and their endpoints are written as the state file and
the job folder's ``landmarks.json`` hold them; read back strictly, other
keys kept (as :mod:`langslice.core.stack.codec` does for the other parts).
"""

from __future__ import annotations

from langslice.core.jsonable import JsonObject, JsonValue, to_jsonable
from langslice.core.stack.json_reads import (
    flag_at,
    integer_at,
    object_at,
    objects_at,
    other_keys,
    point_at,
    set_items,
    text_at,
)
from langslice.core.stack.landmarks import Endpoint, Landmarks, Pair

#: A JSON object as this module writes it.
Written = dict[str, JsonValue]


# --- landmarks ---------------------------------------------------------------

#: The keys of a landmark pair's endpoint.
ENDPOINT_KEYS = (
    "canonical_xy", "picture", "pixel_xy", "registration_context", "geometry",
)  # fmt: skip
#: The keys of a landmark pair.
PAIR_KEYS = (
    "id", "atlas", "atlas_native_xy", "section", "section_fraction_xy",
    "plane", "stale",
)  # fmt: skip


def endpoint_to_json(endpoint: Endpoint) -> Written:
    """Return a landmark pair's endpoint in its JSON form.

    Parameters
    ----------
    endpoint
        One end of a pair, as placed.

    Returns
    -------
    dict
        Its keys in order, then its other keys.
    """
    form = set_items(
        [
            ("canonical_xy", endpoint.canonical_xy),
            ("picture", endpoint.picture),
            ("pixel_xy", endpoint.pixel_xy),
            ("registration_context", endpoint.registration_context),
            ("geometry", endpoint.geometry),
        ]
    )
    form.update(to_jsonable(endpoint.extra))
    return form


def _endpoint(data: JsonObject, key: str) -> Endpoint:
    entry = object_at(data, key)
    return Endpoint(
        canonical_xy=point_at(entry, "canonical_xy"),
        picture=integer_at(entry, "picture"),
        pixel_xy=point_at(entry, "pixel_xy"),
        registration_context=object_at(entry, "registration_context"),
        geometry=object_at(entry, "geometry"),
        extra=other_keys(entry, ENDPOINT_KEYS),
    )


def pair_to_json(pair: Pair) -> Written:
    """Return a landmark pair in its JSON form.

    Parameters
    ----------
    pair
        One pair of matched points.

    Returns
    -------
    dict
        Its keys in order, then its other keys.
    """
    form = set_items(
        [
            ("id", pair.id),
            ("atlas", endpoint_to_json(pair.atlas)),
            ("atlas_native_xy", pair.atlas_native_xy),
            ("section", endpoint_to_json(pair.section)),
            ("section_fraction_xy", pair.section_fraction_xy),
            ("plane", pair.plane),
            ("stale", pair.stale),
        ]
    )
    form.update(to_jsonable(pair.extra))
    return form


def pair_from_json(data: JsonObject) -> Pair:
    """Return the landmark pair a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`pair_to_json` writes.

    Returns
    -------
    Pair
        The pair.
    """
    return Pair(
        id=text_at(data, "id"),
        atlas=_endpoint(data, "atlas"),
        atlas_native_xy=point_at(data, "atlas_native_xy"),
        section=_endpoint(data, "section"),
        section_fraction_xy=point_at(data, "section_fraction_xy"),
        plane=object_at(data, "plane"),
        stale=flag_at(data, "stale"),
        extra=other_keys(data, PAIR_KEYS),
    )


def landmarks_to_json(landmarks: Landmarks | None) -> Written:
    """Return a section's landmarks in their JSON form.

    Parameters
    ----------
    landmarks
        The section's pairs; None for a section that never had one.

    Returns
    -------
    dict
        ``{"next_id", "pairs"}``, or ``{}`` for None.
    """
    form: Written = {}
    if landmarks is not None:
        pairs = [pair_to_json(pair) for pair in landmarks.pairs]
        form = {"next_id": landmarks.next_id, "pairs": list[JsonValue](pairs)}
    return form


def landmarks_from_json(data: JsonObject) -> Landmarks | None:
    """Return the landmarks a JSON object holds.

    Parameters
    ----------
    data
        A form :func:`landmarks_to_json` writes.

    Returns
    -------
    Landmarks or None
        The pairs, or None for ``{}``.
    """
    if not data:
        return None
    pairs = tuple(pair_from_json(item) for item in objects_at(data, "pairs"))
    return Landmarks(next_id=integer_at(data, "next_id"), pairs=pairs)
