"""A section's landmark pairs: points on the section matched to the atlas.

Each pair joins a point on the section to a point on its atlas plane. A
pair keeps both endpoints as they were placed (on which picture, at which
pixel), their canonical coordinates (native atlas pixels; fractions of the
oriented section) and the atlas plane it was placed against. Pair ids are
never reused (``L1``, ``L2``, ...).
"""

from __future__ import annotations

from langslice.core.stack.part import Flag, Object, Part, Point, Whole


class Endpoint(Part):
    """Hold one end of a landmark pair as it was placed.

    Parameters
    ----------
    canonical_xy
        The point in its canonical frame: native atlas-plane pixels for
        the atlas end, fractions of the oriented section for the section
        end.
    picture
        The number of the picture it was placed on.
    pixel_xy
        Where on that picture, in picture pixels.
    registration_context
        The placement that picture was drawn under (its atlas plane).
    geometry
        The picture panel the point was read on (its pixel maps).
    """

    canonical_xy: Point
    picture: Whole
    pixel_xy: Point
    registration_context: Object
    geometry: Object


class Pair(Part):
    """Hold one landmark pair.

    Parameters
    ----------
    id
        The pair's id (``L1``, ``L2``, ...).
    atlas
        The atlas endpoint as placed.
    atlas_native_xy
        The atlas endpoint in native atlas-plane pixels.
    section
        The section endpoint as placed, the same fields as ``atlas``.
    section_fraction_xy
        The section endpoint as fractions of the oriented section.
    plane
        The atlas plane the pair was placed against (atlas, plane,
        position, angles).
    stale
        Whether that plane is not the section's now.
    """

    id: str
    atlas: Endpoint
    atlas_native_xy: Point
    section: Endpoint
    section_fraction_xy: Point
    plane: Object
    stale: Flag


class Landmarks(Part):
    """Hold a section's landmark pairs and the next pair number.

    Parameters
    ----------
    next_id
        The number the next new pair takes.
    pairs
        The pairs, in the order they were made.
    """

    next_id: Whole = 1
    pairs: tuple[Pair, ...] = ()
