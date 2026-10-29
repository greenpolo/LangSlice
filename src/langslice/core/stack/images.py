"""A section's references to its files: made images, masks and layers.

Each names a file under the job folder by its path relative to it. The
files are never deleted, so undoing the write that made one only hides
its reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langslice.core.jsonable import JsonObject, empty_object


@dataclass(frozen=True)
class ImageRef:
    """Hold a made section image (:mod:`langslice.core.sections.images`).

    Parameters
    ----------
    source
        How it was made: ``{"recipe": ...}``, ``{"script": note}`` or
        ``{"opening": ...}``.
    file
        The image file, relative to the job folder.
    digest
        The digest of its pixels.
    made
        When it was made, as ISO 8601 text.
    extra
        Other facts its maker recorded (the weights a recipe used).
    """

    source: JsonObject
    file: str
    digest: str
    made: str
    extra: JsonObject = field(default_factory=empty_object)


@dataclass(frozen=True)
class MaskRef:
    """Hold a section mask (:mod:`langslice.core.sections.masks`).

    Parameters
    ----------
    kind
        ``em`` or ``fit``.
    section_image
        The section image it was made from.
    atlas_image
        The atlas image it was made against, resolved.
    params
        The settings it was made with.
    inputs
        The digest of the section's grid it was made on.
    file
        The mask file, relative to the job folder.
    left_out
        The fraction of the section's tissue it leaves out.
    registration
        The digest of the registration it was made at; None for a mask
        made before it was recorded.
    extra
        Other keys the entry carried, kept as they came.
    """

    kind: str
    section_image: str
    atlas_image: str
    params: JsonObject
    inputs: str
    file: str
    left_out: float
    registration: str | None = None
    extra: JsonObject = field(default_factory=empty_object)


@dataclass(frozen=True)
class LayerRef:
    """Hold a layer a tool plugin left on a section.

    Parameters
    ----------
    id
        ``<tool>/<number>``, unique in the job.
    name
        The layer's name; a newer layer of the same name replaces it.
    kind
        ``points``, ``lines``, ``labels``, ``scalar`` or ``picture``.
    frame
        The frame its coordinates are in.
    file
        The layer file, relative to the job folder.
    tool
        The plugin tool that made it.
    call
        The call that made it.
    created
        When it was made, as ISO 8601 text.
    extra
        Other keys the reference carried, kept as they came.
    """

    id: str
    name: str
    kind: str
    frame: str
    file: str
    tool: str
    call: str
    created: str
    extra: JsonObject = field(default_factory=empty_object)
