"""A section's border trace: the image model's call and what it returned.

``trace_borders`` asks the image model to draw the atlas borders onto the
section (:mod:`langslice.core.fits.trace`). The call's result record is
saved with its attempt in the job folder; the section keeps a
:class:`BorderTrace`, the same record, until a fit reads the traced lines.
The trace does not replace the section's affine or deformation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from langslice.core.jsonable import JsonObject, empty_object


@dataclass(frozen=True)
class BorderTrace:
    """Hold one section's image-model border trace.

    A field that is None was not part of the record: a call still
    running, refused or failed records fewer facts.

    Parameters
    ----------
    id
        The section's filename.
    geometry_fingerprint
        The digest of the placement and image the call was made at.
    status
        ``running``, ``ok`` or ``error``.
    prompt_edited
        Whether the prompt sent differs from the base prompt.
    provider
        The image model's provider.
    model
        The image model.
    profile
        An untested profile's name.
    untested
        True for an untested profile's call.
    include
        The regions whose borders the model was shown alone.
    exclude
        The regions left out of the borders shown.
    output_kind
        What the model was asked for (``border_annotation``).
    fit_performed
        Whether a fit was made inside the call (never, today).
    artifact_dir
        The attempt's folder, relative to the job folder.
    artifact_paths
        The attempt's files by role, relative to the job folder.
    cached
        Whether the record is a saved reply at this geometry reused.
    attempt
        The attempt's number under its call key.
    raw_received
        Whether the model returned an image.
    route
        The transport that carried the call.
    raw_size
        The returned image's ``(width, height)``.
    model_border_pixels
        How many line pixels were read off the returned image.
    error
        A failed call's error code.
    message
        A failed call's message.
    extra
        Other keys the record carried, kept as they came.
    """

    id: str
    geometry_fingerprint: str
    status: str
    prompt_edited: bool | None = None
    provider: str | None = None
    model: str | None = None
    profile: str | None = None
    untested: bool | None = None
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None
    output_kind: str | None = None
    fit_performed: bool | None = None
    artifact_dir: str | None = None
    artifact_paths: Mapping[str, str] | None = None
    cached: bool | None = None
    attempt: int | None = None
    raw_received: bool | None = None
    route: str | None = None
    raw_size: tuple[int, int] | None = None
    model_border_pixels: int | None = None
    error: str | None = None
    message: str | None = None
    extra: JsonObject = field(default_factory=empty_object)
