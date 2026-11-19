"""A section's border trace: the image model's call and what it returned.

``trace_borders`` asks the image model to draw the atlas borders onto the
section (:mod:`langslice.core.fits.trace`). The call's result record is
saved with its attempt in the job folder; the section keeps a
:class:`BorderTrace`, the same record, until a fit reads the traced lines.
The trace does not replace the section's affine or deformation.
"""

from __future__ import annotations

from langslice.core.stack.part import Flag, Part, Whole, omitted


class BorderTrace(Part):
    """Hold one section's image-model border trace.

    Which fields a record has follows its ``status``; a field it does not
    have is None:

    - every record: ``id``, ``geometry_fingerprint``, ``status``.
    - ``running`` (the call made, no reply yet): the call's facts,
      ``prompt_edited``, ``provider``, ``model``, ``output_kind``,
      ``fit_performed``, ``artifact_dir``, ``artifact_paths``, ``cached``,
      ``attempt`` and ``raw_received`` (False); ``include`` and
      ``exclude`` when regions were chosen, ``profile`` and ``untested``
      for an untested profile.
    - ``ok``: the call's facts, ``raw_received`` True, and the reply's:
      ``route``, ``raw_size``, ``model_border_pixels``. A saved reply
      reused at the same geometry is the same record with ``cached`` True.
    - ``error``: ``error`` and ``message``, with the call's facts when the
      call was made and failed (``raw_received`` says whether the model had
      answered), or only ``artifact_dir`` when no call was made (one at
      this geometry was still running or was interrupted).

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
    provider, model
        The image model's provider and the model called.
    profile
        An untested profile's name.
    untested
        True for an untested profile's call.
    include, exclude
        The regions whose borders the model was shown alone, and those left
        out of the borders shown.
    output_kind
        What the model was asked for (``border_annotation``).
    fit_performed
        Whether a fit was made inside the call (never, today).
    artifact_dir, artifact_paths
        The attempt's folder and its files by role, relative to the job
        folder.
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
    error, message
        A failed call's error code and message.

    Other keys the record carried are kept as they came.
    """

    id: str
    geometry_fingerprint: str
    prompt_edited: Flag | None = omitted()
    provider: str | None = omitted()
    model: str | None = omitted()
    profile: str | None = omitted()
    untested: Flag | None = omitted()
    include: tuple[str, ...] | None = omitted()
    exclude: tuple[str, ...] | None = omitted()
    output_kind: str | None = omitted()
    fit_performed: Flag | None = omitted()
    artifact_dir: str | None = omitted()
    artifact_paths: dict[str, str] | None = omitted()
    cached: Flag | None = omitted()
    attempt: Whole | None = omitted()
    raw_received: Flag | None = omitted()
    status: str
    route: str | None = omitted()
    raw_size: tuple[Whole, Whole] | None = omitted()
    model_border_pixels: Whole | None = omitted()
    error: str | None = omitted()
    message: str | None = omitted()
