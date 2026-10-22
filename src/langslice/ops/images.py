"""The ``make_image`` verb and a script's ``add_image``: a section's made images.

A section image (:mod:`langslice.core.section_images`) is a raw channel of the
file, never edited, or a made image: a recipe over the section's other images
(:func:`make_image`) or an array a script computed (:func:`add_image`), saved as
``sections/<stem>/images/<name>.png`` with its entry in ``SliceState.images``.
Every write is one undo step; files are never deleted (undo hides the entry).
:func:`image_names`, :func:`read` and :func:`check_image` are what the fits
and the pictures read through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core import section_images as images
from langslice.core.section_images import NAME_PATTERN, PREPROCESSED
from langslice.ops.refusal import Refused, unknown_sections

if TYPE_CHECKING:
    from langslice.core.section_images import ImageView
    from langslice.core.state import SliceState, StackState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: Names no made image may take: the fits' word for no mask, and ``look``'s
#: word for every layer.
RESERVED = ("none", "all")


@dataclass(frozen=True)
class ImagesMade:
    """What :func:`make_image` did."""

    name: str
    #: The sections the image was made for (or removed from).
    sections: list[str]
    deleted: bool = False
    #: The image as ``status`` lists it now (:func:`langslice.core.section_images.summary`);
    #: empty once it is deleted everywhere.
    in_force: dict[str, Any] = field(default_factory=dict)
    #: Every section image of the stack now, raw channels first.
    images: list[str] = field(default_factory=list)


def stack_images(workspace: Workspace, state: StackState) -> list[str]:
    """Every section image name of the stack: the raw channels (in file order,
    the union over the sections), then ``preprocessed`` and the made ones."""
    out: list[str] = []
    for record in state.in_order():
        for name in images.raw_names(workspace, record.id):
            if name not in out:
                out.append(name)
    for record in state.in_order():
        for name in [PREPROCESSED, *record.images]:
            if name not in out:
                out.append(name)
    return out


def image_names(workspace: Workspace, record: SliceState) -> list[str]:
    """The section's images (:func:`langslice.core.section_images.image_names`)."""
    return images.image_names(workspace, record)


def check_image(
    job: Job, workspace: Workspace, records: Sequence[SliceState], name: Any,
) -> str:
    """*name* as a fit reads it (empty: ``preprocessed``), once every one of
    *records* has that image; ``UNKNOWN_IMAGE`` otherwise."""
    label = str(name or PREPROCESSED).strip()
    for record in records:
        known = images.image_names(workspace, record)
        if label not in known:
            raise Refused("UNKNOWN_IMAGE", image=label, images=known,
                          message=f"{record.id} has no section image {label!r}; its images: "
                          f"{', '.join(known)}.")
    return label


def read(job: Job, workspace: Workspace, record: SliceState, name: str) -> ImageView:
    """Section image *name* of *record* (``UNKNOWN_IMAGE`` when it has none)."""
    check_image(job, workspace, [record], name)
    return images.read_image(workspace, record, str(name))


def _check_name(workspace: Workspace, state: StackState, name: Any) -> str:
    label = str(name or "").strip()
    if label.lower() in RESERVED or not NAME_PATTERN.match(label):
        raise Refused("BAD_ARGS", message=f"an image's name is 1 to {images.MAX_NAME} letters, "
                      "digits, '_' or '-', starting with a letter or digit, and not "
                      f"{' or '.join(map(repr, RESERVED))}.")
    raw = {value for record in state.slices for value in images.raw_names(workspace, record.id)}
    if label in raw:
        raise Refused("BAD_ARGS", message=f"{label!r} is a raw channel of the image files; a raw "
                      "channel is never edited. Make an image from it under another name.")
    return label


def _records(job: Job, sections: Sequence[str] | None) -> list[SliceState]:
    state = job.state
    if not sections:
        return state.in_order()
    found: list[SliceState] = []
    unknown: list[str] = []
    for ref in sections:
        record = state.resolve(ref)
        if record is None:
            unknown.append(str(ref))
        elif record not in found:
            found.append(record)
    if unknown:
        raise unknown_sections(state, unknown)
    return found


def _in_force(workspace: Workspace, job: Job, name: str) -> dict[str, Any]:
    return dict(images.summary(job.state, workspace.spec).get(name) or {})


def make_image(
    job: Job,
    workspace: Workspace,
    name: str,
    sections: Sequence[str] | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    clahe_clip: float = images.DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = images.DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    contrast: Sequence[float] | None = None,
    gamma: float | None = None,
    invert: bool = False,
    delete: bool = False,
) -> ImagesMade:
    """Make, replace or delete the made image *name* of the sections (empty:
    every section). One undo step.

    The recipe (:func:`langslice.core.section_images.validate_recipe`): *weights*
    ``{image name: weight}`` (empty: each raw channel by its tissue coverage),
    CLAHE, *n4*, *denoise*, then *contrast* (fractions 0 to 1), *gamma*,
    *invert*. Each section's image is computed from its own images
    (:func:`langslice.core.section_images.make`) and saved as a file before the
    write. ``preprocessed`` made with every argument at its default is the
    opening image again (the job's own, as it was made when the job opened).
    *delete* removes the entry from the sections (``preprocessed`` cannot be
    deleted: every fit reads it by default).

    Refused (nothing written): ``BAD_ARGS`` (the name: a raw channel's, a
    reserved word, a bad spelling; a bad value; an image made from itself;
    deleting ``preprocessed``), ``UNKNOWN_SLICE_IDS``, ``UNKNOWN_IMAGE`` (a
    weighted image a section lacks; delete of a name no named section holds),
    ``RENDER_FAILED`` (a section's image could not be computed).
    """
    state = job.state
    label = _check_name(workspace, state, name)
    records = _records(job, sections)
    if delete:
        if label == PREPROCESSED:
            raise Refused("BAD_ARGS", message="preprocessed is what every fit reads by "
                          "default and cannot be deleted; make it again instead.")
        holding = [record for record in records if label in record.images]
        if not holding:
            raise Refused("UNKNOWN_IMAGE", image=label,
                          message=f"no section named holds a made image {label!r}.")
        before = job.snapshot()
        for record in holding:
            del record.images[label]
        job.commit(before)
        return ImagesMade(name=label, sections=[record.id for record in holding],
                          deleted=True, in_force=_in_force(workspace, job, label),
                          images=stack_images(workspace, job.state))
    try:
        recipe = images.validate_recipe(
            weights=dict(weights) if weights else None, clahe_clip=clahe_clip,
            clahe_tiles=clahe_tiles, n4=n4, denoise=denoise,
            contrast=list(contrast) if contrast else None, gamma=gamma or None, invert=invert)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    held = recipe["weights"]
    if isinstance(held, Mapping) and label in held:
        raise Refused("BAD_ARGS", message=f"an image is made from other images; {label!r} "
                      "cannot be one of its own sources.")
    if isinstance(held, Mapping):
        for record in records:
            try:
                images.sources_of(workspace, record.id, record.images, recipe)
            except ValueError as exc:
                raise Refused("UNKNOWN_IMAGE", message=str(exc),
                              images=images.image_names(workspace, record)) from exc
    reset = label == PREPROCESSED and recipe == images.OPENING_RECIPE
    entries: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            if reset:
                entries[record.id] = images.opening_entry(workspace, job.layout, record.id)
                continue
            array, used = images.make(workspace, record.id, record.images, recipe)
            extra = {} if isinstance(held, Mapping) else {"weights_used": used}
            entries[record.id] = images.write_file(job.layout, record.id, label, array,
                                                   {"recipe": recipe}, **extra)
        except (OSError, ValueError, RuntimeError) as exc:
            raise Refused("RENDER_FAILED", message=f"{record.id}: {exc}") from exc
    before = job.snapshot()
    for record in records:
        record.images[label] = entries[record.id]
    job.commit(before)
    return ImagesMade(name=label, sections=[record.id for record in records],
                      in_force=_in_force(workspace, job, label),
                      images=stack_images(workspace, job.state))


def add_image(
    job: Job, workspace: Workspace, record: SliceState, name: str, array: Any, note: str = "",
) -> dict[str, Any]:
    """Register *array* as made image *name* of *record*, made by a script (its
    *note* recorded as the source). *array* is ``(H, W)`` on the section's
    working grid, in the file's orientation (as a raw plane is held): ``uint8``,
    or floats from 0 to 1 (scaled to 0-255). One undo step; the file written as
    any made image's. Refused: ``BAD_ARGS`` (the name, the shape or values).
    Returns the entry."""
    label = _check_name(workspace, job.state, name)
    held = np.asarray(array)
    width, height = workspace.working_source(record.id)[0].size
    if held.shape != (height, width):
        raise Refused("BAD_ARGS", message=f"{record.id}'s images are (H, W) = ({height}, "
                      f"{width}) on its working grid, in the file's orientation; got "
                      f"{tuple(held.shape)}.")
    if held.dtype != np.uint8:
        if not np.issubdtype(held.dtype, np.floating) or not np.isfinite(held).all():
            raise Refused("BAD_ARGS", message="an image is uint8, or finite floats from 0 to 1.")
        held = np.rint(np.clip(held, 0.0, 1.0) * 255.0).astype(np.uint8)
    entry = images.write_file(job.layout, record.id, label, held,
                              {"script": str(note or "").strip() or "no note"})
    before = job.snapshot()
    live = job.state.by_id(record.id)
    assert live is not None
    live.images[label] = entry
    job.commit(before)
    return entry
