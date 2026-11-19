"""The ``make_image`` verb and a script's ``add_image``: a section's made images.

A section image (:mod:`langslice.core.sections.images`) is a raw channel of the
file, never edited, or a made image: a recipe over the section's other images
(:func:`make_section_image`, which the ``make_image`` tool,
:func:`make_image_tool`, calls) or an array a script computed
(:func:`add_image`), saved as ``sections/<stem>/images/<name>.png`` with
its entry in
``Section.images``. Every write is one undo step; files are never deleted
(undo hides the entry). :func:`image_names`, :func:`read` and
:func:`check_image` are what the fits and the pictures read through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.hardware import map_processes, map_sections, process_pool
from langslice.core.jsonable import JsonObject
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections import images
from langslice.core.sections.images import NAME_PATTERN, PREPROCESSED
from langslice.core.sections.reading import ITK_THREADS
from langslice.core.stack.codec import image_to_json
from langslice.core.stack.images import ImageRef
from langslice.tools.looking.look import show_result
from langslice.tools.refusal import unknown_sections
from langslice.tools.run import as_list, pictured, spread

if TYPE_CHECKING:
    from langslice.core.sections.images import ImageView
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run

RESERVED = ("none", "all")


@dataclass(frozen=True)
class ImageRecipe:
    """How a made image is computed from the section's other images.

    :func:`make_image_tool` builds it from ``make_image``'s arguments as the
    model gives them and checks the values.
    """

    weights: Mapping[str, float] = field(default_factory=dict[str, float])
    clahe_clip: float = images.DEFAULT_CLAHE_CLIP
    clahe_tiles: int = images.DEFAULT_CLAHE_TILES
    n4: bool = False
    contrast: Sequence[float] = ()
    gamma: float | None = None
    invert: bool = False


@dataclass(frozen=True)
class ImagesMade:
    """What :func:`make_section_image` did."""

    name: str
    sections: list[str]
    deleted: bool = False
    in_force: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)


def stack_images(workspace: Workspace, state: Stack) -> list[str]:
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


def image_names(workspace: Workspace, record: Section) -> list[str]:
    """The section's images (``core.sections.images.image_names``)."""
    return images.image_names(workspace, record)


def check_image(
    job: Job,
    workspace: Workspace,
    records: Sequence[Section],
    name: object,
) -> str:
    """*name* as a fit reads it (empty: ``preprocessed``), once every one of
    *records* has that image; ``NOT_FOUND`` otherwise."""
    label = str(name or PREPROCESSED).strip()
    for record in records:
        known = images.image_names(workspace, record)
        if label not in known:
            raise Refused(
                RefusalCode.NOT_FOUND,
                image=label,
                images=known,
                message=f"{record.id} has no section image {label!r}; its images: "
                f"{', '.join(known)}.",
            )
    return label


def read(job: Job, workspace: Workspace, record: Section, name: str) -> ImageView:
    """Section image *name* of *record* (``NOT_FOUND`` when it has none)."""
    check_image(job, workspace, [record], name)
    return images.read_image(workspace, record, str(name))


def _check_name(workspace: Workspace, state: Stack, name: object) -> str:
    label = str(name or "").strip()
    if label.lower() in RESERVED or not NAME_PATTERN.match(label):
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            message=f"an image's name is 1 to {images.MAX_NAME} letters, "
            "digits, '_' or '-', starting with a letter or digit, and not "
            f"{' or '.join(map(repr, RESERVED))}.",
        )
    raw = {value for record in state.slices for value in images.raw_names(workspace, record.id)}
    if label in raw:
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            message=f"{label!r} is a raw channel of the image files; a raw "
            "channel is never edited. Make an image from it under another name.",
        )
    return label


def _records(job: Job, sections: Sequence[str] | None) -> list[Section]:
    state = job.state
    if not sections:
        return state.in_order()
    found: list[Section] = []
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
    return dict(images.summary(job.state, workspace.settings).get(name) or {})


def make_image_tool(
    run: Run,
    name: str,
    sections: list[str],
    weights: dict[str, float],
    clahe_clip: float,
    clahe_tiles: int,
    n4: bool,
    contrast: list[float],
    gamma: float,
    invert: bool,
    delete: bool,
    *,
    view: bool = True,
) -> dict[str, object]:
    """Make, replace or delete a made image; return the ``make_image`` reply.

    Parameters
    ----------
    run
        The run the tool is called in.
    name, sections, weights, clahe_clip, clahe_tiles, n4, contrast
        The declared arguments as the model sent them
        (``tools/descriptions/looking.py``).
    gamma, invert, delete, view
        More of the declared arguments, as sent.

    Returns
    -------
    dict
        The image, the sections it was made for, its recipe (or
        ``deleted``), every section image of the stack and the pictures.
    """
    job = run.job
    ids = _section_ids(run, as_list(sections))
    try:
        exponent = _exponent(gamma)
    except (TypeError, ValueError):
        raise Refused(RefusalCode.BAD_ARGUMENT, "gamma is a number.") from None
    given = ImageRecipe(
        weights=dict(weights or {}),
        clahe_clip=clahe_clip,
        clahe_tiles=clahe_tiles,
        n4=bool(n4),
        contrast=as_list(contrast),
        gamma=exponent,
        invert=bool(invert),
    )
    done = make_section_image(job, run.workspace, str(name), ids, given, delete=bool(delete))
    reply: dict[str, object] = {
        "status": "ok",
        "image": done.name,
        "sections": ids or "every section",
    }
    if done.deleted:
        reply["deleted"] = True
    else:
        reply["recipe"] = done.in_force
    reply["section_images"] = done.images
    if done.deleted or not run.shows(view):
        return reply
    made = [job.state.by_id(section) for section in done.sections]
    existing = [section for section in made if section is not None]
    spread_out: list[Section] = spread(existing, job.picture_cap)
    pictures = show_result(
        job,
        run.workspace,
        "make_image",
        "section",
        [section.id for section in spread_out],
        images=[done.name],
    )
    return pictured(reply, pictures)


def _section_ids(run: Run, named: list[str]) -> list[str] | None:
    """Return the filenames of the sections named; None when none is named."""
    if not named:
        return None
    scope, unknown = run.resolve_many(named)
    if unknown or not scope:
        raise unknown_sections(run.job.state, unknown)
    return [section.id for section in scope]


def _exponent(gamma: float | None) -> float | None:
    """Return a recipe's gamma as a number; 0 or None is none."""
    if not gamma:
        return None
    return float(gamma)


def make_section_image(
    job: Job,
    workspace: Workspace,
    name: str,
    sections: Sequence[str] | None,
    given: ImageRecipe,
    *,
    delete: bool = False,
) -> ImagesMade:
    """Make, replace or delete the made image *name* of the sections (empty:
    every section). One undo step.

    The recipe *given* (:func:`langslice.core.sections.images.validate_recipe`):
    *weights* ``{image name: weight}`` (empty: each raw channel by its tissue
    coverage), CLAHE, *n4*, then *contrast* (fractions 0 to 1),
    *gamma*, *invert*. Each section's image is computed from its own images
    (:func:`langslice.core.sections.images.make`) and saved as a file before
    the write. ``preprocessed`` made with every argument at its default is the
    opening image again (the job's own, as it was made when the job opened).
    *delete* removes the entry from the sections (``preprocessed`` cannot be
    deleted: every fit reads it by default).

    Refused (nothing written): ``BAD_ARGUMENT`` (the name: a raw channel's, a
    reserved word, a bad spelling; a bad value; an image made from itself;
    deleting ``preprocessed``), ``NOT_FOUND`` (a
    weighted image a section lacks; delete of a name no named section holds),
    ``IO_ERROR`` (a section's image could not be computed).
    """
    state = job.state
    label = _check_name(workspace, state, name)
    records = _records(job, sections)
    if delete:
        if label == PREPROCESSED:
            raise Refused(
                RefusalCode.BAD_ARGUMENT,
                message="preprocessed is what every fit reads by "
                "default and cannot be deleted; make it again instead.",
            )
        holding = [record for record in records if label in record.images]
        if not holding:
            raise Refused(
                RefusalCode.NOT_FOUND,
                image=label,
                message=f"no section named holds a made image {label!r}.",
            )
        before = job.snapshot()
        for record in holding:
            del record.images[label]
        job.commit(before)
        return ImagesMade(
            name=label,
            sections=[record.id for record in holding],
            deleted=True,
            in_force=_in_force(workspace, job, label),
            images=stack_images(workspace, job.state),
        )
    try:
        weights = dict(given.weights) if given.weights else None
        contrast = list(given.contrast) if given.contrast else None
        recipe = images.validate_recipe(
            weights=weights,
            clahe_clip=given.clahe_clip,
            clahe_tiles=given.clahe_tiles,
            n4=given.n4,
            contrast=contrast,
            gamma=given.gamma or None,
            invert=given.invert,
        )
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGUMENT, message=str(exc)) from exc
    held = recipe["weights"]
    if isinstance(held, Mapping) and label in held:
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            message=f"an image is made from other images; {label!r} "
            "cannot be one of its own sources.",
        )
    if isinstance(held, Mapping):
        for record in records:
            try:
                images.sources_of(workspace, record.id, record.images, recipe)
            except ValueError as exc:
                raise Refused(
                    RefusalCode.NOT_FOUND,
                    message=str(exc),
                    images=images.image_names(workspace, record),
                ) from exc
    reset = label == PREPROCESSED and recipe == images.OPENING_RECIPE
    tasks = 2 * len(records) if recipe["n4"] and not reset else 0
    with process_pool(tasks, threads_each=ITK_THREADS) as pool:
        making = _Making(job, workspace, label, recipe, reset, _InProcesses(pool))
        made_all = map_sections(partial(_made, making), records, 1)
    entries = {record.id: entry for record, entry in zip(records, made_all, strict=True)}
    before = job.snapshot()
    for record in records:
        record.images[label] = entries[record.id]
    job.commit(before)
    return ImagesMade(
        name=label,
        sections=[record.id for record in records],
        in_force=_in_force(workspace, job, label),
        images=stack_images(workspace, job.state),
    )


@dataclass(frozen=True)
class _InProcesses:
    """N4 in worker processes (:data:`images.Enhance`)."""

    pool: ProcessPoolExecutor | None

    def __call__(self, planes: list[np.ndarray]) -> list[np.ndarray]:
        return map_processes(images.n4_plane, planes, self.pool)


@dataclass(frozen=True)
class _Making:
    """One ``make_image`` call's image, as every section makes it."""

    job: Job
    workspace: Workspace
    label: str
    recipe: JsonObject
    reset: bool
    enhance: images.Enhance


def _made(making: _Making, section: Section) -> ImageRef:
    """Make *section*'s image and write its file; ``IO_ERROR`` refused."""
    layout, recipe = making.job.layout, making.recipe
    try:
        if making.reset:
            return images.opening_entry(making.workspace, layout, section.id)
        array, used = images.make(
            making.workspace, section.id, section.images, recipe, enhance=making.enhance
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise Refused(
            RefusalCode.IO_ERROR,
            message=f"{section.id}: {exc}; check the section file opens, then try again.",
        ) from exc
    extra = {}
    if not isinstance(recipe["weights"], Mapping):
        extra = {"weights_used": used}
    try:
        return images.write_file(
            layout, section.id, making.label, array, {"recipe": recipe}, extra=extra
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise Refused(
            RefusalCode.IO_ERROR,
            message=f"{section.id}: {exc}; check the section file opens, then try again.",
        ) from exc


def add_image(
    job: Job,
    workspace: Workspace,
    record: Section,
    name: str,
    array: object,
    note: str = "",
) -> dict[str, Any]:
    """Register *array* as made image *name* of *record*, made by a script (its
    *note* recorded as the source). *array* is ``(H, W)`` on the section's
    working grid, in the file's orientation (as a raw plane is held): ``uint8``,
    or floats from 0 to 1 (scaled to 0-255). One undo step; the file written as
    any made image's. Refused: ``BAD_ARGUMENT`` (the name, the shape or values).
    Returns the entry."""
    label = _check_name(workspace, job.state, name)
    held = np.asarray(array)
    width, height = workspace.working_source(record.id)[0].size
    if held.shape != (height, width):
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            message=f"{record.id}'s images are (H, W) = ({height}, "
            f"{width}) on its working grid, in the file's orientation; got "
            f"{tuple(held.shape)}.",
        )
    if held.dtype != np.uint8:
        if not np.issubdtype(held.dtype, np.floating) or not np.isfinite(held).all():
            raise Refused(
                RefusalCode.BAD_ARGUMENT, message="an image is uint8, or finite floats from 0 to 1."
            )
        held = np.rint(np.clip(held, 0.0, 1.0) * 255.0).astype(np.uint8)
    entry = images.write_file(
        job.layout, record.id, label, held, {"script": str(note or "").strip() or "no note"}
    )
    before = job.snapshot()
    live = job.state.by_id(record.id)
    assert live is not None
    live.images[label] = entry
    job.commit(before)
    return image_to_json(entry)
