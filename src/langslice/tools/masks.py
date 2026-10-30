"""The ``make_mask`` verb: the sections' named section-side masks.

A mask is a weight image on the section's fit grid
(:mod:`langslice.core.sections.masks`) of one of two kinds, both the
three-class EM sorting of the tissue: ``em`` (made here, at the section's
current registration) and ``fit`` (copied from the deformation the section
holds, which saved the sorting its fit made). A fit takes a mask by its name
(``fireants_syn``, ``elastix_affine`` and the engines of tool plugins). Nothing
else makes a mask: no threshold and no drawn region; the damage marks are the
atlas side and never mask the section.

:func:`make_mask` writes ``Section.masks`` of each section named (one undo
step for the call) and the weights under the job folder
(``sections/<stem>/masks/<name>.npz``, never deleted: undo hides the state
reference). :func:`read` and :func:`section_weights` are what the fits and the
pictures read through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.atlas.images import resolve_image
from langslice.core.fits import deformation
from langslice.core.fits.deformable.masks import SORTING_KEYS
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.core.sections.images import PREPROCESSED
from langslice.core.sections.masks import NAME_PATTERN, RESERVED_NAMES
from langslice.core.stack.codec import mask_to_json
from langslice.core.stack.images import MaskRef
from langslice.tools.images import check_image
from langslice.tools.refusal import check_frozen, unknown_sections

if TYPE_CHECKING:
    from langslice.core.sections.masks import MaskView
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: The kinds of mask.
SOURCES = ("em", "fit")
#: The name a fit gives for no section mask.
NO_MASK = "none"


@dataclass(frozen=True)
class MasksMade:
    """What :func:`make_mask` did: one row per section named."""

    name: str
    deleted: bool = False
    #: Per section: ``{"id", "status": "ok", "entry"}`` (the state entry now held:
    #: ``kind``, ``section_image``, ``atlas_image``, ``params``, ``left_out``...; none
    #: for a deleted mask), or ``{"id", "status": "error", "error", "message"}``.
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def made(self) -> list[str]:
        """The sections whose mask was written (or removed)."""
        return [row["id"] for row in self.rows
                if row["status"] == ReplyStatus.OK]


def _check_name(name: Any) -> str:
    text = str(name or "").strip()
    if text.lower() in RESERVED_NAMES or not NAME_PATTERN.match(text):
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"name is 1 to 40 letters, digits, '_' or '-' and "
                      f"not {NO_MASK!r} (that is the fits' word for no mask).")
    return text


def _check_sorting(sorting: Mapping[str, Any] | None) -> dict[str, float]:
    given = dict(sorting or {})
    held = [key for key in ("method", "artifact") if key in given]
    if held:  # the keys another engine's sorting takes
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"make_mask's sorting is the fits' EM with an "
                      f"artifact class: it takes no {' or '.join(held)}; its keys are "
                      f"{list(SORTING_KEYS)}.")
    unknown = sorted(set(given) - set(SORTING_KEYS))
    if unknown:
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"sorting takes {list(SORTING_KEYS)}; not {unknown}.")
    try:
        values = {key: float(value) for key, value in given.items()}
    except (TypeError, ValueError) as exc:
        raise Refused(RefusalCode.BAD_ARGS,
                      message="sorting values are numbers.") from exc
    for key, value in values.items():
        if not np.isfinite(value) or value < 0:
            raise Refused(RefusalCode.BAD_ARGS,
                          message=f"sorting {key} is a number from 0 up.")
    return values


def _records(job: Job, sections: Sequence[object] | None) -> list[Section]:
    state = job.state
    if not sections:
        return state.in_order()
    found: list[Section] = []
    unknown: list[str] = []
    for ref in sections:
        section = state.resolve(ref)
        if section is None:
            unknown.append(str(ref))
        elif section not in found:
            found.append(section)
    if unknown:
        raise unknown_sections(state, unknown)
    return found


def make_mask(
    job: Job, workspace: Workspace, name: str, sections: Sequence[Any] | None = None, *,
    source: str = "em", section_image: str = PREPROCESSED, atlas_image: str = "nissl",
    sorting: Mapping[str, Any] | None = None, delete: bool = False,
) -> MasksMade:
    """Make, replace or delete the mask *name* of each section named (empty:
    every section). One undo step for the call.

    ``em``: sort each section's pixels of *section_image* against *atlas_image*
    at its CURRENT registration (the atlas placed through its transform and
    applied deformation), with the optional starting points *sorting*
    (:data:`SORTING_KEYS`); ``fit``: the sorting the section's applied
    deformation saved with its section, copied. The mask is stale only once the
    section is turned or flipped; a new registration only marks it made at an
    earlier one (:func:`langslice.core.sections.masks.read_mask`); making it
    again replaces it. *delete* removes the entry (the weights file stays; undo
    brings the entry back).

    A section that cannot get its mask is that section's row (``NO_POSITION``,
    ``INVALID_LINEAR_PLACEMENT``, ``NO_DEFORMATION`` / ``NO_SORTING`` for source
    ``fit``, ``FIT_FAILED``); the others are written. Refused, nothing written:
    ``UNKNOWN_SLICE_IDS``, ``SECTION_BUSY``, ``BAD_ARGS`` (the name, the source,
    the starting points, an atlas image the atlas does not have), ``UNKNOWN_IMAGE``,
    ``UNKNOWN_MASK`` (delete of a name no section named holds), and, when no
    section could get its mask, the first section's refusal.
    """
    targets = _records(job, sections)
    check_frozen(job, [section.id for section in targets])
    label = _check_name(name)
    if delete:
        holding = [section for section in targets if label in section.masks]
        if not holding:
            raise Refused(RefusalCode.UNKNOWN_MASK,
                          message=f"no section named holds a mask {label!r}.",
                          masks=sorted({mask for section in targets for mask in section.masks}))
        before = job.snapshot()
        for section in holding:
            del section.masks[label]
        job.commit(before)
        return MasksMade(name=label, deleted=True,
                         rows=[{"id": section.id, "status": "ok"} for section in holding])
    kind = str(source or "em").strip().lower()
    if kind not in SOURCES:
        raise Refused(RefusalCode.BAD_ARGS,
                      message=f"source is one of {list(SOURCES)}.")
    params = _check_sorting(sorting)
    image = str(section_image or PREPROCESSED).strip()
    resolved = ""
    if kind == "em":
        try:
            resolved = resolve_image(workspace.atlas, str(atlas_image or "nissl"))
        except ValueError as exc:
            raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
        check_image(job, workspace, targets, image)
    rows: list[dict[str, Any]] = []
    first: Refused | None = None
    before = job.snapshot()  # the mask entries are set by the writes below
    for section in targets:
        try:
            entry = (_from_fit(job, section, label) if kind == "fit" else
                     _from_em(job, workspace, section, label, section_image=image,
                              atlas_image=resolved, params=params))
        except Refused as refusal:
            first = first or refusal
            row = refusal.payload(section_id=section.id)
            row.update(status=ReplyStatus.ERROR, message=refusal.message or refusal.code)
            rows.append(row)
            continue
        rows.append({"id": section.id, "status": "ok", "entry": mask_to_json(entry)})
    if not any(row["status"] == ReplyStatus.OK for row in rows):
        assert first is not None
        raise first
    job.commit(before)
    return MasksMade(name=label, rows=rows)


def _from_fit(job: Job, section: Section, name: str) -> MaskRef:
    from langslice.core.sections.masks import (
        fit_weights,
        grid_digest,
        registration_digest,
        write_mask,
    )

    if job.deformations.current(job.state, section) is None:
        raise Refused(RefusalCode.NO_DEFORMATION,
                      message=f"{section.id} holds no deformation, so there "
                      "is no fit sorting to copy.")
    weights = fit_weights(job.deformations, job.state, section)
    if weights is None:
        raise Refused(RefusalCode.NO_SORTING,
                      message=f"{section.id}'s deformation saved no sorting (its "
                      "engine estimated none).")
    held = job.deformations.current(job.state, section)
    atlas = held.settings.atlas_image if held is not None and held.settings is not None else ""
    footprint = np.asarray(held.footprint, dtype=bool) if held is not None else None
    inside = weights < 0.5
    if footprint is not None and footprint.shape == inside.shape and footprint.any():
        inside = inside[footprint]
    left_out = 0.0
    if inside.size:
        left_out = float(np.count_nonzero(inside)) / inside.size
    facts = MaskRef(kind="fit", section_image="", atlas_image=str(atlas), params={},
                    inputs=grid_digest(section),
                    registration=registration_digest(job.state, section),
                    file="", left_out=left_out)
    return _write(job, section, name, weights, facts, write_mask)


def _from_em(
    job: Job, workspace: Workspace, section: Section, name: str, *, section_image: str,
    atlas_image: str, params: dict[str, float],
) -> MaskRef:
    from langslice.core.sections.masks import (
        em_sorting,
        grid_digest,
        registration_digest,
        write_mask,
    )

    if section.position_mm is None:
        raise Refused(RefusalCode.NO_POSITION,
                      message=f"{section.id} has no position yet.")
    try:
        grid = deformation.fit_grid(job.state, workspace, section)
    except (ValueError, OSError) as exc:
        raise Refused(RefusalCode.INVALID_LINEAR_PLACEMENT,
                      message=f"{section.id}: {exc}") from exc
    try:
        image, _identity = deformation.section_image(workspace, job.state, grid, section_image)
        pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        with deformation.engine_lock():
            weights, facts = em_sorting(workspace, job.state, section, job.deformations,
                                        section_image=pixels, atlas_image=atlas_image,
                                        params=params)
    except Refused:
        raise
    except Exception as exc:  # noqa: BLE001 - the sort's failure is the caller's refusal
        from langslice.core.fits.engine_log import failure_message

        raise Refused(RefusalCode.FIT_FAILED,
                      message=f"{section.id}: " + failure_message(
            f"{type(exc).__name__}: {exc}", job.engine_log,
            f"the sorting of {section.id} failed")) from exc
    made = MaskRef(kind="em", section_image=section_image,
                   atlas_image=atlas_image,
                   params=cast("dict[str, float]", facts.get("params", {})),
                   inputs=grid_digest(section),
                   registration=registration_digest(job.state, section),
                   file="", left_out=cast("float", facts.get("left_out", 0.0)))
    return _write(job, section, name, weights, made, write_mask)


def _write(job: Job, section: Section, name: str, weights: np.ndarray, facts: MaskRef,
           write_mask: Any) -> MaskRef:
    try:
        return write_mask(job.layout, section, name, weights, facts)  # type: ignore[no-any-return]
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc


def read(job: Job, section: Section, name: str) -> MaskView:
    """The mask *name* of *section* with its weights (``UNKNOWN_MASK`` when it
    has none by that name);
    :attr:`~langslice.core.sections.masks.MaskView.stale`
    says whether the section changed since."""
    from langslice.core.sections.masks import read_mask

    try:
        return read_mask(job.layout, job.state, section, name)
    except KeyError as exc:
        raise Refused(RefusalCode.UNKNOWN_MASK,
                      message=f"{section.id} has no mask {name!r}; make it "
                      "with make_mask.", masks=sorted(section.masks)) from exc
    except ValueError as exc:  # the file cannot be read
        raise Refused(RefusalCode.FIT_FAILED, message=str(exc)) from exc


def section_weights(job: Job, section: Section, name: str) -> np.ndarray | None:
    """The weights the fit *name* names reads on *section*, a float array on
    its
    fit grid; None for ``"none"``. Refused: ``UNKNOWN_MASK``, ``STALE_MASK``
    (the section changed since the mask was made)."""
    from langslice.core.sections.masks import section_weights as weights_of

    label = str(name or NO_MASK).strip()
    if label == NO_MASK:
        return None
    view = read(job, section, label)
    if view.stale:
        raise Refused(RefusalCode.STALE_MASK,
                      message=f"{section.id}'s mask {label!r} was made before the "
                      "section changed; make it again with make_mask.")
    return weights_of(view)


def made_earlier(state: Stack, section: Section, name: str) -> bool:
    """Whether the mask *name* of *section* was made at an earlier registration
    (information for the reply and ``status``, never a refusal)."""
    from langslice.core.sections.masks import made_earlier as earlier

    entry = section.masks.get(name)
    return entry is not None and earlier(state, section, entry)
