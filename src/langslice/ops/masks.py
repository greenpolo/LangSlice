"""The ``set_mask`` verb: a section's named section-side masks.

A mask is a weight image on the section's fit grid (:mod:`langslice.core.section_masks`)
of one of two kinds, both the three-class EM sorting of the tissue: ``em`` (made
here, at the section's current registration) and ``fit`` (copied from the
deformation the section holds, which saved the sorting its fit made). A fit
takes a mask by its name (``fireants_syn``, ``elastix_affine`` and the
engines of tool plugins). Nothing else makes a mask: no threshold and no
drawn region; the damage marks are the atlas side and never mask the section.

:func:`set_mask` writes ``SliceState.masks`` (one undo step) and the weights
under the job folder (``sections/<stem>/masks/<name>.npz``, never deleted:
undo hides the state reference). :func:`mask_names` and :func:`read` are what
the fits and the pictures read through.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core import appearance, deformation
from langslice.core.atlas.images import resolve_image
from langslice.core.deformable.masks import SORTING_KEYS
from langslice.core.section_masks import NAME_PATTERN, RESERVED_NAMES
from langslice.ops.refusal import Refused, check_frozen, unknown_sections

if TYPE_CHECKING:
    from langslice.core.section_masks import MaskView
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: The kinds of mask.
SOURCES = ("em", "fit")
#: The name a fit gives for no section mask.
NO_MASK = "none"


@dataclass(frozen=True)
class MaskSet:
    """What :func:`set_mask` did."""

    id: str
    name: str
    deleted: bool = False
    #: The state entry now held (``kind``, ``channel``, ``atlas_image``, ``params``,
    #: ``left_out``...); None for a deleted mask.
    entry: dict[str, Any] | None = None
    record: SliceState | None = field(default=None, repr=False)


def _check_name(name: Any) -> str:
    text = str(name or "").strip()
    if text.lower() in RESERVED_NAMES or not NAME_PATTERN.match(text):
        raise Refused("BAD_ARGS", message=f"name is 1 to 40 letters, digits, '_' or '-' and "
                      f"not {NO_MASK!r} (that is the fits' word for no mask).")
    return text


def _check_sorting(sorting: Mapping[str, Any] | None) -> dict[str, float]:
    given = dict(sorting or {})
    unknown = sorted(set(given) - set(SORTING_KEYS))
    if unknown:
        raise Refused("BAD_ARGS", message=f"sorting takes {list(SORTING_KEYS)}; not {unknown}.")
    try:
        values = {key: float(value) for key, value in given.items()}
    except (TypeError, ValueError) as exc:
        raise Refused("BAD_ARGS", message="sorting values are numbers.") from exc
    for key, value in values.items():
        if not np.isfinite(value) or value < 0:
            raise Refused("BAD_ARGS", message=f"sorting {key} is a number from 0 up.")
    return values


def set_mask(
    job: Job, workspace: Workspace, section: Any, name: str, *, source: str = "em",
    channel: str = appearance.PREPROCESSED, atlas_image: str = "nissl",
    sorting: Mapping[str, Any] | None = None, delete: bool = False,
) -> MaskSet:
    """Make, replace or delete the mask *name* of one section. One undo step.

    ``em``: sort the section's pixels of *channel* against *atlas_image* at its
    CURRENT registration (the atlas placed through its transform and applied
    deformation), with the optional starting points *sorting*
    (:data:`SORTING_KEYS`); ``fit``: the sorting the section's applied
    deformation saved with its record, copied. The mask is stale once the
    section's placement, appearance, damage marks or deformation change
    (:func:`langslice.core.section_masks.read_mask`); making it again replaces it.
    *delete* removes the entry (the weights file stays; undo brings the entry
    back).

    Refused (nothing written): ``UNKNOWN_SLICE_IDS``, ``SECTION_BUSY``,
    ``BAD_ARGS`` (the name, the source, the starting points, an atlas image the
    atlas does not have), ``UNKNOWN_CHANNEL``, ``NO_POSITION``,
    ``INVALID_LINEAR_PLACEMENT``, ``UNKNOWN_MASK`` (delete of a name the
    section does not have), ``NO_DEFORMATION`` / ``NO_SORTING`` (source
    ``fit``), ``FIT_FAILED`` (the sort failed, the message says how).
    """
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    check_frozen(job, [record.id])
    label = _check_name(name)
    if delete:
        if label not in (record.masks or {}):
            raise Refused("UNKNOWN_MASK", message=f"{record.id} has no mask {label!r}.",
                          masks=sorted(record.masks or {}))
        before = job.snapshot()
        del record.masks[label]
        job.commit(before)
        return MaskSet(id=record.id, name=label, deleted=True, record=record)
    kind = str(source or "em").strip().lower()
    if kind not in SOURCES:
        raise Refused("BAD_ARGS", message=f"source is one of {list(SOURCES)}.")
    params = _check_sorting(sorting)
    before = job.snapshot()  # the mask entry is set by the write below
    if kind == "fit":
        entry = _from_fit(job, workspace, record, label)
    else:
        entry = _from_em(job, workspace, record, label, channel=str(channel or "").strip(),
                         atlas_image=str(atlas_image or "nissl"), params=params)
    job.commit(before)
    return MaskSet(id=record.id, name=label, entry=entry, record=record)


def _from_fit(job: Job, workspace: Workspace, record: SliceState, name: str) -> dict[str, Any]:
    from langslice.core.section_masks import fit_weights, registration_digest, write_mask

    if job.deformations.current(job.state, record) is None:
        raise Refused("NO_DEFORMATION", message=f"{record.id} holds no deformation, so there "
                      "is no fit sorting to copy.")
    weights = fit_weights(job.deformations, job.state, record)
    if weights is None:
        raise Refused("NO_SORTING", message=f"{record.id}'s deformation saved no sorting (its "
                      "engine estimated none).")
    held = job.deformations.current(job.state, record)
    atlas = held.settings.atlas_image if held is not None and held.settings is not None else ""
    footprint = np.asarray(held.footprint, dtype=bool) if held is not None else None
    inside = weights < 0.5
    if footprint is not None and footprint.shape == inside.shape and footprint.any():
        inside = inside[footprint]
    meta = {"kind": "fit", "channel": "", "atlas_image": str(atlas), "params": {},
            "inputs": registration_digest(job.state, record),
            "left_out": float(inside.mean()) if inside.size else 0.0}
    return _write(job, record, name, weights, meta, write_mask)


def _from_em(
    job: Job, workspace: Workspace, record: SliceState, name: str, *, channel: str,
    atlas_image: str, params: dict[str, float],
) -> dict[str, Any]:
    from langslice.core.section_masks import em_sorting, registration_digest, write_mask

    if record.position_mm is None:
        raise Refused("NO_POSITION", message=f"{record.id} has no position yet.")
    known = appearance.channel_names(job.state, workspace, record)
    if channel not in known:
        raise Refused("UNKNOWN_CHANNEL", channel=channel, channels=list(known))
    try:
        resolved = resolve_image(workspace.atlas, atlas_image)
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
    except (ValueError, OSError) as exc:
        raise Refused("INVALID_LINEAR_PLACEMENT", message=f"{record.id}: {exc}") from exc
    try:
        image, _identity = deformation.stain_image(workspace, job.state, grid, channel)
        pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        with deformation.engine_lock():
            weights, facts = em_sorting(workspace, job.state, record, job.deformations,
                                        channel_image=pixels, atlas_image=resolved,
                                        params=params)
    except Refused:
        raise
    except Exception as exc:  # noqa: BLE001 - the sort's failure is the caller's refusal
        raise Refused("FIT_FAILED", message=f"{record.id}: {type(exc).__name__}: {exc}") from exc
    meta = {"kind": "em", "channel": channel, "atlas_image": resolved,
            "params": facts.get("params", {}),
            "inputs": registration_digest(job.state, record),
            "left_out": float(facts.get("left_out", 0.0))}
    return _write(job, record, name, weights, meta, write_mask)


def _write(job: Job, record: SliceState, name: str, weights: np.ndarray, meta: dict[str, Any],
           write_mask: Any) -> dict[str, Any]:
    try:
        return write_mask(job.layout, record, name, weights, meta)  # type: ignore[no-any-return]
    except ValueError as exc:
        raise Refused("BAD_ARGS", message=str(exc)) from exc


def read(job: Job, record: SliceState, name: str) -> MaskView:
    """The mask *name* of *record* with its weights (``UNKNOWN_MASK`` when it
    has none by that name); :attr:`~langslice.core.section_masks.MaskView.stale`
    says whether the section changed since."""
    from langslice.core.section_masks import read_mask

    try:
        return read_mask(job.layout, job.state, record, name)
    except KeyError as exc:
        raise Refused("UNKNOWN_MASK", message=f"{record.id} has no mask {name!r}; make it "
                      "with set_mask.", masks=sorted(record.masks or {})) from exc
    except ValueError as exc:  # the file cannot be read
        raise Refused("FIT_FAILED", message=str(exc)) from exc


def section_weights(job: Job, record: SliceState, name: str) -> np.ndarray | None:
    """The weights the fit *name* names reads on *record*, a float array on its
    fit grid; None for ``"none"``. Refused: ``UNKNOWN_MASK``, ``STALE_MASK``
    (the section changed since the mask was made)."""
    from langslice.core.section_masks import section_weights as weights_of

    label = str(name or NO_MASK).strip()
    if label == NO_MASK:
        return None
    view = read(job, record, label)
    if view.stale:
        raise Refused("STALE_MASK", message=f"{record.id}'s mask {label!r} was made before the "
                      "section changed; make it again with set_mask.")
    return weights_of(view)
