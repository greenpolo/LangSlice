"""A deformation made elsewhere, written as a section's deformation.

A tool plugin's engine or a script makes the deformation;
:func:`warp_from_map` builds its record from the full map (the residual over
the section's linear placement, a fixed-point inverse, the usual
diagnostics) without writing it, and :func:`apply_warps` writes each as its
section's deformation: one undo step, under the job's write lock, as an
applied fit is written (:func:`langslice.tools.deformable.fit_deformable`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.fits import deformation
from langslice.core.fits.frame import digest
from langslice.core.refusal import RefusalCode, Refused
from langslice.tools.refusal import frozen_row, unknown_sections
from langslice.tools.stale import stale_row

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


@dataclass(frozen=True)
class MappedWarp:
    """A deformation made elsewhere, as a record not written yet
    (:func:`warp_from_map`): the section, the record and its key, the linear
    placement it sits on (``linear_key``), the step the section's reference
    will list, and the record's numbers
    (:func:`langslice.core.fits.deformation.summary`)."""

    id: str
    key: str
    linear_key: str
    record: deformation.DeformableRecord
    step: dict[str, Any]
    numbers: dict[str, Any]
    #: The key of the deformation it was built on (``start`` ``current``), which
    #: the section must still hold when it is applied; None: built on the
    #: linear placement.
    parent_key: str | None = None


def warp_from_map(
    job: Job,
    workspace: Workspace,
    section: object,
    native_xy: Any,
    *,
    method: str,
    step: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    weights: Any = None,
    start: str = "linear",
    settings: Mapping[str, Any] | None = None,
    engine_settings: Mapping[str, Any] | None = None,
    engine_version: str = "",
) -> MappedWarp:
    """The section's deformation from a full map made elsewhere, not written.

    *native_xy* is a float ``(H, W, 2)`` array: for each pixel centre of the
    section's fit grid (:func:`langslice.core.fits.deformation.fit_grid`'s
    image, ``(W, H)`` its size), the native atlas-plane pixel ``(x, y)`` it
    corresponds to, linear placement and deformation together. It becomes a
    record as every applied deformation is stored: the residual over the
    section's current linear placement
    (:func:`langslice.core.fits.deformation.record_from_map`). *start*
    ``linear``: built on nothing the section held before; ``current``: the map
    was made on top of the deformation the section holds, so that record is its
    parent and the section's steps are kept before its own, as a native fit
    chains (the map is still the whole registration). *method* names it (the
    record's engine and the step's ``method``); *settings* (the call's
    settings, by the tool's names) are the step's ``settings``;
    *engine_settings* (what the engine ran at, in its own units) and
    *engine_version* are the record's ``engine.native_parameters`` and
    ``engine.version``, as a FireANTs record holds them; *step* adds fields to
    the step, *provenance* to the record's. *weights* is the weighting the
    engine estimated for itself, a float ``(H, W)`` array on the fit grid in
    [0, 1] (stored as float16 with the record; ``make_mask`` source ``fit``
    copies it); None when it estimated none.

    Refused (nothing built): ``UNKNOWN_SLICE_IDS``; ``INVALID_LINEAR_PLACEMENT``
    (no position or no usable transform, as a fit refuses); ``BAD_FIELD`` (not
    a float array of the grid's shape with two channels, or not finite;
    *weights* not a finite ``(H, W)`` array); ``BAD_ARGS`` (a *start* that is
    neither, ``current`` on a section that holds no deformation);
    ``DEFORMATION_UNREADABLE``.
    """
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    if start not in tuple(deformation.FitStart):
        raise Refused(
            RefusalCode.BAD_ARGS, message="start is 'linear' or 'current'."
        )
    try:
        grid = deformation.fit_grid(job.state, workspace, record)
    except (ValueError, OSError) as exc:
        raise Refused(
            RefusalCode.INVALID_LINEAR_PLACEMENT, message=f"{record.id}: {exc}"
        ) from exc
    previous: deformation.DeformableRecord | None = None
    if start == deformation.FitStart.CURRENT:
        try:
            previous = job.deformations.current(job.state, record)
        except ValueError as exc:
            raise Refused(
                RefusalCode.DEFORMATION_UNREADABLE,
                message=f"{record.id}: {exc}",
            ) from exc
        if previous is None:
            raise Refused(
                RefusalCode.BAD_ARGS,
                message=f"{record.id} holds no deformation to build on: "
                "start is 'linear'.",
            )
    parent_key = None
    if previous is not None:
        parent_key = record.deformation_key
    array = _checked_map(record.id, native_xy, grid.image.size)
    stored_weights = _stored_weights(record.id, weights, grid.image.size)
    linear = deformation.linear_key(job.state, record)
    array = np.ascontiguousarray(array, dtype=np.float64)
    key = digest(
        {
            "method": method,
            "linear": linear,
            "map": hashlib.sha256(array.tobytes()).hexdigest(),
            **({"parent": parent_key} if parent_key else {}),
            **(
                {
                    "weights": hashlib.sha256(
                        stored_weights.tobytes()
                    ).hexdigest()
                }
                if stored_weights is not None
                else {}
            ),
        }
    )
    outcome = deformation.record_from_map(
        workspace,
        grid,
        array,
        method=method,
        provenance={
            "tool": method,
            "linear_key": linear,
            "start": start,
            **({"settings": dict(settings)} if settings else {}),
            **(provenance or {}),
        },
        native_parameters=dict(engine_settings or {}),
        engine_version=str(engine_version),
    )
    outcome.weights = stored_weights
    if previous is not None:
        outcome.parent = previous
        outcome.step = previous.step + 1
    return MappedWarp(
        id=record.id,
        key=key,
        linear_key=linear,
        record=outcome,
        step={
            "method": method,
            "start": start,
            **(step or {}),
            **({"settings": dict(settings)} if settings else {}),
        },
        numbers=deformation.summary(outcome, previous),
        parent_key=parent_key,
    )


def _checked_map(
    section_id: str, native_xy: object, size: tuple[int, int]
) -> np.ndarray:
    """Return the map as an array.

    Refused (``BAD_FIELD``) unless it is a finite float ``(H, W, 2)`` array
    on the fit grid of ``size`` ``(W, H)``.
    """
    width, height = size
    array = np.asarray(native_xy)
    if not np.issubdtype(array.dtype, np.floating):
        raise Refused(
            RefusalCode.BAD_FIELD,
            message=f"{section_id}: the map is a float array, not "
            f"{array.dtype}.",
        )
    if array.shape != (height, width, 2):
        raise Refused(
            RefusalCode.BAD_FIELD,
            message=f"{section_id}: the map is (H, W, 2) = "
            f"({height}, {width}, 2) on the fit grid; got {array.shape}.",
            expected_shape=[height, width, 2],
        )
    if not np.isfinite(array).all():
        raise Refused(
            RefusalCode.BAD_FIELD,
            message=f"{section_id}: the map has values that are not finite.",
        )
    return array


def _stored_weights(
    section_id: str, weights: object, size: tuple[int, int]
) -> np.ndarray | None:
    """Return the engine's weights as stored (float16 in [0, 1]).

    Refused (``BAD_FIELD``) unless a finite ``(H, W)`` array; None without
    any.
    """
    if weights is None:
        return None
    width, height = size
    given = np.asarray(weights)
    if (
        given.shape != (height, width)
        or not np.isfinite(given.astype(np.float64)).all()
    ):
        raise Refused(
            RefusalCode.BAD_FIELD,
            message=f"{section_id}: weights are a finite (H, W) = "
            f"({height}, {width}) array on the fit grid; got {given.shape}.",
            expected_shape=[height, width],
        )
    return np.clip(given.astype(np.float32), 0.0, 1.0).astype(np.float16)


@dataclass(frozen=True)
class WarpsApplied:
    """What :func:`apply_warps` did: one row per warp (``status`` ``ok`` with
    the record's numbers, or ``error`` with the code) and the sections written."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    written: list[str] = field(default_factory=list)


def apply_warps(job: Job, warps: Sequence[MappedWarp]) -> WarpsApplied:
    """Write each warp as its section's deformation: ONE undo step.

    Under the job's write lock, per warp: a section whose deformation is
    frozen is its row ``FROZEN``; one whose linear placement is no longer the
    one the warp was built on, or (a warp built on the section's deformation,
    ``parent_key``) that no longer holds that deformation, is ``STALE_INPUT``;
    otherwise the record is saved in the section's folder (``job.deformations``)
    and the section's ``deformation`` names it, as an applied fit's does: a
    warp built on the linear placement replaces what the section held, one
    built on its deformation keeps that deformation's steps before its own
    (``undo`` brings the earlier one back). A later change to the placement
    clears it as it clears any deformation.
    """
    store = job.deformations
    rows: list[dict[str, Any]] = []
    written: list[str] = []
    with job.writing():
        before = job.snapshot()
        for warp in warps:
            record = job.state.by_id(warp.id)
            if record is None:
                rows.append(
                    unknown_sections(job.state, [warp.id]).payload(
                        section_id=warp.id
                    )
                )
                continue
            frozen = frozen_row(job, record.id, "deformation")
            if frozen is not None:
                rows.append(frozen)
                continue
            held = record.deformation
            if deformation.linear_key(
                job.state, record
            ) != warp.linear_key or (
                warp.parent_key is not None
                and record.deformation_key != warp.parent_key
            ):
                rows.append(stale_row(record.id))
                continue
            store.put(warp.key, warp.record)
            try:
                folder = store.save(record.id, warp.key, warp.record)
            except (OSError, ValueError) as exc:
                rows.append(
                    Refused(
                        RefusalCode.RECORD_WRITE_FAILED, message=str(exc)
                    ).payload(section_id=record.id)
                )
                continue
            record.deformation = deformation.reference(
                folder=job.layout.relative(folder),
                key=warp.key,
                linear=warp.linear_key,
                record=warp.record,
                step=dict(warp.step),
                previous=held if warp.parent_key is not None else None,
                numbers=warp.numbers,
            )
            written.append(record.id)
            rows.append({"id": record.id, "status": "ok", **warp.numbers})
        if written:
            job.commit(before)
    return WarpsApplied(rows=rows, written=written)
