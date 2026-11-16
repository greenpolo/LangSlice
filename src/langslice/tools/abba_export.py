"""Writing a job's registrations as an ABBA project file.

:func:`export_abba` (the library's ``job.export_abba``, the window's Export
menu; not a model-facing verb, not written at submit) writes, from the stack
as it stands, ``exports/<image folder>.abba``: ABBA 0.25's state file
(:mod:`langslice.job.abba_project`), which ABBA's File > Load State opens on
its Allen CCFv3 atlas with the job's own image files. Each placed section is
a slice at its position and the stack's cutting angles, with LangSlice's
in-plane placement as ABBA's affine registration and an applied deformation
as a BigWarp landmark registration on top (``core/maps/abba.py``'s
``world_landmarks``, in worker processes: each takes 10 to 80 s), so the
project reopens and edits in ABBA without LangSlice. Beside it,
``exports/abba_export.json`` records what was written, the sections left
out and why, and each warp's distance from the deformation.

ABBA slices a whole stack at one pair of angles, so a stack whose sections'
angles differ is refused (``BAD_ARGS``); so is an atlas other than the Allen
mouse CCFv3 or a plane other than coronal. A section without a placement,
or that cannot be carried, is left out with its reason. Nothing in the
state changes; a job that persists nothing writes nothing. A script that
calls it runs its body under ``if __name__ == "__main__":``, as for any
call that starts worker processes.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import time
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from langslice.core.maps.abba import WarpLandmarks, world_landmarks
from langslice.core.maps.abba_world import (
    LOCATION_TO_ATLAS_Z_MM,
    AbbaPlacement,
    abba_placement,
    check_atlas,
    rotations,
)
from langslice.core.maps.section_maps import placement_problem, section_frame
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.worker_threads import started_with_threads, thread_share
from langslice.job.abba_project import (
    AbbaSlice,
    AbbaWarp,
    image_channels,
    project_bytes,
)
from langslice.job.state_file import write_json_atomic
from langslice.tools.exports import applied_record

if TYPE_CHECKING:
    from langslice.core.fits.deformable.record import DeformableRecord
    from langslice.core.maps.section_maps import SectionFrame
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

#: The record of what the export holds, beside the project file.
EXPORT_RECORD = "abba_export.json"
#: Processes placing the deformations' landmark pairs side by side, each
#: with its share of the machine's threads.
WARP_PROCESSES = 6
#: Micrometres per millimetre.
UM_PER_MM = 1000.0
#: What the record tells a person about opening the file.
OPENING_NOTE = (
    "Open in ABBA 0.25 with File > Load State on the Allen Brain Adult Mouse "
    "Brain CCF 2017 (v3p1) atlas, started coronal. Each section opens from "
    "its own image file; a warp is a BigWarp registration, editable with "
    "Edit last registration."
)


@dataclass(frozen=True)
class AbbaExported:
    """What :func:`export_abba` wrote (on a dry run, would write).

    Attributes
    ----------
    path
        The project file ABBA opens.
    sections
        The sections written, with a warp or without.
    warped
        The sections written with a warp step.
    skipped
        ``{"id", "reason"}`` per section left out.
    warp_error_um
        Per warped section, the warp step's distance from LangSlice's
        deformation in micrometres (``median``, ``p99``, ``max``).
    written
        Whether the file was written.
    seconds
        How long the export took.
    """

    path: str
    sections: list[str] = field(default_factory=list)
    warped: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    warp_error_um: dict[str, dict[str, float]] = field(default_factory=dict)
    written: bool = True
    seconds: float = 0.0


@dataclass(frozen=True)
class _Section:
    """One section on its way into the file."""

    section: Section
    frame: SectionFrame
    placement: AbbaPlacement
    item: AbbaSlice
    record: DeformableRecord | None


def project_name(state: Stack) -> str:
    """Name the project file after the image folder.

    Parameters
    ----------
    state
        The stack, whose image folder names the file.

    Returns
    -------
    str
        ``<image folder>.abba``, or ``langslice.abba`` without a folder.
    """
    stem = Path(str(state.image_folder or "")).name or "langslice"
    return f"{stem}.abba"


def warp_pairs(
    frame: SectionFrame, record: DeformableRecord, placement: AbbaPlacement
) -> WarpLandmarks:
    """Place one section's warp landmarks (run in a worker process).

    Parameters
    ----------
    frame
        The section's frame.
    record
        Its applied deformation.
    placement
        Its placement in ABBA's world.

    Returns
    -------
    WarpLandmarks
        The pairs and their measured accuracy.
    """
    return world_landmarks(
        frame, record, placement.file_to_world, placement.native_to_world
    )


def _warp_step(
    section: Section, landmarks: WarpLandmarks
) -> tuple[AbbaWarp, dict[str, float]]:
    """Make the landmarks a warp step, with its error in micrometres."""
    atlas = landmarks.accuracy.atlas
    error = {
        "median": round(atlas.median_mm * UM_PER_MM, 2),
        "p99": round(atlas.p99_mm * UM_PER_MM, 2),
        "max": round(atlas.max_mm * UM_PER_MM, 2),
    }
    reference = section.deformation
    record_path = "" if reference is None else str(reference.record)
    parameters = {
        "langslice_record": record_path,
        "langslice_points": str(len(landmarks.source_mm)),
        "langslice_p99_error_mm": f"{atlas.p99_mm:.6f}",
        "langslice_max_error_mm": f"{atlas.max_mm:.6f}",
    }
    step = AbbaWarp(landmarks.source_mm, landmarks.target_mm, parameters)
    return step, error


def _check_stack(workspace: Workspace, state: Stack) -> None:
    """Refuse a stack ABBA cannot open as one project."""
    try:
        # the orientation is checked per section, from its frame
        check_atlas(workspace.settings.atlas, "asr", state.plane)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, str(exc)) from exc
    if state.mixed_angles:
        raise Refused(
            RefusalCode.BAD_ARGS,
            "ABBA slices a stack at one pair of cutting angles; set every "
            "section to the same angles first",
        )


def _frames(
    workspace: Workspace, state: Stack
) -> tuple[list[tuple[Section, SectionFrame]], list[dict[str, str]]]:
    """Return each section with a placement and its frame, and the rest."""
    skipped: list[dict[str, str]] = []
    framed: list[tuple[Section, SectionFrame]] = []
    for section in state.in_order():
        problem = placement_problem(state, section)
        if problem is not None:
            skipped.append({"id": section.id, "reason": problem})
            continue
        try:
            framed.append((section, section_frame(state, workspace, section)))
        except (OSError, ValueError) as exc:
            skipped.append({"id": section.id, "reason": str(exc)})
    return framed, skipped


def _prepare(
    job: Job,
    workspace: Workspace,
    state: Stack,
    framed: tuple[Section, SectionFrame],
    angles: tuple[float, float],
    warps: bool,
) -> _Section:
    """Carry one section's placement and file into ABBA's terms.

    Raises ``OSError`` or ``ValueError`` saying why it cannot be.
    """
    section, frame = framed
    placement = abba_placement(frame, *angles)
    path = Path(workspace.image_path(section.id))
    channels, rgb = image_channels(path)
    record = applied_record(job, state, section) if warps else None
    size = (int(frame.file_size[0]), int(frame.file_size[1]))
    item = AbbaSlice(
        path=path.resolve(),
        size=size,
        channels=channels,
        rgb=rgb,
        voxel_mm=placement.voxel_mm,
        location_mm=placement.location_mm,
        atlas_z_mm=placement.location_mm - LOCATION_TO_ATLAS_Z_MM,
        affine_mm=placement.affine_mm,
    )
    return _Section(section, frame, placement, item, record)


@dataclass(frozen=True)
class _Warped:
    """The slices with their warp steps, those left out, and the errors."""

    slices: list[AbbaSlice]
    skipped: list[dict[str, str]]
    errors: dict[str, dict[str, float]]


def _with_warps(prepared: list[_Section]) -> _Warped:
    """Add each deformation's warp step, placed in worker processes."""
    skipped: list[dict[str, str]] = []
    errors: dict[str, dict[str, float]] = {}
    warped = [
        (entry, entry.record) for entry in prepared if entry.record is not None
    ]
    futures: dict[str, Future[WarpLandmarks]] = {}
    with ExitStack() as stack:
        if warped:
            share = thread_share(WARP_PROCESSES)
            stack.enter_context(started_with_threads(share))
            context = multiprocessing.get_context("spawn")
            pool = stack.enter_context(
                ProcessPoolExecutor(WARP_PROCESSES, mp_context=context)
            )
            for entry, record in warped:
                futures[entry.section.id] = pool.submit(
                    warp_pairs, entry.frame, record, entry.placement
                )
        slices: list[AbbaSlice] = []
        for entry in prepared:
            future = futures.get(entry.section.id)
            if future is None:
                slices.append(entry.item)
                continue
            try:
                step, error = _warp_step(entry.section, future.result())
            except (OSError, ValueError) as exc:
                logger.warning(
                    "No warp for ABBA: %s", entry.section.id, exc_info=True
                )
                skipped.append({"id": entry.section.id, "reason": str(exc)})
                continue
            errors[entry.section.id] = error
            slices.append(replace(entry.item, warp=step))
    return _Warped(slices, skipped, errors)


def export_abba(
    job: Job, workspace: Workspace, *, warps: bool = True
) -> AbbaExported:
    """Write the job's stack as an ABBA project file (see the module text).

    Parameters
    ----------
    job
        The job whose stack is written, as it stands.
    workspace
        The job's atlas and section files.
    warps
        Whether applied deformations are written as warp steps (False: the
        linear placements only).

    Returns
    -------
    AbbaExported
        The file, the sections written and left out, and each warp's error.

    Raises
    ------
    Refused
        ``BAD_ARGS`` when the sections' cutting angles differ, or for an
        atlas or plane ABBA cannot open; ``NOTHING_EXPORTED`` when no
        section could be written.
    """
    started = time.perf_counter()
    with job.writing():
        state = copy.deepcopy(job.state)
    _check_stack(workspace, state)
    framed, skipped = _frames(workspace, state)
    if not framed:
        raise Refused(
            RefusalCode.NOTHING_EXPORTED,
            "no section has a placement to write",
            skipped=skipped,
        )
    first = framed[0][1]
    angles = rotations(first.pitch_deg, first.yaw_deg)
    prepared: list[_Section] = []
    for entry in framed:
        try:
            prepared.append(
                _prepare(job, workspace, state, entry, angles, warps)
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Not exported to ABBA: %s", entry[0].id, exc_info=True
            )
            skipped.append({"id": entry[0].id, "reason": str(exc)})
    warped = _with_warps(prepared)
    skipped += warped.skipped
    errors, slices = warped.errors, warped.slices
    if not slices:
        raise Refused(
            RefusalCode.NOTHING_EXPORTED,
            "no section could be written for ABBA",
            skipped=skipped,
        )
    target = job.layout.exports_dir / project_name(state)
    names = {item.path.name for item in slices}
    written = [e.section.id for e in prepared if e.item.path.name in names]
    result = AbbaExported(
        path=str(target),
        sections=written,
        warped=sorted(errors),
        skipped=skipped,
        warp_error_um=errors,
        written=job.persist,
        seconds=time.perf_counter() - started,
    )
    if job.persist:
        _write(target, project_bytes(slices, *angles), result, angles)
    return result


def _write(
    target: Path,
    data: bytes,
    result: AbbaExported,
    angles: tuple[float, float],
) -> None:
    """Write the project file atomically, and its record beside it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".abba.part")
    partial.write_bytes(data)
    partial.replace(target)
    write_json_atomic(
        str(target.parent / EXPORT_RECORD),
        {
            "project": target.name,
            "rotation_x_rad": angles[0],
            "rotation_y_rad": angles[1],
            "sections": result.sections,
            "warped": result.warped,
            "skipped": result.skipped,
            "warp_error_um": result.warp_error_um,
            "note": OPENING_NOTE,
        },
    )
