"""The QuickNII and VisuAlign export of a job's stack.

:func:`write_quint` writes, under ``exports/``, ``quicknii.json`` (every
placed section's linear anchoring, made perpendicular), ``visualign.json``
(the same, with VisuAlign markers of each applied deformation and of the
shear the perpendicular anchoring drops) and the section images they name,
one PNG per section (:mod:`langslice.core.exports.quint_images`), from the
stack as it stands. The documents are built by
:func:`langslice.core.exports.quint.job_export`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.exports.quint import job_export
from langslice.core.exports.quint_images import (
    QUICKNII_FILE,
    VISUALIGN_FILE,
    copy_filename,
    quint_copies,
    section_numbers,
    write_copies,
)
from langslice.core.maps.section_maps import placed_sections, residual_markers, section_frame
from langslice.core.sections.images import PREPROCESSED, image_plane
from langslice.core.shared_files import write_json_atomic
from langslice.tools.bookkeeping.applied import applied_warp

if TYPE_CHECKING:
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

QUINT_FILES = ((QUICKNII_FILE, "quicknii", False), (VISUALIGN_FILE, "visualign", True))


def stack_numbers(state: Stack) -> dict[str, int]:
    """Number every section of the stack for the QUINT files.

    Parameters
    ----------
    state
        The stack, every section of it in order.

    Returns
    -------
    dict
        Section id -> its number (:func:`section_numbers`).
    """
    ids = [record.id for record in state.in_order()]
    return dict(zip(ids, section_numbers(ids), strict=True))


def quint_paths(state: Stack, exports_dir: Path) -> list[tuple[str, str]]:
    """List the files :func:`write_quint` would write.

    Parameters
    ----------
    state
        The stack as it stands.
    exports_dir
        The job's exports folder.

    Returns
    -------
    list of tuple
        ``(path, kind)``: the image copy of each placed section, then the
        two documents.
    """
    numbers = stack_numbers(state)
    paths = [
        (str(exports_dir / copy_filename(record.id, numbers[record.id])), "quint_image")
        for record in placed_sections(state)
    ]
    paths += [(str(exports_dir / name), kind) for name, kind, _markers in QUINT_FILES]
    return paths


def write_quint(job: Job, workspace: Workspace, facts: dict[str, Any]) -> list[tuple[str, str]]:
    """Write the stack's QUINT files and image copies; call it under the
    job lock. A section whose frame, deformation or image cannot be read is
    left out and logged. Nothing is written when no section has a placement.

    Parameters
    ----------
    job
        The job, whose state is read as it stands.
    workspace
        The job's atlas and section files.
    facts
        The atlas record of ``registration.json``
        (:func:`langslice.core.pictures.layers.atlas_facts`).

    Returns
    -------
    list of tuple
        ``(path, kind)`` of each file written.
    """
    state = job.state
    exports_dir = job.layout.exports_dir
    files: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    copies: list[tuple[str, np.ndarray]] = []
    numbers = stack_numbers(state)
    for record in placed_sections(state):
        try:
            frame = section_frame(state, workspace, record)
            warp, _stored = applied_warp(job, state, record)
            markers = [] if warp is None else residual_markers(frame, warp)
        except (OSError, ValueError):
            logger.warning("Not exported: %s", record.id, exc_info=True)
            continue
        number = numbers[record.id]
        copy_name = copy_filename(record.id, number)
        rows.append(
            {
                "filename": copy_name,
                "nr": number,
                "width": frame.file_size[0],
                "height": frame.file_size[1],
                "pixel_to_atlas_um": frame.pixel_to_atlas_um(),
                "markers": markers,
            }
        )
        try:
            copies.append((copy_name, image_plane(workspace, record, PREPROCESSED)))
        except (OSError, ValueError):
            logger.warning("No image copy of %s", record.id, exc_info=True)
    if not rows:
        return files
    previous = quint_copies(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    try:
        written = write_copies(exports_dir, copies, previous)
        files += [(str(path), "quint_image") for path in written]
    except OSError:
        logger.warning("Could not write the QUINT image copies", exc_info=True)
    for name, kind, with_markers in QUINT_FILES:
        path = exports_dir / name
        try:
            document = job_export(rows, facts, write_markers=with_markers)
            write_json_atomic(str(path), document)
        except (OSError, ValueError, KeyError):
            logger.warning("Could not write %s", name, exc_info=True)
            continue
        files.append((str(path), kind))
    return files
