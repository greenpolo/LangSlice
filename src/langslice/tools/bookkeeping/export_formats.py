"""The formats a job's registration is written in for other programs.

:data:`EXPORT_FORMATS` is the one list: each :class:`ExportFormat` has a
name, a label for people, the suffix of the file it writes and the function
that writes it. The window's Export menu calls :func:`write_export`; the
formats are written by the modules named below, and nothing else knows
their files:

- ``abba``, ``abba_linear``: an ABBA project, with or without the warps
  (:mod:`langslice.tools.bookkeeping.abba_export`);
- ``quicknii``, ``visualign``: the QUINT files and the section images they
  name, with the rest of what ``submit`` writes
  (:mod:`langslice.tools.bookkeeping.quint_export`,
  :func:`langslice.tools.bookkeeping.exports.export_stack`);
- ``dmc_brainmap``: a DMC-BrainMap animal folder
  (:mod:`langslice.tools.bookkeeping.dmc_export`);
- the per-section atlas-space maps are the ``export_maps`` verb's
  (:func:`langslice.tools.bookkeeping.exports.export_maps`), which writes
  the QUINT files too.
"""

from __future__ import annotations

import functools
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langslice.core.maps.abba import MAX_POINTS
from langslice.core.refusal import RefusalCode, Refused
from langslice.tools.bookkeeping.abba_export import AbbaExported, export_abba
from langslice.tools.bookkeeping.dmc_export import export_dmc_brainmap
from langslice.tools.bookkeeping.exports import export_stack

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


@dataclass(frozen=True)
class Written:
    """What one export wrote: the file to open, in words, and the sections left out."""

    path: str
    message: str
    skipped: int


Writer = Callable[["ExportFormat", "Job", "Workspace", int], Written]


@dataclass(frozen=True)
class ExportFormat:
    """One format of the Export menu.

    Attributes
    ----------
    name
        The format's name, as the page and :func:`write_export` give it.
    label
        What people read.
    about
        One sentence for people on what the file holds and where it opens.
    suffix
        The suffix of the file written.
    write
        Writes the format (given itself, a job, its workspace and the most
        landmark pairs per warp of a format that has warps).
    slow
        What people are told while it is written, for a format that takes
        minutes; blank for one that does not.
    """

    name: str
    label: str
    about: str
    suffix: str
    write: Writer
    slow: str = ""


def _abba(
    warps: bool, _export: ExportFormat, job: Job, workspace: Workspace, landmarks: int
) -> Written:
    done: AbbaExported = export_abba(job, workspace, warps=warps, landmarks=landmarks)
    message = f"{len(done.sections)} sections"
    if done.warp_error_um:
        medians = [row["median"] for row in done.warp_error_um.values()]
        typical = statistics.median(medians)
        message += (
            f", {len(done.warped)} with their warp (at most {landmarks}"
            f" landmarks, {typical:.0f} µm from LangSlice's at the median)"
        )
    return Written(done.path, message, len(done.skipped))


def _quint(export: ExportFormat, job: Job, workspace: Workspace, _landmarks: int) -> Written:
    files = [Path(path) for path, _kind in export_stack(job, workspace).files]
    named = [path for path in files if path.name == export.name + export.suffix]
    if not named:
        raise Refused(
            RefusalCode.PRECONDITION, "no section has a placement to write: place a section first"
        )
    count = sum(1 for path in files if path.suffix == ".png")
    return Written(str(named[0]), f"{count} sections", 0)


def _dmc(_export: ExportFormat, job: Job, workspace: Workspace, _landmarks: int) -> Written:
    done = export_dmc_brainmap(job, workspace)
    return Written(done.folder, f"{len(done.registered)} sections", len(done.skipped))


EXPORT_FORMATS: dict[str, ExportFormat] = {
    export.name: export
    for export in (
        ExportFormat(
            "abba",
            "ABBA project (.abba)",
            "Open in ABBA with File > Load State: each section with its affine and its warp.",
            ".abba",
            functools.partial(_abba, True),
            slow="Each warp becomes landmark pairs; this can take a few minutes.",
        ),
        ExportFormat(
            "abba_linear",
            "ABBA project, linear only (.abba)",
            "Open in ABBA with File > Load State: each section with its affine, no warp.",
            ".abba",
            functools.partial(_abba, False),
        ),
        ExportFormat(
            "quicknii",
            "QuickNII (.json)",
            "The linear placement of each section, with its image.",
            ".json",
            _quint,
        ),
        ExportFormat(
            "visualign",
            "VisuAlign (.json)",
            "The placement and each warp as VisuAlign markers.",
            ".json",
            _quint,
        ),
        ExportFormat(
            "dmc_brainmap",
            "DMC-BrainMap (folder)",
            "An animal folder DMC-BrainMap reads, each section with its registration.",
            "",
            _dmc,
        ),
    )
}


def write_export(
    name: str, job: Job, workspace: Workspace, *, landmarks: int = MAX_POINTS
) -> Written:
    """Write *job*'s registration in the format *name*.

    Parameters
    ----------
    name
        A key of :data:`EXPORT_FORMATS`.
    job, workspace
        The job and its atlas and section files.
    landmarks
        The most landmark pairs per warp of an ABBA project.

    Returns
    -------
    Written
        The file, what it holds, and how many sections were left out.

    Raises
    ------
    Refused
        ``BAD_ARGUMENT`` for a name that is no format; the format's own
        refusals (``PRECONDITION`` when nothing has a placement, and so on).
    """
    export = EXPORT_FORMATS.get(name)
    if export is None:
        raise Refused(RefusalCode.BAD_ARGUMENT, message=f"LangSlice cannot export {name!r}")
    return export.write(export, job, workspace, landmarks)
