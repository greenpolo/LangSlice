"""Writing a job's linear registrations as a DMC-BrainMap animal folder.

:func:`export_dmc_brainmap` (the library's ``job.export_dmc_brainmap``; not
a verb, not written at submit) writes, from the stack as it stands, under
``exports/dmc_brainmap/<animal>/`` (``<animal>`` the image folder's name,
which DMC-BrainMap reads as the animal id):

- ``params.json``: the atlas and its section frame as DMC's params widget
  writes them;
- ``sharpy_track/<channel>/<name>_downsampled.tif``: each section image,
  8-bit gray, resized to the atlas plane (``<name>`` the section folder's
  name);
- ``sharpy_track/<channel>/registration.json``: each placed section's plane
  and point pairs (:mod:`langslice.job.dmc_brainmap`);
- ``langslice_export.json``: what was written, the sections left
  unregistered and why, and the sections whose deformation was left out.

DMC-BrainMap holds a linear registration only, so a section's deformation
is never written: it is listed under ``warps_left_out``. Nothing in the
state changes; a job that persists nothing writes nothing.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
from PIL import Image

import langslice
from langslice.core.maps.section_maps import placement_problem, section_frame
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections.images import image_plane
from langslice.core.space import atlas_space_context
from langslice.job.dmc_brainmap import (
    DMC_CHANNELS,
    DmcFrame,
    DmcPlacement,
    dmc_frame,
    dmc_placement,
    natural_key,
)
from langslice.job.layout import section_dirname
from langslice.job.public_files import applied_deformation
from langslice.job.state_file import write_json_atomic

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

logger = logging.getLogger(__name__)

#: The export's folder under the job's ``exports/``.
DMC_FOLDER = "dmc_brainmap"
#: The record of what the export holds, beside ``params.json``.
EXPORT_RECORD = "langslice_export.json"
#: DMC's suffix for its resized section images.
IMAGE_SUFFIX = "_downsampled.tif"
#: The record's sentence on deformations, which DMC-BrainMap cannot hold.
LINEAR_ONLY_NOTE = (
    "DMC-BrainMap holds a linear registration only. Each section is "
    "written with its linear placement (position, cutting angles, in-plane "
    "affine); a deformation (a warp) applied in LangSlice is left out, so "
    "the atlas regions DMC-BrainMap assigns on the sections listed under "
    "warps_left_out differ from LangSlice's own maps."
)


@dataclass(frozen=True)
class DmcSkipped:
    """A section the export wrote without a registration, or not at all."""

    section: str
    reason: str


@dataclass(frozen=True)
class DmcExported:
    """What :func:`export_dmc_brainmap` wrote (on a dry run, would write)."""

    #: The animal folder DMC-BrainMap opens.
    folder: str
    #: Every file written, absolute paths.
    files: list[str]
    #: The sections written with their registration.
    registered: list[str]
    #: The sections written without one (image only) or not at all.
    skipped: list[DmcSkipped]
    #: The sections whose applied deformation was left out.
    warps_left_out: list[str]
    #: The DMC channel folder and the section image written.
    channel: str
    section_image: str
    written: bool


@dataclass(frozen=True)
class _Row:
    """One section's image file and registration."""

    section: str
    filename: str
    image: np.ndarray | None
    placement: DmcPlacement | None
    reason: str | None


def export_dmc_brainmap(
    job: Job,
    workspace: Workspace,
    *,
    channel: str = "dapi",
    section_image: str = "preprocessed",
) -> DmcExported:
    """Write the job's sections as a DMC-BrainMap animal folder.

    Parameters
    ----------
    job
        The job whose stack is written, as it stands.
    workspace
        The job's atlas and section files.
    channel : {"dapi", "green", "n3", "cy3", "cy5"}
        The channel folder DMC-BrainMap reads the registration from
        (``sharpy_track/<channel>/``).
    section_image
        The section image written for each section: ``preprocessed`` (the
        image LangSlice registered on), a raw channel's name or a made
        image's.

    Returns
    -------
    DmcExported
        The folder, the files, the sections registered, skipped and whose
        deformation was left out.

    Raises
    ------
    Refused
        ``BAD_ARGS`` for a channel DMC-BrainMap does not offer, or an atlas
        whose axes it cannot read.
    """
    if channel not in DMC_CHANNELS:
        raise Refused(
            RefusalCode.BAD_ARGS,
            f"channel {channel!r} is not a DMC-BrainMap channel; "
            f"use one of {', '.join(DMC_CHANNELS)}",
        )
    with job.writing():
        state = copy.deepcopy(job.state)
    frame = _frame(state, workspace)
    animal = Path(state.image_folder).name or "animal"
    folder = job.layout.exports_dir / DMC_FOLDER / animal
    channel_folder = folder / "sharpy_track" / channel

    logger.info("Writing %d sections for DMC-BrainMap", len(state.slices))
    rows = [
        _row(state, workspace, section, frame, section_image)
        for section in state.in_order()
    ]
    written = [row.filename for row in rows if row.image is not None]
    names = sorted(written, key=natural_key)

    files = [
        folder / "params.json",
        *(channel_folder / name for name in names),
        channel_folder / "registration.json",
        folder / EXPORT_RECORD,
    ]
    result = DmcExported(
        folder=str(folder),
        files=[str(path) for path in files],
        registered=[row.section for row in rows if row.placement is not None],
        skipped=[_skipped(row) for row in rows if row.reason is not None],
        warps_left_out=_warped(state),
        channel=channel,
        section_image=section_image,
        written=job.persist,
    )
    if job.persist:
        _write_folder(job, result, rows, names, frame)
    return result


def _write_folder(
    job: Job,
    result: DmcExported,
    rows: list[_Row],
    names: list[str],
    frame: DmcFrame,
) -> None:
    """Write the animal folder: the images, then the three JSON files."""
    folder = Path(result.folder)
    channel_folder = folder / "sharpy_track" / result.channel
    channel_folder.mkdir(parents=True, exist_ok=True)
    _remove_stale_images(channel_folder, set(names))
    for row in rows:
        if row.image is not None:
            tifffile.imwrite(channel_folder / row.filename, row.image)
    _write_registration(channel_folder / "registration.json", rows, names)
    _write_params(folder / "params.json", frame, folder.name, result.channel)
    _write_record(folder / EXPORT_RECORD, job, result, rows)


def _skipped(row: _Row) -> DmcSkipped:
    """Name a section written without a registration, and why."""
    return DmcSkipped(row.section, row.reason or "")


def _warped(state: Stack) -> list[str]:
    """List the sections that hold an applied deformation."""
    return [
        section.id
        for section in state.in_order()
        if applied_deformation(state, section) is not None
    ]


def _frame(state: Stack, workspace: Workspace) -> DmcFrame:
    """Give DMC's frame of the job's atlas and plane.

    ``BAD_ARGS`` when DMC cannot read the atlas.
    """
    context = atlas_space_context(cast("object", workspace.atlas))
    try:
        return dmc_frame(
            context.atlas_name,
            context.orientation,
            context.shape,
            context.resolution_um,
            state.plane,
        )
    except ValueError as error:
        raise Refused(RefusalCode.BAD_ARGS, str(error)) from error


def _row(
    state: Stack,
    workspace: Workspace,
    section: Section,
    frame: DmcFrame,
    section_image: str,
) -> _Row:
    """Read a section's resized image and registration, or why it has none."""
    siblings = [other.id for other in state.slices]
    filename = section_dirname(section.id, siblings) + IMAGE_SUFFIX
    try:
        plane = image_plane(workspace, section, section_image)
    except (OSError, ValueError) as error:
        reason = f"image not written: {error}"
        return _Row(section.id, filename, None, None, reason)
    resized = Image.fromarray(plane).resize(
        (frame.x.size, frame.y.size), Image.Resampling.BILINEAR
    )
    image = np.asarray(resized, dtype=np.uint8)

    problem = placement_problem(state, section)
    if problem is not None:
        return _Row(section.id, filename, image, None, problem)
    try:
        linear = section_frame(state, workspace, section)
        placement = dmc_placement(
            linear.pixel_to_atlas_um(), linear.file_size, frame
        )
    except (OSError, ValueError) as error:
        return _Row(section.id, filename, image, None, str(error))
    return _Row(section.id, filename, image, placement, None)


def _remove_stale_images(channel_folder: Path, names: set[str]) -> None:
    """Remove the images an earlier export left that this one does not write.

    DMC numbers a section by its place among the folder's images.
    """
    for path in channel_folder.glob("*.tif"):
        if path.name not in names:
            path.unlink()


def _write_registration(
    path: Path, rows: list[_Row], names: list[str]
) -> None:
    """Write SHARPy-track's ``registration.json``.

    Every image is named, and each placed section has its plane and pairs,
    keyed by the image's place in the folder.
    """
    location: dict[str, list[float]] = {}
    atlas_dots: dict[str, list[list[float]]] = {}
    sample_dots: dict[str, list[list[float]]] = {}
    image_names: dict[str, str] = {}
    for row in rows:
        if row.image is None:
            continue
        key = str(names.index(row.filename))
        image_names[key] = row.filename
        if row.placement is None:
            continue
        location[key] = row.placement.location()
        atlas_dots[key] = row.placement.atlas_dots
        sample_dots[key] = row.placement.sample_dots
    document = {
        "atlasLocation": location,
        "atlasDots": atlas_dots,
        "sampleDots": sample_dots,
        "imgName": image_names,
    }
    write_json_atomic(str(path), document, indent=None)


def _write_params(
    path: Path, frame: DmcFrame, animal: str, channel: str
) -> None:
    """Write DMC's ``params.json``: the animal, channel and atlas frame."""
    xyz = {
        key: [axis.name, axis.size, axis.resolution_um]
        for key, axis in (("x", frame.x), ("y", frame.y), ("z", frame.z))
    }
    document = {
        "general": {"animal_id": animal, "chans_imaged": [channel]},
        "atlas_info": {
            "atlas": frame.atlas,
            "orientation": frame.plane,
            "resolution": [frame.x.size, frame.y.size],
            "xyz_dict": xyz,
        },
    }
    write_json_atomic(str(path), document)


def _write_record(
    path: Path, job: Job, result: DmcExported, rows: list[_Row]
) -> None:
    """Write ``langslice_export.json``: what the export holds and left out."""
    sections = [
        {
            "section": row.section,
            "file": row.filename,
            "registered": row.placement is not None,
        }
        for row in rows
        if row.image is not None
    ]
    document = {
        "written_by": f"LangSlice {langslice.__version__}",
        "job_folder": str(job.layout.folder),
        "linear_only": True,
        "note": LINEAR_ONLY_NOTE,
        "warps_left_out": result.warps_left_out,
        "channel": result.channel,
        "section_image": result.section_image,
        "sections": sections,
        "not_registered": [
            {"section": skip.section, "reason": skip.reason}
            for skip in result.skipped
        ],
    }
    write_json_atomic(str(path), document)
