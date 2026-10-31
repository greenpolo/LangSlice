"""``abba.prepare``: an ABBA session's snapshots become a prepared job.

The Fiji connector exports calibrated snapshots into the ABBA project's
folder and sends what ABBA knows of them (:class:`AbbaPrepareRequest`). The
job folder beside them (``<snapshots>/langslice/``) gets ``job.json`` and
the reference card, nothing else: no checkpoint, so the job is *prepared*,
and whichever door opens it first takes the saved settings as its defaults
(:func:`langslice.job.host_inputs.prepared_spec`). The saved settings hold
ABBA's inputs (positions in BrainGlobe AP millimetres, the stack-wide
cutting angles, the pixel size, the page names), the registrations the user
made in ABBA frozen (``orientation`` and ``affine`` of every registered
slice, ``deformation`` of every slice with its own warp), and the angle
estimate on. ``host`` holds the session's facts under ``abba``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from langslice.core.sections.discovery import discover_slices
from langslice.core.sections.reading import page_count
from langslice.doors.abba.checkpoints import snapshot_sizes
from langslice.doors.abba.contract import AbbaPrepareRequest
from langslice.doors.reference_card import write_card
from langslice.job.host_inputs import PREPARED_HOST_KIND
from langslice.job.layout import (
    JobLayout,
    check_owner,
    job_folder_for,
    write_job_file,
)
from langslice.job.settings.spec import JobSpec

logger = logging.getLogger(__name__)

#: The format of a prepared job's ``host`` record in ``job.json``.
HOST_FORMAT = 3
#: What a registered ABBA slice keeps: the user's in-plane registration.
REGISTERED_FROZEN = ("orientation", "affine")
#: Why a session whose job already has a checkpoint is not prepared again.
ALREADY_RAN = (
    "This job already ran ({folder} holds its state.json): continue it, or "
    "start over from ABBA with a new job."
)


def prepare_job(request: AbbaPrepareRequest) -> Path:
    """Write the prepared job of an ABBA session's snapshots.

    Parameters
    ----------
    request
        The snapshots' folder and what ABBA knows of them.

    Returns
    -------
    Path
        The job folder, ``<snapshots>/langslice``.

    Raises
    ------
    ValueError
        When a name does not match the exported snapshots, or the job
        folder already holds a checkpoint or another folder's job.
    """
    images = Path(request.image_folder).expanduser().resolve(strict=True)
    if not images.is_dir():
        raise ValueError(f"The snapshot folder {images} is not a directory.")
    layout = JobLayout(job_folder_for(images), images)
    if layout.state_file.exists():
        raise ValueError(ALREADY_RAN.format(folder=layout.folder))
    sizes = snapshot_sizes(images)
    _check_names(request, set(sizes))
    spec = _prepared_settings(request, images)

    # the session's facts, kept with the job
    session = request.abba
    angles = request.angles_deg
    facts = {
        "project_key": session.project_key,
        "atlas": session.atlas,
        "sections": dict(session.sections),
        "registered": sorted(set(request.registered)),
        "existing_warp": sorted(set(request.existing_warp)),
        "angles_deg": {
            "pitch_deg": angles.pitch_deg,
            "yaw_deg": angles.yaw_deg,
        },
        "z_offset_mm": request.z_offset_mm,
        "pixel_size_um": request.pixel_size_um,
        "thickness_um": session.thickness_um,
        "interval_um": session.interval_um,
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    host = {"format": HOST_FORMAT, "kind": PREPARED_HOST_KIND, "abba": facts}

    check_owner(layout)
    layout.ensure()
    write_job_file(layout, spec=spec.to_dict(), host=host)
    write_card(layout)
    logger.info(
        "Prepared the ABBA job in %s (%d sections)", layout.folder, len(sizes)
    )
    return layout.folder


def _check_names(request: AbbaPrepareRequest, snapshots: set[str]) -> None:
    """Refuse a request whose filenames are not the exported snapshots."""
    if not snapshots or set(request.positions_mm) != snapshots:
        raise ValueError(
            "The positions must name every exported snapshot exactly "
            f"(snapshots {sorted(snapshots)}, positions "
            f"{sorted(request.positions_mm)})."
        )
    named = {
        "existing_warp": set(request.existing_warp),
        "registered": set(request.registered),
        "abba.sections": set(request.abba.sections),
    }
    for key, names in named.items():
        unknown = sorted(names - snapshots)
        if unknown:
            raise ValueError(
                f"{key} must name exported snapshots; {unknown} are not."
            )


def _frozen(request: AbbaPrepareRequest) -> dict[str, list[str]]:
    """Freeze the registrations made in ABBA: the agent may not change them."""
    frozen: dict[str, list[str]] = {}
    for name in sorted(set(request.registered) | set(request.existing_warp)):
        kept: list[str] = (
            list(REGISTERED_FROZEN) if name in request.registered else []
        )
        if name in request.existing_warp:
            kept.append("deformation")
        frozen[name] = kept
    return frozen


def _prepared_settings(request: AbbaPrepareRequest, images: Path) -> JobSpec:
    """Make the job's settings: ABBA's inputs, its registrations frozen."""
    positions = dict(request.positions_mm)
    angles = request.angles_deg
    inputs: dict[str, object] = {
        "positions": positions,
        "order": sorted(positions, key=positions.__getitem__),
        "pixel_size_um": request.pixel_size_um,
        "angles": {"pitch": angles.pitch_deg, "yaw": angles.yaw_deg},
    }
    names = _channel_names(request.channel_names)
    if names is not None:
        inputs["channel_names"] = names
    return JobSpec.from_dict(
        {
            "image_folder": str(images),
            "inputs": inputs,
            "host_preprocessing": _host_appearance(images),
            "frozen": _frozen(request),
            "transform": {"angles": True},
            "resume": False,
        }
    )


def _channel_names(names: list[str] | None) -> list[str] | None:
    """Name the pages: a blank name by its page; the names distinct."""
    if names is None:
        return None
    cleaned: list[str] = []
    for index, name in enumerate(names):
        cleaned.append(name.strip() or f"ch{index + 1}")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"channel_names must be distinct; got {cleaned}")
    return cleaned


def _host_appearance(images: Path) -> dict[str, str] | None:
    """Return the snapshots' opening appearance.

    The automatic blend when any snapshot has a page per channel; None (the
    automatic path of a single image) otherwise.
    """
    for path in discover_slices(str(images)):
        if page_count(path) > 1:
            return {"mode": "auto"}
    return None
