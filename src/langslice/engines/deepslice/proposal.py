"""DeepSlice's answer as per-section rows of the job: what the tool proposes.

The geometry is never computed here: each section's position and two cutting
angles, and its in-plane transform, come from LangSlice's own registration
reader (``ToolContext.read_registration``) on the run's QUINT files, read
with ``registered_on="oriented"``: the image DeepSlice saw was the oriented
render, so the six in-plane numbers come out for that orientation.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langslice.engines.deepslice.engine import EngineRun
from langslice.plugin_api import ToolContext

NO_PLANE = "not placed by the registration reader"

Plane = tuple[float, tuple[float, float]]


@dataclass(frozen=True)
class Placement:
    """Hold one section's plane as the registration reader gives it.

    Parameters
    ----------
    position_mm
        Its position along the slicing axis, atlas millimetres.
    pitch_deg, yaw_deg
        Its plane's two cutting angles.
    params
        The six in-plane numbers on the oriented render; None when the job
        has no pixel size for the section.
    """

    position_mm: float
    pitch_deg: float
    yaw_deg: float
    params: tuple[float, ...] | None


@dataclass(frozen=True)
class ProposalRow:
    """Hold one section of DeepSlice's answer.

    Parameters
    ----------
    id
        The section's filename.
    nr
        DeepSlice's section number.
    bad
        Whether it was a bad section.
    final
        Its plane after DeepSlice's batch steps (the stack's angles); None
        when the reader refused it.
    own
        Its own plane right after ``predict``; None when refused.
    anchoring, own_anchoring
        The two QuickNII anchorings it was read from.
    refused
        Why the reader placed it nowhere; None when it did.
    """

    id: str
    nr: int | None
    bad: bool
    final: Placement | None
    own: Placement | None
    anchoring: tuple[float, ...]
    own_anchoring: tuple[float, ...]
    refused: str | None

    @property
    def position_mm(self) -> float | None:
        """Return the proposed position, or None."""
        return None if self.final is None else self.final.position_mm


@dataclass(frozen=True)
class _Read:
    """One QUINT file as the reader placed the run's sections."""

    by_id: dict[str, Placement]
    refused: dict[str, str]


def proposal_rows(context: ToolContext, run: EngineRun) -> list[ProposalRow]:
    """Read a run's answer as one row per section, in DeepSlice's order.

    Parameters
    ----------
    context
        The tool call's context, whose reader places the sections.
    run
        The DeepSlice run.

    Returns
    -------
    list of ProposalRow
        One row per section of the run, in its final order.
    """
    ids = list(run.sections)
    final = _read(context, run.final_json, ids, None)
    own = _read(context, run.own_json, ids, None)
    rows: list[ProposalRow] = []
    for section, answer in run.sections.items():
        final_plane = final.by_id.get(section)
        own_plane = own.by_id.get(section)
        why = final.refused.get(section) or own.refused.get(section)
        if not why and (final_plane is None or own_plane is None):
            why = NO_PLANE
        rows.append(
            ProposalRow(
                id=section,
                nr=answer.nr,
                bad=answer.bad,
                final=final_plane,
                own=own_plane,
                anchoring=answer.anchoring,
                own_anchoring=answer.own_anchoring,
                refused=why or None,
            )
        )
    return rows


def in_plane_params(
    context: ToolContext, run: EngineRun, planes: Mapping[str, Plane]
) -> dict[str, tuple[float, ...] | None]:
    """Read each section's in-plane transform on the plane it would hold.

    Parameters
    ----------
    context
        The tool call's context, whose reader places the sections.
    run
        The DeepSlice run.
    planes
        Each section's plane to read the file's map on.

    Returns
    -------
    dict
        The six numbers by section id (None without a pixel size), for the
        sections the reader placed.
    """
    if not planes:
        return {}
    read = _read(context, run.final_json, list(planes), planes)
    return {section: plane.params for section, plane in read.by_id.items()}


def voted_angles(rows: Sequence[ProposalRow]) -> tuple[float, float] | None:
    """Return the stack's propagated pitch and yaw.

    Parameters
    ----------
    rows
        The run's rows.

    Returns
    -------
    tuple of float, or None
        The median over the placed rows (they agree to 1e-6 degrees), or
        None when no row is placed.
    """
    planes = [row.final for row in rows if row.final is not None]
    if not planes:
        return None
    return (
        statistics.median(plane.pitch_deg for plane in planes),
        statistics.median(plane.yaw_deg for plane in planes),
    )


def _read(
    context: ToolContext,
    path: Path,
    ids: list[str],
    planes: Mapping[str, Plane] | None,
) -> _Read:
    """Read a QUINT file of the run for the sections named."""
    at: dict[object, dict[str, object]] | None = None
    if planes is not None:
        at = {
            section: {"position_mm": position, "angles": angles}
            for section, (position, angles) in planes.items()
        }
    read = context.read_registration(path, sections=ids, at=at, registered_on="oriented")

    rows = cast("list[dict[str, float | str | list[float] | None]]", read["placements"])
    by_id: dict[str, Placement] = {}
    for row in rows:
        params = cast("list[float] | None", row["params"])
        numbers = None if params is None else tuple(params)
        by_id[str(row["id"])] = Placement(
            position_mm=cast("float", row["position_mm"]),
            pitch_deg=cast("float", row["pitch_deg"]),
            yaw_deg=cast("float", row["yaw_deg"]),
            params=numbers,
        )
    refused = cast("dict[str, str]", read["refused"])
    return _Read(by_id=by_id, refused=dict(refused))
