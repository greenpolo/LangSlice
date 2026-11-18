"""What a ``deepslice_place`` call writes, as one undo step.

The positions and the stack's angles go first (``set_positions``); the
angles are refused whole when any section's are frozen, and the positions
then land alone. The in-plane transforms go after them, each read at the
plane its section now holds (``set_transforms``). A proposal writes nothing
and reads the transforms at the proposed planes.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from langslice.engines.deepslice.engine import EngineRun
from langslice.engines.deepslice.proposal import (
    Plane,
    ProposalRow,
    in_plane_params,
)
from langslice.plugin_api import ToolContext, WriteResult

TRANSFORM_NOTE = "DeepSlice in-plane fit"
NO_PIXEL_SIZE = "not written (no pixel size)"
NOT_WRITTEN = "NOT_WRITTEN"


class AnswerPart(enum.StrEnum):
    """Name a part of the answer a call may write (``write``)."""

    POSITIONS = "positions"
    ANGLES = "angles"
    TRANSFORM = "transform"


PROPERTY = {
    AnswerPart.POSITIONS: "position",
    AnswerPart.ANGLES: "angles",
    AnswerPart.TRANSFORM: "affine",
}


@dataclass(frozen=True)
class PartChoice:
    """Hold which parts a call writes.

    Parameters
    ----------
    available
        The parts the description names for this job.
    parts
        The parts the call writes.
    propose_only
        Whether ``write`` was an empty list.
    """

    available: list[AnswerPart]
    parts: list[AnswerPart]
    propose_only: bool

    @property
    def shown(self) -> list[AnswerPart]:
        """Return the parts the reply proposes."""
        return self.available if self.propose_only else self.parts

    @property
    def writes(self) -> bool:
        """Return whether the call writes anything."""
        return not self.propose_only and bool(self.parts)


@dataclass(frozen=True)
class Written:
    """Hold what a call's writes did.

    Parameters
    ----------
    positions
        The sections whose position was written.
    position_errors
        Why each other section's was not, by id.
    transforms
        The sections whose transform was written.
    transform_errors
        Why each other section's was not, by id.
    angles_written
        Whether the stack's angles were written.
    refused
        A write refused whole, as ``CODE: message``; empty when none was.
    clamped
        Each position the atlas range held: id, asked and used mm.
    params
        The in-plane numbers read for each section (None without a pixel
        size).
    """

    positions: frozenset[str] = frozenset()
    position_errors: dict[str, str] = field(default_factory=dict)
    transforms: frozenset[str] = frozenset()
    transform_errors: dict[str, str] = field(default_factory=dict)
    angles_written: bool = False
    refused: str = ""
    clamped: list[tuple[str, float, float]] = field(default_factory=list)
    params: dict[str, tuple[float, ...] | None] = field(default_factory=dict)


def proposed(
    context: ToolContext,
    run: EngineRun,
    rows: Sequence[ProposalRow],
    stack: tuple[float, float] | None,
    choice: PartChoice,
) -> Written:
    """Write nothing; read the proposed transforms at the proposed planes.

    Parameters
    ----------
    context
        The tool call's context.
    run
        The DeepSlice run.
    rows
        DeepSlice's rows.
    stack
        The proposed stack angles; None when no row is placed.
    choice
        The parts the call names.

    Returns
    -------
    Written
        Nothing written, and the transforms read where they are shown.
    """
    if AnswerPart.TRANSFORM not in choice.shown:
        return Written()
    planes: dict[str, Plane] = {}
    for row in rows:
        if row.final is None or row.refused is not None:
            continue
        angles = stack if stack else context.section(row.id).angles
        planes[row.id] = (row.final.position_mm, angles)
    return Written(params=in_plane_params(context, run, planes))


def write(
    context: ToolContext,
    run: EngineRun,
    rows: Sequence[ProposalRow],
    stack: tuple[float, float] | None,
    choice: PartChoice,
) -> Written:
    """Write the parts choice, as one undo step.

    Parameters
    ----------
    context
        The tool call's context.
    run
        The DeepSlice run.
    rows
        DeepSlice's rows.
    stack
        The proposed stack angles; None when no row is placed.
    choice
        The parts the call writes.

    Returns
    -------
    Written
        What was written, what was not and why.
    """
    position_rows: list[dict[str, str | float]] = []
    if AnswerPart.POSITIONS in choice.parts:
        position_rows = [
            {"id": row.id, "position_mm": row.position_mm}
            for row in rows
            if row.position_mm is not None
        ]
    angles = None
    if AnswerPart.ANGLES in choice.parts and stack is not None:
        angles = {"pitch_deg": stack[0], "yaw_deg": stack[1]}

    with context.writing():
        written = _write_positions(context, position_rows, angles)
        if AnswerPart.TRANSFORM in choice.parts:
            written = _write_transforms(context, run, rows, written)
    return written


def _write_positions(
    context: ToolContext,
    position_rows: list[dict[str, str | float]],
    angles: dict[str, float] | None,
) -> Written:
    """Write the positions and the stack's angles, or the positions alone."""
    if not position_rows and angles is None:
        return Written()
    result = context.set_positions(position_rows, cutting_angles=angles)
    if result.ok:
        landed = result.details.get("cutting_angles") is not None
        return _positions_written(
            result, angles_written=angles is not None and landed
        )

    refused = f"{result.error}: {result.message}"
    if angles is None or not position_rows:
        return Written(refused=refused)
    result = context.set_positions(position_rows)
    if not result.ok:
        return Written(refused=refused)
    return replace(
        _positions_written(result, angles_written=False), refused=refused
    )


def _positions_written(
    result: WriteResult, *, angles_written: bool
) -> Written:
    """Return what a position write that was not refused whole did."""
    rows = cast("list[dict[str, str | float]]", result.rows)
    clamped = [
        (
            str(row["id"]),
            float(row["clamped_from_mm"]),
            float(row["position_mm"]),
        )
        for row in rows
        if "clamped_from_mm" in row
    ]
    return Written(
        positions=frozenset(result.written),
        position_errors=_row_errors(result),
        angles_written=angles_written,
        clamped=clamped,
    )


def _write_transforms(
    context: ToolContext,
    run: EngineRun,
    rows: Sequence[ProposalRow],
    written: Written,
) -> Written:
    """Write each in-plane transform, read at the plane it now holds."""
    planes: dict[str, Plane] = {}
    for row in rows:
        here = context.section(row.id)
        if row.refused is None:
            planes[row.id] = (here.position_mm, here.angles)
    params = in_plane_params(context, run, planes)
    errors = {
        section: NO_PIXEL_SIZE
        for section, numbers in params.items()
        if numbers is None
    }
    transform_rows = [
        {"id": section, "params": list(numbers), "note": TRANSFORM_NOTE}
        for section, numbers in params.items()
        if numbers is not None
    ]
    if not transform_rows:
        return replace(written, params=params, transform_errors=errors)

    done = context.set_transforms(transform_rows)
    if not done.ok:
        refused = written.refused or f"{done.error}: {done.message}"
        return replace(
            written, params=params, transform_errors=errors, refused=refused
        )
    return replace(
        written,
        params=params,
        transforms=frozenset(done.written),
        transform_errors={**errors, **_row_errors(done)},
    )


def _row_errors(result: WriteResult) -> dict[str, str]:
    """Return each section a write left out, with its code."""
    rows = cast("list[dict[str, str]]", result.rows)
    landed = set(result.written)
    return {
        row["id"]: str(row.get("error", NOT_WRITTEN))
        for row in rows
        if row["id"] not in landed
    }
