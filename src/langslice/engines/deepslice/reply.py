"""The words of a ``deepslice_place`` reply, and the call's saved record.

The reply (no metric or confidence): ``written``, ``stack`` (the pitch and
yaw now and as proposed, what became of the angles, the spacing used and
whether DeepSlice estimated it, the bad sections and why), then ``columns``
and ``rows``: per section in proposed order its filename, position, change,
own pitch and yaw, whether it is bad, the in-plane change in words where a
transform is proposed, and what was written. ``refused``, ``clamped`` and
``view`` follow when they have something to say.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from langslice.engines.deepslice.engine import EngineRun
from langslice.engines.deepslice.proposal import ProposalRow
from langslice.engines.deepslice.writes import AnswerPart, PartChoice, Written
from langslice.plugin_api import SectionView, ToolResult

NO_PIXEL_SIZE = "no transform: the job has no pixel size for this section"
NO_ORDER = "none: order not given"
INTERVAL_UNUSED = "none: order not given (the job's interval was not used)"
SEE_PROPOSAL = (
    "To see the stack at its current positions, use the positioning view of "
    "look; view true here draws the proposal."
)
SEE_WRITTEN = (
    "The positioning view of look shows the sections at these positions."
)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
COLUMNS = (
    "section",
    "position_mm",
    "change_mm",
    "own_pitch_deg",
    "own_yaw_deg",
    "bad",
)

Cell = str | float | None


@dataclass(frozen=True)
class _Angle:
    now: float | None
    proposed: float | None


@dataclass(frozen=True)
class _Interval:
    interval_um: float | None
    estimated_by_deepslice: bool


@dataclass(frozen=True)
class _Bad:
    id: str
    why: str


@dataclass(frozen=True)
class _Stack:
    pitch_deg: _Angle
    yaw_deg: _Angle
    angles: str
    spacing: _Interval | str
    bad_sections: list[_Bad]


@dataclass(frozen=True)
class CallFacts:
    """Hold what a call ran on, for its reply and its record.

    Parameters
    ----------
    views
        Each section as the job held it before the call.
    run_ids
        The sections run, in the call's order.
    ordered
        Whether the call gave a cutting order.
    bad
        Each bad section's id and why.
    interval_um
        The job's section interval; None when it gives none.
    angles_frozen
        Whether any section's angles are frozen.
    """

    views: Mapping[str, SectionView]
    run_ids: list[str]
    ordered: bool
    bad: dict[str, str]
    interval_um: float | None
    angles_frozen: bool


@dataclass(frozen=True)
class Answer:
    """Hold a call's answer, to be worded.

    Parameters
    ----------
    facts
        What the call ran on.
    choice
        The parts it writes.
    run
        The DeepSlice run.
    rows
        DeepSlice's rows, in proposed order.
    stack
        The proposed stack angles; None when no row is placed.
    written
        What the writes did.
    """

    facts: CallFacts
    choice: PartChoice
    run: EngineRun
    rows: Sequence[ProposalRow]
    stack: tuple[float, float] | None
    written: Written

    @property
    def columns(self) -> list[str]:
        """Return the names of a reply row's cells."""
        columns: list[str] = list(COLUMNS)
        if AnswerPart.TRANSFORM in self.choice.shown:
            columns.append("in_plane_change")
        if not self.choice.propose_only:
            columns.append("written")
        return columns

    @property
    def angles(self) -> str:
        """Return what became of the stack's angles, in words."""
        if self.choice.propose_only:
            return "proposal only"
        if AnswerPart.ANGLES not in self.choice.parts:
            return "not choice"
        if self.written.angles_written:
            return "written"
        if self.facts.angles_frozen:
            return "FROZEN"
        return self.written.refused or "not written"

    @property
    def now(self) -> tuple[float | None, float | None]:
        """Return the stack's angles before the call, rounded."""
        pitch, yaw = self.facts.views[self.facts.run_ids[0]].angles
        return rounded(pitch, 2), rounded(yaw, 2)


def rounded(value: float | None, digits: int) -> float | None:
    """Round a number for the reply.

    Parameters
    ----------
    value
        The number, or None.
    digits
        The decimals kept.

    Returns
    -------
    float or None
        The rounded number (``+ 0.0`` leaves no ``-0.0``), or None.
    """
    return None if value is None else round(float(value), digits) + 0.0


def answered(
    answer: Answer, pictures: list[Image.Image], *, view_sentence: bool
) -> ToolResult:
    """Word a call's reply.

    Parameters
    ----------
    answer
        The call's answer.
    pictures
        The pictures the reply carries.
    view_sentence
        Whether the reply ends with the sentence pointing to ``look``'s
        positioning view (no picture choice, and that view on).

    Returns
    -------
    ToolResult
        The reply, as the module text lays it out, and its pictures.
    """
    if answer.choice.propose_only:
        written: str | list[str] = "nothing: proposal only"
    elif not answer.choice.parts:
        written = "nothing"
    else:
        written = [str(part) for part in answer.choice.parts]
    reply: dict[str, object] = {
        "written": written,
        "stack": asdict(_stack(answer)),
        "columns": answer.columns,
        "rows": table(answer),
    }
    if answer.written.refused:
        reply["refused"] = answer.written.refused
    if answer.written.clamped:
        reply["clamped"] = [
            {"id": section, "asked_mm": requested_mm, "used_mm": used}
            for section, requested_mm, used in answer.written.clamped
        ]
    if view_sentence:
        propose = answer.choice.propose_only
        reply["view"] = SEE_PROPOSAL if propose else SEE_WRITTEN
    return ToolResult(reply, list(pictures))


def _stack(answer: Answer) -> _Stack:
    """Return the reply's ``stack``."""
    now_pitch, now_yaw = answer.now
    pitch = yaw = None
    if answer.stack is not None:
        pitch, yaw = rounded(answer.stack[0], 2), rounded(answer.stack[1], 2)
    return _Stack(
        pitch_deg=_Angle(now=now_pitch, proposed=pitch),
        yaw_deg=_Angle(now=now_yaw, proposed=yaw),
        angles=answer.angles,
        spacing=_spacing(answer),
        bad_sections=[
            _Bad(id=section, why=why)
            for section, why in answer.facts.bad.items()
        ],
    )


def _spacing(answer: Answer) -> _Interval | str:
    """Return the spacing DeepSlice used, or why there is none."""
    if answer.facts.interval_um and not answer.facts.ordered:
        return INTERVAL_UNUSED
    spacing = answer.run.spacing
    if spacing is None:
        return NO_ORDER
    used = abs(spacing.section_thickness_um_used)
    return _Interval(
        interval_um=rounded(used, 1),
        estimated_by_deepslice=spacing.estimated,
    )


def table(answer: Answer) -> list[list[Cell]]:
    """Return one row of cells per section, in proposed order.

    Parameters
    ----------
    answer
        The call's answer.

    Returns
    -------
    list of list
        The cells :attr:`Answer.columns` names, per section.
    """
    return [_cells(answer, row) for row in answer.rows]


def _cells(answer: Answer, row: ProposalRow) -> list[Cell]:
    """Return one section's cells."""
    view = answer.facts.views[row.id]
    change = None
    if row.position_mm is not None:
        change = rounded(row.position_mm - view.position_mm, 3)
    own_pitch = own_yaw = None
    if row.own is not None:
        own_pitch, own_yaw = (
            rounded(row.own.pitch_deg, 1),
            rounded(row.own.yaw_deg, 1),
        )
    cells: list[Cell] = [
        row.id,
        rounded(row.position_mm, 3),
        change,
        own_pitch,
        own_yaw,
        answer.facts.bad.get(row.id, ""),
    ]
    if AnswerPart.TRANSFORM in answer.choice.shown:
        cells.append(_in_plane(answer.written, row, view))
    if not answer.choice.propose_only:
        cells.append(_outcome(answer, row))
    return cells


def _in_plane(written: Written, row: ProposalRow, view: SectionView) -> str:
    """Return a row's in-plane change in words, or why there is none."""
    if row.refused:
        return ""
    params = written.params.get(row.id)
    if params is None:
        return NO_PIXEL_SIZE
    width, height = view.file_size
    if view.rotation_deg in (90, 270):
        width, height = height, width
    current = IDENTITY if view.affine is None else view.affine.matrix
    return in_plane_change(params, current or IDENTITY, width, height)


def _outcome(answer: Answer, row: ProposalRow) -> str:
    """Return what was written for a row, or why it was not placed."""
    if row.refused:
        return "not placed: " + row.refused
    written = answer.written
    parts = (
        (
            AnswerPart.POSITIONS,
            "position",
            written.positions,
            written.position_errors,
        ),
        (
            AnswerPart.TRANSFORM,
            "transform",
            written.transforms,
            written.transform_errors,
        ),
    )
    did: list[str] = []
    for part, label, landed, errors in parts:
        if part not in answer.choice.parts:
            continue
        if row.id in landed:
            did.append(label)
        else:
            did.append(f"{label} {errors.get(row.id, 'not written')}")
    return ", ".join(did) if did else "nothing"


@dataclass(frozen=True)
class _Knobs:
    rotation_deg: float
    scale_x: float
    scale_y: float
    tx: float
    ty: float


def _knobs(params: Sequence[float], width: int, height: int) -> _Knobs:
    """Return six normalized numbers as rotation, scales and shifts.

    In the pixel frame of a ``width`` x ``height`` render, as LangSlice's
    ``decompose_affine``: the rotation counter-clockwise, the shifts as
    shares of the width and height.
    """
    a, b, tx, c, d, ty = (float(value) for value in params)
    b, c = b * width / height, c * height / width
    sx = math.hypot(a, c)
    if sx <= 0:
        return _Knobs(0.0, sx, 0.0, tx, ty)
    turn = math.degrees(math.atan2(-c, a))
    return _Knobs(turn, sx, (a * d - b * c) / sx, tx, ty)


def in_plane_change(
    params: Sequence[float], current: Sequence[float], width: int, height: int
) -> str:
    """Word a proposed in-plane transform against the one a section holds.

    Parameters
    ----------
    params
        The proposed six normalized numbers.
    current
        The section's own six numbers (the identity when it has none).
    width, height
        The oriented render's size in pixels.

    Returns
    -------
    str
        The turn, the scales, and the shift as a share of the width and
        height.
    """
    new = _knobs(params, width, height)
    old = _knobs(current, width, height)
    turn = new.rotation_deg - old.rotation_deg
    sx = new.scale_x / old.scale_x if old.scale_x else 1.0
    sy = new.scale_y / old.scale_y if old.scale_y else 1.0
    dx, dy = new.tx - old.tx, new.ty - old.ty
    way = "counter-clockwise" if turn >= 0 else "clockwise"
    across = "right" if dx >= 0 else "left"
    down = "down" if dy >= 0 else "up"
    return (
        f"turn {abs(turn):.1f} deg {way}, scale x {sx:.2f} y {sy:.2f}, "
        f"shift {abs(dx) * 100:.0f}% {across}, {abs(dy) * 100:.0f}% {down}"
    )


@dataclass(frozen=True)
class Arguments:
    """Hold one call's arguments, as the agent gave them.

    Parameters
    ----------
    sections, order
        Filenames run, and the cutting order; empty for every section, and
        for no order.
    bad_sections
        Filenames DeepSlice leaves out of its vote; None for the marked ones.
    write
        The parts to write; None for every part the job allows.
    view
        Whether the reply carries the positioning picture.
    """

    sections: list[str]
    order: list[str]
    bad_sections: list[str] | None
    write: list[str] | None
    view: bool


@dataclass(frozen=True)
class Settings:
    """Hold what a call ran with, for its record.

    Parameters
    ----------
    arguments
        The call's arguments.
    interval_um_signed_passed
        The signed interval DeepSlice was given; None for its estimate.
    ensemble
        Whether DeepSlice averaged its two networks.
    long_edge
        The long edge of the images written for it.
    """

    arguments: Arguments
    interval_um_signed_passed: float | None
    ensemble: bool
    long_edge: int


def save_record(answer: Answer, settings: Settings, folder: Path) -> None:
    """Write a call's ``settings.json`` and ``rows.json``.

    Parameters
    ----------
    answer
        The call's answer.
    settings
        What the call ran with.
    folder
        The call's folder.
    """
    run = answer.run
    spacing = None if run.spacing is None else asdict(run.spacing)
    record = {
        "arguments": asdict(settings.arguments),
        "run": answer.facts.run_ids,
        "write_parts": [str(part) for part in answer.choice.parts],
        "proposal_only": answer.choice.propose_only,
        "bad_sections_used": answer.facts.bad,
        "interval_um_job": answer.facts.interval_um,
        "interval_um_signed_passed": settings.interval_um_signed_passed,
        "ensemble": settings.ensemble,
        "long_edge": settings.long_edge,
        "versions": asdict(run.versions),
        "timings_s": run.timings_s,
        "spacing": spacing,
        "stack_angles_proposed": answer.stack,
        "stack_angles_now": answer.now,
    }
    _write_json(folder / "settings.json", record)

    proposal = [asdict(_Proposed.of(row)) for row in answer.rows]
    rows = {
        "columns": answer.columns,
        "rows": table(answer),
        "stack": asdict(_stack(answer)),
        "proposal": proposal,
    }
    _write_json(folder / "rows.json", rows)


@dataclass(frozen=True)
class _Proposed:
    """One section of DeepSlice's answer as ``rows.json`` keeps it."""

    id: str
    position_mm: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    own_position_mm: float | None
    own_pitch_deg: float | None
    own_yaw_deg: float | None
    bad: bool
    anchoring: tuple[float, ...]
    own_anchoring: tuple[float, ...]
    refused: str | None

    @classmethod
    def of(cls, row: ProposalRow) -> _Proposed:
        final = row.final
        own = row.own
        return cls(
            id=row.id,
            position_mm=row.position_mm,
            pitch_deg=final and final.pitch_deg,
            yaw_deg=final and final.yaw_deg,
            own_position_mm=own and own.position_mm,
            own_pitch_deg=own and own.pitch_deg,
            own_yaw_deg=own and own.yaw_deg,
            bad=row.bad,
            anchoring=row.anchoring,
            own_anchoring=row.own_anchoring,
            refused=row.refused,
        )


def _write_json(path: Path, value: object) -> None:
    """Write one JSON file of a call's record."""
    text = json.dumps(value, indent=1)
    path.write_text(text, encoding="utf-8", newline="\n")
