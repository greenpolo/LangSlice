"""Where sections sit: positions along the slicing axis, the cutting angles,
and the stack's order, which follows the positions.

:func:`position_sections` is the write (one undo step); the tool's function
(``position_sections_tool``) adds its rows and the positioning picture of what
it wrote.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.core.refusal import RefusalCode, Refused
from langslice.tools.look import show_result
from langslice.tools.refusal import check_frozen, unknown_sections
from langslice.tools.run import as_list, pictured

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.arguments import CuttingAngles, SectionPosition
    from langslice.tools.run import Run


def _write_positions(
    job: Job,
    workspace: Workspace,
    positions: Iterable[tuple[object, float]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float, float]], list[str]]:
    """Write each pair into the state, clamped; ``(written, clamped, unknown)``.
    No undo step: :func:`position_sections` takes it."""
    low, high = workspace.position_range
    written: list[tuple[str, float]] = []
    clamped: list[tuple[str, float, float]] = []
    unknown: list[str] = []
    for ref, requested in positions:
        record = job.state.resolve(ref)
        if record is None:
            unknown.append(str(ref))
            continue
        value = min(high, max(low, requested))
        if value != requested:
            clamped.append((record.id, requested, value))
        record.position_mm = value
        written.append((record.id, value))
    return written, clamped, unknown


def _write_angles(
    job: Job, workspace: Workspace, pitch_deg: float, yaw_deg: float
) -> None:
    """Give the stack one plane and drop every render cached at the old angles.
    No undo step."""
    job.state.set_angles((float(pitch_deg), float(yaw_deg)))
    workspace.render_cache.clear()


# --- position_sections -------------------------------------------------------


@dataclass(frozen=True)
class SectionsPositioned:
    """What :func:`position_sections` wrote."""

    #: ``(section id, position mm)`` as written (clamped), in the order asked.
    written: list[tuple[str, float]] = field(default_factory=list)
    #: ``(section id, requested mm, written mm)`` for each value moved into range.
    clamped: list[tuple[str, float, float]] = field(default_factory=list)
    #: References that name no section.
    unknown: list[str] = field(default_factory=list)
    position_range: tuple[float, float] = (0.0, 0.0)
    #: The stack-wide angles written (``{"pitch", "yaw"}``), None when not asked.
    cutting_angles: dict[str, float] | None = None
    #: The stack's order after the write: ids, anterior first.
    order: list[str] = field(default_factory=list)
    #: The ids whose place in that order changed.
    reordered: list[str] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return [section_id for section_id, _value in self.written]


def renumber(order: Sequence[Section]) -> list[str]:
    """Give *order* corrected indices 0..n-1; the ids whose index changed.

    Only ``index_corrected`` moves; positions and transforms stay where they
    are. No undo step.
    """
    moved: list[str] = []
    for index, record in enumerate(order):
        if record.index_corrected != index:
            moved.append(record.id)
        record.index_corrected = index
    return moved


def order_by_position(job: Job) -> list[str]:
    """Number the stack in the order of the sections' positions; the ids moved.

    Placed sections run by increasing position (ties keep their order) and
    take the places the placed sections held; a section without a position
    keeps its own place. No undo step.
    """
    ordered = job.state.in_order()
    placed = sorted(
        (record for record in ordered if record.position_mm is not None),
        key=lambda record: float(record.position_mm or 0.0),
    )
    queue = iter(placed)
    result = [
        next(queue) if record.position_mm is not None else record
        for record in ordered
    ]
    return renumber(result)


def position_sections(
    job: Job,
    workspace: Workspace,
    sections: Iterable[Any],
    cutting_angles: Any = None,
) -> SectionsPositioned:
    """Set where sections sit and the stack's cutting angles; ONE undo step.

    *sections* is a list of ``{"id", "position_mm"}`` (an id is a filename,
    ``Stack.resolve``; values are clamped into the atlas range
    (``workspace.position_range``) and reported) and *cutting_angles* ``{"pitch_deg",
    "yaw_deg"}`` for the whole stack, or None. Afterwards the stack runs in
    the order of the positions (:func:`order_by_position`); the answer lists
    that order. Either part may be left out; nothing written commits nothing.

    Refused, nothing written: ``BAD_ARGS`` (an entry that is not an id with a
    finite number, angles that are not two finite numbers, nothing asked), and
    ``FROZEN`` (a section named has its position frozen, or the angles are
    asked while any section has its cutting angles frozen: the angles are one
    plane for the whole stack; a task that is off freezes what it supplies, the
    positions without Positioning, the angles without Positioning or
    ``transform.angles``). A frozen position does not stop the angles, nor
    frozen angles the positions. Unknown ids are reported (``unknown``), the
    rest written. The tool door draws the picture of what was written
    (:func:`langslice.tools.look.show_result`).
    """
    import math

    pairs: list[tuple[object, float]] = []
    entries = list(sections) if sections is not None else []
    for entry in entries:
        if not isinstance(entry, Mapping) or "id" not in entry:
            raise Refused(
                RefusalCode.BAD_ARGS,
                message="each section is {id, position_mm}",
            )
        try:
            value = float(entry.get("position_mm"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise Refused(
                RefusalCode.BAD_ARGS, message="position_mm must be a number"
            ) from None
        if not math.isfinite(value):
            raise Refused(
                RefusalCode.BAD_ARGS, message="position_mm must be finite"
            )
        pairs.append((entry["id"], value))
    angles: tuple[float, float] | None = None
    if cutting_angles is not None:
        try:
            angles = (
                float(cutting_angles["pitch_deg"]),
                float(cutting_angles["yaw_deg"]),
            )
        except (TypeError, ValueError, KeyError):
            raise Refused(
                RefusalCode.BAD_ARGS,
                message="cutting_angles is {pitch_deg, yaw_deg}",
            ) from None
        if not all(math.isfinite(value) for value in angles):
            raise Refused(
                RefusalCode.BAD_ARGS, message="cutting angles must be finite"
            )
    if not pairs and angles is None:
        raise Refused(
            RefusalCode.BAD_ARGS, message="give sections or cutting_angles"
        )
    if pairs:
        named = [
            record.id
            for ref, _value in pairs
            if (record := job.state.resolve(ref)) is not None
        ]
        check_frozen(job, named, "position")
    if angles is not None:
        check_frozen(job, [record.id for record in job.state.slices], "angles")

    low, high = workspace.position_range
    before = job.snapshot()
    written, clamped, unknown = _write_positions(job, workspace, pairs)
    if angles is not None:
        _write_angles(job, workspace, *angles)
    reordered = order_by_position(job) if written else []
    if written or angles is not None:
        job.commit(before)
    cutting = None
    if angles is not None:
        pitch, yaw = job.state.stack_angles
        cutting = {"pitch": pitch, "yaw": yaw}
    return SectionsPositioned(
        written=written,
        clamped=clamped,
        unknown=unknown,
        position_range=(low, high),
        cutting_angles=cutting,
        order=[record.id for record in job.state.in_order()],
        reordered=reordered,
    )


# --- the tool ----------------------------------------------------------------

#: Positions closer than this (mm) are one atlas picture in a positioning picture.
SAME_POSITION_MM = 0.02


def distinct_positions(values: Sequence[float]) -> list[float]:
    """Return *values* in order, each once (within :data:`SAME_POSITION_MM`).

    Parameters
    ----------
    values
        Positions along the slicing axis, in mm.

    Returns
    -------
    list of float
        The positions a positioning picture draws an atlas plane at.
    """
    out: list[float] = []
    for value in values:
        if all(abs(value - kept) > SAME_POSITION_MM for kept in out):
            out.append(value)
    return out


def position_sections_tool(
    run: Run,
    *,
    sections: Sequence[SectionPosition] = (),
    cutting_angles: CuttingAngles | None = None,
    view: bool = True,
) -> dict[str, Any]:
    """Run the ``position_sections`` tool: write, answer, picture.

    A call whose every section is unknown, with no angles, is refused
    (``UNKNOWN_SLICE_IDS``); otherwise the unknown names are listed beside
    what was written. The picture is the positioning picture of the
    sections written (none written: the whole stack), or, where the job
    disables ``look:positioning``, their overlays.

    Parameters
    ----------
    run
        The run the tool belongs to.
    sections
        ``{"id", "position_mm"}`` per section, as the model sent them.
    cutting_angles
        ``{"pitch_deg", "yaw_deg"}`` for the whole stack; empty is none.
    view
        Whether to answer with the picture.

    Returns
    -------
    dict
        The reply: what was written, clamped and reordered, and the rows
        the write changed.
    """
    job = run.job
    angles = cutting_angles if cutting_angles else None
    done = position_sections(job, run.workspace, as_list(sections), angles)
    unknown = (
        unknown_sections(job.state, done.unknown) if done.unknown else None
    )
    if (
        unknown is not None
        and not done.written
        and done.cutting_angles is None
    ):
        raise unknown  # every name

    result = _positioned_reply(done, unknown)
    result.update(run.answered(*done.touched))
    if not run.shows(view):
        return result

    in_picture = done.touched or [
        section.id for section in job.state.in_order()
    ]
    in_stack = [job.state.by_id(name) for name in in_picture]
    stack_mm = [
        section.position_mm for section in in_stack if section is not None
    ]
    positions = distinct_positions(
        [float(mm) for mm in stack_mm if mm is not None]
    )
    mode = "overlay" if run.spec.off("look:positioning") else "positioning"
    looked = show_result(
        job,
        run.workspace,
        "position_sections",
        mode,
        in_picture,
        positions_mm=positions,
    )
    return pictured(result, looked)


def _positioned_reply(
    done: SectionsPositioned, unknown: Refused | None
) -> dict[str, Any]:
    """Return the positions written, those clamped into the atlas range (with
    the range), the unknown names, the order and the sections reordered.
    """
    result: dict[str, Any] = {
        "written": [
            {"id": name, "position_mm": round(value, 3)}
            for name, value in done.written
        ],
    }
    if done.clamped:
        result["clamped"] = [
            {
                "id": name,
                "requested_mm": round(requested, 3),
                "clamped_to_mm": round(value, 3),
            }
            for name, requested, value in done.clamped
        ]
        result["atlas_range_mm"] = [round(v, 3) for v in done.position_range]
    if unknown is not None:
        result["unknown_ids"] = done.unknown
        result["unknown_message"] = unknown.message
    result["order"] = done.order
    if done.reordered:
        result["reordered"] = done.reordered
    return result
