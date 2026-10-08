"""Where sections sit: positions along the slicing axis, the cutting angles,
and the stack's order, which follows the positions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job


def _write_positions(
    job: Job, workspace: Workspace, positions: Iterable[tuple[object, float]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float, float]], list[str]]:
    """Write each pair into the state, clamped; ``(written, clamped, unknown)``.
    A written position is the writer's own, so a section's starting-position
    mark (``position_source`` "default") goes even when the value written
    is the starting one. No undo step: :func:`position_sections` takes it."""
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
        record.position_source = ""
        written.append((record.id, value))
    return written, clamped, unknown


def _write_angles(job: Job, workspace: Workspace, pitch_deg: float, yaw_deg: float) -> None:
    """Give the stack one plane and drop every render cached at the old angles.
    No undo step."""
    job.state.cutting_angles_deg = {"pitch": float(pitch_deg), "yaw": float(yaw_deg)}
    workspace.render_cache.clear()


# --- position_sections ----------------------------------------------------------------


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


def renumber(order: Sequence[SliceState]) -> list[str]:
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
    placed = sorted((record for record in ordered if record.position_mm is not None),
                    key=lambda record: float(record.position_mm or 0.0))
    queue = iter(placed)
    result = [next(queue) if record.position_mm is not None else record for record in ordered]
    return renumber(result)


def position_sections(
    job: Job, workspace: Workspace, sections: Iterable[Any], cutting_angles: Any = None,
) -> SectionsPositioned:
    """Set where sections sit and the stack's cutting angles; ONE undo step.

    *sections* is a list of ``{"id", "position_mm"}`` (an id is a filename,
    ``StackState.resolve``; values are clamped into the atlas range
    (``workspace.position_range``) and reported) and *cutting_angles* ``{"pitch_deg",
    "yaw_deg"}`` for the whole stack, or None. Afterwards the stack runs in
    the order of the positions (:func:`order_by_position`); the answer lists
    that order. Either part may be left out; nothing written commits nothing.

    Refused, nothing written: ``BAD_ARGS`` (an entry that is not an id with a
    finite number, angles that are not two finite numbers, nothing asked),
    ``POSITIONS_SUPPLIED`` (the host supplies the positions: the spec has no
    ``position`` task) and ``ANGLES_SUPPLIED`` (angles asked while neither the
    ``position`` task nor ``transform.angles`` is on). A supplied position
    does not stop the angles when ``transform.angles`` is on. Unknown ids are
    reported (``unknown``), the rest written. The tool door draws the
    picture of what was written (:func:`langslice.ops.look.show_result`).
    """
    import math

    spec = job.spec
    pairs: list[tuple[object, float]] = []
    entries = list(sections) if sections is not None else []
    for entry in entries:
        if not isinstance(entry, Mapping) or "id" not in entry:
            raise Refused("BAD_ARGS", message="each section is {id, position_mm}")
        try:
            value = float(entry.get("position_mm"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise Refused("BAD_ARGS", message="position_mm must be a number") from None
        if not math.isfinite(value):
            raise Refused("BAD_ARGS", message="position_mm must be finite")
        pairs.append((entry["id"], value))
    angles: tuple[float, float] | None = None
    if cutting_angles is not None:
        try:
            angles = (float(cutting_angles["pitch_deg"]), float(cutting_angles["yaw_deg"]))
        except (TypeError, ValueError, KeyError):
            raise Refused("BAD_ARGS", message="cutting_angles is {pitch_deg, yaw_deg}") from None
        if not all(math.isfinite(value) for value in angles):
            raise Refused("BAD_ARGS", message="cutting angles must be finite")
    if not pairs and angles is None:
        raise Refused("BAD_ARGS", message="give sections or cutting_angles")
    if pairs and not spec.has("position"):
        raise Refused("POSITIONS_SUPPLIED", message="The positions were supplied by the "
                      "host; they cannot be changed here.")
    if angles is not None and not (spec.has("position") or spec.transform.angles):
        raise Refused("ANGLES_SUPPLIED", message="The cutting angles were supplied by the "
                      "host; they cannot be changed here.")

    low, high = workspace.position_range
    before = job.snapshot()
    written, clamped, unknown = _write_positions(job, workspace, pairs)
    if angles is not None:
        _write_angles(job, workspace, *angles)
    reordered = order_by_position(job) if written else []
    if written or angles is not None:
        job.commit(before)
    return SectionsPositioned(
        written=written, clamped=clamped, unknown=unknown, position_range=(low, high),
        cutting_angles=dict(job.state.cutting_angles_deg) if angles is not None else None,
        order=[record.id for record in job.state.in_order()], reordered=reordered)
