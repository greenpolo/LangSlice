"""Where sections sit: positions along the slicing axis, and the stack's cutting angles."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langslice.ops.refusal import Refused

if TYPE_CHECKING:
    from langslice.core.display import DisplayOptions
    from langslice.core.state import SliceState
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.ops.views import PlacementView


@dataclass(frozen=True)
class PositionsWritten:
    """What :func:`set_positions` wrote, in the order asked."""

    #: ``(section id, position mm)`` as written (clamped).
    written: list[tuple[str, float]] = field(default_factory=list)
    #: ``(section id, requested mm, written mm)`` for each value moved into range.
    clamped: list[tuple[str, float, float]] = field(default_factory=list)
    #: References that name no section.
    unknown: list[str] = field(default_factory=list)
    #: The atlas range the values were clamped into, ``(low, high)`` mm.
    position_range: tuple[float, float] = (0.0, 0.0)
    #: With display options: the written placements pictured
    #: (:func:`langslice.ops.views.placement_view`), and the ids of those
    #: *show* left out.
    view: PlacementView | None = None
    not_shown: list[str] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return [section_id for section_id, _value in self.written]


def set_positions(
    job: Job, workspace: Workspace, positions: Iterable[tuple[object, float]], *,
    options: DisplayOptions | None = None,
    show: Callable[[SliceState, float], bool] | None = None,
) -> PositionsWritten:
    """Write each ``(filename or corrected index, millimetres)`` pair.

    A value outside the atlas range (``workspace.position_range``) is
    clamped into it and reported. The same section twice is written twice
    (the last value stays). One undo step for the whole call; nothing is
    committed when nothing was written.

    With *options*, each written placement is then pictured in the call's
    placement mode, those *show* keeps (all without it; the tool door
    leaves out what the model has already seen at the same geometry). A
    picture that fails leaves the write standing.
    """
    low, high = workspace.position_range
    before = job.snapshot()
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
    if written:
        job.commit(before)
    view: PlacementView | None = None
    not_shown: list[str] = []
    if options is not None and written:
        from langslice.ops.views import placement_view

        pairs: list[tuple[SliceState, float]] = []
        for name, value in written:
            record = job.state.by_id(name)
            if record is None:
                continue
            if show is None or show(record, value):
                pairs.append((record, value))
            else:
                not_shown.append(name)
        view = placement_view(job, workspace, pairs, options, regions_report=False)
    return PositionsWritten(written=written, clamped=clamped, unknown=unknown,
                            position_range=(low, high), view=view, not_shown=not_shown)


def set_cutting_angles(job: Job, workspace: Workspace, pitch_deg: float, yaw_deg: float) -> None:
    """Set the stack-wide cutting angles (degrees); one undo step.

    Every render cached at the old angles is dropped.
    """
    before = job.snapshot()
    job.state.cutting_angles_deg = {"pitch": float(pitch_deg), "yaw": float(yaw_deg)}
    workspace.render_cache.clear()
    job.commit(before)


# --- searches: read, never written ----------------------------------------------------


def search_position(
    job: Job, workspace: Workspace, ref: object, window_mm: Any, *, angles: bool,
) -> dict[str, Any]:
    """Search the atlas around a section's current position; writes nothing.

    Scores the section (its 512 px working render) against resampled atlas
    planes within *window_mm* of its position (:func:`langslice.core.oblique.fit_oblique`)
    and returns the best: ``id``, ``current_position_mm``, ``position_mm``,
    ``pitch_deg``, ``yaw_deg``, ``score``, ``searched_window_mm``,
    ``searched_angles``. *angles* also searches the cutting angles (±15
    degrees); otherwise they are held at the stack's. Refused:
    ``UNKNOWN_SLICE_IDS``, ``NO_POSITION``, ``BAD_ARGS`` (a window that is
    not a number), ``FIT_FAILED``.
    """
    import logging

    from langslice.core.oblique import fit_oblique
    from langslice.core.sections import render_slice
    from langslice.core.space import Plane

    state = job.state
    record = state.resolve(ref)
    if record is None:
        raise Refused("UNKNOWN_SLICE_IDS", unknown=[ref])
    if record.position_mm is None:
        raise Refused("NO_POSITION", id=record.id)
    try:
        window = max(0.0, float(window_mm))
    except (TypeError, ValueError):
        raise Refused("BAD_ARGS") from None
    pitch, yaw = state.pitch_deg, state.yaw_deg
    bounds = ((-15.0, 15.0), (-15.0, 15.0)) if angles else ((pitch, pitch), (yaw, yaw))
    section = render_slice(workspace, record, long_edge=512)
    try:
        fit = fit_oblique(
            workspace.atlas,
            section,
            record.position_mm,
            cast(Plane, state.plane),
            pitch_bounds=bounds[0],
            yaw_bounds=bounds[1],
            position_window_mm=window,
            allow_mirror=False,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("search_position failed for %s: %s", record.id, exc)
        raise Refused("FIT_FAILED", message=str(exc)) from exc
    return {
        "id": record.id,
        "current_position_mm": round(record.position_mm, 3),
        "position_mm": round(float(fit["position_mm"]), 3),
        "pitch_deg": round(float(fit["pitch_deg"]), 3),
        "yaw_deg": round(float(fit["yaw_deg"]), 3),
        "score": round(float(fit["score"]), 4),
        "searched_window_mm": round(window, 3),
        "searched_angles": bool(angles),
    }


def run_deepslice(
    job: Job, workspace: Workspace, slice_ids: list[str], *, allow_angle_change: bool,
) -> dict[str, Any]:
    """Seed positions (and optionally angles) with DeepSlice: the seam only
    (:mod:`langslice.core.deepslice`), which answers ``UNAVAILABLE``."""
    from langslice.core.deepslice import run_deepslice as seam

    return seam(job.state, workspace, slice_ids=[str(item) for item in slice_ids],
                allow_angle_change=bool(allow_angle_change))
