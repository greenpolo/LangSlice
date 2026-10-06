"""Where sections sit: positions along the slicing axis, and the cutting angles."""

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

    Every section gets the one plane: a stack whose sections carried
    different angles (a registration supplied per section) has one angle
    afterwards, and undo restores each section's own. Every render cached
    at the old angles is dropped.
    """
    before = job.snapshot()
    job.state.cutting_angles_deg = {"pitch": float(pitch_deg), "yaw": float(yaw_deg)}
    workspace.render_cache.clear()
    job.commit(before)


# --- searches: read, never written ----------------------------------------------------


#: The coarse position grid's largest spacing (mm) in :func:`search_position`.
SEARCH_STEP_MM = 0.25


def search_position(
    job: Job, workspace: Workspace, ref: object, window_mm: Any, *, angles: bool,
    around_mm: Any = None,
) -> dict[str, Any]:
    """Search the atlas for a section's position; writes nothing.

    Scores the section (its 512 px working render) against resampled atlas
    planes (:func:`langslice.core.oblique.fit_oblique`, the coarse position
    grid at most :data:`SEARCH_STEP_MM` apart) and returns the best: ``id``,
    ``current_position_mm`` (None for a section without one),
    ``position_mm``, ``pitch_deg``, ``yaw_deg``, ``score``,
    ``searched_window_mm``, ``searched_range_mm`` and ``searched_angles``.
    The search runs within *window_mm* of *around_mm* (None or negative:
    the section's position), clamped to the atlas's valid range; a section
    with neither is searched over the whole valid range. With the section's
    pixel size known, a plane whose brain is too small to hold its tissue
    is never chosen (``fit_oblique``'s ``section_um_per_px``). *angles* also
    searches the cutting angles (±15 degrees); otherwise they are held at
    the section's own (the stack's unless a registration was supplied per
    section). Refused: ``UNKNOWN_SLICE_IDS``, ``BAD_ARGS`` (a window or
    centre that is not a number), ``FIT_FAILED``.
    """
    import logging
    import math

    from langslice.core.oblique import fit_oblique
    from langslice.core.sections import canvas_um_per_px, render_slice
    from langslice.core.space import Plane

    state = job.state
    record = state.resolve(ref)
    if record is None:
        raise Refused("UNKNOWN_SLICE_IDS", unknown=[ref])
    try:
        window = max(0.0, float(window_mm))
        centre = None if around_mm is None else float(around_mm)
    except (TypeError, ValueError):
        raise Refused("BAD_ARGS") from None
    if not math.isfinite(window) or (centre is not None and not math.isfinite(centre)):
        raise Refused("BAD_ARGS")
    if centre is None or centre < 0:
        centre = record.position_mm
    low, high = workspace.position_range
    if centre is None:  # nothing to search around: the whole valid range
        start, stop = low, high
    else:
        start, stop = max(low, centre - window), min(high, centre + window)
        if start > stop:  # a centre outside the range: its nearest end
            start = stop = min(max(centre, low), high)
    pitch, yaw = record.angles
    bounds = ((-15.0, 15.0), (-15.0, 15.0)) if angles else ((pitch, pitch), (yaw, yaw))
    section = render_slice(workspace, record, long_edge=512)
    um_per_px, _source = canvas_um_per_px(workspace, record, long_edge=512)
    try:
        fit = fit_oblique(
            workspace.atlas,
            section,
            (start + stop) / 2.0,
            cast(Plane, state.plane),
            pitch_bounds=bounds[0],
            yaw_bounds=bounds[1],
            position_window_mm=(stop - start) / 2.0,
            allow_mirror=False,
            position_step_mm=SEARCH_STEP_MM,
            section_um_per_px=um_per_px,
        )
        if not math.isfinite(float(fit["score"])):
            raise ValueError("no atlas plane in the searched range can hold the section")
    except Exception as exc:
        logging.getLogger(__name__).warning("search_position failed for %s: %s", record.id, exc)
        raise Refused("FIT_FAILED", message=str(exc)) from exc
    current = record.position_mm
    return {
        "id": record.id,
        "current_position_mm": round(current, 3) if current is not None else None,
        "position_mm": round(float(fit["position_mm"]), 3),
        "pitch_deg": round(float(fit["pitch_deg"]), 3),
        "yaw_deg": round(float(fit["yaw_deg"]), 3),
        "score": round(float(fit["score"]), 4),
        "searched_window_mm": round((stop - start) / 2.0, 3),
        "searched_range_mm": [round(start, 3), round(stop, 3)],
        "searched_angles": bool(angles),
    }
