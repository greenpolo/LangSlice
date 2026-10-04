"""The read verbs: what each viewing tool shows, as plain pictures and data.

One function per viewing tool (``status``, ``view_slices``, ``view_atlas``,
``view_placement``, ``view_stack``), so a script or the CLI gets the very
picture the tool shows. Each takes the job (and the workspace) plus checked
arguments and the call's :class:`~langslice.core.display.DisplayOptions`,
writes nothing and takes no undo step. The pictures are the core's
(:mod:`langslice.core.pictures`, :mod:`langslice.core.placement`): captions
burned in, each noted with what it shows (:mod:`langslice.core.layers`), so
the job saves them with their layers whichever door shows them
(:meth:`langslice.job.views.ViewStore.shown`). The look-before-commit gates
and the delivery bookkeeping are the tool door's, never applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core import placement
from langslice.core.display import DisplayOptions, regions_in_plane
from langslice.core.pictures import atlas_view_picture, section_picture, stack_review
from langslice.core.sizes import MAX_IMAGES_PER_CALL
from langslice.core.state import SliceState
from langslice.core.status import stack_angles_entry, status_rows

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: Sections (or section-position pairs) one viewing call shows.
MAX_VIEW_SLICES = MAX_IMAGES_PER_CALL


# --- status ------------------------------------------------------------------------


@dataclass(frozen=True)
class StackStatus:
    """The stack as it stands: one row per section in corrected order
    (:func:`langslice.core.status.status_rows`), the cutting angles and the
    interval breaks. ``cutting_angles_deg`` is the stack's ``{"pitch",
    "yaw"}``, or ``"per section"`` when the sections' angles differ (each
    row then carries its own)."""

    rows: list[dict[str, Any]]
    cutting_angles_deg: dict[str, float] | str
    interval_breaks: list[int]


def status(job: Job) -> StackStatus:
    """The status table (``status``)."""
    state = job.state
    return StackStatus(rows=status_rows(state), cutting_angles_deg=stack_angles_entry(state),
                       interval_breaks=list(state.interval_breaks))


# --- sections and atlas sections ---------------------------------------------------


@dataclass(frozen=True)
class SectionsView:
    """``view_slices``: one picture per section shown, in order."""

    pictures: list[Image.Image] = field(default_factory=list)
    #: The sections pictured, in the pictures' order.
    shown: list[str] = field(default_factory=list)
    #: ``{"id", "message"}`` per section whose picture failed (only with
    #: ``keep_going``; otherwise the failure propagates).
    failed: list[dict[str, str]] = field(default_factory=list)
    #: Mode ``channels``: each section's raw channel names.
    channels: dict[str, list[str]] = field(default_factory=dict)


def view_slices(
    job: Job, workspace: Workspace, records: list[SliceState], options: DisplayOptions,
    *, keep_going: bool = False,
) -> SectionsView:
    """Each section as corrected, tissue-framed and labelled
    (:func:`langslice.core.pictures.section_picture`); mode ``channels``:
    its raw channels side by side. *keep_going* records a failed picture and
    goes on to the next section."""
    pictures: list[Image.Image] = []
    shown: list[str] = []
    failed: list[dict[str, str]] = []
    for record in records:
        try:
            pictures.append(section_picture(workspace, job.state, record, options))
        except Exception as exc:
            if not keep_going:
                raise
            failed.append({"id": record.id, "message": str(exc)})
            continue
        shown.append(record.id)
    channels = ({record.id: list(workspace.section_channels(record.id)[0]) for record in records}
                if options.mode == "channels" else {})
    return SectionsView(pictures=pictures, shown=shown, failed=failed, channels=channels)


def regions_not_in_plane(
    job: Job, workspace: Workspace, position_mm: float, options: DisplayOptions,
    angles: tuple[float, float],
) -> list[str]:
    """The call's highlighted regions with no pixel in the plane at
    *position_mm* and *angles* (a section's own; ``view_atlas``'s, the
    stack's view angles)."""
    present = regions_in_plane(workspace, job.state, position_mm, options, angles=angles)
    return [name for name, _ids in options.regions if name not in present]


@dataclass(frozen=True)
class AtlasView:
    """``view_atlas``: one atlas picture per position, in order."""

    pictures: list[Image.Image]
    #: Per position (``"<mm:.2f>"``), the highlighted regions not in its plane.
    regions_not_in_plane: dict[str, list[str]] = field(default_factory=dict)


def view_atlas(
    job: Job, workspace: Workspace, positions: list[float], options: DisplayOptions,
) -> AtlasView:
    """The atlas alone at each position (already clamped into the atlas
    range), at the stack's cutting angles (``StackState.view_angles``: the
    median of the sections' when they differ), tissue-framed and labelled."""
    state = job.state
    pictures = [atlas_view_picture(workspace, state, position, options)
                for position in positions]
    absent: dict[str, list[str]] = {}
    if options.regions:
        absent = {
            f"{position:.2f}": missing for position in positions
            if (missing := regions_not_in_plane(job, workspace, position, options,
                                                state.view_angles))
        }
    return AtlasView(pictures=pictures, regions_not_in_plane=absent)


# --- placements ------------------------------------------------------------------------


@dataclass(frozen=True)
class PairShown:
    """One section-position pair pictured."""

    record: SliceState
    position: float
    #: The pair's facts (:class:`langslice.core.placement.Placed` ``row``):
    #: calibration, the transform drawn, whether a deformation was drawn and,
    #: in ``side_by_side``, ``image_indexes`` (``section`` / ``atlas``).
    row: dict[str, Any]
    #: Where its pictures are in the call's list: indexes, or the
    #: ``side_by_side`` mapping.
    indexes: Any
    regions_not_in_plane: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlacementView:
    """``view_placement`` (and ``set_positions``'s pictures)."""

    pictures: list[Image.Image] = field(default_factory=list)
    shown: list[PairShown] = field(default_factory=list)
    #: ``(record, position, message)`` per pair whose pictures failed.
    failed: list[tuple[SliceState, float, str]] = field(default_factory=list)


def placement_view(
    job: Job, workspace: Workspace, pairs: list[tuple[SliceState, float]],
    options: DisplayOptions, *, regions_report: bool = True,
) -> PlacementView:
    """The pictures of each section-position pair in the call's placement
    mode (:func:`langslice.core.placement.placement_pictures`).

    ``side_by_side`` shows a section once per call however many positions it
    is shown at, and one atlas per pair. A pair whose pictures fail is listed
    in ``failed``, its pictures dropped; the others go on. *regions_report*
    lists, per pair, the highlighted regions not in its plane.
    """
    state = job.state
    pictures: list[Image.Image] = []
    shown: list[PairShown] = []
    failed: list[tuple[SliceState, float, str]] = []
    section_indexes: dict[str, int] = {}
    working: placement.Working = {}
    for record, position in pairs:
        first = len(pictures)
        try:
            placed = placement.placement_pictures(workspace, state, record, position, options,
                                                  working, store=job.deformations)
        except Exception as exc:
            failed.append((record, position, str(exc)))
            continue
        row = placed.row
        if placed.separate:
            tissue_image, atlas_image = placed.images
            if record.id not in section_indexes:
                section_indexes[record.id] = len(pictures)
                pictures.append(tissue_image)
            row["image_indexes"] = {"section": section_indexes[record.id],
                                    "atlas": len(pictures)}
            pictures.append(atlas_image)
        else:
            pictures.extend(placed.images)
        absent = (regions_not_in_plane(job, workspace, position, options, record.angles)
                  if regions_report else [])
        shown.append(PairShown(
            record=record, position=position, row=row,
            indexes=row.get("image_indexes", list(range(first, len(pictures)))),
            regions_not_in_plane=absent,
        ))
    return PlacementView(pictures=pictures, shown=shown, failed=failed)


def view_placement(
    job: Job, workspace: Workspace, pairs: list[tuple[SliceState, float]],
    options: DisplayOptions,
) -> PlacementView:
    """``view_placement``: each section at each position (already clamped into
    the atlas range) under its complete current registration, in the call's
    mode (:func:`placement_view`)."""
    return placement_view(job, workspace, pairs, options)


# --- the stack ------------------------------------------------------------------------


@dataclass(frozen=True)
class StackView:
    """``view_stack``: the contact sheet and the spacing plot."""

    sheet: Image.Image
    plot: Image.Image
    #: The status rows in written-position order (unplaced last).
    rows: list[dict[str, Any]]


def view_stack(job: Job, workspace: Workspace, options: DisplayOptions) -> StackView:
    """Every section in written-position order over its atlas match, and the
    spacing plot (:func:`langslice.core.pictures.stack_review`)."""
    state = job.state
    sheet, plot = stack_review(workspace, state, options)
    ordered = sorted(
        status_rows(state), key=lambda r: (r["position_mm"] is None, r["position_mm"] or 0.0)
    )
    return StackView(sheet=sheet, plot=plot, rows=ordered)
