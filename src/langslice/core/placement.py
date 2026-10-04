"""A section on its physical canvas at a placement: every placement picture.

The core of `view_placement`, `set_positions`, `fit_affine` and
`adjust_transforms`'s pictures, taken out of the tool door (layered refactor,
phase 3b). Plain inputs in (the workspace, the stack state, a section
record, numbers, the call's :class:`~langslice.core.display.DisplayOptions`),
plain PIL pictures with their captions burned in out, plus the metadata a
door words its reply from. No undo, no gates, no message types.

- :func:`draw_canvas` — the ONE renderer of a section at a placement on its
  millimetre-true canvas (:func:`langslice.core.canvas.physical_views`),
  drawn from the render the options ask for at the call's picture size. It
  returns a :class:`Canvas`: the panels and the :class:`CanvasFrame` they
  were drawn in (everything that fixes the canvas, kept so the job folder can
  later be given the picture's layers without re-deriving the geometry).
- :func:`stored_placement` — a section's stored in-plane transform as a
  drawable map; :func:`current_warp` — its applied deformation, when it
  still sits on its placement.
- :func:`placement_pictures` — one section-position pair in a placement mode
  (:data:`PLACEMENT_MODES`): the physical canvas under the complete current
  registration, the stacked picture, or the two separate references.
- :func:`stage` / :class:`Staged` / :func:`staged_views` — the interactive
  transform's section, its calibrated canvas and the resolved pivot, and its
  pictures at any knobs or matrix.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from PIL import Image

from langslice.core.affine import denormalized_affine, physical_affine_matrix
from langslice.core.canvas import (
    VIEW_MODES,
    CanvasGeometry,
    PanelFrame,
    canvas_geometry,
    physical_views,
    pivot_on_canvas,
)
from langslice.core.captions import caption
from langslice.core.display import (
    MODE_RULES,
    DisplayOptions,
    atlas_caption,
    atlas_image_picture,
    framed_atlas,
    framed_section,
)
from langslice.core.layers import note
from langslice.core.pictures import reference_atlas_picture, reference_section_picture
from langslice.core.sections import (
    PREVIEW_LONG_EDGE,
    render_slice,
    rescale_section_matrix,
    shown_section,
)
from langslice.core.sheets import stacked
from langslice.core.space import Plane
from langslice.core.state import IDENTITY_PARAMS, SliceState, StackState
from langslice.core.transform import calibrate
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: Placement pictures (``view_placement``, ``set_positions``): the physical
#: views plus ``stacked`` — the section over the atlas, each tissue-framed.
PLACEMENT_MODES = ("template", "stacked", *[m for m in VIEW_MODES if m != "template"])
#: Placement modes whose pictures are tissue-framed rather than drawn on the
#: physical canvas (no zoom, no stored placement drawn).
FRAMED_PLACEMENT_MODES = ("stacked", "side_by_side")
#: Placement modes that draw the section under its stored placement, where an
#: applied deformation can be drawn too.
WARPED_PLACEMENT_MODES = ("overlay", "checkerboard", "outlines", "section")


# --- the physical canvas --------------------------------------------------------


@dataclass(frozen=True)
class CanvasFrame:
    """Everything one physical picture's canvas is drawn from.

    The canvas itself is :func:`langslice.core.canvas.canvas_geometry` of
    (``section.size``, ``um_per_px``, the atlas, ``position_mm``, ``plane``,
    ``pitch_deg``, ``yaw_deg``); the section sits on it under ``params``
    (the knobs about ``pivot`` / ``pivot_in_section``, or a ready 2x3 on
    ``section``'s frame), or under ``spline``, then ``warp`` (an applied
    deformation record) resamples it; ``zoom`` crops the canvas and
    ``long_edge`` sizes each panel. Holding these, the section as placed, the
    atlas labels, the border mask and the pixel-to-atlas map can all be drawn
    in the same frame as the picture.
    """

    section_id: str
    #: The render the picture shows, before any warp (``shown_section``).
    section: Image.Image
    um_per_px: float
    position_mm: float
    plane: str
    pitch_deg: float
    yaw_deg: float
    params: dict[str, float] | np.ndarray
    pivot: tuple[float, float] | None
    pivot_in_section: tuple[float, float] | None
    spline: dict[str, Any] | None
    warp: Any
    zoom: list[float] | None
    long_edge: int


@dataclass(frozen=True)
class Canvas:
    """One placement's panels (captioned), the frame they were drawn in, and
    where each panel's pixels sit (:class:`~langslice.core.canvas.PanelFrame`,
    one per image: what :mod:`langslice.core.layers` computes the layers from)."""

    images: list[Image.Image]
    frame: CanvasFrame
    panels: list[PanelFrame] = field(default_factory=list)


def canvas_label(label: str, options: DisplayOptions) -> str:
    """A physical picture's caption head: what of the section, and any atlas under it."""
    under = ""
    if (options.atlas_images and options.atlas_opacity > 0
            and MODE_RULES[options.mode].opacity):
        under = f"  atlas {options.atlas_name()} under at {options.atlas_opacity:g}"
    return label + options.section_tag() + under


def draw_canvas(
    ws: Workspace,
    state: StackState,
    record: SliceState,
    section: Any,
    um_per_px: float,
    position: float,
    params: dict[str, float] | np.ndarray,
    options: DisplayOptions,
    *,
    mode: str | None = None,
    pivot: tuple[float, float] | None = None,
    section_offset: tuple[int, int] = (0, 0),
    label: str = "",
    spline: dict[str, Any] | None = None,
    long_edge: int | None = None,
    matrix_label: str = "fitted matrix",
    warp: Any = None,
    left: np.ndarray | None = None,
) -> Canvas:
    """The physical canvas pictures of one section at one placement.

    *section* is the working frame a transform is computed on; the
    picture is drawn from the render the options ask for (the view
    appearance or a raw channel), large enough that each panel, zoom
    included, comes out at *long_edge* (None: ``options.long_edge``)
    unless the section's working copy has fewer pixels, with a matrix and
    the pivot carried onto it. The one renderer of every placement
    picture: `view_placement`, `set_positions`, `adjust_transforms` and
    `fit_affine` all draw through here. *warp* (a `DeformableRecord` on
    this placement) resamples the section into its placed-atlas frame
    first, so the picture shows the full registration: linear placement
    plus deformation. *left* is a fit's own side split for one-sided
    regions (:class:`~langslice.core.transform.FitFrame`).
    """
    edge = int(long_edge or options.long_edge)
    window = options.window
    # The canvas is at least the section on each axis, so a section
    # render 1/span times the panel puts at least the panel's pixels
    # inside the zoom (render_slice stops at the working copy).
    span = (min(max(window[2] - window[0], 1e-3), max(window[3] - window[1], 1e-3), 1.0)
            if window else 1.0)
    render_edge = int(math.ceil(edge / span))
    if render_edge > PREVIEW_LONG_EDGE:
        # Past the working copy every request is the same render: one
        # cache entry, however deep the zoom.
        render_edge = min(render_edge, max(ws.working_source(record.id)[0].size))
    shown, shown_um, (fx, fy) = shown_section(
        ws, record, section, um_per_px, options.look(state, record),
        long_edge=render_edge,
    )
    in_section: tuple[float, float] | None = None
    if shown is not section:
        if isinstance(params, np.ndarray):
            params = rescale_section_matrix(params, fx, fy)
        if pivot is not None:
            ox, oy = section_offset
            in_section = ((pivot[0] - ox) * fx, (pivot[1] - oy) * fy)
    frame = CanvasFrame(
        section_id=record.id, section=shown, um_per_px=shown_um, position_mm=position,
        plane=state.plane, pitch_deg=record.pitch_deg, yaw_deg=record.yaw_deg, params=params,
        pivot=pivot if in_section is None else None, pivot_in_section=in_section,
        spline=spline, warp=warp, zoom=options.window, long_edge=edge,
    )
    if warp is not None:
        from langslice.core.deformable import warp_section_image

        shown = warp_section_image(shown, warp)
    panels: list[PanelFrame] = []
    drawn_mode = mode or options.mode
    images, _iou = physical_views(
        shown, shown_um, ws.atlas, position, cast(Plane, state.plane),
        record.pitch_deg, record.yaw_deg, params,
        mode=drawn_mode, zoom=options.window,
        atlas_opacity=options.atlas_opacity, outlines=options.layer,
        border_color=options.border_color, border_thickness=options.border_thickness,
        pivot=frame.pivot, pivot_in_section=in_section,
        label=canvas_label(label or record.id, options), spline=spline, long_edge=edge,
        atlas_picture=atlas_image_picture(ws, state, options.atlas_channels, position,
                                          angles=record.angles),
        atlas_name=options.atlas_name(), regions=options.regions,
        matrix_label=matrix_label, template_lines=options.borders, left=left,
        panel_frames=panels,
    )
    applied = (record.deformation or {}).get("record") if warp is not None else None
    for image, panel in zip(images, panels, strict=True):
        note(image, sections=(record.id,), mode=drawn_mode, frame=frame, panel=panel,
             deformation=applied if isinstance(applied, str) else None)
    return Canvas(images=images, frame=frame, panels=panels)


def stored_placement(record: SliceState, section: Any) -> tuple[Any, Any, str]:
    """``(params or matrix, spline, kind)`` of the section's in-plane transform.

    The six stored numbers are the exact map (shear included); a section
    without a transform is drawn at identity.
    """
    transform = record.transform or {}
    values = transform.get("params")
    if values is not None and len(values) == 6:
        return (denormalized_affine(values, section.size), transform.get("spline"),
                str(transform.get("kind") or "stored"))
    return dict(IDENTITY_PARAMS), None, "identity"


def current_warp(store: Any, state: StackState, record: SliceState) -> Any:
    """The section's applied deformation record from *store* (a
    :class:`~langslice.core.deformation.RecordStore`), or None (none held,
    stale, or unreadable)."""
    if not record.deformation:
        return None
    try:
        return store.current(state, record)
    except Exception:  # a missing record must not break a placement picture
        logger.warning("Deformation record unreadable for %s", record.id, exc_info=True)
        return None


# --- one section-position pair --------------------------------------------------


@dataclass
class Placed:
    """One section-position pair's pictures, in the call's placement mode.

    ``separate`` (mode ``side_by_side``): ``images`` is ``[section, atlas]``,
    two independently tissue-framed references, the section's shared by
    every pair of the same id. ``canvas``: the physical canvas, when the mode
    draws one. ``row``: the pair's facts (calibration, the transform drawn,
    whether a deformation was drawn).
    """

    images: list[Image.Image]
    row: dict[str, Any]
    separate: bool = False
    canvas: Canvas | None = None


#: Per call, each section's working frame: ``(render, um/px, calibration source)``.
Working = dict[str, tuple[Any, float, str]]


def placement_pictures(
    ws: Workspace,
    state: StackState,
    record: SliceState,
    position: float,
    options: DisplayOptions,
    working: Working,
    *,
    store: Any = None,
) -> Placed:
    """One section-position pair's pictures.

    ``side_by_side``: separate tissue-framed references, the section and the
    atlas (the doors map them by index). ``stacked``: one image, the framed
    section over the framed atlas. Every other mode: the physical canvas, the
    section under its complete current registration — the stored in-plane
    transform (identity when it has none) and, at the position it was fitted
    at, the applied deformation from *store* (unless ``view.deformation`` is
    ``none``). *working* caches each section's working frame for the call.
    """
    if record.id not in working:
        section = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
        working[record.id] = (section, *calibrate(state, ws, record, section))
    section, um_per_px, source = working[record.id]
    row: dict[str, Any] = {"calibration": {"um_per_px": round(um_per_px, 3), "source": source}}
    default_atlas = options.atlas_images == ("ara",) and not options.lines
    if options.mode == "side_by_side":
        atlas_image = (
            reference_atlas_picture(ws, state, position, long_edge=options.long_edge,
                                    angles=record.angles)
            if default_atlas else caption(
                framed_atlas(ws, state, position, options, angles=record.angles),
                atlas_caption(state, position, options, angles=record.angles),
            )
        )
        tissue_image = reference_section_picture(
            ws, record, long_edge=options.long_edge, look=options.look(state, record),
        )
        note(tissue_image, sections=(record.id,), mode="side_by_side",
             extra={"panel": "section"})
        note(atlas_image, sections=(record.id,), mode="side_by_side",
             extra={"panel": "atlas", "position_mm": float(position)})
        return Placed(images=[tissue_image, atlas_image], row=row, separate=True)
    if options.mode == "stacked":
        # One picture: the atlas is drawn to the section's long edge so
        # the two read at the same size, as in `view_stack`.
        top = framed_section(ws, state, record, options)
        picture = stacked(
            top, framed_atlas(ws, state, position, options, long_edge=max(top.size), fill=True,
                              angles=record.angles),
        )
        name = options.atlas_name()
        return Placed(images=[note(caption(
            picture, f"{record.id}{options.section_tag()} over atlas {position:.2f} mm"
            + ("" if name == "template" else f" ({name})"),
        ), sections=(record.id,), mode="stacked", extra={"position_mm": float(position)})],
            row=row)
    params, spline, kind = stored_placement(record, section)
    # The section under its full placement: the stored warp too, at the
    # position it was fitted at (the atlas-only view needs no section).
    warp = (current_warp(store, state, record)
            if store is not None and options.deformation == "applied"
            and position == record.position_mm and options.mode in WARPED_PLACEMENT_MODES
            else None)
    canvas = draw_canvas(
        ws, state, record, section, um_per_px, position, params, options,
        label=f"{record.id} vs atlas {position:.2f} mm",
        spline=spline,
        matrix_label=f"{kind} transform" + (" + deformation" if warp is not None else ""),
        warp=warp,
    )
    row["transform"] = kind
    if warp is not None:
        row["deformation_drawn"] = True
    elif (options.mode in WARPED_PLACEMENT_MODES and record.deformation
          and "keep_linear" not in record.deformation and position == record.position_mm
          and options.deformation == "applied"):
        # Held but not drawable (stale or unreadable): say so, never pretend.
        row["deformation_drawn"] = False
    return Placed(images=canvas.images, row=row, canvas=canvas)


# --- the interactive transform --------------------------------------------------


class StageFailure(ValueError):
    """A section that cannot be staged: ``code`` is ``ATLAS_RENDER_FAILED``
    (no atlas plane at its position) or ``BAD_PIVOT``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Staged:
    """One section ready to be drawn or measured on its physical canvas."""

    record: SliceState
    section: Any
    um_per_px: float
    calibration_source: str
    geometry: CanvasGeometry
    params: dict[str, float]
    #: Rotation/scale centre in CANVAS pixels; None is the canvas centre.
    pivot: tuple[float, float] | None
    #: The same pivot as fractions of the canvas, for the payload.
    pivot_frac: list[float] = field(default_factory=list)
    #: What was asked for: "canvas", "tissue" or "fractions".
    pivot_mode: str = "canvas"

    @property
    def pivot_in_section(self) -> tuple[float, float] | None:
        """The pivot on the SECTION's frame, which the six numbers live on."""
        if self.pivot is None:
            return None
        ox, oy = self.geometry.section_offset
        return (self.pivot[0] - ox, self.pivot[1] - oy)

    @property
    def calibration(self) -> dict[str, Any]:
        return {
            "section_um_per_px": round(self.um_per_px, 4),
            "source": self.calibration_source,
        }

    def matrix(self, params: dict[str, float] | None = None) -> Any:
        """The canvas-pixel 2x3 of *params* (default: the staged ones).

        Built on the canvas frame directly: width and height cancel out of the
        physical translation, so this is the very map the picture shows.
        """
        return physical_affine_matrix(
            size=self.geometry.size,
            um_per_px=self.um_per_px,
            pivot=self.pivot,
            **(params if params is not None else self.params),
        )


def stage(
    ws: Workspace, state: StackState, record: SliceState, params: dict[str, float], pivot: Any,
) -> Staged:
    """*record* (which has a position) with the knobs *params*, its working
    frame, calibration and canvas, and *pivot* ("canvas", "tissue" or
    ``[fx, fy]`` fractions of the canvas) resolved onto it.

    Raises :class:`StageFailure` when the atlas plane cannot be drawn there
    or the pivot is not one of those.
    """
    section = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
    um_per_px, source = calibrate(state, ws, record, section)
    try:
        geometry = canvas_geometry(
            section.size,
            um_per_px,
            ws.atlas,
            float(record.position_mm or 0.0),
            cast(Plane, state.plane),
            record.pitch_deg,
            record.yaw_deg,
        )
    except Exception as exc:
        logger.warning("transform: atlas render failed for %s: %s", record.id, exc)
        raise StageFailure("ATLAS_RENDER_FAILED", str(exc)) from exc
    try:
        point = pivot_on_canvas(pivot, section, geometry)
    except ValueError as exc:
        raise StageFailure("BAD_PIVOT", str(exc)) from exc
    width, height = geometry.size
    centre = point or (width / 2.0, height / 2.0)
    mode = "canvas"
    if isinstance(pivot, str):
        mode = (pivot.strip().lower() or "canvas")
    elif pivot:
        mode = "fractions"
    return Staged(
        record=record,
        section=section,
        um_per_px=um_per_px,
        calibration_source=source,
        geometry=geometry,
        params=params,
        pivot=point,
        pivot_frac=[round(centre[0] / width, 4), round(centre[1] / height, 4)],
        pivot_mode=mode,
    )


def staged_views(
    ws: Workspace,
    state: StackState,
    staged: Staged,
    params: dict[str, float] | np.ndarray,
    options: DisplayOptions,
    *,
    mode: str,
    pivot: tuple[float, float] | None,
    label: str = "",
    spline: dict[str, Any] | None = None,
) -> Canvas:
    """One staged section's canvas pictures at *params* (knobs or a 2x3 on
    its working frame) about *pivot* (canvas pixels)."""
    return draw_canvas(
        ws, state, staged.record, staged.section, staged.um_per_px,
        float(staged.record.position_mm or 0.0), params, options,
        mode=mode, pivot=pivot, section_offset=staged.geometry.section_offset,
        label=label, spline=spline,
    )


# --- the transform tools' pictures -------------------------------------------------


def fit_picture(
    ws: Workspace, state: StackState, record: SliceState, frame: Any, options: DisplayOptions,
) -> list[Image.Image]:
    """A fitted section under its new transform (``fit_affine``'s picture).

    Drawn from the fit's working frame and matrix (*frame*, a
    :class:`~langslice.core.transform.FitFrame`). One-sided regions are
    highlighted with the sides the fit resolved (``frame.left``), so the
    picture shows what the fit used even after a large turn.
    """
    return draw_canvas(
        ws, state, record, frame.section, frame.um_per_px,
        float(record.position_mm or 0.0), frame.matrix, options, label=record.id,
        left=frame.left,
    ).images


def ab_reference(
    staged: Staged, previous: dict[str, Any] | None,
) -> tuple[Any, tuple[float, float] | None]:
    """What an ``ab`` picture's B side draws: ``(params, pivot)``.

    The six numbers the section carried before the call, when it had them:
    they are the exact map, where the knobs (shear included) are rounded.
    Knobs alone are drawn as given, about their stored pivot; no transform
    at all is identity.
    """
    stored = (previous or {}).get("physical")
    before = (previous or {}).get("params")
    other: Any = dict(IDENTITY_PARAMS)
    if before is not None and len(before) == 6:
        other = denormalized_affine(before, staged.section.size)
    elif isinstance(stored, dict):
        other = {key: float(stored[key]) for key in IDENTITY_PARAMS}
        if stored.get("shear"):
            other["shear"] = float(stored["shear"])
    other_pivot = staged.pivot
    if isinstance(stored, dict) and stored.get("pivot"):
        fractions = [float(value) for value in stored["pivot"]]
        other_pivot = (
            fractions[0] * staged.geometry.size[0],
            fractions[1] * staged.geometry.size[1],
        )
    return other, other_pivot


def transform_views(
    ws: Workspace, state: StackState, staged: Staged, options: DisplayOptions,
    previous: dict[str, Any] | None,
) -> list[Image.Image]:
    """``adjust_transforms``'s pictures of one staged section.

    Mode ``ab``: two overlays at the same crop, the staged knobs (labelled
    "candidate") then what the section carried before the call, *previous*
    (labelled "stored"; "identity" when it had none, :func:`ab_reference`).
    Any other mode: the staged knobs in that mode.
    """
    record = staged.record
    if options.mode != "ab":
        return staged_views(ws, state, staged, staged.params, options, mode=options.mode,
                            pivot=staged.pivot).images
    other, other_pivot = ab_reference(staged, previous)
    held = previous is not None
    return staged_views(
        ws, state, staged, staged.params, options, mode="overlay", pivot=staged.pivot,
        label=f"{record.id} candidate",
    ).images + staged_views(
        ws, state, staged, other, options, mode="overlay", pivot=other_pivot,
        label=f"{record.id} {'stored' if held else 'identity'}",
        spline=(previous or {}).get("spline"),
    ).images
