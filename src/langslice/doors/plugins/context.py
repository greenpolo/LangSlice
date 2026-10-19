"""The context a plugin tool's handler gets: what it may read and write.

One :class:`ToolContext` per call (a plugin must not keep it). Reads give
plain data in LangSlice's documented units and frames; writes go through the
same operations the native tools use (``ops.positions``, ``ops.transforms``,
``ops.deformable``, ``ops.damage``), so every write is one undo step that checkpoints the state
and rewrites ``registration.json``, frozen properties are refused (``FROZEN``),
and a long tool's section that changed meanwhile is refused (``STALE_INPUT``).
A tool declared ``read`` cannot write; a ``write`` tool writes the section
properties named in its :attr:`~langslice.core.plugins.ToolSpec.writes` only.
Files go under :meth:`ToolContext.work_dir` and nowhere else.
"""

from __future__ import annotations

import contextlib
import copy
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core import frozen as freeze
from langslice.core.plugins import Layer, ToolError, ToolResult, ToolSpec
from langslice.core.sizes import PICTURE_EDGES
from langslice.core.state import SliceState, StackState
from langslice.ops.inputs import STALE_INPUT, section_inputs, stale_row
from langslice.ops.refusal import Refused, frozen_row, unknown_sections

if TYPE_CHECKING:
    from langslice.doors.tools.door import Door


# --- what a handler reads ---------------------------------------------------------------


@dataclass(frozen=True)
class AffineView:
    """A section's in-plane transform: ``kind`` (``interactive``, ``elastix``,
    ``plugin:<tool>``...), the six normalized numbers ``[a, b, tx, c, d, ty]``
    (``docs/file_formats.md``), the knobs they are in (``rotation_deg``,
    ``scale_x``, ``scale_y``, ``shear``, ``translate_x_mm``, ``translate_y_mm``,
    ``pivot``), and whether it mirrors the section."""

    kind: str | None
    matrix: tuple[float, ...] | None
    knobs: dict[str, Any] | None
    mirrored: bool | None


@dataclass(frozen=True)
class SectionView:
    """One section as the job holds it now."""

    id: str
    #: Its place in the stack (0-based, in corrected order).
    index: int
    #: The image file's size in pixels, ``(width, height)``.
    file_size: tuple[int, int]
    pixel_size_um: float | None
    #: ``file``, ``host`` or ``""`` (neither says).
    pixel_size_source: str
    position_mm: float | None
    #: ``(pitch_deg, yaw_deg)`` of its atlas plane.
    angles: tuple[float, float]
    flip: bool
    rotation_deg: int
    affine: AffineView | None
    #: The applied deformation's summary (``key``, ``steps``...), or None.
    deformation: dict[str, Any] | None
    damaged_regions: tuple[str, ...]
    damage_note: str
    #: Names of the layers it shows.
    layers: tuple[str, ...]
    #: The properties frozen on it (``langslice.core.frozen.PROPERTIES``).
    frozen: tuple[str, ...]


@dataclass(frozen=True)
class AtlasView:
    name: str
    #: Voxel size in micrometres along the atlas's axes.
    resolution_um: tuple[float, ...]
    orientation: str
    species: str
    plane: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class FitFrame:
    """A section and the atlas plane it is placed on, on the grid LangSlice's
    fits work on.

    ``image`` is the section as its orientation sets it, at its current plane,
    with no in-plane transform (RGB, ``uint8``).
    ``atlas`` holds the plane at the section's position and cutting angles on
    the native atlas grid: ``annotation`` (region ids) and the atlas's images
    asked for (``template`` and any additional reference the BrainGlobe atlas
    ships, by its name). ``atlas_to_section`` (3x3 on ``[x, y, 1]``)
    maps native atlas pixel centres to ``image`` pixel centres. ``plane`` says
    where the native plane lies in atlas micrometres: ``center_um``, ``dx``,
    ``dy`` (the column and row directions), ``px_um``, ``shape`` and
    ``resolution_um``. ``placed_atlas_to_section`` is the same map under the
    section's stored in-plane transform (its current linear placement, where an
    engine that refines it starts), None while it has no transform.
    ``excluded`` is the native plane's pixels of the section's marked damage
    regions (sides included), the regions every native fit leaves out; None
    while the section has none marked. ``damage_zone`` is ``excluded`` widened
    by the margin the native fits also leave out of their comparison
    (:data:`langslice.core.deformable.masks.EXCLUSION_MARGIN_MM`, scaled to the
    atlas's brain size); None with ``excluded``. ``stain`` is the section's
    preprocessed channel on the same grid as ``image`` (RGB ``uint8``), the
    picture the native fits read; None unless asked for.
    """

    id: str
    image: np.ndarray
    atlas: dict[str, np.ndarray]
    atlas_to_section: np.ndarray
    plane: dict[str, Any]
    position_mm: float
    angles: tuple[float, float]
    placed_atlas_to_section: np.ndarray | None = None
    applied_map: np.ndarray | None = None
    excluded: np.ndarray | None = None
    damage_zone: np.ndarray | None = None
    stain: np.ndarray | None = None


@dataclass(frozen=True)
class TransformProposal:
    """A transform a method found, in the forms LangSlice stores it:
    ``params`` are the six normalized numbers, ``knobs`` the same transform as
    ``interactive_transform`` takes it about ``pivot``, ``record`` the stored
    record. :meth:`row` is a row for :meth:`ToolContext.set_transforms`."""

    id: str
    params: tuple[float, ...]
    knobs: dict[str, float]
    pivot: tuple[float, float]
    mirrored: bool
    calibration: dict[str, Any]
    record: dict[str, Any]

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "transform": self.record}


@dataclass(frozen=True)
class DeformationProposal:
    """A deformation a method found, as LangSlice stores it, not written yet
    (:meth:`ToolContext.deformation_from_map`): ``summary`` holds its numbers
    (``displacement_mm`` max and median over the placed atlas footprint, in
    millimetres, ``fold_fraction``, ``flags``) and ``inverse_consistency_mm``
    how well the inverse LangSlice computed undoes it. :meth:`row` is a row
    for :meth:`ToolContext.set_deformations`;
    ``overlay_picture(id, proposal={"deformation": proposal})`` draws it."""

    id: str
    key: str
    summary: dict[str, Any]
    inverse_consistency_mm: dict[str, float]
    _warp: Any = field(repr=False, compare=False)

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "deformation": self}


@dataclass(frozen=True)
class SectionRaster:
    """A section image together with its frame.

    ``array`` is the section as its orientation sets it (quarter turn, then
    left-right flip) and at most the asked size: its pixels are NOT the image
    file's. ``section_px`` (what layers' points and lines are in) is the file's
    own pixels, unturned, unflipped and full size. :meth:`to_section_px` and
    :meth:`from_section_px` convert coordinates both ways;
    :meth:`mask_to_section_px` turns a raster computed on ``array`` into one in
    the file's orientation (a raster covers the whole file whatever its size, so
    needs no scaling); ``Layer.labels(mask, frame="oriented")`` does that when
    the layer is added.
    """

    id: str
    array: np.ndarray
    #: The image file's size, ``(width, height)``.
    file_size: tuple[int, int]
    rotation_deg: int
    flip: bool

    def _matrix(self) -> np.ndarray:
        """3x3: oriented edge fractions to file edge fractions."""
        from langslice.core.landmarks import affine_from_points
        from langslice.core.native import unoriented_fractions

        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        x, y = unoriented_fractions(corners[:, 0], corners[:, 1], self.rotation_deg, self.flip)
        return affine_from_points(np.stack([x, y], axis=1))

    def to_section_px(self, xy: Any) -> np.ndarray:
        """``[x, y]`` pixels of ``array`` as ``section_px`` (file pixels)."""
        from langslice.core.landmarks import transform

        height, width = self.array.shape[:2]
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array([width, height])
        theirs = transform(self._matrix(), fractions)
        return theirs * np.array(self.file_size) - 0.5

    def from_section_px(self, xy: Any) -> np.ndarray:
        """``section_px`` (file pixels) as ``[x, y]`` pixels of ``array``."""
        from langslice.core.landmarks import transform

        height, width = self.array.shape[:2]
        fractions = (np.asarray(xy, dtype=np.float64) + 0.5) / np.array(self.file_size)
        return transform(np.linalg.inv(self._matrix()), fractions) * np.array(
            [width, height]) - 0.5

    def mask_to_section_px(self, raster: Any) -> np.ndarray:
        """A raster in ``array``'s orientation as one in the file's."""
        return unorient_raster(np.asarray(raster), self.rotation_deg, self.flip)


def unorient_raster(raster: np.ndarray, rotation_deg: int, flip: bool) -> np.ndarray:
    """*raster*, drawn as a section's orientation sets it, in the file's orientation."""
    from langslice.core.native import unorient

    kind = "F" if np.issubdtype(raster.dtype, np.floating) else "I"
    image = Image.fromarray(raster.astype(np.float32 if kind == "F" else np.int32), mode=kind)
    return np.asarray(unorient(image, rotation_deg, flip)).astype(raster.dtype)


@dataclass(frozen=True)
class LayerRef:
    """A layer registered on a section."""

    id: str
    section: str
    name: str
    kind: str
    frame: str
    file: str


@dataclass
class WriteResult:
    """What a write did. ``ok`` is False when the call was refused whole
    (``error`` is the code, ``message`` the words; nothing was written);
    otherwise ``rows`` has one row per section asked, ``status`` ``ok`` or
    ``error`` with the code (``FROZEN``, ``STALE_INPUT``, ``UNKNOWN_SLICE_IDS``,
    ...), and ``written`` the ids written (one undo step)."""

    ok: bool = True
    error: str | None = None
    message: str = ""
    written: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def refused(cls, refusal: Refused) -> WriteResult:
        return cls(ok=False, error=refusal.code, message=str(refusal),
                   details={key: value for key, value in refusal.details.items()
                            if key != "message"})

    def reply(self) -> dict[str, Any]:
        """The result as reply fields: ``written`` and ``rows``, or the refusal."""
        if not self.ok:
            return {"status": "error", "error": self.error, "message": self.message,
                    **self.details}
        return {"written": list(self.written), "rows": [dict(row) for row in self.rows]}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- the context -------------------------------------------------------------------------


class ToolContext:
    """What one plugin tool call can read and write (see the module text)."""

    def __init__(self, door: Door, tool: ToolSpec, call: str) -> None:
        self._door = door
        self._tool = tool
        #: The call's id, as the layers it makes record it.
        self.call = call
        self._grids: dict[str, Any] = {}
        self._depth = 0  # open ``writing()`` blocks
        #: The call's undo step so far: the mark it began at, ``job.pushed``
        #: after it and the undo entry on top then; None before any write.
        self._step: tuple[int, int, dict[str, Any]] | None = None

    # --- the job ----------------------------------------------------------------------

    @property
    def tool(self) -> str:
        """The tool's name."""
        return self._tool.name

    @property
    def spec(self) -> Mapping[str, Any]:
        """The job's spec (``JobSpec.to_dict``), read-only."""
        return types.MappingProxyType(copy.deepcopy(self._door.spec.to_dict()))

    @property
    def job_dir(self) -> Path:
        """The job folder."""
        return Path(self._door.job.folder)

    def work_dir(self) -> Path:
        """``<job>/plugins/<tool>/``: the one place this tool writes files
        (scratch, logs, an engine's inputs and outputs). Made on first use."""
        path = self._door.job.layout.plugin_dir(self._tool.name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_on(self, task: str) -> bool:
        """Whether the job's *task* (``position``, ``transform``, ``nonlinear``) is on."""
        return bool(self._door.spec.has(task))

    # --- reading ------------------------------------------------------------------------

    def _record(self, section: object) -> SliceState:
        state = self._door.state
        record = state.resolve(section)
        if record is None:
            raise ToolError("UNKNOWN_SLICE_IDS", **unknown_sections(state, [str(section)]).details)
        return record

    def _view(self, record: SliceState) -> SectionView:
        from langslice.core.maps import render_sizes

        workspace = self._door.ctx
        transform = record.transform or {}
        affine = None
        if transform:
            params = transform.get("params")
            affine = AffineView(
                kind=transform.get("kind"),
                matrix=tuple(float(v) for v in params) if params else None,
                knobs=copy.deepcopy(transform.get("physical")) if transform.get("physical")
                else None,
                mirrored=transform.get("mirrored"))
        size = render_sizes(workspace, record.id).file_size
        pixel_size, source = workspace.calibration(record.id)
        held = record.deformation
        return SectionView(
            id=record.id, index=int(record.index_corrected), file_size=(size[0], size[1]),
            pixel_size_um=pixel_size, pixel_size_source=source,
            position_mm=record.position_mm, angles=record.angles, flip=bool(record.flip),
            rotation_deg=int(record.rotation_deg), affine=affine,
            deformation=None if held is None else {
                key: copy.deepcopy(held[key]) for key in ("key", "kind", "steps")
                if key in held},
            damaged_regions=tuple(record.damaged_regions), damage_note=record.damage_note,
            layers=tuple(str(ref["name"]) for ref in record.layers),
            frozen=tuple(name for name in freeze.PROPERTIES
                         if name in self._door.spec.frozen_for(record.id)))

    def sections(self) -> list[SectionView]:
        """Every section, in stack order."""
        return [self._view(record) for record in self._door.state.in_order()]

    def section(self, section: object) -> SectionView:
        """One section by filename (the extension may be left off)."""
        return self._view(self._record(section))

    def atlas(self) -> AtlasView:
        """The job's atlas."""
        atlas = self._door.ctx.atlas
        return AtlasView(
            name=str(self._door.state.atlas),
            resolution_um=tuple(float(v) for v in np.asarray(atlas.resolution, dtype=float)),
            orientation=str(getattr(atlas, "orientation", "")),
            species=self._door.ctx.species, plane=str(self._door.state.plane),
            shape=tuple(int(v) for v in np.shape(atlas.annotation)))

    def section_inputs(self, section: object, *, deformation: bool = False) -> str:
        """A digest of what a computation on the section reads (its placement,
        appearance and damage marks; with *deformation* its deformation too).
        Take it before a long computation and hand it back as ``expect=`` to a
        write: a section that changed meanwhile comes back ``STALE_INPUT``."""
        return section_inputs(self._door.state, self._record(section), deformation=deformation)

    def section_image(self, section: object, *, level: str | int = "auto",
                      channels: Sequence[str] | None = None) -> np.ndarray:
        """The pixels of :meth:`section_raster`. Their frame is NOT the image
        file's (turned and flipped as the orientation sets it, and reduced):
        convert coordinates with :meth:`section_raster`'s methods before they
        go in a ``section_px`` layer."""
        return self.section_raster(section, level=level, channels=channels).array

    def section_raster(self, section: object, *, level: str | int = "auto",
                       channels: Sequence[str] | None = None) -> SectionRaster:
        """The section as its orientation sets it (RGB ``uint8`` ``array``,
        never larger than the source) with the frame it is in
        (:class:`SectionRaster`). *level* is the long edge in pixels, or ``low`` /
        ``medium`` / ``high`` / ``auto`` (the job's picture size). *channels*:
        raw channel names, or ``["preprocessed"]`` for the channel the fits
        read; None is the default appearance."""
        from langslice.core.appearance import preprocessed_settings
        from langslice.core.channels import with_properties
        from langslice.core.maps import render_sizes
        from langslice.core.sections import render_slice
        from langslice.core.sizes import picture_edge

        record = self._record(section)
        if isinstance(level, int):
            edge = int(level)
        elif level == "auto":
            edge = picture_edge(self._door.ctx)
        elif level in PICTURE_EDGES:
            edge = PICTURE_EDGES[level][1]
        else:
            raise ToolError("BAD_ARGS", f"level is a pixel size or one of "
                            f"{', '.join(PICTURE_EDGES)}; got {level!r}")
        look: Any = None
        if channels:
            names = [str(name) for name in channels]
            look = (preprocessed_settings(self._door.state, record.id)
                    if names == ["preprocessed"]
                    else with_properties(self._door.state, {"overlay": names}))
        image = render_slice(self._door.ctx, record, long_edge=edge, look=look)
        return SectionRaster(
            id=record.id, array=np.asarray(image.convert("RGB")),
            file_size=tuple(render_sizes(self._door.ctx, record.id).file_size),  # type: ignore[arg-type]
            rotation_deg=int(record.rotation_deg), flip=bool(record.flip))

    def _grid(self, record: SliceState) -> Any:
        from langslice.core import deformation
        from langslice.core.affine import IDENTITY_PARAMS

        if record.position_mm is None:
            raise ToolError("NO_POSITION", f"{record.id} has no position yet")
        if record.id not in self._grids:
            try:
                self._grids[record.id] = deformation.fit_grid(
                    self._door.state, self._door.ctx, record,
                    transform={"params": list(IDENTITY_PARAMS)})
            except ValueError as exc:
                raise ToolError("INVALID_LINEAR_PLACEMENT", f"{record.id}: {exc}") from exc
        return self._grids[record.id]

    def fit_frame(self, section: object, *, atlas: Sequence[str] = (
            "template", "annotation"), stain: bool = False) -> FitFrame:
        """The section and its atlas plane on the fit grid (:class:`FitFrame`).
        Needs a position. *atlas* picks the atlas planes to sample:
        ``annotation`` and any of the atlas's images
        (:func:`langslice.core.atlas.images.atlas_images`); a name the atlas
        does not have is skipped. *stain* adds the section's preprocessed
        channel on the grid."""
        from langslice.core import deformation
        from langslice.core.atlas.images import atlas_images
        from langslice.core.atlas.size import brain_scale
        from langslice.core.damage import exclusions
        from langslice.core.deformable.masks import EXCLUSION_MARGIN_MM, dilate
        from langslice.core.deformable.plane_images import (
            native_intensity,
            native_labels,
            placement_left,
            regions_mask,
        )

        record = self._record(section)
        grid = self._grid(record)
        image = grid.image.convert("RGB")
        placement = grid.placement
        workspace = self._door.ctx
        planes: dict[str, np.ndarray] = {}
        if "annotation" in atlas:
            planes["annotation"] = np.asarray(native_labels(workspace.atlas, placement))
        for kind in atlas_images(workspace.atlas):
            if kind in atlas:
                planes[kind] = np.asarray(native_intensity(workspace.atlas, placement, kind),
                                          dtype=np.float32)
        placed = None
        if record.transform:
            try:
                placed = np.asarray(deformation.fit_grid(
                    self._door.state, workspace, record).placement.atlas_to_section,
                    dtype=np.float64)
            except ValueError:  # a stored transform no fit can use (stale, turned over)
                placed = None
        applied = None
        if record.deformation:
            from langslice.core.deformable.render import composed_native_grid

            held = self._door.job.deformations.current(self._door.state, record)
            if held is not None and tuple(held.section_size) == image.size:
                applied = np.stack(composed_native_grid(held, 1), axis=-1)
        excluded = zone = None
        _, marked = exclusions(record)
        if marked:
            labels = planes.get("annotation")
            if labels is None:
                labels = np.asarray(native_labels(workspace.atlas, placement))
            excluded = regions_mask(workspace.atlas, labels, marked,
                                    placement_left(workspace.atlas, placement, marked))
            spacing_mm = float(np.mean(_plane_frame(workspace.atlas, placement)["px_um"])) / 1000.0
            zone = dilate(excluded, EXCLUSION_MARGIN_MM * brain_scale(workspace.atlas)
                          / spacing_mm)
        return FitFrame(
            id=record.id, image=np.asarray(image), atlas=planes,
            atlas_to_section=np.asarray(placement.atlas_to_section, dtype=np.float64),
            plane=_plane_frame(workspace.atlas, placement),
            position_mm=float(placement.position_mm),
            angles=(float(placement.pitch_deg), float(placement.yaw_deg)),
            placed_atlas_to_section=placed, applied_map=applied, excluded=excluded,
            damage_zone=zone,
            stain=(np.asarray(deformation.stain_image(
                workspace, self._door.state, grid, deformation.FIT_LOOK)[0].convert("RGB"))
                if stain else None))

    def angles_from_plane(self, normal_um: Sequence[float], point_um: Sequence[float],
                          ) -> dict[str, float]:
        """The placement of the plane through *point_um* with normal
        *normal_um* (atlas micrometres, the atlas's axis order, as
        :attr:`FitFrame.plane` gives a plane): ``position_mm`` (where it crosses
        the volume's in-plane centre, the stack's position convention),
        ``pitch_deg`` and ``yaw_deg`` (the stack's cutting-angle convention).
        Nothing is written. ``ToolError`` ``BAD_PLANE`` for a plane tilted
        past what a cutting angle can say."""
        from langslice.core.import_geometry import plane_angles, plane_position_mm
        from langslice.core.space import Plane

        atlas = self._door.ctx.atlas
        plane = cast(Plane, self._door.state.plane)
        normal = np.asarray(normal_um, dtype=np.float64)
        point = np.asarray(point_um, dtype=np.float64)
        if normal.shape != (3,) or point.shape != (3,) or not (
                np.isfinite(normal).all() and np.isfinite(point).all()) or not normal.any():
            raise ToolError("BAD_PLANE", "normal_um and point_um are three finite numbers, "
                            "the normal not zero")
        try:
            pitch, yaw = plane_angles(normal, atlas, plane)
        except ValueError as exc:
            raise ToolError("BAD_PLANE", str(exc)) from exc
        return {"position_mm": float(plane_position_mm(point, normal, atlas, plane)),
                "pitch_deg": float(pitch), "yaw_deg": float(yaw)}

    def atlas_slice(self, section: object, *, kind: str = "template") -> np.ndarray:
        """The atlas plane at the section's position and cutting angles, on
        the native atlas grid (:class:`FitFrame` ``atlas``): *kind* is
        ``annotation`` or one of the atlas's images."""
        from langslice.core.atlas.images import atlas_images

        offered = ("annotation", *atlas_images(self._door.ctx.atlas))
        if kind not in offered:
            raise ToolError("BAD_ARGS", f"kind is one of {', '.join(offered)}; got {kind!r}")
        frame = self.fit_frame(section, atlas=(kind,))
        return frame.atlas[kind]

    def transform_from_matrix(self, section: object, matrix: Any, *,
                              note: str = "") -> TransformProposal:
        """A method's linear result as a transform LangSlice stores: *matrix*
        is the 2x3 map from the fit grid's pixels (:attr:`FitFrame.image`) to
        native atlas pixels (:attr:`FitFrame.atlas`), the form registration
        engines return. The result is a refinement of the section's current
        placement exactly as an Elastix fit is, in the six normalized numbers
        and the knobs about the section's stored pivot (else the canvas
        centre). ``ToolError`` ``BAD_MATRIX`` for a degenerate map."""
        from langslice.core.affine import IDENTITY_PARAMS, denormalized_affine, normalized_affine

        record = self._record(section)
        grid = self._grid(record)
        width, height = grid.image.size
        m = np.vstack([np.asarray(matrix, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
        if not np.isfinite(m).all() or abs(np.linalg.det(m[:2, :2])) < 1e-9:
            raise ToolError("BAD_MATRIX", "the map is degenerate or not finite")
        start = np.vstack([denormalized_affine(list(IDENTITY_PARAMS), (width, height)),
                           [0.0, 0.0, 1.0]])
        step = np.asarray(grid.placement.atlas_to_section, dtype=np.float64) @ m
        params = normalized_affine((start @ step)[:2], (width, height))
        return self.transform_from_params(record.id, params, note=note)

    def transform_from_params(self, section: object, params: Sequence[float], *,
                              note: str = "") -> TransformProposal:
        """The stored transform for the six normalized numbers *params*
        (``[a, b, tx, c, d, ty]``, on the section as its orientation sets it):
        the knobs about the section's stored pivot (else the canvas centre),
        its calibration and whether it mirrors the section."""
        from langslice.core.affine import decompose_affine, denormalized_affine
        from langslice.core.canvas import canvas_geometry
        from langslice.core.sections import PREVIEW_LONG_EDGE, render_slice
        from langslice.core.transform import calibrate, physical_params

        record = self._record(section)
        values = [float(v) for v in params]
        if len(values) != 6 or not np.isfinite(values).all():
            raise ToolError("BAD_ARGS", "a transform is six finite numbers [a, b, tx, c, d, ty]")
        if record.position_mm is None:
            raise ToolError("NO_POSITION", f"{record.id} has no position yet")
        workspace, state = self._door.ctx, self._door.state
        rendered = render_slice(workspace, record, long_edge=PREVIEW_LONG_EDGE)
        um_per_px, source = calibrate(state, workspace, record, rendered)
        try:
            geometry = canvas_geometry(rendered.size, um_per_px, workspace.atlas,
                                       record.position_mm, state.plane,  # type: ignore[arg-type]
                                       record.pitch_deg, record.yaw_deg)
        except ValueError as exc:
            raise ToolError("INVALID_LINEAR_PLACEMENT", f"{record.id}: {exc}") from exc
        in_section = denormalized_affine(values, rendered.size)
        ox, oy = (float(v) for v in geometry.section_offset)
        shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
        back = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
        on_canvas = (shift @ np.vstack([in_section, [0.0, 0.0, 1.0]]) @ back)[:2]
        canvas_w, canvas_h = geometry.size
        held_pivot = ((record.transform or {}).get("physical") or {}).get("pivot")
        pivot = (float(held_pivot[0]), float(held_pivot[1])) if held_pivot else (0.5, 0.5)
        knobs = physical_params(on_canvas, pivot=(canvas_w * pivot[0], canvas_h * pivot[1]),
                                um_per_px=um_per_px)
        mirrored = bool(decompose_affine(values)["mirrored"])
        calibration = {"section_um_per_px": round(float(um_per_px), 4), "source": source}
        stored = {
            "kind": f"plugin:{self._tool.name}", "params": values,
            "physical": {**knobs, "pivot": list(pivot)}, "calibration": calibration,
            "mirrored": mirrored, "note": str(note or "").strip(),
        }
        return TransformProposal(id=record.id, params=tuple(values), knobs=dict(knobs),
                                 pivot=pivot, mirrored=mirrored, calibration=calibration,
                                 record=stored)

    def deformation_from_map(self, section: object, native_xy: Any, *, note: str = "",
                             meta: Mapping[str, Any] | None = None) -> DeformationProposal:
        """A method's deformable result as the deformation LangSlice stores:
        *native_xy* is a float ``(H, W, 2)`` array giving, for each pixel
        centre ``(x, y)`` of the fit grid (:attr:`FitFrame.image`, ``H`` by
        ``W``), the native atlas pixel ``(x, y)`` (:attr:`FitFrame.atlas`) it
        corresponds to: the whole map, linear placement and deformation
        composed, in the frames :meth:`transform_from_matrix` takes. It is
        stored as every deformation is, the residual over the section's
        current linear placement, with an inverse LangSlice computes; it
        replaces whatever deformation the section holds. *note* and *meta*
        (JSON values) are kept with it. ``ToolError`` ``BAD_FIELD`` for an
        array of another shape, not float or not finite;
        ``INVALID_LINEAR_PLACEMENT`` for a section without a position or a
        transform."""
        import json

        from langslice.ops import deformable

        record = self._record(section)
        plain_meta = _plain(dict(meta or {}))
        try:
            json.dumps(plain_meta)
        except (TypeError, ValueError) as exc:
            raise ToolError("BAD_ARGS", f"meta is JSON values: {exc}") from exc
        text = str(note or "").strip()
        try:
            warp = deformable.warp_from_map(
                self._door.job, self._door.ctx, record.id, native_xy,
                method=f"plugin:{self._tool.name}", step={"note": text} if text else {},
                provenance={"plugin": {"tool": self._tool.name, "call": self.call,
                                       "version": self._tool.version, "note": text,
                                       "meta": plain_meta},
                            "input": "full map: fit-grid pixel centres to native atlas "
                                     "pixels"})
        except Refused as refusal:
            raise ToolError(refusal.code, str(refusal), **{
                key: value for key, value in refusal.details.items()
                if key != "message"}) from refusal
        consistency = warp.record.diagnostics.get("inverse_consistency_mm") or {}
        return DeformationProposal(
            id=record.id, key=warp.key, summary=copy.deepcopy(warp.numbers),
            inverse_consistency_mm={k: round(float(v), 4) for k, v in consistency.items()},
            _warp=warp)

    def read_registration(self, path: str | Path, *, sections: Sequence[str] = (),
                          ) -> dict[str, Any]:
        """A linear registration file made elsewhere (QuickNII or VisuAlign
        JSON/XML, DeepSlice CSV/JSON/XML, a LangSlice ``registration.json``)
        as placements for this job's sections: ``{"placements": [{"id",
        "position_mm", "pitch_deg", "yaw_deg"}], "refused": {id: reason},
        "missing": [id]}``. Nothing is written."""
        from langslice.job.imports import import_placements

        state = self._door.state
        names = [self._record(ref).id for ref in sections] or [r.id for r in state.slices]
        result = import_placements(
            Path(path), self._door.ctx, section_ids=names,
            orientations={r.id: (r.rotation_deg, r.flip) for r in state.slices})
        return {
            "placements": [{"id": p.section_id, "position_mm": round(float(p.position_mm), 4),
                            "pitch_deg": round(float(p.pitch_deg), 3),
                            "yaw_deg": round(float(p.yaw_deg), 3)}
                           for p in result.placements],
            "refused": dict(result.refused), "missing": list(result.missing)}

    # --- output for the agent ---------------------------------------------------------------

    def overlay_picture(self, section: object, *, proposal: Mapping[str, Any] | None = None,
                        resolution: int | None = None) -> Image.Image:
        """The overlay ``look`` draws of the section (borders on the section,
        captioned), as it stands, or with a *proposal* not yet written:
        ``{"transform": record or params, "position_mm": mm, "angles":
        (pitch, yaw)}``, any of them, or ``{"deformation": proposal or map}``
        alone (a :class:`DeformationProposal`, or the map
        :meth:`deformation_from_map` takes), drawn on the section's current
        placement. Nothing is saved or written here; the picture is the
        handler's to return."""
        from langslice.core import look as looks

        record = self._record(section)
        state = self._door.state
        store = self._door.job.deformations
        if proposal and proposal.get("deformation") is not None:
            if set(proposal) - {"deformation"}:
                raise ToolError("BAD_ARGS", "a deformation proposal is drawn on the "
                                "section's current placement: give it alone")
            warped = proposal["deformation"]
            if not isinstance(warped, DeformationProposal):
                warped = self.deformation_from_map(record.id, warped)
            elif warped.id != record.id:
                raise ToolError("BAD_ARGS", f"the proposal is {warped.id}'s, not {record.id}'s")
            store = self._proposal_store(warped)
            state = StackState.from_dict(copy.deepcopy(state.to_dict()))
            held = state.by_id(record.id)
            assert held is not None
            held.deformation = {"key": warped.key, "linear_key": warped._warp.linear_key,
                                "steps": [dict(warped._warp.step)]}
            record = held
        elif proposal:
            state = StackState.from_dict(copy.deepcopy(state.to_dict()))
            held = state.by_id(record.id)
            assert held is not None
            transform = proposal.get("transform")
            if isinstance(transform, TransformProposal):
                transform = transform.record
            elif transform is not None and not isinstance(transform, Mapping):
                transform = self.transform_from_params(record.id, transform).record
            if transform is not None:
                held.transform = dict(transform)
                held.deformation = None
            if proposal.get("position_mm") is not None:
                held.position_mm = float(proposal["position_mm"])
                held.deformation = None
            if proposal.get("angles") is not None:
                held.cutting_angles_deg = {"pitch": float(proposal["angles"][0]),
                                           "yaw": float(proposal["angles"][1])}
                held.deformation = None
            record = held
        edge = int(resolution) if resolution else None
        request = looks.LookRequest(mode="overlay", sections=(record.id,), long_edge=edge)
        try:
            drawn = looks.look(self._door.ctx, state, request, store=store)
        except looks.LookError as exc:
            raise ToolError(exc.code, str(exc)) from exc
        return drawn[0].image

    def _proposal_store(self, proposal: DeformationProposal) -> Any:
        """A record store that holds *proposal*'s record (in memory only) and
        reads every other from the job's."""
        from langslice.core.deformation import RecordStore

        held = self._door.job.deformations
        store = RecordStore(root=held.root, folder_of=held.folder_of)
        store.put(proposal.key, proposal._warp.record)
        return store

    @staticmethod
    def caption(image: Image.Image, text: str) -> Image.Image:
        """*image* with *text* in a caption band below it (pictures reach a
        model as bare attachments: what they show is said in the picture)."""
        from langslice.core.captions import caption

        return caption(image, text)

    # --- background work --------------------------------------------------------------------

    def start_work(self, sections: Sequence[object],
                   land: Callable[[ToolContext], ToolResult]) -> str:
        """Start *land* as a piece of the job's background work on *sections*
        and return its id at once (``w3``); the tool returns meanwhile.

        On a work thread ``land(ctx)`` computes and writes through *ctx*, a
        context of its own for the same tool (``with ctx.writing():`` makes
        its writes one undo step; pass ``expect=`` digests taken at the
        start, as a long tool does). Its :class:`ToolResult` is the work's
        notice: ``reply["notice"]`` (one paragraph) its text, the reply its
        result and the pictures saved and numbered with the job's. A
        :class:`ToolError` or :class:`Refused` it raises fails the work with
        its message. The next tool reply carries the notice
        (``background``); ``status`` lists running work; ``submit`` waits
        for it. While it runs, a write to any of *sections*, by any tool, is
        refused ``SECTION_BUSY``; so is starting it while other work runs
        there."""
        from langslice.job.background import DONE, FAILED, Landed

        job = self._door.job
        ids = [self._record(section).id for section in sections]
        busy = job.busy_refusal(ids)
        if busy is not None:
            raise Refused(busy.pop("error"), **busy)
        door, tool = self._door, self._tool

        def landing() -> Landed:
            context = ToolContext(door, tool, f"{self.call}-bg")
            try:
                result = land(context)
            except (ToolError, Refused) as exc:
                return Landed(status=FAILED, text=str(getattr(exc, "message", "") or exc))
            reply = dict(result.reply)
            return Landed(status=DONE, text=str(reply.pop("notice", "") or "done."),
                          pictures=[item if isinstance(item, Image.Image)
                                    else Image.fromarray(np.asarray(item))
                                    for item in result.pictures],
                          result=_plain(reply))

        return job.background.start(tool.name, ids, landing, wait_for_images=False).id

    # --- writing ----------------------------------------------------------------------------

    def _require(self, *properties: str) -> None:
        if self._tool.kind != "write":
            raise ToolError("READ_TOOL_WRITES", f"{self._tool.name} is declared a read tool and "
                            "writes nothing; declare kind='write'")
        outside = [name for name in properties if name not in self._tool.writes]
        if outside:
            raise ToolError("WRITE_NOT_DECLARED", f"{self._tool.name} writes {outside} but "
                            f"declares writes={list(self._tool.writes)}")

    @contextlib.contextmanager
    def writing(self) -> Iterator[ToolContext]:
        """Hold the job's write lock around several writes: everything
        written inside is ONE undo step, and so is everything one call
        writes, unless another write landed on the job between two of the
        call's (then they stay separate steps, so undo never takes back
        someone else's write). A long tool computes first and applies
        inside, passing ``expect=`` to the writes so a section that changed
        meanwhile is refused."""
        job = self._door.job
        with job.writing():
            outermost = self._depth == 0
            mark = job.pushed
            if outermost and self._step is not None:
                start, pushed, top = self._step
                if job.pushed == pushed and job.undo_stack and job.undo_stack[-1] is top:
                    mark = start  # nothing else wrote since the call's last write
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                job.merge_steps(mark)
                if outermost and job.pushed > mark and job.undo_stack:
                    self._step = (mark, job.pushed, job.undo_stack[-1])

    def _stale(self, ids: Sequence[str], expect: Mapping[str, str] | None,
               ) -> dict[str, dict[str, Any]]:
        """The rows of sections whose inputs differ from *expect*."""
        rows: dict[str, dict[str, Any]] = {}
        for name in ids:
            if expect is None or name not in expect:
                continue
            record = self._door.state.by_id(name)
            if record is None:
                continue
            now = {section_inputs(self._door.state, record, deformation=flag)
                   for flag in (False, True)}  # taken with or without the deformation
            if expect[name] not in now:
                rows[name] = stale_row(name)
        return rows

    def set_positions(self, rows: Sequence[Mapping[str, Any]], *,
                      cutting_angles: Mapping[str, float] | None = None,
                      expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set positions (``rows``: ``{"id", "position_mm"}``; atlas
        millimetres, clamped into the atlas range and reported) and, with
        *cutting_angles* (``{"pitch_deg", "yaw_deg"}``), the stack's angles.
        The stack then runs in the order of the positions. Needs ``position``
        (and ``angles``) in the tool's ``writes``."""
        from langslice.ops import positions

        self._require("position", *(["angles"] if cutting_angles is not None else []))
        job, workspace = self._door.job, self._door.ctx
        with self.writing():
            out: dict[str, dict[str, Any]] = {}
            todo: list[Mapping[str, Any]] = []
            ids = [self._resolve_id(row.get("id")) for row in rows]
            stale = self._stale([name for name in ids if name], expect)
            for row, name in zip(rows, ids, strict=True):
                if not name:
                    out[str(row.get("id"))] = {"id": str(row.get("id")), "status": "error",
                                               "error": "UNKNOWN_SLICE_IDS"}
                elif name in stale:
                    out[name] = stale[name]
                elif (frozen := frozen_row(job, name, "position")) is not None:
                    out[name] = frozen
                else:
                    todo.append(row)
            try:
                done = positions.position_sections(job, workspace, todo, cutting_angles) \
                    if todo or cutting_angles is not None else None
            except Refused as refusal:
                return WriteResult.refused(refusal)
            written = [] if done is None else [name for name, _mm in done.written]
            for name, mm in ([] if done is None else done.written):
                out[name] = {"id": name, "status": "ok", "position_mm": round(float(mm), 4)}
            for name, asked, used in ([] if done is None else done.clamped):
                out[name]["clamped_from_mm"] = round(float(asked), 4)
                out[name]["position_mm"] = round(float(used), 4)
        reply = WriteResult(written=written, rows=list(out.values()))
        if done is not None and done.cutting_angles is not None:
            reply.details["cutting_angles"] = dict(done.cutting_angles)
        return reply

    def _resolve_id(self, ref: object) -> str:
        record = self._door.state.resolve(ref)
        return record.id if record is not None else ""

    def set_transforms(self, rows: Sequence[Mapping[str, Any]], *,
                       expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set in-plane transforms. A row is ``{"id", "transform": record}``
        (a :class:`TransformProposal`'s ``record``, or :meth:`TransformProposal.row`)
        or ``{"id", "params": [six numbers], "note"?}``. A frozen affine, or
        a transform that would turn over a section whose orientation is
        frozen, comes back as that section's row (``FROZEN``); the others are
        written as one undo step. Needs ``affine`` in the tool's ``writes``."""
        from langslice.ops import transforms

        self._require("affine")
        job = self._door.job
        with self.writing():
            out: dict[str, dict[str, Any]] = {}
            ids = [self._resolve_id(row.get("id")) for row in rows]
            stale = self._stale([name for name in ids if name], expect)
            written: list[str] = []
            for row, name in zip(rows, ids, strict=True):
                if not name:
                    out[str(row.get("id"))] = {"id": str(row.get("id")), "status": "error",
                                               "error": "UNKNOWN_SLICE_IDS"}
                    continue
                if name in stale:
                    out[name] = stale[name]
                    continue
                record = row.get("transform")
                try:
                    if record is None:
                        if "params" not in row:
                            raise ToolError("BAD_ARGS", "a row is {id, transform} or "
                                            "{id, params}")
                        record = self.transform_from_params(
                            name, row["params"], note=str(row.get("note") or "")).record
                    done = transforms.set_transforms(job, {name: dict(record)})
                except ToolError as exc:
                    out[name] = {"id": name, "status": "error", "error": exc.code,
                                 "message": exc.message}
                    continue
                except Refused as refusal:
                    out[name] = {"id": name, "status": "error", **refusal.payload()}
                    out[name].pop("ids", None)
                    continue
                written += done
                out[name] = {"id": name, "status": "ok"}
        return WriteResult(written=written, rows=list(out.values()))

    def set_deformations(self, rows: Sequence[Mapping[str, Any]], *,
                         expect: Mapping[str, str] | None = None) -> WriteResult:
        """Set deformations. A row is ``{"id", "map": native_xy, "note"?,
        "meta"?}`` (the full map :meth:`deformation_from_map` takes) or
        ``{"id", "deformation": proposal}`` (:meth:`DeformationProposal.row`).
        Each replaces the section's deformation, as the deformation a native
        fit applies is stored (one undo step for the call; a later change to
        the section's placement clears it). Per section: a frozen deformation
        is ``FROZEN``, a map that cannot be stored ``BAD_FIELD``, a section
        without a position or transform ``INVALID_LINEAR_PLACEMENT``, one whose
        inputs differ from *expect* (or whose placement moved since its
        proposal was made) ``STALE_INPUT``. A map is converted before the
        job's write lock is taken, so a long tool holds it only to write.
        The ok rows carry the numbers ``DeformationProposal.summary`` has.
        Needs ``deformation`` in the tool's ``writes``."""
        from langslice.ops import deformable

        self._require("deformation")
        job = self._door.job
        out: dict[str, dict[str, Any]] = {}
        proposals: list[DeformationProposal] = []
        for row in rows:
            name = self._resolve_id(row.get("id"))
            if not name:
                out[str(row.get("id"))] = {"id": str(row.get("id")), "status": "error",
                                           "error": "UNKNOWN_SLICE_IDS"}
                continue
            frozen = frozen_row(job, name, "deformation")
            if frozen is not None:
                out[name] = frozen
                continue
            proposal = row.get("deformation")
            try:
                if proposal is None:
                    if "map" not in row:
                        raise ToolError("BAD_ARGS", "a row is {id, map, note?, meta?} or "
                                        "{id, deformation}")
                    proposal = self.deformation_from_map(
                        name, row["map"], note=str(row.get("note") or ""),
                        meta=row.get("meta"))
                elif not isinstance(proposal, DeformationProposal) or proposal.id != name:
                    raise ToolError("BAD_ARGS", "deformation is this section's "
                                    "DeformationProposal (deformation_from_map)")
            except ToolError as exc:
                out[name] = {"id": name, "status": "error", "error": exc.code,
                             "message": exc.message}
                continue
            out[name] = {"id": name}
            proposals.append(proposal)
        written: list[str] = []
        with self.writing():
            stale = self._stale([proposal.id for proposal in proposals], expect)
            out.update(stale)
            done = deformable.apply_warps(
                job, [proposal._warp for proposal in proposals if proposal.id not in stale])
            for applied in done.rows:
                out[applied["id"]] = applied
            written = list(done.written)
        return WriteResult(written=written, rows=list(out.values()))

    def mark_damage(self, rows: Sequence[Mapping[str, Any]], *,
                    expect: Mapping[str, str] | None = None) -> WriteResult:
        """Mark atlas regions a section has lost (``rows``: ``{"id", "regions",
        "note"?}``; empty regions clear the marks), as ``mark_damage`` does."""
        from langslice.ops import damage

        self._require()
        job, workspace = self._door.job, self._door.ctx
        with self.writing():
            out: list[dict[str, Any]] = []
            written: list[str] = []
            for row in rows:
                name = self._resolve_id(row.get("id"))
                if not name:
                    out.append({"id": str(row.get("id")), "status": "error",
                                "error": "UNKNOWN_SLICE_IDS"})
                    continue
                stale = self._stale([name], expect)
                if name in stale:
                    out.append(stale[name])
                    continue
                try:
                    done = damage.mark_damage(job, workspace, name, list(row.get("regions") or ()),
                                              str(row.get("note") or ""))
                except Refused as refusal:
                    out.append({"id": name, "status": "error", **refusal.payload()})
                    continue
                out.append({"id": name, "status": "ok", "regions": list(done.regions),
                            "written": done.written})
                if done.written:
                    written.append(name)
        return WriteResult(written=written, rows=out)

    def add_layer(self, section: object, layer: Layer) -> LayerRef:
        """Leave *layer* on the section so ``look`` can draw it (``layers=``)
        and ``status`` lists it. The layer's file goes under this tool's
        folder; the reference joins the section's state, so undoing the call
        that made it hides it. A layer of the same name on the section
        replaces the earlier one. Only a ``write`` tool leaves layers."""
        from langslice.job.plugin_store import register_layer

        self._require()
        layer.check()
        record_id = self._record(section).id
        job = self._door.job
        if layer.frame == "oriented":  # a raster computed on a section_image array
            held = self._record(section)
            layer = replace(layer, frame="section_px", raster=unorient_raster(
                np.asarray(layer.raster), held.rotation_deg, held.flip))
        with self.writing():
            record = self._door.state.by_id(record_id)
            assert record is not None
            before = job.snapshot()
            ref = register_layer(job.layout, record, self._tool.name, self.call, layer,
                                 persist=job.persist)
            job.commit(before)
        return LayerRef(id=ref["id"], section=record.id, name=ref["name"], kind=ref["kind"],
                        frame=ref["frame"], file=ref["file"])

    def layers(self, section: object) -> list[LayerRef]:
        """The layers the section shows now."""
        record = self._record(section)
        return [LayerRef(id=ref["id"], section=record.id, name=ref["name"], kind=ref["kind"],
                         frame=ref["frame"], file=ref["file"]) for ref in record.layers]


def _plane_frame(atlas: Any, placement: Any) -> dict[str, Any]:
    """The native atlas plane's frame in atlas micrometres (``center_um``,
    ``dx``, ``dy``: the display column and row directions, ``px_um``,
    ``shape``, ``resolution_um``; ``native_to_um``: 3x3 from a native pixel
    centre ``[x, y, 1]`` to its point in atlas micrometres, exactly the plane
    sampled, voxel centres at index times resolution), the plane
    :func:`langslice.core.oblique.sample_oblique_plane` samples."""
    from langslice.core.oblique import build_rotation_matrix, plane_axes

    normal_axis, row_axis, col_axis = plane_axes(atlas, placement.plane)
    rotation = build_rotation_matrix(placement.pitch_deg, placement.yaw_deg,
                                     row_axis=row_axis, col_axis=col_axis)
    res = np.asarray(atlas.resolution, dtype=np.float64)
    shape = tuple(int(v) for v in np.shape(atlas.annotation))
    centre = np.zeros(3)
    centre[normal_axis] = float(placement.position_mm) * 1000.0
    centre[row_axis] = (shape[row_axis] - 1) / 2.0 * res[row_axis]
    centre[col_axis] = (shape[col_axis] - 1) / 2.0 * res[col_axis]
    unit = np.eye(3)
    d_row, d_col = rotation @ unit[row_axis], rotation @ unit[col_axis]
    if placement.plane == "coronal":
        dx, dy = d_col, d_row
        px = [float(res[col_axis]), float(res[row_axis])]
        size = [shape[row_axis], shape[col_axis]]
    else:
        dx, dy = d_row, d_col
        px = [float(res[row_axis]), float(res[col_axis])]
        size = [shape[col_axis], shape[row_axis]]
    from langslice.core.oblique import plane_index_affine

    to_index = plane_index_affine(atlas, float(placement.position_mm), placement.plane,
                                  float(placement.pitch_deg), float(placement.yaw_deg))
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # [x, y, 1] -> [row, col, 1]
    native_to_um = np.diag(res) @ to_index @ swap
    return {"center_um": centre.tolist(), "dx": dx.tolist(), "dy": dy.tolist(),
            "px_um": px, "shape": size, "resolution_um": res.tolist(),
            "native_to_um": native_to_um.tolist()}


__all__ = [
    "AffineView", "AtlasView", "DeformationProposal", "FitFrame", "LayerRef", "SectionRaster",
    "SectionView", "ToolContext",
    "TransformProposal", "WriteResult", "STALE_INPUT",
]
