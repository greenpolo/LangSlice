"""The atlas frames of a plugin context: a section's plane, any plane, points.

:class:`ContextFrames` adds to the reads the fit grid and the atlas plane a
section is placed on (:class:`FitFrame`), an atlas plane at any placement
(:class:`AtlasPlane`), the whole atlas volume, and section points carried
to atlas micrometres and back through the section's registration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from langslice.core.plugins import ToolError
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.section import Section
from langslice.doors.plugins.reads import ContextReads, as_tool_error
from langslice.tools.stale import section_inputs


@dataclass(frozen=True)
class FitFrame:
    """A section and the atlas plane it is placed on, on the grid LangSlice's
    fits work on.

    ``image`` is the section as its orientation sets it, at its current plane,
    with no in-plane transform (RGB, ``uint8``). ``atlas`` holds the plane at
    the section's position and cutting angles on the native atlas grid:
    ``annotation`` (region ids) and the atlas's images asked for (``template``
    and any additional reference the BrainGlobe atlas ships, by its name).
    ``atlas_to_section`` (3x3 on ``[x, y, 1]``) maps native atlas pixel centres
    to ``image`` pixel centres. ``plane`` says where the native plane lies in
    atlas micrometres: ``center_um``, ``dx``, ``dy`` (the column and row
    directions), ``px_um``, ``shape`` and ``resolution_um``.
    ``placed_atlas_to_section`` is the same map under the section's stored
    in-plane transform (its current linear placement, where an engine that
    refines it starts), None while it has no transform. ``excluded`` is the
    native plane's pixels of the section's marked damage regions (sides
    included), the regions every native fit leaves out; None while the section
    has none marked. ``damage_zone`` is ``excluded`` widened by the margin the
    native fits also leave out of their comparison
    (:data:`langslice.core.fits.deformable.masks.EXCLUSION_MARGIN_MM`, scaled
    to the atlas's brain size); None with ``excluded``. ``section_image`` is
    the section image ``fit_frame(image=)`` asked for, on the same grid as
    ``image`` (RGB ``uint8``, three equal planes) and ``section_image_ref``
    what names it (``{name, kind, file, digest}``, a fit reply's
    ``inputs.section_image``); ``stain`` is the section's ``preprocessed``
    image the same way (``stain=True`` is ``image="preprocessed"``), the
    picture the native fits read by default; each None unless asked for.
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
    #: The margin ``damage_zone`` adds around ``excluded``, in millimetres (always
    #: set: the margin the native fits report, whether or not damage is marked).
    damage_margin_mm: float | None = None
    stain: np.ndarray | None = None
    section_image: np.ndarray | None = None
    section_image_ref: dict[str, Any] | None = None
    #: ``{asked name: the atlas image it resolved to}`` for each image in
    #: ``atlas`` (``"nissl"`` is the atlas's Nissl image, else its template).
    atlas_resolved: dict[str, str] = field(default_factory=dict)
    #: What the answer to the default ``"nissl"`` request was when it resolved to the
    #: ``template`` (a Nissl template honours it; an atlas without a Nissl falls back); None
    #: when ``"nissl"`` was not asked for or has an image of its own.
    atlas_image_note: str | None = None


@dataclass(frozen=True)
class AtlasPlane:
    """An atlas plane at a placement of your choosing
    (:meth:`ToolContext.atlas_plane`), on its native grid: ``atlas`` the
    planes sampled (``annotation`` and the atlas's images asked for), ``plane``
    where it lies in atlas micrometres (as :attr:`FitFrame.plane`:
    ``native_to_um``, ``px_um``, ``shape``, ...), at ``position_mm`` and
    ``angles`` ``(pitch_deg, yaw_deg)``."""

    position_mm: float
    angles: tuple[float, float]
    atlas: dict[str, np.ndarray]
    plane: dict[str, Any]


class ContextFrames(ContextReads):
    """Read a section's atlas plane, any atlas plane, and points on them."""

    def _grid(self, record: Section) -> Any:
        from langslice.core.fits import deformation
        from langslice.core.geometry.affine import IDENTITY_PARAMS

        if record.position_mm is None:
            raise ToolError(RefusalCode.NO_POSITION,
                            f"{record.id} has no position yet")
        # Keyed by what the grid is drawn from: a script's context outlives a
        # placement change.
        key = (record.id, section_inputs(self._run.state, record))
        if key not in self._grids:
            try:
                self._grids[key] = deformation.fit_grid(
                    self._run.state, self._run.ctx, record,
                    params=list(IDENTITY_PARAMS))
            except ValueError as exc:
                raise ToolError(RefusalCode.INVALID_LINEAR_PLACEMENT,
                                f"{record.id}: {exc}") from exc
        return self._grids[key]

    def fit_frame(self, section: object, *, atlas: Sequence[str] = (
            "template", "annotation"), image: str | None = None,
            stain: bool = False) -> FitFrame:
        """The section and its atlas plane on the fit grid (:class:`FitFrame`).
        Needs a position. *atlas* picks the atlas planes to sample:
        ``annotation`` and any of the atlas's images
        (:func:`langslice.core.atlas.images.atlas_images`), and ``"nissl"``,
        the atlas's Nissl image where it has one, else its template
        (:func:`langslice.core.atlas.images.default_image`), under the name
        asked for and under the image's own; a name the atlas does not have is
        skipped. *image* adds that section image on the grid: a raw channel,
        ``"preprocessed"`` or another made image (:meth:`image_names`);
        ``ToolError`` ``UNKNOWN_IMAGE`` for another. *stain* True is
        ``image="preprocessed"`` and also fills ``stain``."""
        from langslice.core.atlas.brain_size import brain_scale
        from langslice.core.atlas.images import atlas_images, nissl_note, resolve_image
        from langslice.core.fits import deformation
        from langslice.core.fits.deformable.masks import (
            EXCLUSION_MARGIN_MM,
            dilate,
        )
        from langslice.core.fits.deformable.plane_images import (
            native_intensity,
            native_labels,
            placement_left,
            regions_mask,
        )
        from langslice.core.sections.damage import exclusions

        record = self._record(section)
        grid = self._grid(record)
        oriented = grid.image.convert("RGB")
        placement = grid.placement
        workspace = self._run.ctx
        planes: dict[str, np.ndarray] = {}
        if "annotation" in atlas:
            planes["annotation"] = np.asarray(native_labels(workspace.atlas, placement))
        offered = set(atlas_images(workspace.atlas))
        resolved_names: dict[str, str] = {}
        for asked in atlas:
            if asked == "annotation" or asked in planes:
                continue
            try:
                kind = resolve_image(workspace.atlas, asked)
            except ValueError:  # an image the atlas does not have is skipped
                continue
            if kind not in offered and asked not in offered:
                continue
            if kind not in planes:
                planes[kind] = np.asarray(native_intensity(workspace.atlas, placement, kind),
                                          dtype=np.float32)
            planes[asked] = planes[kind]
            resolved_names[asked] = kind
        placed = None
        if record.transform:
            try:
                placed = np.asarray(deformation.fit_grid(
                    self._run.state, workspace, record).placement.atlas_to_section,
                    dtype=np.float64)
            except ValueError:  # a stored transform no fit can use (stale, turned over)
                placed = None
        applied = None
        if record.deformation:
            from langslice.core.fits.deformable.render import (
                composed_native_grid,
            )

            held = self._run.job.deformations.current(self._run.state, record)
            if held is not None and tuple(held.section_size) == oriented.size:
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
        wanted = "preprocessed" if stain and image is None else image
        picture: np.ndarray | None = None
        reference: dict[str, Any] | None = None
        if wanted is not None:
            from langslice.core.sections.images import read_image

            known = self.image_names(record.id)
            if wanted not in known:
                raise ToolError(RefusalCode.UNKNOWN_IMAGE,
                                f"{record.id} has no section image "
                                f"{wanted!r}; its images: {', '.join(known)}",
                                images=list(known))
            picture = np.asarray(deformation.section_image(
                workspace, self._run.state, grid, wanted)[0].convert("RGB"))
            reference = read_image(workspace, record, wanted).payload()
        preprocessed = picture if wanted == "preprocessed" else (
            np.asarray(deformation.section_image(
                workspace, self._run.state, grid, "preprocessed")[0].convert("RGB"))
            if stain else None)
        return FitFrame(
            id=record.id, image=np.asarray(oriented), atlas=planes,
            atlas_to_section=np.asarray(placement.atlas_to_section, dtype=np.float64),
            plane=_plane_frame(workspace.atlas, placement),
            position_mm=float(placement.position_mm),
            angles=(float(placement.pitch_deg), float(placement.yaw_deg)),
            placed_atlas_to_section=placed, applied_map=applied, excluded=excluded,
            damage_zone=zone, damage_margin_mm=round(
                float(EXCLUSION_MARGIN_MM * brain_scale(workspace.atlas)), 4),
            stain=preprocessed, section_image=picture, section_image_ref=reference,
            atlas_resolved=resolved_names,
            atlas_image_note=nissl_note(workspace.atlas, "nissl") if "nissl" in resolved_names
            else None)

    def angles_from_plane(self, normal_um: Sequence[float], point_um: Sequence[float],
                          ) -> dict[str, float]:
        """The placement of the plane through *point_um* with normal
        *normal_um* (atlas micrometres, the atlas's axis order, as
        :attr:`FitFrame.plane` gives a plane): ``position_mm`` (where it crosses
        the volume's in-plane centre, the stack's position convention),
        ``pitch_deg`` and ``yaw_deg`` (the stack's cutting-angle convention).
        Nothing is written. ``ToolError`` ``BAD_PLANE`` for a plane tilted
        past what a cutting angle can say."""
        from langslice.core.maps.imported import (
            plane_angles,
            plane_position_mm,
        )
        from langslice.core.space import Plane

        atlas = self._run.ctx.atlas
        plane = cast(Plane, self._run.state.plane)
        normal = np.asarray(normal_um, dtype=np.float64)
        point = np.asarray(point_um, dtype=np.float64)
        if normal.shape != (3,) or point.shape != (3,) or not (
                np.isfinite(normal).all() and np.isfinite(point).all()) or not normal.any():
            raise ToolError(RefusalCode.BAD_PLANE,
                            "normal_um and point_um are three finite numbers, "
                            "the normal not zero")
        try:
            pitch, yaw = plane_angles(normal, atlas, plane)
        except ValueError as exc:
            raise ToolError(RefusalCode.BAD_PLANE, str(exc)) from exc
        return {"position_mm": float(plane_position_mm(point, normal, atlas, plane)),
                "pitch_deg": float(pitch), "yaw_deg": float(yaw)}

    def atlas_slice(self, section: object, *, kind: str = "template") -> np.ndarray:
        """The atlas plane at the section's position and cutting angles, on
        the native atlas grid (:class:`FitFrame` ``atlas``): *kind* is
        ``annotation`` or one of the atlas's images."""
        from langslice.core.atlas.images import atlas_images

        offered = ("annotation", *atlas_images(self._run.ctx.atlas))
        if kind not in offered:
            raise ToolError(RefusalCode.BAD_ARGS,
                            f"kind is one of {', '.join(offered)}; got {kind!r}")
        frame = self.fit_frame(section, atlas=(kind,))
        return frame.atlas[kind]

    def atlas_plane(self, position_mm: float, *, pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                    kinds: Sequence[str] = ("template", "annotation")) -> AtlasPlane:
        """The atlas plane at *position_mm* (atlas millimetres along the
        slicing axis, the stack's convention) and cutting angles *pitch_deg*
        / *yaw_deg*, whichever section holds it, on the native atlas grid
        (:class:`AtlasPlane`): ``annotation`` and the atlas's images among
        *kinds* (a name the atlas does not have is skipped). Nothing is
        written."""
        from langslice.core.atlas.images import atlas_images
        from langslice.core.fits.deformable.geometry import Placement
        from langslice.core.fits.deformable.plane_images import (
            native_intensity,
            native_labels,
        )

        workspace = self._run.ctx
        values = [float(position_mm), float(pitch_deg), float(yaw_deg)]
        if not np.isfinite(values).all():
            raise ToolError(RefusalCode.BAD_ARGS,
                            "position_mm, pitch_deg and yaw_deg are finite numbers")
        placement = Placement(atlas_name=str(self._run.state.atlas), position_mm=values[0],
                              plane=self._run.state.plane,  # type: ignore[arg-type]
                              pitch_deg=values[1], yaw_deg=values[2],
                              atlas_to_section=np.eye(3), section_mm_per_px=1.0)
        planes: dict[str, np.ndarray] = {}
        if "annotation" in kinds:
            planes["annotation"] = np.asarray(native_labels(workspace.atlas, placement))
        for kind in atlas_images(workspace.atlas):
            if kind in kinds:
                planes[kind] = np.asarray(native_intensity(workspace.atlas, placement, kind),
                                          dtype=np.float32)
        return AtlasPlane(position_mm=values[0], angles=(values[1], values[2]), atlas=planes,
                          plane=_plane_frame(workspace.atlas, placement))

    def atlas_volume(self, kind: str = "template") -> np.ndarray:
        """The atlas's whole volume *kind* (``annotation`` or one of its
        images) on its own grid, the atlas's axis order (read only: copy it
        before changing it)."""
        from langslice.core.atlas.images import atlas_images
        from langslice.core.atlas.images import atlas_volume as volume

        atlas = self._run.ctx.atlas
        if kind == "annotation":
            return np.asarray(atlas.annotation)
        if kind not in atlas_images(atlas):
            raise ToolError(RefusalCode.BAD_ARGS,
                            f"kind is annotation or one of "
                            f"{', '.join(atlas_images(atlas))}; got {kind!r}")
        return volume(atlas, kind)

    def to_atlas_um(self, section: object, xy: Any, *, frame: str = "file") -> np.ndarray:
        """Section points *xy* (``(N, 2)`` ``[x, y]`` pixels: of the image file
        as stored with *frame* ``file``, the ``section_px`` of layers; of the
        fit grid, :attr:`FitFrame.image`, with ``fit``) in atlas micrometres,
        ``(N, 3)`` in the atlas's axis order, through the section's current
        registration, its deformation included; NaN outside the atlas volume."""
        from langslice.tools.points import atlas_points

        record = self._record(section)
        return self._points(atlas_points, record.id, xy, frame).um

    def from_atlas_um(self, section: object, xyz_um: Any, *, frame: str = "file",
                      ) -> np.ndarray:
        """Atlas points *xyz_um* (``(N, 3)`` micrometres, the atlas's axis
        order) as ``[x, y]`` pixels of the section (*frame* as
        :meth:`to_atlas_um`), each projected onto the section's atlas plane
        and carried back through its current registration, its deformation's
        inverse included. :meth:`plane_distance_um` says how far off the
        plane each point lies."""
        from langslice.tools.points import section_points

        record = self._record(section)
        return self._points(section_points, record.id, xyz_um, frame).xy

    def plane_distance_um(self, section: object, xyz_um: Any) -> np.ndarray:
        """How far each atlas point *xyz_um* lies off the section's atlas
        plane, in micrometres, ``(N,)``."""
        from langslice.tools.points import section_points

        record = self._record(section)
        return self._points(section_points, record.id, xyz_um, "file").off_plane_um

    def region_at(self, section: object, xy: Any, *, frame: str = "file",
                  ) -> list[dict[str, Any] | None]:
        """The atlas region under each section point *xy* (*frame* as
        :meth:`to_atlas_um`) through the section's current registration:
        ``{"id", "acronym", "name"}`` per point, None outside the atlas."""
        from langslice.tools.points import atlas_points

        record = self._record(section)
        found = self._points(atlas_points, record.id, xy, frame)
        structures = self._run.ctx.atlas.structures
        out: list[dict[str, Any] | None] = []
        for value in found.ids.tolist():
            if not value:
                out.append(None)
                continue
            try:
                entry = structures[int(value)]
                out.append({"id": int(value), "acronym": str(entry["acronym"]),
                            "name": str(entry["name"])})
            except KeyError:
                out.append({"id": int(value), "acronym": None, "name": None})
        return out

    def _points(self, function: Callable[..., Any], name: str, values: Any, frame: str) -> Any:
        try:
            return function(self._run.job, self._run.ctx, name, values, frame=frame)
        except Refused as refusal:
            raise as_tool_error(refusal) from refusal


def _plane_frame(atlas: Any, placement: Any) -> dict[str, Any]:
    """The native atlas plane's frame in atlas micrometres (``center_um``,
    ``dx``, ``dy``: the display column and row directions, ``px_um``,
    ``shape``, ``resolution_um``; ``native_to_um``: 3x3 from a native pixel
    centre ``[x, y, 1]`` to its point in atlas micrometres, exactly the plane
    sampled, voxel centres at index times resolution), the plane
    :func:`langslice.core.geometry.oblique.sample_oblique_plane` samples."""
    from langslice.core.geometry.oblique import (
        build_rotation_matrix,
        plane_axes,
    )

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
    from langslice.core.geometry.oblique import plane_index_affine

    to_index = plane_index_affine(atlas, float(placement.position_mm), placement.plane,
                                  float(placement.pitch_deg), float(placement.yaw_deg))
    # [x, y, 1] -> [row, col, 1]
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    native_to_um = np.diag(res) @ to_index @ swap
    return {"center_um": centre.tolist(), "dx": dx.tolist(), "dy": dy.tolist(),
            "px_um": px, "shape": size, "resolution_um": res.tolist(),
            "native_to_um": native_to_um.tolist()}
