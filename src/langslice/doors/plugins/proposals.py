"""The proposals of a plugin context: a method's result as LangSlice stores it.

:class:`ContextProposals` turns a method's answer into the transform or
deformation the job would store (:class:`TransformProposal`,
:class:`DeformationProposal`) without writing it, and reads a linear
registration file made elsewhere as placements for the job's sections.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.jsonable import to_jsonable
from langslice.core.plugins import ToolError
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.affine import (
    Calibration,
    PhysicalAffine,
    StoredAffine,
)
from langslice.core.stack.codec import affine_to_json
from langslice.core.stack.deformation import DeformationRef
from langslice.doors.plugins.frames import ContextFrames
from langslice.doors.plugins.reads import as_tool_error

if TYPE_CHECKING:
    from langslice.tools.warps import MappedWarp


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
    #: The stored record as the section holds it, for LangSlice's own use.
    affine: StoredAffine | None = field(
        default=None, repr=False, compare=False)

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
    _warp: MappedWarp = field(repr=False, compare=False)

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "deformation": self}

    def reference(self) -> DeformationRef:
        """Return the deformation a section holds while the proposal is drawn.

        Returns
        -------
        DeformationRef
            Its key, the linear placement it sits on and its one step; no
            record folder, since it is not saved.
        """
        return DeformationRef(key=self.key, linear_key=self._warp.linear_key,
                              steps=(dict(self._warp.step),))


class ContextProposals(ContextFrames):
    """Make transforms and deformations to write, and read placements."""

    def transform_from_matrix(self, section: object, matrix: Any, *,
                              note: str = "") -> TransformProposal:
        """A method's linear result as a transform LangSlice stores: *matrix*
        is the 2x3 map from the fit grid's pixels (:attr:`FitFrame.image`) to
        native atlas pixels (:attr:`FitFrame.atlas`), the form registration
        engines return. The result is a refinement of the section's current
        placement exactly as an Elastix fit is, in the six normalized numbers
        and the knobs about the section's stored pivot (else the canvas
        centre). ``ToolError`` ``BAD_MATRIX`` for a degenerate map."""
        from langslice.core.geometry.affine import (
            IDENTITY_PARAMS,
            denormalized_affine,
            normalized_affine,
        )

        record = self._record(section)
        grid = self._grid(record)
        width, height = grid.image.size
        m = np.vstack([np.asarray(matrix, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
        if not np.isfinite(m).all() or abs(np.linalg.det(m[:2, :2])) < 1e-9:
            raise ToolError(RefusalCode.BAD_MATRIX,
                            "the map is degenerate or not finite")
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
        from langslice.core.fits.affine import calibrate, physical_params
        from langslice.core.geometry.affine import (
            decompose_affine,
            denormalized_affine,
        )
        from langslice.core.pictures.canvas import canvas_geometry
        from langslice.core.sections.renders import (
            PREVIEW_LONG_EDGE,
            render_slice,
        )

        record = self._record(section)
        values = [float(v) for v in params]
        if len(values) != 6 or not np.isfinite(values).all():
            raise ToolError(RefusalCode.BAD_ARGS,
                            "a transform is six finite numbers [a, b, tx, c, d, ty]")
        if record.position_mm is None:
            raise ToolError(RefusalCode.NO_POSITION,
                            f"{record.id} has no position yet")
        workspace, state = self._run.ctx, self._run.state
        rendered = render_slice(workspace, record, long_edge=PREVIEW_LONG_EDGE)
        um_per_px, source = calibrate(state, workspace, record, rendered)
        try:
            geometry = canvas_geometry(rendered.size, um_per_px, workspace.atlas,
                                       record.position_mm, state.plane,  # type: ignore[arg-type]
                                       record.pitch_deg, record.yaw_deg)
        except ValueError as exc:
            raise ToolError(RefusalCode.INVALID_LINEAR_PLACEMENT,
                            f"{record.id}: {exc}") from exc
        in_section = denormalized_affine(values, rendered.size)
        ox, oy = (float(v) for v in geometry.section_offset)
        shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
        back = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
        on_canvas = (shift @ np.vstack([in_section, [0.0, 0.0, 1.0]]) @ back)[:2]
        canvas_w, canvas_h = geometry.size
        knobs_now = None if record.transform is None else record.transform.physical
        held_pivot = None if knobs_now is None else knobs_now.pivot
        pivot = (float(held_pivot[0]), float(held_pivot[1])) if held_pivot else (0.5, 0.5)
        knobs = physical_params(on_canvas, pivot=(canvas_w * pivot[0], canvas_h * pivot[1]),
                                um_per_px=um_per_px)
        mirrored = bool(decompose_affine(values)["mirrored"])
        scale = round(float(um_per_px), 4)
        calibration = {"section_um_per_px": scale, "source": source}
        physical = PhysicalAffine(
            rotation_deg=knobs["rotation_deg"],
            scale_x=knobs["scale_x"],
            scale_y=knobs["scale_y"],
            translate_x_mm=knobs["translate_x_mm"],
            translate_y_mm=knobs["translate_y_mm"],
            shear=knobs["shear"],
            pivot=pivot,
        )
        affine = StoredAffine(
            kind=self._caller.label,
            params=tuple(values),
            physical=physical,
            calibration=Calibration(section_um_per_px=scale, source=source),
            mirrored=mirrored,
            note=str(note or "").strip(),
        )
        return TransformProposal(id=record.id, params=tuple(values), knobs=dict(knobs),
                                 pivot=pivot, mirrored=mirrored, calibration=calibration,
                                 record=affine_to_json(affine), affine=affine)

    def deformation_from_map(self, section: object, native_xy: Any, *, note: str = "",
                             meta: Mapping[str, Any] | None = None,
                             weights: Any = None,
                             provenance: Mapping[str, Any] | None = None,
                             start: str = "linear",
                             settings: Mapping[str, Any] | None = None,
                             engine_settings: Mapping[str, Any] | None = None,
                             engine_version: str = "") -> DeformationProposal:
        """A method's deformable result as the deformation LangSlice stores:
        *native_xy* is a float ``(H, W, 2)`` array giving, for each pixel
        centre ``(x, y)`` of the fit grid (:attr:`FitFrame.image`, ``H`` by
        ``W``), the native atlas pixel ``(x, y)`` (:attr:`FitFrame.atlas`) it
        corresponds to: the whole map, linear placement and deformation
        composed, in the frames :meth:`transform_from_matrix` takes. It is
        stored as every deformation is, the residual over the section's
        current linear placement, with an inverse LangSlice computes. *start*
        ``linear`` (the map was made from the linear placement): it replaces
        whatever deformation the section holds; ``current`` (the map was made
        on top of the deformation the section holds, as read through
        :attr:`FitFrame.applied_map`): that deformation is its record's
        parent and its steps stay before this one, as a native fit chains, and
        it is written only while the section still holds it (``STALE_INPUT``);
        ``BAD_ARGS`` when it holds none. *settings* (the call's settings, by
        the tool's names) are the step's, which ``status`` and the
        registration list show; *engine_settings* (what the engine ran at, in
        its own units) and *engine_version* go to the record's ``engine``, as
        a FireANTs record holds its own. *note* and *meta* (JSON values) are
        kept with it. *weights*: the weighting the engine
        estimated for itself, a float ``(H, W)`` array on the fit grid in
        [0, 1] (its tissue sorting; stored as float16 with the record, and
        what ``make_mask(source="fit")`` copies); None when it estimated none.
        *provenance*: JSON values added to the record's provenance (the
        ``inputs`` and ``estimated`` blocks of a fit,
        :mod:`langslice.core.fits.inputs`). ``ToolError`` ``BAD_FIELD`` for an
        array of another shape, not float or not finite;
        ``INVALID_LINEAR_PLACEMENT`` for a section without a position or a
        transform."""
        import json

        from langslice.tools import warps

        record = self._record(section)
        try:
            plain_meta = to_jsonable(dict(meta or {}), strict=True)
            json.dumps(plain_meta)
        except (TypeError, ValueError) as exc:
            raise ToolError(RefusalCode.BAD_ARGS,
                            f"meta is JSON values: {exc}") from exc
        text = str(note or "").strip()
        given = to_jsonable(dict(provenance or {}), strict=True)
        fitted_with = to_jsonable(dict(settings or {}), strict=True)
        engine = to_jsonable(dict(engine_settings or {}), strict=True)
        try:
            warp = warps.warp_from_map(
                self._run.job, self._run.ctx, record.id, native_xy,
                method=self._caller.label, step={"note": text} if text else {},
                provenance={**self._caller.provenance(self.call, text, plain_meta),
                            "input": "full map: fit-grid pixel centres to native atlas "
                                     "pixels", **given},
                weights=weights, start=str(start),
                settings=fitted_with or None, engine_settings=engine,
                engine_version=str(engine_version or ""))
        except Refused as refusal:
            raise as_tool_error(refusal) from refusal
        consistency = warp.record.diagnostics.get("inverse_consistency_mm") or {}
        return DeformationProposal(
            id=record.id, key=warp.key, summary=copy.deepcopy(warp.numbers),
            inverse_consistency_mm={k: round(float(v), 4) for k, v in consistency.items()},
            _warp=warp)

    def read_registration(self, path: str | Path, *, sections: Sequence[str] = (),
                          at: Mapping[object, Mapping[str, Any]] | None = None,
                          registered_on: str = "file") -> dict[str, Any]:
        """A linear registration file made elsewhere (QuickNII or VisuAlign
        JSON/XML, DeepSlice CSV/JSON/XML, a LangSlice ``registration.json``)
        as placements for this job's sections. Nothing is written. Returns
        ``{"placements": [...], "refused": {id: reason}, "missing": [id],
        "unmatched": [file names]}``; a placement is ``{"id", "position_mm",
        "pitch_deg", "yaw_deg", "rotation_deg", "flip", "params",
        "pixel_size_um", "pixel_size_source", "out_of_plane_um",
        "in_plane"}``.

        Each section is read in its CURRENT orientation (quarter turn and flip
        as the agent set them), so ``rotation_deg`` and ``flip`` are the
        section's own and ``params`` (six normalized numbers ``[a, b, tx, c, d,
        ty]``) are for the oriented render: pass them to :meth:`set_transforms`
        as ``{"id", "params"}`` (or :meth:`transform_from_params`) once the
        section holds the placement they were read at. ``in_plane`` reads them
        as rotation, scales and shear on that render
        (:func:`~langslice.core.geometry.affine.decompose_affine`). *at*
        ``{section: {"position_mm", "angles": (pitch, yaw)}}`` reads the file's
        map on a plane of your choosing (the section's final position and
        angles, e.g. a stack's propagated angle): ``params`` are then the
        file's map projected onto that plane and ``out_of_plane_um`` is how far
        the map lies off it; without *at* the file's own plane is used and
        ``out_of_plane_um`` is about 0. ``params`` is None, with the reason in
        ``params_note``, when the job has no pixel size for the section (the
        six numbers are relative to a pixel size, so they mean nothing to a job
        that would calibrate otherwise).

        *registered_on* is the image the file's entries were made on:
        ``"file"`` (default) the image file as stored, ``"oriented"`` the
        section as its orientation turns and flips it, which is what a method
        sees when the plugin hands it :meth:`section_raster`; the six numbers
        are for the oriented render either way."""
        from langslice.job.imports.placements import import_placements

        self._fresh()
        state = self._run.state
        names = [self._record(ref).id for ref in sections] or [r.id for r in state.slices]
        planes: dict[str, tuple[float, float, float]] = {}
        for ref, plane in (at or {}).items():
            name = self._record(ref).id
            record = state.by_id(name)
            assert record is not None
            position = plane.get("position_mm", record.position_mm)
            angles = plane.get("angles") or (record.pitch_deg, record.yaw_deg)
            if position is None:
                raise ToolError(RefusalCode.NO_POSITION,
                                f"{name} has no position to read the file at")
            planes[name] = (float(position), float(angles[0]), float(angles[1]))
        result = import_placements(
            Path(path), self._run.ctx, section_ids=names,
            orientations={r.id: (r.rotation_deg, r.flip) for r in state.slices},
            planes=planes, registered_on=registered_on)
        rows = []
        for p in result.placements:
            calibrated = p.pixel_size_source != "imported"
            rows.append({
                "id": p.section_id, "position_mm": round(float(p.position_mm), 4),
                "pitch_deg": round(float(p.pitch_deg), 3),
                "yaw_deg": round(float(p.yaw_deg), 3),
                "rotation_deg": int(p.rotation_deg), "flip": bool(p.flip),
                "params": [float(v) for v in p.params] if calibrated else None,
                **({} if calibrated else {
                    "params_note": "the job has no pixel size for this section; the file's "
                                   "numbers imply one, which this job would not use"}),
                "pixel_size_um": float(p.pixel_size_um),
                "pixel_size_source": p.pixel_size_source,
                "out_of_plane_um": round(float(p.placement.out_of_plane_um), 3),
                "in_plane": {k: (round(float(v), 5) if isinstance(v, float) else v)
                             for k, v in p.placement.in_plane.items()},
            })
        return {"placements": rows, "refused": dict(result.refused),
                "missing": list(result.missing), "unmatched": list(result.unmatched)}
