"""The proposals of a plugin context (``ctx.proposals``): a method's result as
LangSlice stores it.

:class:`Proposals` turns a method's answer into the transform or
deformation the job would store (:class:`TransformProposal`,
:class:`DeformationProposal`) without writing it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.geometry.affine import IDENTITY_PARAMS
from langslice.core.jsonable import to_jsonable
from langslice.core.plugins import ToolError
from langslice.core.refusal import RefusalCode
from langslice.core.stack.affine import (
    Calibration,
    PhysicalAffine,
    StoredAffine,
)
from langslice.core.stack.deformation import DeformationRef
from langslice.plugins.caller import Call
from langslice.plugins.frames import Grids
from langslice.plugins.reads import Records

if TYPE_CHECKING:
    from langslice.core.fits.deformation import Grid
    from langslice.tools.changing.warps import MappedWarp
    from langslice.tools.run import Run


@dataclass(frozen=True)
class TransformProposal:
    """A transform a method found, in the forms LangSlice stores it:
    ``params`` are the six normalized numbers, ``knobs`` the same transform as
    ``interactive_transform`` takes it about ``pivot``, ``record`` the stored
    record. :meth:`row` is a row for ``ctx.writes.set_transforms``."""

    id: str
    params: tuple[float, ...]
    knobs: dict[str, float]
    pivot: tuple[float, float]
    mirrored: bool
    calibration: dict[str, Any]
    record: dict[str, Any]
    affine: StoredAffine | None = field(default=None, repr=False, compare=False)

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "transform": self.record}


@dataclass(frozen=True)
class DeformationProposal:
    """A deformation a method found, as LangSlice stores it, not written yet
    (``ctx.proposals.deformation_from_map``): ``summary`` holds its numbers
    (``displacement_mm`` max and median over the placed atlas footprint, in
    millimetres, ``fold_fraction``, ``flags``) and ``inverse_consistency_mm``
    how well the inverse LangSlice computed undoes it. :meth:`row` is a row
    for ``ctx.writes.set_deformations``;
    ``overlay_picture(id, proposal={"deformation": proposal})`` draws it."""

    id: str
    key: str
    summary: dict[str, Any]
    inverse_consistency_mm: dict[str, float]
    warp: MappedWarp = field(repr=False, compare=False)

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
        return DeformationRef(
            key=self.key, linear_key=self.warp.linear_key, steps=(dict(self.warp.step),)
        )


DEGENERATE_DET = 1e-9


def is_invertible(matrix: Any) -> bool:
    """Whether the 2x2 or 3x3 *matrix* is finite and has a determinant
    (of its 2x2 block) of at least :data:`DEGENERATE_DET`."""
    m = np.asarray(matrix, dtype=np.float64)
    return bool(np.isfinite(m).all() and abs(np.linalg.det(m[:2, :2])) >= DEGENERATE_DET)


class Proposals:
    """A method's result as LangSlice stores it, not written: ``ctx.proposals``.

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    call
        The call: who it works for and its id, which a record keeps.
    records
        How a section is found by filename.
    grids
        The fit grids of the context's sections.
    """

    def __init__(self, run: Run, call: Call, records: Records, grids: Grids) -> None:
        self._run = run
        self._call = call
        self._records = records
        self._grids = grids

    def transform_from_matrix(
        self, section: object, matrix: Any, *, note: str = ""
    ) -> TransformProposal:
        """A method's linear result as a transform LangSlice stores: *matrix*
        is the 2x3 map from the fit grid's pixels (:attr:`FitFrame.image`) to
        native atlas pixels (:attr:`FitFrame.atlas`), the form registration
        engines return. The result is a refinement of the section's current
        placement exactly as an Elastix fit is, in the six normalized numbers
        and the knobs about the section's stored pivot (else the canvas
        centre). ``ToolError`` ``BAD_ARGUMENT`` for a degenerate map."""
        record = self._records.get(section)
        grid = self._grids.of(record)
        m = np.vstack([np.asarray(matrix, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
        if not is_invertible(m):
            raise ToolError(RefusalCode.BAD_ARGUMENT, "the map is degenerate or not finite")
        step = np.asarray(grid.placement.atlas_to_section, dtype=np.float64) @ m
        params = cast("Grid", grid).params_after(IDENTITY_PARAMS, step)
        return self.transform_from_params(record.id, params, note=note)

    def transform_from_params(
        self, section: object, params: Sequence[float], *, note: str = ""
    ) -> TransformProposal:
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

        record = self._records.get(section)
        values = [float(v) for v in params]
        if len(values) != 6 or not np.isfinite(values).all():
            raise ToolError(
                RefusalCode.BAD_ARGUMENT, "a transform is six finite numbers [a, b, tx, c, d, ty]"
            )
        workspace, state = self._run.workspace, self._run.state
        rendered = render_slice(workspace, record, long_edge=PREVIEW_LONG_EDGE)
        um_per_px, source = calibrate(state, workspace, record, rendered)
        try:
            geometry = canvas_geometry(
                rendered.size,
                um_per_px,
                workspace.atlas,
                record.position_mm,
                state.plane,
                record.pitch_deg,
                record.yaw_deg,
            )
        except ValueError as exc:
            raise ToolError(RefusalCode.PRECONDITION, f"{record.id}: {exc}") from exc
        in_section = denormalized_affine(values, rendered.size)
        ox, oy = (float(v) for v in geometry.section_offset)
        shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
        back = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
        on_canvas = (shift @ np.vstack([in_section, [0.0, 0.0, 1.0]]) @ back)[:2]
        canvas_w, canvas_h = geometry.size
        knobs_now = None if record.transform is None else record.transform.physical
        held_pivot = None if knobs_now is None else knobs_now.pivot
        pivot = (float(held_pivot[0]), float(held_pivot[1])) if held_pivot else (0.5, 0.5)
        knobs = physical_params(
            on_canvas, pivot=(canvas_w * pivot[0], canvas_h * pivot[1]), um_per_px=um_per_px
        )
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
            kind=self._call.caller.label,
            params=tuple(values),
            physical=physical,
            calibration=Calibration(section_um_per_px=scale, source=source),
            mirrored=mirrored,
            note=str(note or "").strip(),
        )
        return TransformProposal(
            id=record.id,
            params=tuple(values),
            knobs=dict(knobs),
            pivot=pivot,
            mirrored=mirrored,
            calibration=calibration,
            record=affine.to_json(),
            affine=affine,
        )

    def deformation_from_map(
        self,
        section: object,
        native_xy: Any,
        *,
        note: str = "",
        meta: Mapping[str, Any] | None = None,
        weights: Any = None,
        provenance: Mapping[str, Any] | None = None,
        start: str = "linear",
        settings: Mapping[str, Any] | None = None,
        engine_settings: Mapping[str, Any] | None = None,
        engine_version: str = "",
    ) -> DeformationProposal:
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
        ``BAD_ARGUMENT`` when it holds none. *settings* (the call's settings, by
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
        :mod:`langslice.core.fits.inputs`). ``ToolError`` ``BAD_ARGUMENT`` for an
        array of another shape, not float or not finite;
        ``PRECONDITION`` for a section without a position or a
        transform."""
        import json

        from langslice.tools.changing import warps

        self._call.begin()
        record = self._records.get(section)
        try:
            plain_meta = to_jsonable(dict(meta or {}), strict=True)
            json.dumps(plain_meta)
        except (TypeError, ValueError) as exc:
            raise ToolError(RefusalCode.BAD_ARGUMENT, f"meta is JSON values: {exc}") from exc
        text = str(note or "").strip()
        given = to_jsonable(dict(provenance or {}), strict=True)
        fitted_with = to_jsonable(dict(settings or {}), strict=True)
        engine = to_jsonable(dict(engine_settings or {}), strict=True)
        warp = warps.warp_from_map(
            self._run.job,
            self._run.workspace,
            record.id,
            native_xy,
            method=self._call.caller.label,
            step={"note": text} if text else {},
            provenance={
                **self._call.caller.provenance(self._call.id, text, plain_meta),
                "input": "full map: fit-grid pixel centres to native atlas pixels",
                **given,
            },
            weights=weights,
            start=str(start),
            settings=fitted_with or None,
            engine_settings=engine,
            engine_version=str(engine_version or ""),
        )
        consistency = warp.record.diagnostics.get("inverse_consistency_mm") or {}
        return DeformationProposal(
            id=record.id,
            key=warp.key,
            summary=copy.deepcopy(warp.numbers),
            inverse_consistency_mm={k: round(float(v), 4) for k, v in consistency.items()},
            warp=warp,
        )
