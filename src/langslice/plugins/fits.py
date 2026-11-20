"""The engine runs of a plugin context (``ctx.fits``): an outside engine's files, the
native fits.

:class:`Fits` writes what an engine outside LangSlice registers and reads
its answer back as a proposal, reads a registration file made elsewhere, and
runs the native engines (FireANTs SyN, the Elastix affine) with settings of
the caller's, each returning a proposal that is not written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core.plugins import ToolError
from langslice.core.refusal import RefusalCode
from langslice.plugins.frames import Frames, Grids
from langslice.plugins.proposals import (
    DeformationProposal,
    Proposals,
    TransformProposal,
    is_invertible,
)
from langslice.plugins.reads import Records
from langslice.tools.changing import warps

if TYPE_CHECKING:
    from langslice.tools.run import Run


class Fits:
    """Files and engines outside the context, as proposals: ``ctx.fits``.

    Writes what an engine outside LangSlice registers and reads its answer
    back (a transform file, a displacement field, a QuickNII or DeepSlice
    registration file), and runs the native engines (FireANTs SyN, the
    Elastix affine) with settings of the caller's, each returning a proposal
    that is not written.

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    records
        How a section is found by filename.
    grids
        The fit grids of the context's sections.
    frames
        The context's atlas frames.
    proposals
        The context's proposals.
    """

    def __init__(
        self, run: Run, records: Records, grids: Grids, frames: Frames, proposals: Proposals
    ) -> None:
        self._run = run
        self._records = records
        self._grids = grids
        self._frames = frames
        self._proposals = proposals

    def write_frame(
        self,
        section: object,
        folder: str | Path,
        *,
        format: str = "tif",
    ) -> dict[str, Path]:
        """Write what an engine outside LangSlice registers, into *folder*:
        on the fit grid (:attr:`FitFrame.image`'s pixels), ``section``
        (RGB), ``stain`` (the preprocessed image the native fits read,
        gray) and ``placed_template`` / ``placed_annotation`` (the atlas
        plane drawn on that grid at the section's current placement, so the
        two images already overlap where the placement puts them); on the
        native atlas grid ``template`` and ``annotation``; and
        ``frame.json`` (the sizes, ``placed_atlas_to_section``,
        ``atlas_to_section``, ``native_to_um``, the fit grid's millimetres
        per pixel, the files). *format* is the images' suffix (``tif``, or
        ``nii.gz``, ``nii``, ``mha``, ``nrrd`` through SimpleITK); every
        image has spacing 1 and origin 0, so an engine's answer is in pixels
        (:meth:`map_from_itk` reads it back). Returns ``{name: path}``.
        Writes nothing in the job."""
        import json

        import cv2

        from langslice.core.maps.itk import IMAGE_FORMATS, write_image

        if format not in IMAGE_FORMATS:
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                f"format is one of {', '.join(IMAGE_FORMATS)}; got {format!r}",
            )
        record = self._records.get(section)
        f = self._frames.fit_frame(record.id, stain=True)
        height, width = f.image.shape[:2]
        placed = (
            f.placed_atlas_to_section
            if f.placed_atlas_to_section is not None
            else f.atlas_to_section
        )
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        stain = np.asarray(Image.fromarray(np.asarray(f.stain)).convert("L"))
        images: dict[str, np.ndarray] = {
            "section": np.asarray(f.image, dtype=np.uint8),
            "stain": stain,
            "template": np.asarray(f.atlas["template"], dtype=np.float32),
            "annotation": np.asarray(f.atlas["annotation"]).astype(np.uint32),
            "placed_template": cv2.warpAffine(
                np.asarray(f.atlas["template"], dtype=np.float32),
                placed[:2],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderValue=0.0,
            ),
            "placed_annotation": cv2.warpAffine(
                np.asarray(f.atlas["annotation"]).astype(np.float32),
                placed[:2],
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderValue=0.0,
            ).astype(np.uint32),
        }
        written = {
            name: write_image(target / f"{name}.{format}", array) for name, array in images.items()
        }
        grid = self._grids.of(record)
        info = {
            "section": record.id,
            "fit_size": [int(width), int(height)],
            "native_size": [
                int(f.atlas["annotation"].shape[1]),
                int(f.atlas["annotation"].shape[0]),
            ],
            "fit_mm_per_px": float(grid.placement.section_mm_per_px),
            "placed_atlas_to_section": np.asarray(placed).tolist(),
            "atlas_to_section": np.asarray(f.atlas_to_section).tolist(),
            "native_to_um": f.plane["native_to_um"],
            "position_mm": f.position_mm,
            "angles": list(f.angles),
            "files": {name: path.name for name, path in written.items()},
            "convention": "Pixel centres at integers, [x, y] = [column, row]; every image "
            "spacing 1, origin 0. section, stain and placed_* on the fit grid; "
            "template and annotation on the native atlas grid. "
            "placed_atlas_to_section (3x3 on [x, y, 1]) takes a native atlas "
            "pixel to the fit-grid pixel the current placement puts it on; "
            "native_to_um a native pixel [x, y, 1] to atlas micrometres.",
        }
        frame_file = target / "frame.json"
        frame_file.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
        return {**written, "frame": frame_file}

    def map_from_itk(
        self,
        section: object,
        source: Any,
        *,
        fixed: str = "section",
        atlas: str = "placed",
        spacing: float = 1.0,
    ) -> np.ndarray:
        """An outside engine's answer as the full map
        ``ctx.proposals.deformation_from_map`` and ``ctx.writes.set_deformations`` take
        (``(H, W, 2)`` native atlas pixels of every fit-grid pixel centre).

        *source*: a displacement-field image file (``.nii.gz``, ``.mha``,
        ``.nrrd``...), a transform file (``.mat``, ``.txt``, ``.tfm``,
        ``.h5``: ANTs' and ITK's), a ``SimpleITK`` transform or vector image,
        or a float array ``(H, W, 2)`` of displacements ``(dx, dy)`` in
        fit-grid pixels. Read in ITK's convention: it maps each FIXED image
        point to the MOVING point it corresponds to. *fixed* says which image
        was fixed: ``section`` (the section's ``stain`` or ``section`` of
        :meth:`write_frame`) or ``atlas`` (then the map is inverted, on the
        fit grid). *atlas* says which atlas image the engine used:
        ``placed`` (``placed_template``, on the fit grid) or ``native``
        (``template``; with *fixed* ``section`` only). *spacing* is the
        images' spacing per pixel when they were not :meth:`write_frame`'s."""
        from langslice.core.maps.itk import inverted, mapped_points

        _check_images(fixed, atlas)
        if fixed == "atlas" and atlas == "native":
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                "a field with the atlas fixed is read on the fit grid: "
                "register against placed_template (atlas='placed')",
            )
        record = self._records.get(section)
        width, height = self._grids.of(record).image.size
        try:
            points = mapped_points(source, (width, height), spacing=float(spacing))
            if fixed == "atlas":
                points = inverted(points)
        except (ValueError, RuntimeError) as exc:
            raise ToolError(RefusalCode.BAD_ARGUMENT, f"{record.id}: {exc}") from exc
        if atlas == "native":
            return points
        to_native = np.linalg.inv(self._grids.placed(record))
        return np.stack(
            [
                to_native[0, 0] * points[..., 0]
                + to_native[0, 1] * points[..., 1]
                + to_native[0, 2],
                to_native[1, 0] * points[..., 0]
                + to_native[1, 1] * points[..., 1]
                + to_native[1, 2],
            ],
            axis=-1,
        )

    def transform_from_itk(
        self,
        section: object,
        source: Any,
        *,
        fixed: str = "section",
        atlas: str = "placed",
        spacing: float = 1.0,
        note: str = "",
    ) -> TransformProposal:
        """An outside engine's affine (a transform file such as ANTs'
        ``0GenericAffine.mat``, or a ``SimpleITK`` transform; *fixed*,
        *atlas* and *spacing* as :meth:`map_from_itk`, either way round) as
        a :class:`TransformProposal`, like ``ctx.proposals.transform_from_matrix``.
        ``BAD_ARGUMENT`` for a transform that is not affine."""
        from langslice.core.maps.itk import linear_matrix

        _check_images(fixed, atlas)
        record = self._records.get(section)
        size = self._grids.of(record).image.size
        try:
            matrix = linear_matrix(source, (int(size[0]), int(size[1])), spacing=float(spacing))
        except (ValueError, RuntimeError) as exc:
            raise ToolError(RefusalCode.BAD_ARGUMENT, f"{record.id}: {exc}") from exc
        if fixed == "atlas":
            if not is_invertible(matrix):
                raise ToolError(RefusalCode.BAD_ARGUMENT, f"{record.id}: the map is degenerate")
            matrix = np.linalg.inv(matrix)
        if atlas == "placed":
            matrix = np.linalg.inv(self._grids.placed(record)) @ matrix
        return self._proposals.transform_from_matrix(record.id, matrix[:2], note=note)

    def fit_fireants(
        self,
        section: object,
        *,
        restrict_to: Sequence[str] = (),
        exclude: Sequence[str] = (),
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        mask: str = "none",
        smoothness_um: float = 80.0,
        metric: str = "mutual_information",
        edges: bool = True,
        detail: str = "standard",
        start: str = "linear",
        marked: bool = True,
        note: str = "",
    ) -> DeformationProposal:
        """Run FireANTs SyN (the engine ``fireants_syn`` runs) on the section
        with settings of your own and return the result as a
        :class:`DeformationProposal`, NOT written (``ctx.writes.set_deformations``
        writes it, recorded as this caller's, with the settings and the
        ``inputs`` / ``estimated`` blocks in its provenance).

        The settings are ``fireants_syn``'s, by its names and with its
        defaults: *restrict_to* (region entries, ``"CTX:left"`` for one side:
        fit these regions' neighbourhood only), *section_image*,
        *atlas_image* (``"nissl"`` is the atlas's Nissl image where it has
        one, else its template), *smoothness_um*, *metric*
        (``mutual_information`` / ``local_correlation``), *edges*, *detail*
        (``coarse`` / ``standard`` / ``fine``); *exclude* adds regions to
        leave out. *mask*: ``"none"`` or the NAME of a mask of the section
        (``make_mask`` made it; ``ctx.reads.mask`` reads one): its weights at or above
        0.5 are compared, the others left out. An array is refused
        (``BAD_ARGUMENT``): make the mask with ``make_mask`` so the job holds it.
        *start* ``linear`` fits from the linear placement, ``current`` on
        top of the applied deformation; *marked* False leaves the section's
        marked damage regions in the fit."""
        from langslice.core.atlas.images import resolve_image
        from langslice.core.fits.deformable.render import composed_native_grid
        from langslice.tools.fitting import deformable, fit_regions

        record = self._records.get(section)
        workspace = self._run.workspace
        label = _mask_name(mask)
        if str(metric) not in deformable.METRICS:
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                f"metric is one of {list(deformable.METRICS)}; got {metric!r}",
            )
        if str(detail) not in deformable.DETAILS:
            raise ToolError(
                RefusalCode.BAD_ARGUMENT,
                f"detail is one of {list(deformable.DETAILS)}; got {detail!r}",
            )
        try:
            regions = fit_regions.region_entries(workspace, self._run.job.state, list(restrict_to))
            left_out = fit_regions.region_entries(workspace, self._run.job.state, list(exclude))
            chosen = replace(
                warps.default_settings(workspace),
                smoothness_um=float(smoothness_um),
                atlas_image=str(atlas_image),
                stain_metric=str(metric),
                stain_edges=bool(edges),
                detail=str(detail),
                structures=regions,
                exclude=left_out,
            )
            resolve_image(workspace.atlas, str(chosen.atlas_image))
        except (TypeError, ValueError) as exc:
            raise ToolError(RefusalCode.BAD_ARGUMENT, str(exc)) from exc
        fitted = warps.fit_record(
            self._run.job,
            workspace,
            record.id,
            chosen,
            start=start,
            mask=label,
            marked=bool(marked),
            section_image=str(section_image),
        )
        native_xy = np.stack(composed_native_grid(fitted, 1), axis=-1).astype(np.float64)
        settings = {
            "restrict_to": list(regions),
            "exclude": list(left_out),
            "section_image": str(section_image),
            "atlas_image": str(atlas_image),
            "mask": label,
            "smoothness_um": float(smoothness_um),
            "metric": str(metric),
            "edges": bool(edges),
            "detail": str(detail),
            "marked": bool(marked),
        }
        meta = {
            "engine": "fireants",
            "start": start,
            "runtime_s": round(float(fitted.engine.get("runtime_s", 0.0)), 1),
        }
        held = fitted.provenance
        return self._proposals.deformation_from_map(
            record.id,
            native_xy,
            note=note or "FireANTs SyN, settings of the caller's",
            meta=meta,
            weights=fitted.weights,
            settings=settings,
            start=start,
            engine_settings=dict(fitted.engine.get("native_parameters") or {}),
            engine_version=str(fitted.engine.get("version") or ""),
            provenance={key: held.get(key) for key in ("inputs", "estimated")},
        )

    def fit_elastix(
        self,
        section: object,
        *,
        restrict_to: Sequence[str] = (),
        exclude: Sequence[str] = (),
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        mask: str = "none",
        edges: bool = True,
        marked: bool = True,
        note: str = "",
        position_mm: float | None = None,
        angles: tuple[float, float] | None = None,
    ) -> TransformProposal:
        """Run the Elastix affine (the engine ``elastix_affine`` runs) on the
        section from its current placement and return the result as a
        :class:`TransformProposal`, NOT written (``ctx.writes.set_transforms``
        writes it; its record keeps the call's settings under ``fit``).

        With *position_mm* and/or *angles* ``(pitch_deg, yaw_deg)`` the fit
        is made at that candidate plane instead (from the section's current
        transform, its deformation left out), still writing nothing: compare
        candidates with ``ctx.pictures.overlay_picture(section,
        proposal={"transform": p.record, "position_mm": mm, "angles": a})``,
        and keep one by writing its position (``set_positions`` or
        ``position_sections``) and then the transform. The record's ``fit``
        holds the ``position_mm`` and ``angles`` it was made at.
        The settings are ``elastix_affine``'s, by its names and with its
        defaults: *restrict_to* (region entries, ``"CTX:left"`` for one side),
        *section_image*, *atlas_image* (as for :meth:`fit_fireants`), *mask*
        (``"none"`` or the NAME of a mask of the section, as for
        :meth:`fit_fireants`; an array is refused ``BAD_ARGUMENT``), *edges*;
        *exclude* adds regions to leave out; *marked* False keeps the section's
        marked damage regions in the fit."""
        from langslice.tools.fitting.affine_fit import elastix_fit

        record = self._records.get(section)
        label = _mask_name(mask)
        fitted = elastix_fit(
            self._run.job,
            self._run.workspace,
            record.id,
            restrict_to=list(restrict_to),
            exclude=list(exclude),
            atlas_image=str(atlas_image),
            mask=label,
            marked=bool(marked),
            section_image=str(section_image),
            edges=bool(edges),
            position_mm=position_mm,
            angles=angles,
        )
        proposal = self._proposals.transform_from_params(
            record.id, fitted.params, note=note or "Elastix affine, settings of the caller's"
        )
        read = fitted.inputs
        resolved, inputs = atlas_image, None
        if read is not None:
            resolved, inputs = read.atlas_image_resolved, read.payload()
        stored = {
            **proposal.record,
            "fit": {
                "engine": "elastix",
                "restrict_to": list(fitted.report.get("include") or ()),
                "exclude": list(fitted.report.get("exclude") or ()),
                "atlas_image": resolved,
                "section_image": str(section_image),
                "edges": bool(edges),
                "marked": bool(marked),
                "mask": label,
                "inputs": inputs,
                "position_mm": record.position_mm if position_mm is None else float(position_mm),
                "angles": list(record.angles if angles is None else angles),
            },
        }
        return replace(proposal, record=stored)

    def read_registration(
        self,
        path: str | Path,
        *,
        sections: Sequence[str] = (),
        at: Mapping[object, Mapping[str, Any]] | None = None,
        registered_on: str = "file",
    ) -> dict[str, Any]:
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
        ty]``) are for the oriented render: pass them to ``ctx.writes.set_transforms``
        as ``{"id", "params"}`` (or ``ctx.proposals.transform_from_params``) once the
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
        sees when the plugin hands it ``ctx.reads.section_raster``; the six numbers
        are for the oriented render either way."""
        from langslice.job.imports.placements import import_placements

        self._records.fresh()
        state = self._run.state
        names = [self._records.get(ref).id for ref in sections] or [r.id for r in state.slices]
        planes: dict[str, tuple[float, float, float]] = {}
        for ref, plane in (at or {}).items():
            name = self._records.get(ref).id
            record = state.by_id(name)
            assert record is not None
            position = plane.get("position_mm", record.position_mm)
            angles = plane.get("angles") or (record.pitch_deg, record.yaw_deg)
            planes[name] = (float(position), float(angles[0]), float(angles[1]))
        result = import_placements(
            Path(path),
            self._run.workspace,
            section_ids=names,
            orientations={r.id: (r.rotation_deg, r.flip) for r in state.slices},
            planes=planes,
            registered_on=registered_on,
        )
        rows = []
        for p in result.placements:
            calibrated = p.pixel_size_source != "imported"
            rows.append(
                {
                    "id": p.section_id,
                    "position_mm": round(float(p.position_mm), 4),
                    "pitch_deg": round(float(p.pitch_deg), 3),
                    "yaw_deg": round(float(p.yaw_deg), 3),
                    "rotation_deg": int(p.rotation_deg),
                    "flip": bool(p.flip),
                    "params": [float(v) for v in p.params] if calibrated else None,
                    **(
                        {}
                        if calibrated
                        else {
                            "params_note": "the job has no pixel size for this section; the file's "
                            "numbers imply one, which this job would not use"
                        }
                    ),
                    "pixel_size_um": float(p.pixel_size_um),
                    "pixel_size_source": p.pixel_size_source,
                    "out_of_plane_um": round(float(p.placement.out_of_plane_um), 3),
                    "in_plane": {
                        k: (round(float(v), 5) if isinstance(v, float) else v)
                        for k, v in p.placement.in_plane.items()
                    },
                }
            )
        return {
            "placements": rows,
            "refused": dict(result.refused),
            "missing": list(result.missing),
            "unmatched": list(result.unmatched),
        }


def _check_images(fixed: str, atlas: str) -> None:
    """Refuse a *fixed* that is not ``section`` or ``atlas``, or an *atlas*
    that is not ``placed`` or ``native``."""
    if fixed not in ("section", "atlas") or atlas not in ("placed", "native"):
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "fixed is 'section' or 'atlas'; atlas is 'placed' or 'native'",
        )


def _mask_name(mask: Any) -> str:
    """*mask* as a fit takes it: ``"none"`` or a mask's name. An array (a
    script's own weights) is refused: a mask a fit reads is one the job holds
    (``make_mask``), so every door leaves the same trace."""
    if isinstance(mask, str):
        return mask.strip() or "none"
    raise ToolError(
        RefusalCode.BAD_ARGUMENT,
        "mask is 'none' or the NAME of one of the section's "
        "masks; an array is not taken (a fit reads only a mask "
        "the job holds).",
    )
