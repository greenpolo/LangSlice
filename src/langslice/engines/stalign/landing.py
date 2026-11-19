"""Running STalign on the sections of an ``stalign_lddmm`` call and landing each result."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.core.failures import ENGINE_FAILURES
from langslice.core.refusal import Refused
from langslice.engines import _blocks
from langslice.engines._pixels import (
    outlined,
    placed_matrix,
    placed_outline,
)
from langslice.engines.stalign import engine
from langslice.engines.stalign.prepare import (
    LddmmCall,
    Prepared,
    grid_query,
    native_map,
    prepare_section,
    sorting_settings,
)
from langslice.engines.stalign.settings import (
    ENGINE_VERSION,
    FITS,
    LOSS_COLUMNS,
    LOSS_PLOT,
    MIN_SCALE,
    PICTURES,
    RESOLUTIONS,
    SWAP,
    Result,
)
from langslice.plugin_api import (
    DeformationProposal,
    FitFrame,
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    flag_words,
)


def _sorting_picture(
    ctx: ToolContext, frame: FitFrame, weights: np.ndarray, title: str
) -> Image.Image | None:
    """The section's image with every set-aside patch outlined, beside the weight map
    (white compared, black ignored); None when the sorting set nothing aside (a map of
    all white says nothing, so no panel is drawn)."""
    stain = _blocks.section_pixels(frame)
    gh, gw = stain.shape[:2]
    w = cv2.resize(np.asarray(weights, np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
    ignored = w < 0.5
    if not ignored.any():
        return None
    left = outlined(stain, ignored)
    right = np.repeat((np.clip(w, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    gap = np.full((gh, PICTURES.gap_px, 3), 255, np.uint8)
    image = Image.fromarray(np.concatenate([left, gap, right], axis=1))
    image.thumbnail((PICTURES.long_edge, PICTURES.long_edge))
    return ctx.pictures.caption(image, title, sections=[frame.id])


def _loss_picture(energy: np.ndarray, path: Path) -> None:
    """STalign's mismatch record as a small line plot (a file in the job, not sent)."""
    h, w, pad = LOSS_PLOT.height, LOSS_PLOT.width, LOSS_PLOT.pad
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    if len(energy) > 1:
        y = energy[:, 1]
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, MIN_SCALE)
        pts = [
            (pad + (w - 2 * pad) * i / (len(y) - 1), h - pad - (h - 2 * pad) * (v - lo) / span)
            for i, v in enumerate(y)
        ]
        draw.line(pts, fill="black", width=1)
        draw.text(
            (pad, 6),
            f"STalign image mismatch per iteration ({len(y)}); min {lo:.4g}, max {hi:.4g}",
            fill="black",
        )
    image.save(path)


def run_engine(call: Callable[[], Result], folder: Path) -> Result:
    """Run an engine call in this process, its failures refused as ``FIT_FAILED``."""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        return call()
    except ENGINE_FAILURES as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise ToolError(RefusalCode.FIT_FAILED, f"STalign failed: {failure}") from exc


@dataclass(frozen=True)
class _Landing:
    """Fit one section and write the result, as the section's background work."""

    call: LddmmCall
    section: Prepared

    def __call__(self, lctx: ToolContext) -> ToolResult:
        """Run the fit, save its files, write the registration and report it."""
        call, section = self.call, self.section
        here = call.folder / section.sid
        request = engine.LddmmRequest(
            source=section.source.image,
            source_rows_um=section.source.rows_um,
            source_cols_um=section.source.cols_um,
            target=section.target.image,
            target_rows_um=section.target.rows_um,
            target_cols_um=section.target.cols_um,
            start=section.source.start,
            query_um=grid_query(section.target.size, section.target.um_per_px),
            settings=call.fit,
            machine=call.machine,
        )
        result = run_engine(partial(engine.run_lddmm, request), here)
        native = native_map(result.source_um, section.source, section.target)
        note = self._note()
        weights = (
            _blocks.on_fit_grid(result.match_weights, section.target.size) if call.sorts else None
        )
        outline = placed_outline(section.frame, section.target.size)
        estimated = _blocks.estimated_block(weights, outline)
        engine_settings = {
            "target_um": RESOLUTIONS.target_um,
            "atlas_um": RESOLUTIONS.source_um,
            **result.chosen,
            "device": result.device,
        }
        blocks = {
            "inputs": section.inputs,
            "estimated": estimated,
            "settings": call.settings,
            "engine_settings": engine_settings,
        }
        self._save(here, result, blocks)
        summary = self._write(lctx, result, native, note, weights, estimated, engine_settings)
        return self._report(lctx, here, result, note, estimated, summary, blocks)

    def _note(self) -> str:
        """Say what the fit did, for the registration's record."""
        start = "the deformation it held" if self.section.source.on_grid else "its linear placement"
        what = "affine and bend" if self.call.fit.refit_transform else "bend"
        return f"STalign {what} on {start}, atlas {self.section.resolved}"

    def _save(self, here: Path, result: engine.LddmmResult, blocks: Mapping[str, Any]) -> None:
        """Save the call's files: the energy record, its plot, the settings and the weights."""
        energy = result.energy
        with open(here / "energy.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(LOSS_COLUMNS[: 1 + energy.shape[1]])
            for i, row in enumerate(energy):
                writer.writerow([i, *[f"{v:.6g}" for v in row]])
        _loss_picture(energy, here / "mismatch.png")
        asked = {**self.call.settings, **self.call.fit.keywords()}
        (here / "settings.json").write_text(
            json.dumps(
                {
                    "asked": asked,
                    **blocks,
                    "chosen": result.chosen,
                    "device": result.device,
                    "seconds": result.seconds,
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
            newline="\n",
        )
        np.savez_compressed(
            here / "weights.npz",
            WM=result.match_weights,
            WB=result.background_weights,
            WA=result.artifact_weights,
        )

    def _write(
        self,
        lctx: ToolContext,
        result: engine.LddmmResult,
        native: np.ndarray,
        note: str,
        weights: np.ndarray | None,
        estimated: Mapping[str, Any] | None,
        engine_settings: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write the section's transform (fit ``both``) and its deformation; return the
        deformation's summary."""
        section = self.section
        sid = section.sid
        affine = self.call.fit.refit_transform
        meta = {"inputs": section.inputs, "estimated": estimated}

        def proposal() -> DeformationProposal:
            return lctx.proposals.deformation_from_map(
                sid,
                native,
                note=note,
                meta=meta,
                weights=weights,
                start="current" if section.source.on_grid else "linear",
                settings=self.call.settings,
                engine_settings=engine_settings,
                engine_version=ENGINE_VERSION,
            )

        warp = None if affine else proposal()
        with lctx.writes.writing():
            if affine:
                self._write_transform(lctx, result, note)
            if warp is None:
                warp = proposal()
            done = lctx.writes.set_deformations(
                [warp.row()], expect=None if affine else {sid: section.digest}
            )
            if sid not in done.written:
                raise done.refusal(sid)
        return warp.summary

    def _write_transform(self, lctx: ToolContext, result: engine.LddmmResult, note: str) -> None:
        """Write the transform STalign refitted (fit ``both``)."""
        section = self.section
        sid = section.sid
        px_x, px_y = section.source.atlas_px_um
        um = section.target.um_per_px
        a_xy = SWAP @ result.affine @ SWAP
        placed = np.diag([1 / um, 1 / um, 1.0]) @ a_xy @ np.diag([px_x, px_y, 1.0])
        proposal = lctx.proposals.transform_from_matrix(sid, np.linalg.inv(placed)[:2], note=note)
        done = lctx.writes.set_transforms([proposal.row()], expect={sid: section.digest})
        if sid not in done.written:
            raise done.refusal(sid)

    def _report(
        self,
        lctx: ToolContext,
        here: Path,
        result: engine.LddmmResult,
        note: str,
        estimated: Mapping[str, Any] | None,
        summary: dict[str, Any],
        blocks: Mapping[str, Any],
    ) -> ToolResult:
        """Word the landing's notice and draw its pictures."""
        sid = self.section.sid
        relative = _blocks.job_path(lctx, here)
        left = ""
        if estimated and estimated["left_out_fraction"] is not None:
            left = (
                f"; its sorting left out {estimated['left_out_fraction']:.0%} of the tissue "
                "and is saved with the deformation"
            )
        waited = result.gpu_wait_s
        queue = {"gpu_wait_s": round(waited)} if waited >= 1.0 else {}
        text = (
            f"{note}; displacement median {summary['displacement_mm']['median']} mm, "
            f"max {summary['displacement_mm']['max']} mm; fold fraction "
            f"{summary.get('fold_fraction', 0)}; flags {flag_words(summary)}{left}. "
            + (f"It waited {round(waited)} s for the machine's GPU queue. " if queue else "")
            + f"STalign's mismatch record: "
            f"{relative}/energy.csv (plot: {relative}/mismatch.png)."
        )
        pictures: list[Image.Image] = []
        if self.call.view:
            pictures = [lctx.pictures.caption(lctx.pictures.overlay_picture(sid), f"{sid}: {note}")]
            sorting = _sorting_picture(
                lctx,
                self.section.frame,
                result.match_weights,
                f"{sid}: STalign's final sorting (outlined: set aside; right: white compared)",
            )
            if sorting is None:
                text += (
                    " Its final sorting set nothing aside, so every pixel was compared "
                    "and no sorting picture is sent."
                )
            else:
                pictures.append(sorting)
        return ToolResult({"notice": text, "id": sid, **summary, **queue, **blocks}, pictures)


def _start(ctx: ToolContext, call: LddmmCall, section: Prepared) -> dict[str, object]:
    """Start one section's fit as background work; return its row."""
    try:
        work = ctx.start_work([section.sid], _Landing(call, section))
    except (Refused, *ENGINE_FAILURES) as exc:
        payload = (
            exc.payload()
            if isinstance(exc, Refused)
            else {"error": RefusalCode.ENGINE_UNAVAILABLE, "message": str(exc)}
        )
        failed = {key: value for key, value in payload.items() if key != "status"}
        return {"id": section.sid, "status": "error", **failed}
    return {
        "id": section.sid,
        "status": "started",
        "work": work,
        "start": "deformation" if section.source.on_grid else "linear placement",
        "inputs": section.inputs,
        "estimated": None,
    }


def lddmm_tool(
    ctx: ToolContext,
    machine: engine.Machine,
    sections: list[str],
    section_image: str,
    atlas_image: str,
    kernel_um: float,
    fit: str,
    displacement_um: float,
    sorting: dict[str, Any],
    iterations: int,
    view: bool,
) -> ToolResult:
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "name one to twenty sections")
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    spreads, classes = sorting_settings(sorting)
    settings = {
        "section_image": section_image,
        "atlas_image": atlas_image,
        "mask": "none",
        "kernel_um": kernel_um,
        "fit": fit,
        "displacement_um": displacement_um,
        "sorting": sorting or {"method": "classic"},
        "iterations": iterations,
        "device": machine.device,
    }
    call = LddmmCall(
        settings=settings,
        fit=engine.LddmmSettings(
            kernel_um=float(kernel_um),
            displacement_um=float(displacement_um),
            iterations=int(iterations),
            refit_transform=fit == "both",
            sorting=spreads,
            mixture_classes=classes,
        ),
        machine=machine,
        atlas_image=atlas_image,
        section_image=section_image,
        folder=_blocks.call_folder(ctx, "lddmm"),
        view=view,
        sorts=sorting.get("method", "classic") != "none",
    )
    rows: list[dict[str, object]] = []
    for name in sections:
        prepared = prepare_section(ctx, call, name)
        rows.append(prepared if isinstance(prepared, dict) else _start(ctx, call, prepared))
    return ToolResult({"rows": rows, "settings": settings})


def target_to_atlas(frame: FitFrame, um: float) -> np.ndarray:
    """4 x 4: target (z, row um, col um) -> atlas um, through the section's linear placement
    and its plane (z along the unit normal)."""
    to_um = np.asarray(frame.plane["native_to_um"], np.float64)
    from_grid = np.linalg.inv(placed_matrix(frame))
    m = to_um @ from_grid
    d_row, d_col = m[:, 1] / um, m[:, 0] / um
    n = np.cross(d_col, d_row)
    n /= np.linalg.norm(n)
    basis = np.eye(4)
    basis[:3, 0], basis[:3, 1], basis[:3, 2], basis[:3, 3] = n, d_row, d_col, m[:, 2]
    return basis
