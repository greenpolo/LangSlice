"""Landing the stack's fit: each section's result written, as the call's background work."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from langslice.core.failures import ENGINE_FAILURES
from langslice.core.refusal import Refused
from langslice.engines import _blocks
from langslice.engines.emlddmm import engine
from langslice.engines.emlddmm.files import FileRecord, write_files
from langslice.engines.emlddmm.run import (
    Lattice,
    expand,
    run_engine,
    share_set_aside,
    summary_flags,
    weights_on_grid,
)
from langslice.engines.emlddmm.settings import ENGINE_VERSION, GETS, OVERLAYS, Array, Options
from langslice.engines.emlddmm.stack import Placed, Stack
from langslice.plugin_api import (
    DeformationProposal,
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    flag_words,
)


@dataclass(frozen=True)
class StackCall:
    """Hold what one ``emlddmm_stack`` call shares across its landing.

    Parameters
    ----------
    options
        The plugin's options, the device resolved.
    stack
        The stack.
    request
        The engine's request.
    lattices
        Where each section's map is read.
    settings
        The call's settings, as the reply and the deformation record them.
    asked
        What the call asked, for ``settings.json``.
    chosen
        What the plugin picked, for ``settings.json`` and the deformation record.
    routes
        How each section's result is written.
    digests
        Each section's registration digest when the call began.
    rigid, warp
        Whether the in-plane transform and the warp are fitted.
    view
        Whether pictures are sent.
    folder
        The call's folder in the job.
    note
        The words the registration's record carries.
    """

    options: Options
    stack: Stack
    request: engine.FitRequest
    lattices: list[Lattice]
    settings: dict[str, Any]
    asked: dict[str, Any]
    chosen: dict[str, Any]
    routes: dict[str, str]
    digests: dict[str, str]
    rigid: bool
    warp: bool
    view: bool
    folder: Path
    note: str


@dataclass(frozen=True)
class _Record:
    """Hold one section's result of the fit, read on its own plane.

    Parameters
    ----------
    placed
        The section on the stack.
    native
        The native atlas pixels at the lattice, ``(nh, nw, 2)``, with the warp if fitted.
    matrix
        The in-plane transform the fit implies, fit-grid pixels to native atlas pixels.
    weights
        The final sorting on the fit grid, or None without a warp.
    estimated
        The ``estimated`` block.
    off_plane_um
        The most the section's points lie off its own plane, um.
    moved_mm
        The median and the largest move inside the tissue, mm.
    set_aside
        The share of the tissue set aside as background and as artifact.
    """

    placed: Placed
    native: Array
    matrix: Array
    weights: np.ndarray | None
    estimated: dict[str, Any] | None
    off_plane_um: float
    moved_mm: tuple[float, float]
    set_aside: dict[str, float]

    @property
    def id(self) -> str:
        """Return the section's filename."""
        return self.placed.id


@dataclass(frozen=True)
class _Row:
    """Hold what one section's landing wrote.

    Parameters
    ----------
    record
        The section's result.
    summary
        The written deformation's summary, empty without one.
    wrote
        What was written, in words.
    problem
        Why it was not (fully) written; empty when it was.
    """

    record: _Record
    summary: dict[str, Any]
    wrote: str
    problem: str


@dataclass
class _Outcome:
    """Keep what the writes of one landing gave."""

    written: dict[str, str] = field(default_factory=dict)
    failed: dict[str, dict[str, Any]] = field(default_factory=dict)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class _StackLanding:
    """Fit the stack and write each section's result, as the stack's background work."""

    call: StackCall

    def __call__(self, lctx: ToolContext) -> ToolResult:
        """Run the fit, read it per section, save the files, write and report."""
        call = self.call
        result = run_engine(call.request, call.folder)
        records = self._records(result)
        write_files(
            call.stack,
            result,
            call.folder,
            FileRecord(call.asked, call.chosen, call.options.settings(), {}),
        )
        outcome = self._write(lctx, records)
        rows = self._rows(records, outcome)
        self._sections_csv(rows)
        return self._report(lctx, result, rows, outcome)

    def _records(self, result: engine.FitResult) -> list[_Record]:
        """Read the fit on each section's own plane."""
        call = self.call
        records = []
        for i, (placed, lattice) in enumerate(zip(call.stack.sections, call.lattices, strict=True)):
            section = placed.section
            n_points = len(lattice.points)
            nat_full, off_plane = section.native_from_atlas(result.warped_um[i], lattice.shape)
            nat_lin, _ = section.native_from_atlas(result.linear_um[i], lattice.shape)
            gxy = np.concatenate([lattice.pixels.reshape(-1, 2), np.ones((n_points, 1))], 1)
            matrix = np.linalg.lstsq(gxy, nat_lin.reshape(-1, 2), rcond=None)[0].T
            w, h = section.size
            inside = section.outline[
                np.clip(lattice.pixels[..., 1].round().astype(int), 0, h - 1),
                np.clip(lattice.pixels[..., 0].round().astype(int), 0, w - 1),
            ]
            used = nat_full if call.warp else nat_lin
            moved = np.hypot(*(used - lattice.native).transpose(2, 0, 1)) * section.native_mm
            moved = moved[inside] if inside.any() else moved.ravel()
            weights = (
                weights_on_grid(call.stack, placed, result.match_weights[placed.slot])
                if call.warp
                else None
            )
            records.append(
                _Record(
                    placed=placed,
                    native=used,
                    matrix=matrix,
                    weights=weights,
                    estimated=_blocks.estimated_block(weights, section.outline),
                    off_plane_um=float(np.abs(off_plane).max()),
                    moved_mm=(float(np.median(moved)), float(moved.max())),
                    set_aside=share_set_aside(result, placed),
                )
            )
        return records

    def _proposal(
        self, lctx: ToolContext, record: _Record, *, from_current: bool
    ) -> DeformationProposal:
        """Propose the section's cut of the 3-D warp as its deformation."""
        call = self.call
        section = record.placed.section
        sid = record.id
        current = from_current and section.applied is not None
        return lctx.proposals.deformation_from_map(
            sid,
            expand(record.native, *section.size),
            note=call.note,
            meta={"inputs": section.inputs, "estimated": record.estimated},
            weights=record.weights,
            settings=call.settings,
            engine_settings=call.chosen,
            engine_version=ENGINE_VERSION,
            start="current" if current else "linear",
        )

    def _write(self, lctx: ToolContext, records: list[_Record]) -> _Outcome:
        """Write the transforms, then the deformations, as one undo step."""
        call = self.call
        routes, digests = call.routes, call.digests
        outcome = _Outcome()
        transforms = {
            rec.id: lctx.proposals.transform_from_matrix(rec.id, rec.matrix, note=call.note).row()
            for rec in records
            if routes[rec.id] == "transform"
        }
        proposals = {
            rec.id: self._proposal(lctx, rec, from_current=True)
            for rec in records
            if routes[rec.id] == "deformation"
        }
        with lctx.writes.writing():
            if transforms:
                done = lctx.writes.set_transforms(
                    list(transforms.values()), expect={sid: digests[sid] for sid in transforms}
                )
                for row in done.rows:
                    if row.get("status") == "ok":
                        outcome.written[row["id"]] = "transform"
                    else:
                        outcome.failed[row["id"]] = row
            for rec in records:
                if routes[rec.id] == "transform" and call.warp and rec.id not in outcome.failed:
                    proposals[rec.id] = self._proposal(lctx, rec, from_current=False)
            if proposals:
                done = lctx.writes.set_deformations(
                    [p.row() for p in proposals.values()],
                    expect={sid: digests[sid] for sid in proposals if routes[sid] == "deformation"},
                )
                for row in done.rows:
                    if row.get("status") == "ok":
                        outcome.summaries[row["id"]] = proposals[row["id"]].summary
                        outcome.written[row["id"]] = self._words(
                            outcome.written.get(row["id"]) == "transform"
                        )
                    else:
                        outcome.failed[row["id"]] = row
        return outcome

    def _words(self, transform_too: bool) -> str:
        """Say what a section's landing wrote when its deformation was written."""
        if transform_too:
            return "transform and deformation"
        return "deformation (rigid folded in)" if self.call.rigid else "deformation"

    def _rows(self, records: list[_Record], outcome: _Outcome) -> list[_Row]:
        """One row per section: what was written and why not, if it was not."""
        rows = []
        for rec in records:
            sid = rec.id
            code = outcome.failed.get(sid, {}).get("error")
            if sid in outcome.written and sid not in outcome.failed:
                problem, wrote = "", outcome.written[sid]
            elif sid in outcome.written:
                problem, wrote = f"transform only, deformation {code}", "transform only"
            else:
                problem, wrote = code or RefusalCode.FROZEN, "not written"
            rows.append(_Row(rec, outcome.summaries.get(sid, {}), wrote, problem))
        return rows

    def _sections_csv(self, rows: list[_Row]) -> None:
        """Save the per-section results as ``sections.csv``."""
        with open(self.call.folder / "sections.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "section",
                    "slot",
                    "snap_um",
                    "moved_median_mm",
                    "moved_max_mm",
                    "off_plane_max_um",
                    "displacement_median_mm",
                    "displacement_max_mm",
                    "fold_fraction",
                    "flags",
                    "set_aside_background",
                    "set_aside_artifact",
                    "written",
                    "problem",
                ]
            )
            for r in rows:
                rec = r.record
                disp = r.summary.get("displacement_mm", {})
                writer.writerow(
                    [
                        rec.id,
                        rec.placed.slot,
                        round(rec.placed.snap_um, 2),
                        round(rec.moved_mm[0], 4),
                        round(rec.moved_mm[1], 4),
                        round(rec.off_plane_um, 1),
                        disp.get("median", ""),
                        disp.get("max", ""),
                        r.summary.get("fold_fraction", ""),
                        " ".join(summary_flags(r.summary)),
                        rec.set_aside["WB"],
                        rec.set_aside["WA"],
                        r.wrote,
                        r.problem,
                    ]
                )

    def _report(
        self, lctx: ToolContext, result: engine.FitResult, rows: list[_Row], outcome: _Outcome
    ) -> ToolResult:
        """Word the landing's notice and draw the overlays."""
        call = self.call
        problems = [f"{r.record.id} ({r.problem})" for r in rows if r.problem]
        flagged = [
            f"{r.record.id} ({flag_words(r.summary)})" for r in rows if summary_flags(r.summary)
        ]
        shown = sorted(
            (r for r in rows if r.record.id in outcome.written),
            key=lambda r: (not summary_flags(r.summary), -r.record.moved_mm[0]),
        )[:OVERLAYS]
        drawn = ", ".join(f"{r.record.id} ({r.record.moved_mm[0]:.2f})" for r in shown)
        relative = _blocks.job_path(lctx, call.folder)
        left = [
            f"{r.record.id} ({r.record.estimated['left_out_fraction']:.0%})"
            for r in rows
            if r.record.estimated and r.record.estimated["left_out_fraction"] is not None
        ]
        energy = (
            f"emlddmm's energy record: {relative}/energy.csv (plot: {relative}/energy.png)"
            if result.energy_captured
            else "emlddmm's energy record was not captured"
        )
        waited = result.gpu_wait_s
        text = (
            f"{call.note}: written for {len(outcome.written)} of {len(rows)} sections"
            + (
                f"; it waited {round(waited)} s for the machine's GPU queue"
                if waited >= 1.0
                else ""
            )
            + (f"; not (fully) written: {', '.join(problems)}" if problems else "")
            + (f"; warp flags: {', '.join(flagged)}" if flagged else "")
            + (
                f"; the sorting emlddmm estimated left out, of the tissue, {', '.join(left)}, "
                "saved with each deformation"
                if left
                else ""
            )
            + (
                f". Overlays, with how far each moved (median, mm): {drawn}"
                if call.view and shown
                else ""
            )
            + f". {energy}; per-section results: {relative}/sections.csv; "
            f"the images it saw: {relative}/images/."
        )
        pictures = (
            [
                lctx.pictures.caption(
                    lctx.pictures.overlay_picture(r.record.id), f"{r.record.id}: after emlddmm"
                )
                for r in shown
            ]
            if call.view
            else []
        )
        return ToolResult(
            {
                "notice": text,
                "files": relative,
                "settings": call.settings,
                **({"gpu_wait_s": round(waited)} if waited >= 1.0 else {}),
                "rows": [
                    {
                        "id": r.record.id,
                        "inputs": r.record.placed.section.inputs,
                        "estimated": r.record.estimated,
                    }
                    for r in rows
                ],
            },
            pictures,
        )


def start_stack(ctx: ToolContext, call: StackCall) -> ToolResult:
    """Start the stack's fit as background work and answer at once."""
    stack = call.stack
    try:
        work = ctx.start_work([placed.id for placed in stack.sections], _StackLanding(call))
    except (Refused, *ENGINE_FAILURES) as exc:
        payload = (
            exc.payload()
            if isinstance(exc, Refused)
            else {"error": RefusalCode.ENGINE_UNAVAILABLE, "message": str(exc)}
        )
        details = {k: v for k, v in payload.items() if k not in ("status", "error", "message")}
        raise ToolError(
            str(payload.get("error")), str(payload.get("message") or exc), **details
        ) from exc
    other = [
        {"id": sid, "gets": GETS[route]}
        for sid, route in call.routes.items()
        if route != "transform"
    ]
    device = call.options.device
    return ToolResult(
        {
            "work": work,
            "stack": len(stack.sections),
            "left_out": stack.left_out,
            "settings": call.settings,
            "rows": [
                {"id": placed.id, "inputs": placed.section.inputs, "estimated": None}
                for placed in stack.sections
            ],
            **({"other_routes": other} if other else {}),
            "largest_snap_um": round(max(abs(p.snap_um) for p in stack.sections), 1),
            "device": device + (" (the fit takes tens of minutes)" if device == "cpu" else ""),
            "files": _blocks.job_path(ctx, call.folder),
        }
    )
