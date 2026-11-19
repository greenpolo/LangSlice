"""Running SPOmiAlign on the sections of a call and landing each result."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.core.failures import ENGINE_FAILURES
from langslice.engines import _blocks
from langslice.engines._pixels import outlined, sample_pixels
from langslice.engines.spomialign import engine, matching
from langslice.engines.spomialign.canvas import Pair, pair_inputs
from langslice.engines.spomialign.settings import (
    BRIGHTNESS,
    ENGINE_VERSION,
    FITS,
    MATCHER,
    MATCHES_PICTURE,
    MIN_MATCHES,
    WORK_H,
    WORK_W,
    WRITES,
    Options,
    mode_of,
)
from langslice.plugin_api import (
    FitFrame,
    RefusalCode,
    SectionView,
    ToolContext,
    ToolError,
    ToolResult,
)


def _engine(options: Options, mode: str, matches: int, pairs: list[Pair]) -> engine.EngineRun:
    """Run the engine on every section's pair in this process (one model load)."""
    absent = matching.weights_missing()
    try:
        return engine.fit_pairs(
            [pair.engine_pair() for pair in pairs],
            mode=mode,
            top_k=matches,
            device_name=options.device,
        )
    except ENGINE_FAILURES as error:
        missing = [name for name in matching.weights_missing() if name in absent]
        if missing:
            raise ToolError(
                RefusalCode.ENGINE_UNAVAILABLE,
                "SPOmiAlign's matcher weights are not in torch's "
                "hub cache and could not be downloaded: "
                + ", ".join(matching.WEIGHTS[name] for name in missing)
                + ". Use another method, or ask the user to install it.",
            ) from error
        first = str(error).strip().splitlines()[:1] or ["no message"]
        raise ToolError(
            RefusalCode.FIT_FAILED, f"SPOmiAlign did not run: {type(error).__name__}: {first[0]}"
        ) from error


def gpu_words(waited_s: float, retries: int) -> dict[str, Any]:
    """What a row says of the machine's GPU queue, as fireants_syn's rows say it:
    ``gpu_wait_s`` from a second's wait, ``retried`` when the GPU ran out of memory."""
    out: dict[str, Any] = {}
    if waited_s >= 1.0:
        out["gpu_wait_s"] = round(waited_s)
    if retries:
        times = "once" if retries == 1 else f"{retries} times"
        out["retried"] = (
            f"The GPU ran out of memory {times} (another process held it); "
            "the fit waited and ran again from the start, so this result is "
            "whole."
        )
    return out


def _to_native(pair: Pair, canvas_xy: np.ndarray) -> np.ndarray:
    """Target-canvas points (2 x N) as native atlas px (2 x N)."""
    grid = (pair.from_canvas @ np.vstack([canvas_xy, np.ones(canvas_xy.shape[1])]))[:2]
    if pair.applied is None:
        return grid
    return sample_pixels(pair.applied, grid.T).T


def _full_map(frame: FitFrame, pair: Pair, matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Native atlas px of every fit-grid pixel under SPOmiAlign's full map.

    A fit-grid pixel u sits at canvas px c = to_canvas u; SPOmiAlign takes c to
    M c + R(c) on the target canvas (R zero off the canvas, as ITK's field is), which
    :func:`_to_native` reads as native atlas px."""
    gh, gw = _blocks.section_pixels(frame).shape[:2]
    ys, xs = np.mgrid[0:gh, 0:gw]
    u = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]).astype(np.float64)
    c = (pair.to_canvas @ u)[:2]
    linear = engine.homogeneous(matrix)
    moved = linear[:2, :2] @ c + linear[:2, 2:3]
    cx = c[0].reshape(gh, gw).astype(np.float32)
    cy = c[1].reshape(gh, gw).astype(np.float32)
    for axis in range(2):
        moved[axis] += cv2.remap(
            np.ascontiguousarray(residual[..., axis]),
            cx,
            cy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).ravel()
    return _to_native(pair, moved).T.reshape(gh, gw, 2)


def _pairs_picture(
    pair: Pair,
    used: np.ndarray,
    dropped: np.ndarray,
    path: Path,
) -> None:
    """The two canvases side by side, the most confident pairs as dots of one colour on both
    (coloured by where they sit on the section) and the pairs passed over in the damage zone
    as red crosses on the atlas (a file in the job, not sent)."""
    left, right = pair.target, pair.source
    gap = MATCHES_PICTURE.gap_px
    image = Image.fromarray(
        np.concatenate([left, np.full((left.shape[0], gap, 3), 255, np.uint8), right], axis=1)
    )
    draw = ImageDraw.Draw(image)
    shift = left.shape[1] + gap
    r = max(3, left.shape[1] // 250)
    height = right.shape[0]
    shown = MATCHES_PICTURE.drawn_pairs
    for ax, ay, sx, sy, _ in used[:shown]:
        hue = np.uint8(170 * sy / max(height, 1)), np.uint8(255), np.uint8(230)
        rgb = cv2.cvtColor(np.array([[hue]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        colour = tuple(int(v) for v in rgb)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=colour, outline="black")
        draw.ellipse([sx + shift - r, sy - r, sx + shift + r, sy + r], fill=colour, outline="black")
    cross = MATCHES_PICTURE.cross_colour
    for ax, ay, *_ in dropped:
        draw.line([(ax - r, ay - r), (ax + r, ay + r)], fill=cross, width=2)
        draw.line([(ax - r, ay + r), (ax + r, ay - r)], fill=cross, width=2)
    draw.text(
        (6, 6),
        f"atlas (left) and section (right) as SPOmiAlign saw them; the "
        f"{min(shown, len(used))} most confident of {len(used)} pairs, each a dot of "
        f"one colour on both; red crosses: "
        f"{len(dropped)} passed over (marked damage or the mask)",
        fill="black",
    )
    edge = MATCHES_PICTURE.long_edge
    image.thumbnail((edge, edge))
    image.save(path)


def _files(here: Path, pair: Pair, got: engine.EngineAnswer, settings: Mapping[str, Any]) -> None:
    """Save the call's files for one section: the two canvases, the settings and the pairs."""
    here.mkdir(parents=True, exist_ok=True)
    Image.fromarray(outlined(pair.target, ~pair.keep)).save(here / "atlas.png")
    Image.fromarray(pair.source).save(here / "section.png")
    (here / "settings.json").write_text(
        json.dumps(settings, indent=1, default=str), encoding="utf-8", newline="\n"
    )
    none = np.zeros((0, 5), np.float32)
    used = none if got.used is None else got.used
    dropped = none if got.dropped_pairs is None else got.dropped_pairs
    with open(here / "matches.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atlas_x", "atlas_y", "section_x", "section_y", "confidence", "dropped"])
        for rows, flag in ((used, 0), (dropped, 1)):
            for row in rows:
                writer.writerow([*(f"{v:.2f}" for v in row[:4]), f"{row[4]:.4g}", flag])
    _pairs_picture(pair, used, dropped, here / "matches.png")


def _refusal(view_of: SectionView, frame: FitFrame, mode: str) -> dict[str, object] | None:
    """Why the engine *mode* cannot start on this section, before the engine runs."""
    sid = view_of.id
    frozen = [prop for prop in WRITES[mode] if prop in view_of.frozen]
    if frozen:
        refusal = ToolError(
            RefusalCode.FROZEN, f"the section's {' and '.join(frozen)} may not be changed"
        )
        return {"id": sid, **refusal.payload()}
    if frame.placed_atlas_to_section is None:
        refusal = ToolError(
            RefusalCode.PRECONDITION,
            "SPOmiAlign starts from the section's placement: give it a transform first",
        )
        return {"id": sid, **refusal.payload()}
    if mode != engine.WARP and view_of.deformation:
        refusal = ToolError(
            RefusalCode.PRECONDITION,
            "this fit would write a new transform, which clears the "
            'section\'s deformation; fit "warp" to build on it',
        )
        return {"id": sid, **refusal.payload()}
    return None


@dataclass(frozen=True)
class _Call:
    """Hold what every section of one ``spomialign`` call shares.

    Parameters
    ----------
    mode
        The engine's mode (``rigid``, ``affine``, ``warp`` or ``affine_warp``).
    section_image, atlas_image, mask
        The images and mask asked for.
    smoothness_um
        The warp's smoothing, um.
    matches
        How many of the most confident pairs are used.
    view
        Whether pictures are sent.
    options
        The plugin's options.
    settings
        The call's settings, as the reply and the deformation record them.
    """

    mode: str
    section_image: str
    atlas_image: str
    mask: str
    smoothness_um: float
    matches: int
    view: bool
    options: Options
    settings: dict[str, Any]


@dataclass(frozen=True)
class _Ready:
    """Hold one section ready to match.

    Parameters
    ----------
    sid
        The section's filename.
    frame
        Its fit frame.
    pair
        The pair SPOmiAlign sees.
    digest
        Its registration's digest when the call began.
    inputs
        The ``inputs`` block of its reply.
    resolved
        The atlas image that was used.
    """

    sid: str
    frame: FitFrame
    pair: Pair
    digest: str
    inputs: dict[str, Any]
    resolved: str


def _call(
    options: Options,
    fit: str,
    rigid: bool,
    section_image: str,
    atlas_image: str,
    mask: str,
    smoothness_um: float,
    matches: int,
    view: bool,
    sections: list[str],
) -> _Call:
    """Check a call's arguments and gather them."""
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    if rigid and fit != "transform":
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            "rigid fits the transform alone: SPOmiAlign's warp "
            "follows an affine or the registration the section holds; give fit "
            '"transform" with rigid, or leave rigid false',
        )
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "name one to twenty sections")
    if int(matches) < MIN_MATCHES:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"matches is at least {MIN_MATCHES}")
    if not float(smoothness_um) > 0:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "smoothness_um is above 0")
    settings = {
        "fit": fit,
        "rigid": bool(rigid),
        "section_image": section_image,
        "atlas_image": atlas_image,
        "mask": mask,
        "smoothness_um": float(smoothness_um),
        "matches": int(matches),
    }
    return _Call(
        mode=mode_of(fit, bool(rigid)),
        section_image=section_image,
        atlas_image=atlas_image,
        mask=mask,
        smoothness_um=float(smoothness_um),
        matches=int(matches),
        view=view,
        options=options,
        settings=settings,
    )


def _read_sections(
    ctx: ToolContext, call: _Call, sections: list[str]
) -> tuple[list[dict[str, object]], list[_Ready]]:
    """Read every section's pair; a section that cannot start has its refusal as a row."""
    rows: list[dict[str, object]] = []
    ready: list[_Ready] = []
    for name in sections:
        view_of = ctx.reads.section(name)
        frame = _blocks.frame_for(ctx, view_of.id, call.atlas_image, call.section_image)
        plane_image, resolved = _blocks.atlas_plane(frame, call.atlas_image)
        refused = _refusal(view_of, frame, call.mode)
        if refused:
            rows.append(refused)
            continue
        held = _blocks.section_mask(ctx, view_of.id, call.mask)
        on_grid = call.mode == engine.WARP and frame.applied_map is not None
        pair = pair_inputs(
            frame,
            plane_image,
            weights=None if held is None else np.asarray(held.weights, np.float32),
            on_grid=on_grid,
            smoothness_um=call.smoothness_um,
        )
        inputs = _blocks.inputs_block(
            ctx,
            frame,
            section_image=call.section_image,
            atlas_image=call.atlas_image,
            resolved=resolved,
            mask=held,
            brightness=BRIGHTNESS,
        )
        digest = ctx.reads.section_inputs(view_of.id, deformation=on_grid)
        ready.append(_Ready(view_of.id, frame, pair, digest, inputs, resolved))
    return rows, ready


def _land(
    ctx: ToolContext,
    call: _Call,
    folder: Path,
    section: _Ready,
    got: engine.EngineAnswer,
    wait_s: float,
) -> tuple[dict[str, object], Image.Image | None]:
    """Save one section's files, write its registration and give its row and picture."""
    sid, pair, inputs = section.sid, section.pair, section.inputs
    mode = call.mode
    here = folder / Path(sid).stem
    start = "the deformation it held" if pair.on_grid else "its linear placement"
    by_mask = got.dropped_by_mask
    in_damage = got.dropped - by_mask
    _files(
        here,
        pair,
        got,
        {
            **call.settings,
            "inputs": inputs,
            "atlas_image_resolved": section.resolved,
            "smoothing_px": pair.smoothing_px,
            "start": start,
            "canvas_px": pair.target.shape[:2],
            "device": call.options.device,
            "pairs_used": got.pairs,
            "pairs_dropped_in_damage": in_damage,
            "pairs_dropped_by_mask": by_mask,
        },
    )
    relative = _blocks.job_path(ctx, here)
    warps = mode in (engine.WARP, engine.AFFINE_WARP)
    matrix, residual = got.matrix, got.residual
    if got.error or matrix is None or (warps and residual is None):
        failed = ToolError(RefusalCode.FIT_FAILED, got.error)
        return {**failed.payload(section_id=sid), "files": relative}, None
    what = mode.replace("_", " and ") if mode != engine.WARP else f"warp on {start}"
    note = (
        f"SPOmiAlign {what}, atlas {section.resolved}, {got.pairs} pairs, "
        f"{in_damage} in marked damage dropped"
        + (f", {by_mask} outside the mask dropped" if call.mask != "none" else "")
    )
    engine_settings = {
        "matcher": MATCHER,
        "mode": mode,
        "working_grid_px": [WORK_W, WORK_H],
        "working_um": round(float(pair.work_um), 3),
        "smoothness_um": call.smoothness_um,
        "smoothing_sigma_px": round(float(pair.smoothing_px), 3),
        "matches": call.matches,
        "canvas_px": list(pair.target.shape[:2]),
    }
    row: dict[str, Any] = {
        "id": sid,
        "status": "ok",
        "fitted": note,
        "files": relative,
        "inputs": inputs,
        "estimated": None,
        "engine_settings": engine_settings,
        **gpu_words(wait_s, got.oom_retries),
    }
    if call.mask != "none":
        row["mask_note"] = "approximated by dropping matches"
    if mode != engine.WARP:
        to_native = (pair.from_canvas @ engine.homogeneous(matrix) @ pair.to_canvas)[:2]
        proposal = ctx.proposals.transform_from_matrix(sid, to_native, note=note)
        written = ctx.writes.set_transforms([proposal.row()], expect={sid: section.digest})
        if sid not in written.written:
            return {**written.rows[0], "id": sid}, None
        row["transform"] = proposal.knobs
    if warps and residual is not None:
        native = _full_map(section.frame, pair, matrix, residual)
        warp = ctx.proposals.deformation_from_map(
            sid,
            native,
            note=note,
            meta={"inputs": inputs, "estimated": None},
            start="current" if pair.on_grid else "linear",
            settings=call.settings,
            engine_settings=engine_settings,
            engine_version=ENGINE_VERSION,
        )
        written = ctx.writes.set_deformations(
            [warp.row()], expect={sid: section.digest} if mode == engine.WARP else None
        )
        if sid not in written.written:
            return {**written.rows[0], "id": sid}, None
        row.update(warp.summary)
    picture = (
        ctx.pictures.caption(ctx.pictures.overlay_picture(sid), f"{sid}: {note}")
        if call.view
        else None
    )
    return row, picture


def run_call(
    ctx: ToolContext,
    options: Options,
    sections: list[str],
    fit: str,
    rigid: bool,
    section_image: str,
    atlas_image: str,
    mask: str,
    smoothness_um: float,
    matches: int,
    view: bool,
) -> ToolResult:
    call = _call(
        options,
        fit,
        rigid,
        section_image,
        atlas_image,
        mask,
        smoothness_um,
        matches,
        view,
        sections,
    )
    rows, ready = _read_sections(ctx, call, sections)
    if not ready:
        return ToolResult({"rows": rows, "settings": call.settings})
    folder = _blocks.call_folder(ctx, call.mode)
    run = _engine(options, call.mode, call.matches, [section.pair for section in ready])
    pictures: list[Image.Image] = []
    with ctx.writes.writing():
        for section, got in zip(ready, run.answers, strict=True):
            row, picture = _land(ctx, call, folder, section, got, run.gpu_wait_s)
            rows.append(row)
            if picture is not None:
                pictures.append(picture)
    return ToolResult({"rows": rows, "settings": call.settings}, pictures)
