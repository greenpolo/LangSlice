"""SPOmiAlign as a LangSlice tool: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.spomialign.plugin:tools``);
it is imported from this package (no ``PYTHONPATH``). Option
(``plugin_config``): ``device`` (the torch device; by default the first GPU
when torch sees one, else the CPU). The method itself runs in this process,
in ``engine.py`` and ``matching.py``, on RoMa (``romatch``, LangSlice's
``engines`` extra), imported only when a call runs; a call without it is
refused ``ENGINE_MISSING``. This module imports only
``langslice.plugin_api`` and the engine.

Each call puts the section on SPOmiAlign's canvas, runs the engine on (atlas
image, section), and maps the answer back: the linear part through
``transform_from_matrix``, the full map through ``deformation_from_map``.
Canvas: for rigid, affine and affine_warp the atlas plane's grid, upsampled
to about :data:`CANVAS_EDGE` px, with the section resampled onto it through
its current transform (as the published user's hand-picked plate and
rotation did); for warp on a section that holds a deformation, the fit grid,
with the atlas resampled onto it through that deformation
(``FitFrame.applied_map``), so the warp starts on top of a landmark warp, as
``stalign_lddmm`` does. A new transform clears a deformation, so rigid,
affine and affine_warp are refused on a section that holds one
(``HAS_DEFORMATION``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from langslice.engines import _blocks
from langslice.engines.spomialign import engine, matching
from langslice.plugin_api import (
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
)

DEFAULTS = {"device": ""}
MISSING = (
    "SPOmiAlign cannot run here: its matcher, RoMa (the romatch package), "
    "is not installed; install LangSlice's engines extra: "
    'pip install "langslice[engines]"'
)
CANVAS_EDGE = 1536
FITS = ("transform", "warp", "both")
WRITES = {
    engine.RIGID: ("affine",),
    engine.AFFINE: ("affine",),
    engine.WARP: ("deformation",),
    engine.AFFINE_WARP: ("affine", "deformation"),
}
ENGINE_VERSION = "SPOmiAlign c7f8b7a"
MATCHER = "RoMa roma_outdoor, coarse resolution 560"


def mode_of(fit: str, rigid: bool) -> str:
    """The engine mode of a call's *fit* and *rigid*: ``rigid`` or ``affine``
    for the transform, ``warp``, or ``affine_warp`` for both."""
    if fit == "transform":
        return engine.RIGID if rigid else engine.AFFINE
    return engine.WARP if fit == "warp" else engine.AFFINE_WARP


NISSL = _blocks.NISSL
WORK_W, WORK_H = matching.W, matching.H
DRAWN_PAIRS = 300


def spomialign(
    sections: list[str],
    fit: Literal["transform", "warp", "both"],
    rigid: bool = False,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    smoothness_um: float = 150.0,
    matches: int = 5000,
    view: bool = True,
) -> dict[str, Any]:
    """Register sections to the atlas with SPOmiAlign, on their current registration.

    SPOmiAlign (Wang et al. 2026, Briefings in Bioinformatics) was built to align
    spatial-omics sections (transcriptomics, proteomics, H&E) to each other and to
    the Allen Nissl plates. Instead of comparing brightness it matches points with
    RoMa (Edstedt et al. 2024), a dense feature matcher pretrained on DINOv2
    features, so only the tissue's shape and structure need agree, not its stain.

    SPOmiAlign matches distinctive points between a section image (by
    name) and an atlas image with a pretrained dense matcher (RoMa), weights
    the matches toward tissue edges, keeps the most confident (matches), and
    fits the chosen transform to the pairs; it compares no intensities pixel by
    pixel and runs no iterations; a call takes under a minute. It starts from
    the section's current placement and does not search from scratch~
    [[?position_sections|interactive_transform :~
    [[?position_sections  place the section with position_sections]]~
    [[?position_sections,interactive_transform  and]]~
    [[?interactive_transform  set its flip, quarter turn and rough transform with ~
    interactive_transform]] first]]. The section's marked damage regions are left
    out automatically[[?mark_damage ; where tissue is missing or lies folded over ~
    other anatomy, mark those regions with mark_damage first (tissue that is ~
    intact but shifted is not damage)]].

    fit chooses what is fitted:
    - "transform": the section's transform: rotation, shift, scale and shear
      (an affine), or with rigid rotation and shift alone, the section keeping
      its size. Written as the section's transform.
    - "warp": a smooth local deformation on top of the registration the section
      holds (its transform, and its deformation when it has one[[?set_landmarks ~
      , a landmark warp included, so the landmark move is kept]]); the transform
      stays. Written as the section's deformation.
    - "both": an affine, then a warp on top of it (SPOmiAlign's own default).
      Writes both.
    A new transform clears the section's deformation, so "transform" and
    "both" are refused on a section that holds one: fit "warp" on it instead.

    The warp is SPOmiAlign's smoothed match field: each pair's displacement,
    filled in between the pairs and smoothed by a Gaussian of smoothness_um.
    Where the section and the atlas image share little tissue (a damaged or
    partial section), the affine follows wrong matches, because it weighs
    every pair equally and rejects none; the sign is an atlas shrunk off the
    tissue in the returned overlay. For such a section, fit "transform" with
    rigid, check the returned overlay, then fit "warp". The warp bends whole
    regions and does not refine individual borders.

    SPOmiAlign reads one image pair per section, nothing else, and has no
    weighting: it estimates no mask of its own, and a section mask can only
    drop matches. With a mask named, every match whose end on the section lies
    where the mask is below one half is dropped and the next most confident
    take their place; the mask is approximated by dropping matches, not
    applied to the pixels. The atlas side's marked damage is dropped the same
    way. The section is read as 8-bit RGB with the atlas image drawn as a
    brightfield Nissl, dark on white.

    Each call is one undoable write.

    Args:
        sections: One to twenty filenames (the extension may be left off).
        fit: "transform", "warp" or "both".
        rigid: True fits the transform as a rotation and a shift alone (fit
            "transform" only). Default false.
        section_image: The section image to match, by name: "preprocessed", a raw channel~
            [[?make_image , or an image made with make_image]]. Default "preprocessed".
        atlas_image: One of the atlas's images (the job statement lists
            them: "template" and any other reference images the atlas ships,
            by their names), or "nissl", the atlas's Nissl (its template when
            it ships none). Default "nissl"; the reply names the image used.
        mask: "none", or the name of a mask of each section[[?make_mask  ~
            (made with make_mask)]]; matches ending where it is below one half
            are dropped. Default "none".
        smoothness_um: The width (sigma) of the Gaussian that smooths the
            warp's displacement field, in micrometres of the atlas; larger
            bends broader regions as one. Default 150.
        matches: How many of the most confident matched pairs the fit uses.
            Default 5000 (SPOmiAlign's).
        view: True returns the picture described below; false returns none.

    Returns:
        Per section what was fitted and on what, what the fit used (inputs:
        the section image, the atlas image as asked and as used, the mask, the
        damage, the brightness scaling), what SPOmiAlign ran at
        (engine_settings: the matcher, its working grid and micrometres per
        pixel, the smoothing in its pixels, the pairs), the transform written
        (rotation_deg, scale_x, scale_y, shear, translate_x_mm,
        translate_y_mm about the canvas centre) when one was, for a warp its
        displacement (median and max, mm), fold fraction and plausibility
        flags, the pairs used and dropped, and the folder in the job holding
        the call's files (the two images SPOmiAlign saw, its settings and the
        matched pairs as a table and a picture[[?read_file , which read_file ~
        opens]]); and a picture of each section under its new registration
        with the atlas borders on it. SPOmiAlign estimates no weighting, so
        estimated is null.
    """
    ...


def _canvas_scale(shape: tuple[int, int]) -> int:
    return max(1, round(CANVAS_EDGE / max(shape)))


def _scale(k: int) -> np.ndarray:
    """Grid px -> canvas px (pixel centres), canvas k times finer."""
    return np.array([[k, 0.0, 0.5 * k - 0.5], [0.0, k, 0.5 * k - 0.5], [0.0, 0.0, 1.0]])


def _homog(m: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(m, np.float64)[:2], [0.0, 0.0, 1.0]])


def _atlas_picture(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The atlas image as SPOmiAlign's published reference looks, a brightfield Nissl
    plate: white background, dense tissue dark (0.5-99.5 percentile window inside the
    brain), uint8 RGB. The same rendering as the method-alone arm
    (``comparators/rigid/spomialign_runner.py`` ``atlas_render``)."""
    inside = labels > 0
    values = image[inside] if inside.any() else image.ravel()
    lo, hi = np.percentile(values.astype(np.float32), [0.5, 99.5])
    scaled = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    grey = (255 * (1.0 - scaled)).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def _slide_fill(stain: np.ndarray) -> tuple[float, ...]:
    border = np.concatenate([stain[0], stain[-1], stain[:, 0], stain[:, -1]])
    return tuple(float(v) for v in np.median(border, axis=0))


def _inputs(
    frame: Any, plane_image: np.ndarray, on_grid: bool, weights: np.ndarray | None = None
) -> dict[str, Any]:
    """The pair SPOmiAlign sees, on one canvas: over the atlas plane through the linear
    placement, or (``on_grid``) over the fit grid through the section's deformation.
    ``to_canvas`` takes fit-grid px to canvas px; ``from_canvas`` takes canvas px to the
    grid of ``lookup`` (native atlas px, or with ``on_grid`` the fit grid, read through
    ``applied`` to native atlas px). *weights* (a section mask on the fit grid) becomes
    ``section_keep`` on the section's canvas, True where the weight is at least one half;
    ``um_per_px`` is the atlas micrometres one canvas pixel spans."""
    labels = frame.atlas["annotation"]
    picture = _atlas_picture(plane_image, labels)
    if frame.excluded is not None:
        picture[frame.excluded] = 255
    zone = frame.damage_zone if frame.damage_zone is not None else np.zeros(labels.shape, bool)
    stain = _blocks.section_pixels(frame)
    plane_px = float(np.mean(frame.plane["px_um"]))
    if on_grid:
        applied = np.asarray(frame.applied_map, np.float32)
        gh, gw = applied.shape[:2]
        k = _canvas_scale((gh, gw))
        size = (gw * k, gh * k)
        picture = cv2.remap(
            picture,
            applied[..., 0],
            applied[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        zone = cv2.remap(
            zone.astype(np.uint8),
            applied[..., 0],
            applied[..., 1],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        source = cv2.resize(stain, size, interpolation=cv2.INTER_LINEAR)
        K = _scale(k)
        to_canvas = K
        placed = np.asarray(frame.placed_atlas_to_section, np.float64)
        um_per_px = plane_px / float(np.sqrt(abs(np.linalg.det(placed[:2, :2])))) / k
        section_keep = (
            None
            if weights is None
            else cv2.resize(np.asarray(weights, np.float32), size, interpolation=cv2.INTER_LINEAR)
            >= 0.5
        )
        extra: dict[str, Any] = {"applied": applied}
    else:
        h, w = labels.shape
        k = _canvas_scale((h, w))
        size = (w * k, h * k)
        K = _scale(k)
        start = frame.placed_atlas_to_section
        to_canvas = K @ np.linalg.inv(start)
        source = cv2.warpAffine(
            stain,
            (start @ np.linalg.inv(K))[:2],
            size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_slide_fill(stain),
        )
        um_per_px = plane_px / k
        section_keep = (
            None
            if weights is None
            else cv2.warpAffine(
                np.asarray(weights, np.float32),
                (start @ np.linalg.inv(K))[:2],
                size,
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            >= 0.5
        )
        extra = {"start": start}
    target = cv2.resize(picture, size, interpolation=cv2.INTER_LINEAR)
    keep = ~cv2.resize(np.asarray(zone, np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(
        bool
    )
    return {
        "target": target,
        "source": source,
        "keep": keep,
        "K": K,
        "to_canvas": to_canvas,
        "from_canvas": np.linalg.inv(K),
        "on_grid": on_grid,
        "um_per_px": um_per_px,
        "section_keep": section_keep,
        "work_um": um_per_px * float(np.mean([size[0] / WORK_W, size[1] / WORK_H])),
        **extra,
    }


def _engine(
    config: dict[str, Any], mode: str, matches: int, pairs: list[dict[str, Any]]
) -> engine.EngineRun:
    """Run the engine on every section's pair in this process (one model load)."""
    inputs = [
        engine.EnginePair(
            target=cast("npt.NDArray[np.uint8]", pair["target"]),
            source=cast("npt.NDArray[np.uint8]", pair["source"]),
            keep=cast("npt.NDArray[np.bool_]", pair["keep"]),
            smoothing_px=cast(float, pair["smoothing_px"]),
            section_keep=cast("npt.NDArray[np.bool_] | None", pair["section_keep"]),
        )
        for pair in pairs
    ]
    absent = matching.weights_missing()
    try:
        return engine.fit_pairs(
            inputs, mode=mode, top_k=matches, device_name=str(config.get("device") or "")
        )
    except Exception as error:
        missing = [name for name in matching.weights_missing() if name in absent]
        if missing:
            raise ToolError(
                RefusalCode.ENGINE_MISSING,
                "SPOmiAlign's matcher weights are not in torch's "
                "hub cache and could not be downloaded: "
                + ", ".join(matching.WEIGHTS[name] for name in missing),
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


def _to_native(pair: dict[str, Any], canvas_xy: np.ndarray) -> np.ndarray:
    """Target-canvas points (2 x N) as native atlas px (2 x N)."""
    grid = (pair["from_canvas"] @ np.vstack([canvas_xy, np.ones(canvas_xy.shape[1])]))[:2]
    if not pair["on_grid"]:
        return grid
    applied = pair["applied"]
    n = grid.shape[1]
    rows = -(-n // 4096)
    pad = np.zeros((2, rows * 4096 - n), np.float32)
    gx, gy = (
        np.concatenate([grid[c].astype(np.float32), pad[c]]).reshape(rows, 4096) for c in range(2)
    )
    return np.stack(
        [
            cv2.remap(
                np.ascontiguousarray(applied[..., c]),
                gx,
                gy,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).ravel()[:n]
            for c in range(2)
        ]
    )


def _full_map(
    frame: Any, pair: dict[str, Any], matrix: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Native atlas px of every fit-grid pixel under SPOmiAlign's full map.

    A fit-grid pixel u sits at canvas px c = to_canvas u; SPOmiAlign takes c to
    M c + R(c) on the target canvas (R zero off the canvas, as ITK's field is), which
    :func:`_to_native` reads as native atlas px."""
    gh, gw = _blocks.section_pixels(frame).shape[:2]
    ys, xs = np.mgrid[0:gh, 0:gw]
    u = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]).astype(np.float64)
    c = (pair["to_canvas"] @ u)[:2]
    moved = _homog(matrix)[:2, :2] @ c + _homog(matrix)[:2, 2:3]
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


def _outline(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (230, 30, 160), max(1, image.shape[1] // 600))
    return out


def _pairs_picture(pair: dict[str, Any], used: np.ndarray, dropped: np.ndarray, path: Path) -> None:
    """The two canvases side by side, the most confident pairs as dots of one colour on both
    (coloured by where they sit on the section) and the pairs passed over in the damage zone
    as red crosses on the atlas (a file in the job, not sent)."""
    left, right = pair["target"], pair["source"]
    gap = 16
    image = Image.fromarray(
        np.concatenate([left, np.full((left.shape[0], gap, 3), 255, np.uint8), right], axis=1)
    )
    draw = ImageDraw.Draw(image)
    shift = left.shape[1] + gap
    r = max(3, left.shape[1] // 250)
    height = right.shape[0]
    for ax, ay, sx, sy, _ in used[:DRAWN_PAIRS]:
        hue = np.uint8(170 * sy / max(height, 1)), np.uint8(255), np.uint8(230)
        rgb = cv2.cvtColor(np.array([[hue]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        colour = tuple(int(v) for v in rgb)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=colour, outline="black")
        draw.ellipse([sx + shift - r, sy - r, sx + shift + r, sy + r], fill=colour, outline="black")
    for ax, ay, *_ in dropped:
        draw.line([(ax - r, ay - r), (ax + r, ay + r)], fill=(220, 30, 30), width=2)
        draw.line([(ax - r, ay + r), (ax + r, ay - r)], fill=(220, 30, 30), width=2)
    draw.text(
        (6, 6),
        f"atlas (left) and section (right) as SPOmiAlign saw them; the "
        f"{min(DRAWN_PAIRS, len(used))} most confident of {len(used)} pairs, each a dot of "
        f"one colour on both; red crosses: "
        f"{len(dropped)} passed over (marked damage or the mask)",
        fill="black",
    )
    image.thumbnail((2400, 2400))
    image.save(path)


def _files(
    here: Path, pair: dict[str, Any], got: engine.EngineAnswer, settings: dict[str, Any]
) -> None:
    here.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_outline(pair["target"], ~pair["keep"])).save(here / "atlas.png")
    Image.fromarray(pair["source"]).save(here / "section.png")
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


def _refusal(view_of: Any, frame: Any, mode: str) -> dict[str, object] | None:
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
            RefusalCode.INVALID_LINEAR_PLACEMENT,
            "SPOmiAlign starts from the section's placement: give it a transform first",
        )
        return {"id": sid, **refusal.payload()}
    if mode != engine.WARP and view_of.deformation:
        refusal = ToolError(
            RefusalCode.HAS_DEFORMATION,
            "this fit would write a new transform, which clears the "
            'section\'s deformation; fit "warp" to build on it',
        )
        return {"id": sid, **refusal.payload()}
    return None


def _run(
    ctx: ToolContext,
    config: dict[str, Any],
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
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGS, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    if rigid and fit != "transform":
        raise ToolError(
            RefusalCode.BAD_ARGS,
            "rigid fits the transform alone: SPOmiAlign's warp "
            "follows an affine or the registration the section holds; give fit "
            '"transform" with rigid, or leave rigid false',
        )
    mode = mode_of(fit, bool(rigid))
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGS, "name one to twenty sections")
    if int(matches) < 3:
        raise ToolError(RefusalCode.BAD_ARGS, "matches is at least 3")
    if not float(smoothness_um) > 0:
        raise ToolError(RefusalCode.BAD_ARGS, "smoothness_um is above 0")
    matches, smoothness_um = int(matches), float(smoothness_um)
    settings = {
        "fit": fit,
        "rigid": bool(rigid),
        "section_image": section_image,
        "atlas_image": atlas_image,
        "mask": mask,
        "smoothness_um": smoothness_um,
        "matches": matches,
    }
    rows: list[dict[str, Any]] = []
    ready: list[tuple[str, Any, dict[str, Any], str, dict[str, Any], str]] = []
    for name in sections:
        view_of = ctx.section(name)
        frame = _blocks.frame_for(ctx, view_of.id, atlas_image, section_image)
        plane_image, resolved = _blocks.atlas_plane(frame, atlas_image)
        refused = _refusal(view_of, frame, mode)
        if refused:
            rows.append(refused)
            continue
        held = _blocks.section_mask(ctx, view_of.id, mask)
        on_grid = mode == engine.WARP and frame.applied_map is not None
        pair = _inputs(
            frame,
            plane_image,
            on_grid,
            None if held is None else np.asarray(held.weights, np.float32),
        )
        pair["smoothing_px"] = smoothness_um / pair["work_um"]
        inputs = _blocks.inputs_block(
            ctx,
            frame,
            section_image=section_image,
            atlas_image=atlas_image,
            resolved=resolved,
            mask=held,
            brightness="8-bit, Nissl rendered dark on white",
        )
        ready.append(
            (
                view_of.id,
                frame,
                pair,
                ctx.section_inputs(view_of.id, deformation=on_grid),
                inputs,
                resolved,
            )
        )
    if not ready:
        return ToolResult({"rows": rows, "settings": settings})
    folder = _blocks.call_folder(ctx, mode)
    run = _engine(config, mode, matches, [item[2] for item in ready])
    pictures = []
    with ctx.writing():
        for (sid, frame, pair, digest, inputs, resolved), got in zip(
            ready, run.answers, strict=True
        ):
            here = folder / Path(sid).stem
            start = "the deformation it held" if pair["on_grid"] else "its linear placement"
            by_mask = got.dropped_by_mask
            in_damage = got.dropped - by_mask
            _files(
                here,
                pair,
                got,
                {
                    **settings,
                    "inputs": inputs,
                    "atlas_image_resolved": resolved,
                    "smoothing_px": pair["smoothing_px"],
                    "start": start,
                    "canvas_px": pair["target"].shape[:2],
                    "device": config.get("device") or "SPOmiAlign's choice",
                    "pairs_used": got.pairs,
                    "pairs_dropped_in_damage": in_damage,
                    "pairs_dropped_by_mask": by_mask,
                },
            )
            relative = _blocks.job_path(ctx, here)
            if got.error or got.matrix is None:
                failed = ToolError(RefusalCode.FIT_FAILED, got.error)
                rows.append({**failed.payload(section_id=sid), "files": relative})
                continue
            what = mode.replace("_", " and ") if mode != engine.WARP else f"warp on {start}"
            note = (
                f"SPOmiAlign {what}, atlas {resolved}, {got.pairs} pairs, "
                f"{in_damage} in marked damage dropped"
                + (f", {by_mask} outside the mask dropped" if mask != "none" else "")
            )
            engine_settings = {
                "matcher": MATCHER,
                "mode": mode,
                "working_grid_px": [WORK_W, WORK_H],
                "working_um": round(float(pair["work_um"]), 3),
                "smoothness_um": smoothness_um,
                "smoothing_sigma_px": round(float(pair["smoothing_px"]), 3),
                "matches": matches,
                "canvas_px": list(pair["target"].shape[:2]),
            }
            row: dict[str, Any] = {
                "id": sid,
                "status": "ok",
                "fitted": note,
                "files": relative,
                "inputs": inputs,
                "estimated": None,
                "engine_settings": engine_settings,
                **gpu_words(run.gpu_wait_s, got.oom_retries),
            }
            if mask != "none":
                row["mask_note"] = "approximated by dropping matches"
            if mode != engine.WARP:
                to_native = (pair["from_canvas"] @ _homog(got.matrix) @ pair["to_canvas"])[:2]
                proposal = ctx.transform_from_matrix(sid, to_native, note=note)
                written = ctx.set_transforms([proposal.row()], expect={sid: digest})
                if sid not in written.written:
                    rows.append({**written.rows[0], "id": sid})
                    continue
                row["transform"] = proposal.knobs
            if mode in (engine.WARP, engine.AFFINE_WARP):
                assert got.residual is not None
                native = _full_map(frame, pair, got.matrix, got.residual)
                warp = ctx.deformation_from_map(
                    sid,
                    native,
                    note=note,
                    meta={"sitk_max_px": got.sitk_max_px, "inputs": inputs, "estimated": None},
                    start="current" if pair["on_grid"] else "linear",
                    settings=settings,
                    engine_settings=engine_settings,
                    engine_version=ENGINE_VERSION,
                )
                written = ctx.set_deformations(
                    [warp.row()], expect={sid: digest} if mode == engine.WARP else None
                )
                if sid not in written.written:
                    rows.append({**written.rows[0], "id": sid})
                    continue
                row.update(warp.summary)
            rows.append(row)
            if view:
                pictures.append(ctx.caption(ctx.overlay_picture(sid), f"{sid}: {note}"))
    return ToolResult({"rows": rows, "settings": settings}, pictures)


def tools(config: dict[str, Any] | None = None) -> list[ToolSpec]:
    options = {**DEFAULTS, **(config or {})}

    def handler(
        ctx: ToolContext,
        sections: list[str],
        fit: str,
        rigid: bool = False,
        section_image: str = "preprocessed",
        atlas_image: str = NISSL,
        mask: str = "none",
        smoothness_um: float = 150.0,
        matches: int = 5000,
        view: bool = True,
    ) -> ToolResult:
        if not matching.is_installed():
            raise ToolError(RefusalCode.ENGINE_MISSING, MISSING)
        return _run(
            ctx,
            options,
            sections,
            fit,
            rigid,
            section_image,
            atlas_image,
            mask,
            smoothness_um,
            matches,
            view,
        )

    return [
        ToolSpec(
            "spomialign",
            spomialign,
            handler,
            kind="write",
            group="Nonlinear",
            writes=("affine", "deformation"),
            limits={"sections": 20},
            long=True,
            version="2",
            meta={
                "method": "SPOmiAlign",
                "paper": "Wang et al. 2026, Brief Bioinform",
                "commit": "c7f8b7a",
                "engine": "langslice.engines.spomialign",
            },
        )
    ]
