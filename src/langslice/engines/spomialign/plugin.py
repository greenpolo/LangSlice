"""SPOmiAlign as a LangSlice tool: the adapter, run in LangSlice's process.

Load it with ``--tool-plugin langslice.engines.spomialign.plugin:tools`` or
``create_job(..., plugins=["langslice.engines.spomialign.plugin:tools"])``;
it is imported from this package (no ``PYTHONPATH``). Options (``plugin_config``): ``python``
(the ``rigid_spomialign`` interpreter), ``src`` (SPOmiAlign's ``SPOmiAlign`` folder),
``device``, ``timeout_s``. The method itself runs in ``engine.py`` under that
interpreter; this module imports only ``langslice.plugin_api``.

Each call puts the section on SPOmiAlign's canvas, runs the engine on (atlas image,
section), and maps the answer back: the linear part through ``transform_from_matrix``, the
full map through ``deformation_from_map``. Canvas: for rigid, affine and affine_warp the
atlas plane's grid, upsampled to about :data:`CANVAS_EDGE` px, with the section resampled
onto it through its current transform (as the published user's hand-picked plate and
rotation did); for warp on a section that holds a deformation, the fit grid, with the atlas
resampled onto it through that deformation (``FitFrame.applied_map``), so the warp starts on
top of a landmark warp, as ``stalign_lddmm`` does. A new transform clears a deformation, so
rigid, affine and affine_warp are refused on a section that holds one (``HAS_DEFORMATION``).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.plugin_api import ToolContext, ToolError, ToolResult, ToolSpec

HERE = Path(__file__).resolve().parent
NAME = "SPOmiAlign"
#: Where SliceBench keeps the vendored sources the engines read by path.
COMPARATOR_SRC = Path("~/Software/SliceBench/comparators/src").expanduser()
REF = "langslice.engines.spomialign.plugin:tools"
DEFAULTS = {
    "python": str(Path("~/.claude-science/conda/envs/rigid_spomialign/bin/python").expanduser()),
    "src": str(COMPARATOR_SRC / "spomialign" / "SPOmiAlign"),
    "device": "",
    "timeout_s": 900,
}
#: Long edge of the canvas SPOmiAlign receives, px: above its 1152 x 864 grid, so its own
#: resize brings both images down to that grid as with its published inputs.
CANVAS_EDGE = 1536
TRANSFORMS = ("rigid", "affine", "warp", "affine_warp")
#: The section properties each transform writes (``frozen`` names).
WRITES = {"rigid": ("affine",), "affine": ("affine",), "warp": ("deformation",),
          "affine_warp": ("affine", "deformation")}
#: RoMa's weights, which ``roma_outdoor`` downloads into torch's hub cache on first use.
WEIGHTS = {
    "roma_outdoor.pth": "https://github.com/Parskatt/storage/releases/download/roma/"
                        "roma_outdoor.pth",
    "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
                                  "dinov2_vitl14_pretrain.pth",
}
#: The default atlas image: LangSlice's borrowed Nissl where the job offers it (an Allen CCFv3
#: atlas), else the template (on ccfv3augmented_mouse already an averaged Nissl).
NISSL = "nissl"
BORROWED_NISSL = "ccfv3augmented_nissl"
#: How many of the most confident pairs ``matches.png`` draws (the CSV holds them all).
DRAWN_PAIRS = 300


def spomialign(
    sections: list[str],
    transform: Literal["rigid", "affine", "warp", "affine_warp"],
    atlas_image: str = "nissl",
    matches: int = 5000,
    smoothing_px: float = 15.0,
    view: bool = True,
) -> dict[str, Any]:
    """Register sections to the atlas with SPOmiAlign, on their current registration.

    SPOmiAlign matches distinctive points between the section's preprocessed
    channel and an atlas image with a pretrained dense matcher (RoMa), weights
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

    transform chooses what is fitted:
    - "rigid": rotation and shift; the section keeps its size. Written as the
      section's transform.
    - "affine": rotation, shift, scale and shear. Written as the section's
      transform.
    - "warp": a smooth local deformation on top of the registration the section
      holds (its transform, and its deformation when it has one[[?set_landmarks ~
      , a landmark warp included, so the landmark move is kept]]); the transform
      stays. Written as the section's deformation.
    - "affine_warp": an affine, then a warp on top of it (SPOmiAlign's own
      default). Writes both.
    A new transform clears the section's deformation, so rigid, affine and
    affine_warp are refused on a section that holds one: fit warp on it instead.

    The warp is SPOmiAlign's smoothed match field: each pair's displacement,
    filled in between the pairs and blurred by smoothing_px. Where the section
    and the atlas image share little tissue (a damaged or partial section), the
    affine follows wrong matches, because it weighs every pair equally and
    rejects none; the sign is an atlas shrunk off the tissue in the returned
    overlay. For such a section, fit "rigid", check the returned overlay, then
    fit "warp". The warp bends whole regions and does not refine individual
    borders.

    Each call is one undoable write.

    Args:
        sections: One to four filenames (the extension may be left off).
        transform: "rigid", "affine", "warp" or "affine_warp".
        atlas_image: One of the atlas's images (the job statement lists
            them: "template" and any other reference images the atlas ships,
            by their names); pick the one that looks most like the sections.
            Default "nissl": ccfv3augmented_nissl where the job lists it,
            else "template".
        matches: How many of the most confident matched pairs the fit uses.
            Default 5000 (SPOmiAlign's).
        smoothing_px: How widely the warp's pair displacements are blurred, in
            pixels of SPOmiAlign's 1152 x 864 working image; larger bends
            broader regions as one. Default 15 (SPOmiAlign's).
        view: True returns the picture described below; false returns none.

    Returns:
        Per section what was fitted and on what, the transform written
        (rotation_deg, scale_x, scale_y, shear, translate_x_mm,
        translate_y_mm about the canvas centre) when one was, for a warp its
        displacement (median and max, mm), fold fraction and plausibility
        flags, the pairs used and dropped, and the folder in the job holding
        the call's files (the two images SPOmiAlign saw, its settings and the
        matched pairs as a table and a picture[[?read_file , which read_file ~
        opens]]); and a picture of each section under its new registration
        with the atlas borders on it.
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


def _inputs(frame: Any, atlas_image: str, on_grid: bool) -> dict[str, Any]:
    """The pair SPOmiAlign sees, on one canvas: over the atlas plane through the linear
    placement, or (``on_grid``) over the fit grid through the section's deformation.
    ``to_canvas`` takes fit-grid px to canvas px; ``from_canvas`` takes canvas px to the
    grid of ``lookup`` (native atlas px, or with ``on_grid`` the fit grid, read through
    ``applied`` to native atlas px)."""
    labels = frame.atlas["annotation"]
    picture = _atlas_picture(frame.atlas[atlas_image], labels)
    if frame.excluded is not None:  # marked damage painted as background, as native fits do
        picture[frame.excluded] = 255
    zone = frame.damage_zone if frame.damage_zone is not None else np.zeros(labels.shape, bool)
    stain = frame.stain
    if on_grid:
        applied = np.asarray(frame.applied_map, np.float32)
        gh, gw = applied.shape[:2]
        k = _canvas_scale((gh, gw))
        size = (gw * k, gh * k)
        picture = cv2.remap(picture, applied[..., 0], applied[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        zone = cv2.remap(zone.astype(np.uint8), applied[..., 0], applied[..., 1],
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        source = cv2.resize(stain, size, interpolation=cv2.INTER_LINEAR)
        K = _scale(k)
        to_canvas = K
        extra: dict[str, Any] = {"applied": applied}
    else:
        h, w = labels.shape
        k = _canvas_scale((h, w))
        size = (w * k, h * k)
        K = _scale(k)
        start = frame.placed_atlas_to_section
        to_canvas = K @ np.linalg.inv(start)  # fit-grid px -> canvas px
        source = cv2.warpAffine(stain, (start @ np.linalg.inv(K))[:2], size,
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=_slide_fill(stain))
        extra = {"start": start}
    target = cv2.resize(picture, size, interpolation=cv2.INTER_LINEAR)
    keep = ~cv2.resize(np.asarray(zone, np.uint8), size,
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return {"target": target, "source": source, "keep": keep, "K": K,
            "to_canvas": to_canvas, "from_canvas": np.linalg.inv(K), "on_grid": on_grid,
            **extra}


def _call_folder(ctx: ToolContext, transform: str) -> Path:
    calls = ctx.work_dir() / "calls"
    calls.mkdir(parents=True, exist_ok=True)
    n = len([p for p in calls.iterdir() if p.is_dir()])
    return calls / f"{n:04d}_{transform}"


def _weights_missing() -> list[str]:
    hub = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch")
    return [name for name in WEIGHTS if not (hub / "hub" / "checkpoints" / name).exists()]


def _engine(config: dict[str, Any], folder: Path, mode: str, matches: int,
            smoothing_px: float, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run engine.py on every section's pair in one process (one model load); its inputs,
    outputs and log are kept in *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    src, dst = folder / "in.npz", folder / "out.npz"
    arrays: dict[str, Any] = {"mode": np.array(mode), "n": np.array(len(pairs)),
                              "matches": np.array(matches),
                              "smoothing_px": np.array(smoothing_px)}
    for i, pair in enumerate(pairs):
        arrays.update({f"target_{i}": pair["target"], f"source_{i}": pair["source"],
                       f"keep_{i}": pair["keep"]})
    np.savez_compressed(src, **arrays)
    env = {**os.environ, "SPOMIALIGN_SRC": config["src"]}
    if config.get("device"):
        env["SPOMIALIGN_DEVICE"] = str(config["device"])
    done = subprocess.run([config["python"], "-I", str(HERE / "engine.py"), str(src), str(dst)],
                          capture_output=True, text=True, env=env, cwd=str(folder),
                          timeout=float(config["timeout_s"]))
    (folder / "log.txt").write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not dst.exists():
        missing = _weights_missing()
        if missing:
            raise ToolError("ENGINE_FAILED", "SPOmiAlign's matcher weights are not in torch's "
                            "hub cache and could not be downloaded: "
                            + ", ".join(WEIGHTS[name] for name in missing))
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise ToolError("ENGINE_FAILED", f"SPOmiAlign did not run: {tail[0]}")
    out = np.load(dst)
    results = []
    for i in range(len(pairs)):
        got = {key[: -len(f"_{i}")]: out[key] for key in out.files if key.endswith(f"_{i}")}
        results.append(got)
    return results


def _to_native(pair: dict[str, Any], canvas_xy: np.ndarray) -> np.ndarray:
    """Target-canvas points (2 x N) as native atlas px (2 x N)."""
    grid = (pair["from_canvas"] @ np.vstack([canvas_xy, np.ones(canvas_xy.shape[1])]))[:2]
    if not pair["on_grid"]:
        return grid
    applied = pair["applied"]
    n = grid.shape[1]
    rows = -(-n // 4096)  # remap takes maps under 32767 px a side: fold the points into rows
    pad = np.zeros((2, rows * 4096 - n), np.float32)
    gx, gy = (np.concatenate([grid[c].astype(np.float32), pad[c]]).reshape(rows, 4096)
              for c in range(2))
    return np.stack([cv2.remap(np.ascontiguousarray(applied[..., c]), gx, gy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).ravel()[:n] for c in range(2)])


def _full_map(frame: Any, pair: dict[str, Any], matrix: np.ndarray,
              residual: np.ndarray) -> np.ndarray:
    """Native atlas px of every fit-grid pixel under SPOmiAlign's full map.

    A fit-grid pixel u sits at canvas px c = to_canvas u; SPOmiAlign takes c to
    M c + R(c) on the target canvas (R zero off the canvas, as ITK's field is), which
    :func:`_to_native` reads as native atlas px."""
    gh, gw = frame.stain.shape[:2]
    ys, xs = np.mgrid[0:gh, 0:gw]
    u = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]).astype(np.float64)
    c = (pair["to_canvas"] @ u)[:2]
    moved = _homog(matrix)[:2, :2] @ c + _homog(matrix)[:2, 2:3]
    cx = c[0].reshape(gh, gw).astype(np.float32)
    cy = c[1].reshape(gh, gw).astype(np.float32)
    for axis in range(2):
        moved[axis] += cv2.remap(np.ascontiguousarray(residual[..., axis]), cx, cy,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0).ravel()
    return _to_native(pair, moved).T.reshape(gh, gw, 2)


def _require(config: dict[str, Any], *keys: str) -> None:
    """Refuse at call time when an environment or source folder the engine needs is missing."""
    for key in keys:
        path = Path(str(config[key])).expanduser()
        if not path.exists():
            raise ToolError("ENGINE_UNAVAILABLE",
                            f"{NAME} cannot run here: {key} {str(path)!r} does not exist (set "
                            f"it in the plugin_config of {REF})")


# --- job files --------------------------------------------------------------------------------


def _outline(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (230, 30, 160), max(1, image.shape[1] // 600))
    return out


def _pairs_picture(pair: dict[str, Any], used: np.ndarray, dropped: np.ndarray,
                   path: Path) -> None:
    """The two canvases side by side, the most confident pairs as dots of one colour on both
    (coloured by where they sit on the section) and the pairs passed over in the damage zone
    as red crosses on the atlas (a file in the job, not sent)."""
    left, right = pair["target"], pair["source"]
    gap = 16
    image = Image.fromarray(np.concatenate(
        [left, np.full((left.shape[0], gap, 3), 255, np.uint8), right], axis=1))
    draw = ImageDraw.Draw(image)
    shift = left.shape[1] + gap
    r = max(3, left.shape[1] // 250)
    height = right.shape[0]
    for ax, ay, sx, sy, _ in used[:DRAWN_PAIRS]:
        hue = np.uint8(170 * sy / max(height, 1)), np.uint8(255), np.uint8(230)
        rgb = cv2.cvtColor(np.array([[hue]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        colour = tuple(int(v) for v in rgb)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=colour, outline="black")
        draw.ellipse([sx + shift - r, sy - r, sx + shift + r, sy + r], fill=colour,
                     outline="black")
    for ax, ay, *_ in dropped:
        draw.line([(ax - r, ay - r), (ax + r, ay + r)], fill=(220, 30, 30), width=2)
        draw.line([(ax - r, ay + r), (ax + r, ay - r)], fill=(220, 30, 30), width=2)
    draw.text((6, 6), f"atlas (left) and section (right) as SPOmiAlign saw them; the "
              f"{min(DRAWN_PAIRS, len(used))} most confident of {len(used)} pairs, each a dot of "
              f"one colour on both; red crosses: "
              f"{len(dropped)} passed over in marked damage", fill="black")
    image.thumbnail((2400, 2400))
    image.save(path)


def _files(here: Path, pair: dict[str, Any], got: dict[str, Any],
           settings: dict[str, Any]) -> None:
    here.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_outline(pair["target"], ~pair["keep"])).save(here / "atlas.png")
    Image.fromarray(pair["source"]).save(here / "section.png")
    (here / "settings.json").write_text(json.dumps(settings, indent=1, default=str))
    used = np.asarray(got.get("used", np.zeros((0, 5))))
    dropped = np.asarray(got.get("dropped_pairs", np.zeros((0, 5))))
    with open(here / "matches.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atlas_x", "atlas_y", "section_x", "section_y", "confidence",
                         "dropped_in_damage"])
        for rows, flag in ((used, 0), (dropped, 1)):
            for row in rows:
                writer.writerow([*(f"{v:.2f}" for v in row[:4]), f"{row[4]:.4g}", flag])
    _pairs_picture(pair, used, dropped, here / "matches.png")


# --- the tool ---------------------------------------------------------------------------------


def _refusal(view_of: Any, frame: Any, transform: str) -> dict[str, Any] | None:
    """Why *transform* cannot start on this section, before the engine runs."""
    sid = view_of.id
    frozen = [prop for prop in WRITES[transform] if prop in view_of.frozen]
    if frozen:
        return {"id": sid, "status": "error", "error": "FROZEN",
                "message": f"the section's {' and '.join(frozen)} may not be changed"}
    if frame.placed_atlas_to_section is None:
        return {"id": sid, "status": "error", "error": "INVALID_LINEAR_PLACEMENT",
                "message": "SPOmiAlign starts from the section's placement: give it a "
                           "transform first"}
    if transform != "warp" and view_of.deformation:
        return {"id": sid, "status": "error", "error": "HAS_DEFORMATION",
                "message": f"{transform} would write a new transform, which clears the "
                           "section's deformation; fit warp to build on it"}
    return None


def _run(ctx: ToolContext, config: dict[str, Any], sections: list[str], transform: str,
         atlas_image: str = NISSL, matches: int = 5000, smoothing_px: float = 15.0,
         view: bool = True) -> ToolResult:
    if transform not in TRANSFORMS:
        raise ToolError("BAD_ARGS", f"transform is one of {', '.join(TRANSFORMS)}; "
                        f"got {transform!r}")
    if not sections:
        raise ToolError("BAD_ARGS", "name one to four sections")
    if int(matches) < 3:
        raise ToolError("BAD_ARGS", "matches is at least 3")
    if not float(smoothing_px) > 0:
        raise ToolError("BAD_ARGS", "smoothing_px is above 0")
    matches, smoothing_px = int(matches), float(smoothing_px)
    rows: list[dict[str, Any]] = []
    ready: list[tuple[str, Any, dict[str, Any], str]] = []
    for name in sections:
        view_of = ctx.section(name)
        asked = (BORROWED_NISSL, "template") if atlas_image == NISSL else (atlas_image,)
        frame = ctx.fit_frame(view_of.id, atlas=(*asked, "annotation"), stain=True)
        atlas_image = next((name for name in asked if name in frame.atlas), atlas_image)
        if atlas_image not in frame.atlas:
            raise ToolError("BAD_ARGS", f"the atlas has no image {atlas_image!r}; "
                            "the job statement lists its images")
        refused = _refusal(view_of, frame, transform)
        if refused:
            rows.append(refused)
            continue
        on_grid = transform == "warp" and frame.applied_map is not None
        ready.append((view_of.id, frame, _inputs(frame, atlas_image, on_grid),
                      ctx.section_inputs(view_of.id, deformation=on_grid)))
    if not ready:
        return ToolResult({"rows": rows})
    folder = _call_folder(ctx, transform)
    answers = _engine(config, folder, transform, matches, smoothing_px,
                      [pair for _, _, pair, _ in ready])
    pictures = []
    with ctx.writing():
        for (sid, frame, pair, digest), got in zip(ready, answers, strict=True):
            here = folder / Path(sid).stem
            start = "the deformation it held" if pair["on_grid"] else "its linear placement"
            _files(here, pair, got, {
                "atlas_image": atlas_image, "transform": transform, "matches": matches,
                "smoothing_px": smoothing_px, "start": start, "canvas_px": pair["target"].shape[:2],
                "device": config.get("device") or "SPOmiAlign's choice",
                "pairs_used": int(got.get("pairs", 0)),
                "pairs_dropped": int(got.get("dropped", 0))})
            relative = here.relative_to(ctx.job_dir) if here.is_relative_to(ctx.job_dir) else here
            if "error" in got:
                rows.append({"id": sid, "status": "error", "error": "FIT_FAILED",
                             "message": str(got["error"]), "files": str(relative)})
                continue
            what = transform.replace("_", " and ") if transform != "warp" else f"warp on {start}"
            note = (f"SPOmiAlign {what}, atlas {atlas_image}, {int(got['pairs'])} pairs, "
                    f"{int(got['dropped'])} in marked damage dropped")
            row: dict[str, Any] = {"id": sid, "status": "ok", "fitted": note,
                                   "files": str(relative)}
            if transform != "warp":
                to_native = (pair["from_canvas"] @ _homog(got["matrix"]) @ pair["to_canvas"])[:2]
                proposal = ctx.transform_from_matrix(sid, to_native, note=note)
                written = ctx.set_transforms([proposal.row()], expect={sid: digest})
                if sid not in written.written:
                    rows.append({**written.rows[0], "id": sid})
                    continue
                row["transform"] = proposal.knobs
            if transform in ("warp", "affine_warp"):
                native = _full_map(frame, pair, got["matrix"], got["residual"])
                warp = ctx.deformation_from_map(sid, native, note=note,
                                                meta={"sitk_max_px": float(got["sitk_max_px"])})
                written = ctx.set_deformations(
                    [warp.row()], expect={sid: digest} if transform == "warp" else None)
                if sid not in written.written:
                    rows.append({**written.rows[0], "id": sid})
                    continue
                row.update(warp.summary)
            rows.append(row)
            if view:
                pictures.append(ctx.caption(ctx.overlay_picture(sid), f"{sid}: {note}"))
    return ToolResult({"rows": rows}, pictures)


def tools(config: dict[str, Any] | None = None) -> list[ToolSpec]:
    options = {**DEFAULTS, **(config or {})}

    def handler(ctx: ToolContext, sections: list[str], transform: str,
                atlas_image: str = NISSL, matches: int = 5000,
                smoothing_px: float = 15.0, view: bool = True) -> ToolResult:
        _require(options, "python", "src")
        return _run(ctx, options, sections, transform, atlas_image, matches, smoothing_px, view)

    return [ToolSpec("spomialign", spomialign, handler, kind="write", group="Nonlinear",
                     writes=("affine", "deformation"), limits={"sections": 4},
                     version="2",
                     meta={"method": "SPOmiAlign", "paper": "Wang et al. 2026, Brief Bioinform",
                           "commit": "c7f8b7a", "engine": "langslice.engines.spomialign"})]
