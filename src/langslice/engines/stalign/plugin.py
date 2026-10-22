"""STalign as LangSlice tools: the adapter, run in LangSlice's process.

Load it with ``--tool-plugin langslice.engines.stalign.plugin:tools`` or
``create_job(..., plugins=["langslice.engines.stalign.plugin:tools"])``;
it is imported from this package (no ``PYTHONPATH``). Options (``plugin_config``): ``python`` (the
``stalign2`` interpreter), ``device`` (``auto``: the GPU at float32 when there is one, else
the CPU at float64), ``threads``, ``timeout_s``. STalign (GPL-3.0) runs in ``engine.py`` under
that interpreter; this module imports only ``langslice.plugin_api``.

Two tools (design: SliceBench/_local/notes/2026-10-18_stalign_plugin_design/fable_proposal.md):

- ``stalign_lddmm``: STalign's 2-D fit, one piece of background work per section, building on
  the section's current registration (a landmark warp included);
- ``stalign_plane_search``: STalign 2.0's 3-D plane search, proposing cutting angles and
  positions, nothing written.

Inputs, as every engine takes them: ``section_image`` (a channel by name, default
``preprocessed``), ``atlas_image`` (default ``nissl``, resolved by the job's atlas; the reply
names the image used) and ``mask`` (STalign takes none: it sorts the section itself, and its final
sorting is saved with the deformation). The reply carries ``settings`` and, per section,
``inputs`` and ``estimated``.

Frames. The target is the section's image (a channel by name) on the fit grid, grey, scaled so
the slide (median of the image border) is 0 and the tissue (median inside the placed atlas
outline) is 1, whatever the stain's polarity, area-averaged to 10 um. Target micrometres are
atlas micrometres at the section's linear scale. The source is the atlas image on its native
plane, area-averaged to 50 um (the published atlas voxel), I / mean|I| plus (I - mean I)^2 (the
published atlas recipe); on a section that holds a deformation, the atlas resampled onto the
fit grid through it (``FitFrame.applied_map``), so the bend starts on top of a landmark warp.
Marked damage (with the native margin) is painted as atlas background.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from langslice.engines import _blocks
from langslice.plugin_api import ToolContext, ToolError, ToolResult, ToolSpec, flag_words

HERE = Path(__file__).resolve().parent
NAME = "STalign"
REF = "langslice.engines.stalign.plugin:tools"
DEFAULTS = {
    "python": str(Path("~/.claude-science/conda/envs/stalign2/bin/python").expanduser()),
    "device": "auto",
    "threads": 4,
    "timeout_s": 1800,
}
#: The published target raster and atlas voxel, um (merfish-allen3Datlas-alignment.ipynb).
TARGET_UM, SOURCE_UM = 10.0, 50.0
#: The plane search's trial planes reach this far (STalign 2.0, _prefit.py).
SEARCH_TILT_DEG, SEARCH_SHIFT_UM = 6.0, 300.0
SORTING_METHODS = ("classic", "em", "none")
SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (x, y) <-> (row, col)


# --- the agent-facing declarations ----------------------------------------------------------


def stalign_lddmm(
    sections: list[str],
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    smoothness_um: float = 500.0,
    affine: bool = False,
    displacement_um: float = 94.0,
    sorting: dict[str, Any] = {},  # noqa: B006 (a declaration's default, as the native stubs)
    iterations: int = 2000,
    view: bool = True,
) -> dict[str, Any]:
    """Fit the atlas onto sections with STalign, on their current registration.

    STalign (Clifton et al. 2023, Nature Communications) is LDDMM registration
    (large deformation diffeomorphic metric mapping) built to align single-cell
    spatial transcriptomics, as rasterised cell images or histology, to each other
    and to the Allen CCF. Its distinctive parts are an estimated contrast map, so
    the two images need not share brightness, and its own per-pixel sorting into
    matching tissue, background and artifact. It is a gradient descent that finds
    the nearest fit, so it needs a close start.

    STalign bends the atlas image onto each section's image by gradient
    descent on their brightness mismatch, with a linear contrast fit between
    the two, a smoothness scale (smoothness_um: tissue closer than that bends
    together) and a displacement scale (displacement_um: how far a point may
    be carried before the penalty grows). It starts from the section's
    current registration (its linear placement, and its deformation when it
    has one) and refines it; it does not search from scratch~
    [[?interactive_transform : set the orientation and a rough transform with ~
    interactive_transform first]]. With affine false the transform stays and
    only the bend is fitted, on top of the deformation the section holds[[?set_landmarks ~
    , a landmark warp included, so the landmark move is kept]]. With affine true
    it also refits the section's in-plane transform and writes it; refused on a
    section that holds a deformation, since a new transform clears it. The
    section's marked damage regions are left out automatically[[?mark_damage ; ~
    mark them with mark_damage first]].

    STalign takes no section mask: it sorts the section's pixels itself. Before
    each comparison every pixel is weighted between matching (compared),
    background (slide, ignored) and artifact (debris, ignored), from the
    mismatch and from the background and artifact means (sorting, below). The
    final sorting is saved with the deformation[[?set_mask , and set_mask can ~
    take it as a mask for another fit (source fit)]]. A named mask is refused
    (MASK_UNSUPPORTED); the engines that take one are fireants_syn,
    elastix_affine, emlddmm_stack and spomialign. STalign fits the whole
    section: it takes no regions[[?fireants_syn ; refine single regions afterwards ~
    with fireants_syn, which builds on this fit]]. It keeps its own brightness
    scaling: slide 0, typical tissue 1.

    Starts one piece of background work per section and returns at once with
    their ids, so you can carry on with other sections. A write to a section
    whose fit still runs is refused until it lands. When a section's fit
    lands, the next tool reply starts with its notice: what was fitted, the
    displacement, fold fraction and flags, what STalign used and estimated,
    and the file in the job folder that holds STalign's own mismatch record
    for the fit[[?read_file , which read_file opens]]; then its pictures. Each
    landing is its own undo step. status lists the work still running; submit
    waits for it. Judge the overlay against the internal anatomy.

    Args:
        sections: One to twenty filenames (the extension may be left off).
        section_image: The section channel to fit, by name: "preprocessed"
            or another channel of the stack[[?set_channel  (set_channel makes ~
            and changes them)]]. STalign reads it as grey.
        atlas_image: One of the atlas's images (the job statement lists
            them: "template" and any other reference images the atlas ships,
            by their names), or "nissl", the atlas's Nissl (its template when
            it ships none). Default "nissl"; the reply names the image used.
        mask: "none". STalign takes no section mask (see above).
        smoothness_um: The bend's smoothness scale in micrometres (STalign's
            a); larger bends broader patches as one. Default 500.
        affine: True also refits the in-plane transform (turn, stretch, shift)
            and writes it; false keeps the current transform. Default false.
        displacement_um: The displacement scale in micrometres; larger allows
            a bigger bend. Default 94.
        sorting: {"method": "classic" | "em" | "none"} and any of "match_sd",
            "background_sd", "artifact_sd" (spreads), "background_mean",
            "artifact_mean" (left out: estimated) and "artifact" (true adds the
            artifact bin to "em"). Brightness units: slide 0, typical tissue 1.
            "classic" is STalign's published sorting with fixed spreads; "em"
            measures the matching spread itself and takes the background mean
            from the slide at the image border, for sections with missing
            tissue; "none" compares every pixel. Default {"method": "classic"}
            with STalign's own spreads.
        iterations: Gradient-descent steps. Default 2000.
        view: True sends the pictures described below with each notice; false
            sends none.

    Returns:
        Per section its work id and what the fit uses (inputs: the channel,
        the atlas image as asked and as used, the mask, the damage, the
        brightness scaling), or why it was not started. Each notice brings,
        per section, the numbers above, what STalign estimated (its sorting,
        saved with the deformation, and the share of the tissue it left out)
        and the section under its new registration with the atlas borders on
        it; when the sorting set some tissue aside, also the final sorting
        picture (set-aside patches outlined; white compared, black ignored),
        and when it set nothing aside the notice says so and sends no picture.
    """
    ...


def stalign_plane_search(
    sections: list[str],
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    view: bool = True,
) -> dict[str, Any]:
    """Estimate the cutting angles with STalign's plane search. Writes nothing.

    For each section, STalign tilts the atlas plane the section sits on by -6,
    0 and 6 degrees about each in-plane axis and shifts it by -300, 0 and 300
    um along the slicing axis (27 trial planes), fits each roughly to the
    section's image at low resolution, refines the best, and
    reports where that plane lies: its pitch and yaw in this stack's
    convention, and the position it implies. It starts from the section's
    current position, the stack's cutting angles and its transform, so place
    and orient the section first. Marked damage is left out automatically.
    One search reaches about 6 degrees from the start; from further away it
    lands short, so write the angles and search again from them until two
    passes agree within a degree or two.

    The stack has one plane. Search sections with plenty of anatomy spread
    along the stack, take the middle value of the pitches and of the yaw
    magnitudes, leave out a section flagged at_search_edge or whose picture
    at the found plane looks wrong, and write the pair[[?position_sections ~
    with position_sections]]. Keep the positions you placed unless a
    section's picture shows its position change is right. A section mounted
    mirrored reverses the sign of its yaw and keeps its pitch, and the
    hemispheres often cannot be told apart by any other sign, so many
    sections of a stack, up to half, may be mirrored: the reply sorts the
    sections by the sign of their yaw. Make every section agree on one sign
    (mirror the smaller group[[?interactive_transform  with ~
    interactive_transform]] and search them again) before writing the angles.

    Args:
        sections: One to eight filenames (the extension may be left off).
        section_image: The section channel to search with, as for the fits.
        atlas_image: One of the atlas's images, or "nissl", as for the fits.
        view: True returns the pictures described below; false returns none.

    Returns:
        Per section pitch_deg, yaw_deg and position_mm of the found plane,
        change (the differences from the current pitch, yaw and position)
        and at_search_edge; the sections grouped by the sign of their yaw,
        with the middle pitch and yaw magnitude of the call; and a picture of
        each section with the atlas borders drawn at the found plane, not
        written.
    """
    ...


# --- frames -----------------------------------------------------------------------------------


def _homog(m: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(m, np.float64)[:2], [0.0, 0.0, 1.0]])


def _down(image: np.ndarray, factor: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Area-average by *factor* (never up): the image and (sx, sy), down pixel k sitting at
    original pixel (k + 0.5) * s - 0.5."""
    h, w = image.shape[:2]
    if factor <= 1.0:
        return image, (1.0, 1.0)
    wd, hd = max(1, round(w / factor)), max(1, round(h / factor))
    return cv2.resize(image, (wd, hd), interpolation=cv2.INTER_AREA), (w / wd, h / hd)


def _axis(n: int, s: float, um: float) -> np.ndarray:
    return ((np.arange(n) + 0.5) * s - 0.5) * um


def _placed(frame: Any) -> np.ndarray:
    placed = frame.placed_atlas_to_section
    return np.asarray(frame.atlas_to_section if placed is None else placed, np.float64)


def _scale_um(frame: Any) -> float:
    """Micrometres per fit-grid pixel at the section's linear scale (atlas micrometres)."""
    px = float(np.mean(frame.plane["px_um"]))
    return px / float(np.sqrt(abs(np.linalg.det(_placed(frame)[:2, :2]))))


def _target(frame: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """The section's image, slide 0 and tissue 1, at ~10 um; its frame."""
    grey = _blocks.channel_image(frame).astype(np.float64).mean(axis=2)
    gh, gw = grey.shape
    border = np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]])
    slide = float(np.median(border))
    outline = cv2.warpAffine((frame.atlas["annotation"] > 0).astype(np.uint8),
                             _placed(frame)[:2], (gw, gh), flags=cv2.INTER_NEAREST) > 0
    tissue = float(np.median(grey[outline])) if outline.any() else float(grey.mean())
    if abs(tissue - slide) < 1e-6:
        raise ToolError("NO_TISSUE", "the section's tissue and slide look the same")
    unit = (grey - slide) / (tissue - slide)
    s_um = _scale_um(frame)
    down, (sx, sy) = _down(unit.astype(np.float32), TARGET_UM / s_um)
    return down.astype(np.float64)[None], {
        "s_um": s_um, "x0": _axis(down.shape[0], sy, s_um), "x1": _axis(down.shape[1], sx, s_um),
        "outline": cv2.resize(outline.astype(np.uint8), (down.shape[1], down.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0,
        "grid": (gw, gh)}


def _source_image(image: np.ndarray) -> np.ndarray:
    image = image[None] / max(float(np.mean(np.abs(image))), 1e-12)
    return np.concatenate((image, (image - image.mean()) ** 2))


def _source(frame: Any, plane_image: np.ndarray, s_um: float) -> tuple[np.ndarray, dict[str, Any]]:
    """The atlas image (damage painted as background) at ~50 um, its axes, the start affine
    (source um -> target um, row/col) and how its points become native pixels."""
    atlas = np.asarray(plane_image, np.float32).copy()
    if frame.damage_zone is not None:
        atlas[frame.damage_zone] = 0.0
    if frame.applied_map is not None:  # on the grid, through the section's deformation
        applied = np.asarray(frame.applied_map, np.float32)
        image = cv2.remap(atlas, applied[..., 0], applied[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        down, (sx, sy) = _down(image, SOURCE_UM / s_um)
        return _source_image(down.astype(np.float64)), {
            "x0": _axis(down.shape[0], sy, s_um), "x1": _axis(down.shape[1], sx, s_um),
            "A0": np.eye(3), "on_grid": True, "applied": applied}
    px_x, px_y = (float(v) for v in frame.plane["px_um"])
    down, (sx, sy) = _down(atlas, SOURCE_UM / min(px_x, px_y))
    a_xy = np.diag([s_um, s_um, 1.0]) @ _placed(frame) @ np.diag([1 / px_x, 1 / px_y, 1.0])
    return _source_image(down.astype(np.float64)), {
        "x0": _axis(down.shape[0], sy, px_y), "x1": _axis(down.shape[1], sx, px_x),
        "A0": SWAP @ a_xy @ SWAP, "on_grid": False, "px": (px_x, px_y)}


def _grid_query(gw: int, gh: int, s_um: float) -> np.ndarray:
    yy, xx = np.indices((gh, gw), dtype=np.float64)
    return np.stack([yy.ravel() * s_um, xx.ravel() * s_um], -1)


def _native_map(source_um: np.ndarray, src: dict[str, Any], gw: int, gh: int,
                s_um: float) -> np.ndarray:
    """STalign's source points of every fit-grid pixel as native atlas pixels (x, y)."""
    pts = source_um.reshape(gh, gw, 2)
    if src["on_grid"]:
        gx = (pts[..., 1] / s_um).astype(np.float32)
        gy = (pts[..., 0] / s_um).astype(np.float32)
        applied = src["applied"]
        return np.stack([cv2.remap(np.ascontiguousarray(applied[..., c]), gx, gy,
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                         for c in range(2)], -1).astype(np.float64)
    px_x, px_y = src["px"]
    return np.stack([pts[..., 1] / px_x, pts[..., 0] / px_y], -1)


def _sorting_params(sorting: dict[str, Any]) -> dict[str, Any]:
    sorting = dict(sorting or {})
    method = str(sorting.pop("method", "classic"))
    if method not in SORTING_METHODS:
        raise ToolError("BAD_ARGS", f"sorting method is one of {', '.join(SORTING_METHODS)}; "
                        f"got {method!r}")
    known = {"match_sd": "sigmaM", "background_sd": "sigmaB", "artifact_sd": "sigmaA",
             "background_mean": "muB", "artifact_mean": "muA"}
    artifact = bool(sorting.pop("artifact", False))
    unknown = [key for key in sorting if key not in known]
    if unknown:
        raise ToolError("BAD_ARGS", f"unknown sorting keys {unknown}; the keys are method, "
                        f"{', '.join(known)}, artifact")
    params: dict[str, Any] = {known[key]: float(value) for key, value in sorting.items()}
    if method == "em":
        params["mixture"] = {"classes": "MAB" if artifact else "MB"}
    elif method == "none":
        params["mixture"] = {"classes": "M"}
    return params


def _require(config: dict[str, Any], *keys: str) -> None:
    """Refuse at call time when an environment or source folder the engine needs is missing."""
    for key in keys:
        path = Path(str(config[key])).expanduser()
        if not path.exists():
            raise ToolError("ENGINE_UNAVAILABLE",
                            f"{NAME} cannot run here: {key} {str(path)!r} does not exist (set "
                            f"it in the plugin_config of {REF})")


# --- the engine -------------------------------------------------------------------------------


def _engine(config: dict[str, Any], folder: Path, arrays: dict[str, Any],
            params: dict[str, Any]) -> dict[str, Any]:
    """Run engine.py on *arrays* in *folder* (inputs, outputs and log kept there)."""
    folder.mkdir(parents=True, exist_ok=True)
    src, dst = folder / "in.npz", folder / "out.npz"
    np.savez_compressed(src, params=np.array(json.dumps(
        {**params, "device": config["device"], "threads": int(config["threads"])})), **arrays)
    env = {**os.environ, "OMP_NUM_THREADS": str(config["threads"])}
    done = subprocess.run([config["python"], "-I", str(HERE / "engine.py"), str(src), str(dst)],
                          capture_output=True, text=True, env=env, cwd=str(folder),
                          timeout=float(config["timeout_s"]))
    (folder / "log.txt").write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not dst.exists():
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise ToolError("ENGINE_FAILED", f"STalign did not run: {tail[0]}")
    out = dict(np.load(dst, allow_pickle=False))
    if "error" in out:
        raise ToolError("ENGINE_FAILED", f"STalign failed: {out['error']}")
    return out


def _frame(ctx: ToolContext, name: str, atlas_image: str, section_image: str,
           ) -> tuple[Any, np.ndarray, str]:
    """The section's fit frame, the atlas image asked for on it and the name that resolved to."""
    frame = _blocks.frame_for(ctx, name, atlas_image, section_image)
    plane_image, resolved = _blocks.atlas_plane(frame, atlas_image)
    return frame, plane_image, resolved


# --- pictures ---------------------------------------------------------------------------------


def _sorting_picture(ctx: ToolContext, frame: Any, weights: np.ndarray,
                     title: str) -> Image.Image | None:
    """The section's image with every set-aside patch outlined, beside the weight map
    (white compared, black ignored); None when the sorting set nothing aside (a map of
    all white says nothing, so no panel is drawn)."""
    stain = _blocks.channel_image(frame)
    gh, gw = stain.shape[:2]
    w = cv2.resize(np.asarray(weights, np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
    ignored = (w < 0.5).astype(np.uint8)
    if not ignored.any():
        return None
    left = stain.copy()
    contours, _ = cv2.findContours(ignored, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, contours, -1, (230, 30, 160), max(1, gw // 600))
    right = np.repeat((np.clip(w, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    gap = np.full((gh, 8, 3), 255, np.uint8)
    image = Image.fromarray(np.concatenate([left, gap, right], axis=1))
    image.thumbnail((1600, 1600))
    return ctx.caption(image, title)


def _loss_picture(energy: np.ndarray, path: Path) -> None:
    """STalign's mismatch record as a small line plot (a file in the job, not sent)."""
    h, w, pad = 240, 640, 30
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    if len(energy) > 1:
        y = energy[:, 1]
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, 1e-12)
        pts = [(pad + (w - 2 * pad) * i / (len(y) - 1), h - pad - (h - 2 * pad) * (v - lo) / span)
               for i, v in enumerate(y)]
        draw.line(pts, fill="black", width=1)
        draw.text((pad, 6), f"STalign image mismatch per iteration ({len(y)}); "
                  f"min {lo:.4g}, max {hi:.4g}", fill="black")
    image.save(path)


# --- stalign_lddmm ----------------------------------------------------------------------------


def _lddmm(ctx: ToolContext, config: dict[str, Any], sections: list[str], section_image: str,
           atlas_image: str, mask: str, smoothness_um: float, affine: bool,
           displacement_um: float, sorting: dict[str, Any], iterations: int,
           view: bool) -> ToolResult:
    if not sections:
        raise ToolError("BAD_ARGS", "name one to twenty sections")
    sorted_params = _sorting_params(sorting)
    params = {**sorted_params, "a": float(smoothness_um),
              "sigma_disp": float(displacement_um), "niter": int(iterations), "nt": 4}
    if not affine:
        params.update(epL=0.0, epT=0.0)
    estimates = (sorting or {}).get("method", "classic") != "none"
    settings = {"section_image": section_image, "atlas_image": atlas_image, "mask": mask,
                "smoothness_um": smoothness_um, "affine": affine,
                "displacement_um": displacement_um,
                "sorting": sorting or {"method": "classic"}, "iterations": iterations,
                "device": config["device"]}
    folder = _blocks.call_folder(ctx, "lddmm")
    rows = []
    for name in sections:
        view_of = ctx.section(name)
        sid = view_of.id
        if affine and view_of.deformation:
            rows.append({"id": sid, "status": "error", "error": "HAS_DEFORMATION",
                         "message": "affine true would refit the transform, which clears the "
                         "section's deformation; fit with affine false to bend on top of it"})
            continue
        frame, plane_image, resolved = _frame(ctx, sid, atlas_image, section_image)
        if frame.placed_atlas_to_section is None:
            rows.append({"id": sid, "status": "error", "error": "INVALID_LINEAR_PLACEMENT",
                         "message": "the section needs a transform first"})
            continue
        digest = ctx.section_inputs(sid, deformation=True)
        inputs = _blocks.inputs_block(ctx, frame, section_image=section_image,
                                      atlas_image=atlas_image, resolved=resolved, mask=None,
                                      brightness="slide 0, tissue 1")
        J, tgt = _target(frame)
        I, src = _source(frame, plane_image, tgt["s_um"])  # noqa: E741
        gw, gh = tgt["grid"]
        arrays = {"mode": np.array("lddmm"), "I": I, "xI0": src["x0"], "xI1": src["x1"],
                  "J": J, "xJ0": tgt["x0"], "xJ1": tgt["x1"], "A0": src["A0"],
                  "query": _grid_query(gw, gh, tgt["s_um"])}
        here = folder / sid
        start = "the deformation it held" if src["on_grid"] else "its linear placement"

        def land(lctx: ToolContext, sid: str = sid, frame: Any = frame, tgt: dict = tgt,
                 src: dict = src, arrays: dict = arrays, here: Path = here, start: str = start,
                 digest: str = digest, gw: int = gw, gh: int = gh, resolved: str = resolved,
                 inputs: dict = inputs) -> ToolResult:
            out = _engine(config, here, arrays, params)
            native = _native_map(out["source_um"], src, gw, gh, tgt["s_um"])
            what = ("affine and bend" if affine else "bend") + f" on {start}"
            note = f"STalign {what}, atlas {resolved}"
            weights = _blocks.on_fit_grid(out["WM"], (gw, gh)) if estimates else None
            inside_grid = np.asarray(frame.atlas["annotation"] > 0, np.uint8)
            outline = cv2.warpAffine(inside_grid, _placed(frame)[:2], (gw, gh),
                                     flags=cv2.INTER_NEAREST) > 0
            estimated = _blocks.estimated_block(weights, outline)
            blocks = {"inputs": inputs, "estimated": estimated, "settings": settings}
            energy = np.asarray(out.get("energy", np.zeros((0, 3))))
            with open(here / "energy.csv", "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["iteration", "total", "image_mismatch", "smoothness",
                                 "landmarks"][:2 + energy.shape[1]])
                for i, row in enumerate(energy):
                    writer.writerow([i, *[f"{v:.6g}" for v in row]])
            _loss_picture(energy, here / "mismatch.png")
            (here / "settings.json").write_text(json.dumps(
                {"asked": {**settings, **params}, **blocks,
                 "chosen": json.loads(str(out["settings"])), "device": str(out["device"]),
                 "seconds": float(out["seconds"])}, indent=1, default=str))
            np.savez_compressed(here / "weights.npz", WM=out["WM"], WB=out["WB"], WA=out["WA"])
            # bend only: the map converts before the lock (with affine the conversion reads the
            # new transform, so it waits for it)
            warp = None if affine else lctx.deformation_from_map(
                sid, native, note=note, meta=blocks, weights=weights)
            with lctx.writing():
                if affine:
                    a_xy = SWAP @ np.asarray(out["A"]) @ SWAP  # source um -> target um (x, y)
                    px_x, px_y = src["px"]
                    placed = (np.diag([1 / tgt["s_um"], 1 / tgt["s_um"], 1.0]) @ a_xy
                              @ np.diag([px_x, px_y, 1.0]))
                    proposal = lctx.transform_from_matrix(sid, np.linalg.inv(placed)[:2],
                                                          note=note)
                    done = lctx.set_transforms([proposal.row()], expect={sid: digest})
                    if sid not in done.written:
                        raise ToolError(done.rows[0].get("error", "NOT_WRITTEN"),
                                        done.rows[0].get("message", ""))
                if warp is None:
                    warp = lctx.deformation_from_map(sid, native, note=note, meta=blocks,
                                                     weights=weights)
                done = lctx.set_deformations([warp.row()],
                                             expect=None if affine else {sid: digest})
                if sid not in done.written:
                    raise ToolError(done.rows[0].get("error", "NOT_WRITTEN"),
                                    done.rows[0].get("message", ""))
            summary = warp.summary
            relative = here.relative_to(lctx.job_dir) if here.is_relative_to(lctx.job_dir) else here
            flags = flag_words(summary)
            left = (f"; its sorting left out {estimated['left_out_fraction']:.0%} of the tissue "
                    "and is saved with the deformation") if estimated and (
                        estimated["left_out_fraction"] is not None) else ""
            waited = float(out.get("gpu_wait_s", 0.0))
            queue = {"gpu_wait_s": round(waited)} if waited >= 1.0 else {}
            text = (f"{note}; displacement median {summary['displacement_mm']['median']} mm, "
                    f"max {summary['displacement_mm']['max']} mm; fold fraction "
                    f"{summary.get('fold_fraction', 0)}; flags {flags}{left}. "
                    + (f"It waited {round(waited)} s for the machine's GPU queue. "
                       if queue else "")
                    + f"STalign's mismatch record: "
                    f"{relative / 'energy.csv'} (plot: {relative / 'mismatch.png'}).")
            pictures = []
            if view:
                pictures = [lctx.caption(lctx.overlay_picture(sid), f"{sid}: {note}")]
                sorting = _sorting_picture(lctx, frame, out["WM"],
                                           f"{sid}: STalign's final sorting (outlined: set aside; "
                                           "right: white compared)")
                if sorting is None:
                    text += (" Its final sorting set nothing aside, so every pixel was compared "
                             "and no sorting picture is sent.")
                else:
                    pictures.append(sorting)
            return ToolResult({"notice": text, "id": sid, **summary, **queue, **blocks},
                              pictures)

        try:
            work = ctx.start_work([sid], land)
        except Exception as exc:  # SECTION_BUSY and the like: that section's row
            as_payload = getattr(exc, "payload", None)
            payload = as_payload() if as_payload else {"error": "NOT_STARTED",
                                                       "message": str(exc)}
            rows.append({"id": sid, "status": "error", **{k: v for k, v in payload.items()
                                                          if k != "status"}})
            continue
        rows.append({"id": sid, "status": "started", "work": work,
                     "start": "deformation" if src["on_grid"] else "linear placement",
                     "inputs": inputs, "estimated": None})
    return ToolResult({"rows": rows, "settings": settings})


# --- stalign_plane_search ---------------------------------------------------------------------


def _target_to_atlas(frame: Any, s_um: float) -> np.ndarray:
    """4 x 4: target (z, row um, col um) -> atlas um, through the section's linear placement
    and its plane (z along the unit normal)."""
    to_um = np.asarray(frame.plane["native_to_um"], np.float64)
    from_grid = np.linalg.inv(_placed(frame))  # fit-grid px -> native px
    m = to_um @ from_grid  # fit-grid [x, y, 1] -> atlas um
    d_row, d_col = m[:, 1] / s_um, m[:, 0] / s_um
    n = np.cross(d_col, d_row)
    n /= np.linalg.norm(n)
    B = np.eye(4)
    B[:3, 0], B[:3, 1], B[:3, 2], B[:3, 3] = n, d_row, d_col, m[:, 2]
    return B


def _plane_search(ctx: ToolContext, config: dict[str, Any], sections: list[str],
                  section_image: str, atlas_image: str, view: bool) -> ToolResult:
    if not sections:
        raise ToolError("BAD_ARGS", "name one to eight sections")
    atlas = ctx.atlas()
    folder = _blocks.call_folder(ctx, "plane")
    params = {"nt": 4, "sigmaM": 2.0, "sigmaB": 2.0, "sigmaA": 2.0, "muA": 3.0, "muB": 0.0}
    rows, pictures = [], []
    for name in sections:
        view_of = ctx.section(name)
        sid = view_of.id
        frame, _plane_image, resolved = _frame(ctx, sid, atlas_image, section_image)
        inputs = _blocks.inputs_block(ctx, frame, section_image=section_image,
                                      atlas_image=atlas_image, resolved=resolved, mask=None,
                                      brightness="slide 0, tissue 1")
        J, tgt = _target(frame)
        gw, gh = tgt["grid"]
        B0 = _target_to_atlas(frame, tgt["s_um"])
        arrays = {"mode": np.array("plane"), "atlas": np.array(atlas.name),
                  "image": np.array(resolved), "J": J, "xJ0": tgt["x0"], "xJ1": tgt["x1"],
                  "A0": np.linalg.inv(B0)}
        out = _engine(config, folder / sid, arrays, params)
        B = np.linalg.inv(np.asarray(out["A"], np.float64))
        centre = B @ np.array([0.0, (gh - 1) / 2 * tgt["s_um"], (gw - 1) / 2 * tgt["s_um"], 1.0])
        found = ctx.angles_from_plane(np.cross(B[:3, 2], B[:3, 1]).tolist(),
                                      centre[:3].tolist())
        prefit = json.loads(str(out["prefit"]))
        starts, best = prefit.get("starts") or [], prefit.get("best")
        edge = False
        if starts and best is not None:
            tilt_a, tilt_b, shift = starts[int(best)]
            edge = (abs(abs(tilt_a) - SEARCH_TILT_DEG) < 1e-6 or abs(abs(tilt_b) - SEARCH_TILT_DEG)
                    < 1e-6 or abs(abs(shift) - SEARCH_SHIFT_UM) < 1e-6)
        (folder / sid / "plane.json").write_text(json.dumps(
            {"found": found, "prefit": prefit, "device": str(out["device"])}, indent=1,
            default=float))
        pitch0, yaw0 = (float(v) for v in view_of.angles)
        row = {"id": sid, "status": "ok", "inputs": inputs, "estimated": None,
               "pitch_deg": round(found["pitch_deg"], 2),
               "yaw_deg": round(found["yaw_deg"], 2),
               "position_mm": round(found["position_mm"], 3),
               "change": {"pitch_deg": round(found["pitch_deg"] - pitch0, 2),
                          "yaw_deg": round(found["yaw_deg"] - yaw0, 2),
                          "position_mm": round(found["position_mm"]
                                               - float(view_of.position_mm or 0.0), 3)},
               "at_search_edge": bool(edge)}
        rows.append(row)
        if view:
            proposal = {"position_mm": found["position_mm"],
                        "angles": (found["pitch_deg"], found["yaw_deg"])}
            pictures.append(ctx.caption(ctx.overlay_picture(sid, proposal=proposal),
                                        f"{sid}: atlas at the plane STalign found (pitch "
                                        f"{row['pitch_deg']}, yaw {row['yaw_deg']}), not written"))
    ok = [row for row in rows if row["status"] == "ok"]
    groups = {"positive_yaw": [r["id"] for r in ok if r["yaw_deg"] > 0],
              "negative_yaw": [r["id"] for r in ok if r["yaw_deg"] < 0],
              "zero_yaw": [r["id"] for r in ok if r["yaw_deg"] == 0]}
    summary = {"sections_by_yaw_sign": groups,
               "middle_pitch_deg": round(float(np.median([r["pitch_deg"] for r in ok])), 2)
               if ok else None,
               "middle_yaw_magnitude_deg": round(float(np.median([abs(r["yaw_deg"])
                                                                 for r in ok])), 2) if ok else None}
    return ToolResult({"rows": rows, "stack": summary,
                       "settings": {"section_image": section_image, "atlas_image": atlas_image}},
                      pictures)


# --- the factory ------------------------------------------------------------------------------


def tools(config: dict[str, Any] | None = None) -> list[ToolSpec]:
    options = {**DEFAULTS, **(config or {})}
    meta = {"method": "STalign 2.0.0", "paper": "Clifton et al. 2023, Nat Commun",
            "commit": "8a84bf1f", "engine": "langslice.engines.stalign"}

    def lddmm(ctx: ToolContext, sections: list[str], section_image: str = "preprocessed",
              atlas_image: str = "nissl", mask: str = "none", smoothness_um: float = 500.0,
              affine: bool = False, displacement_um: float = 94.0,
              sorting: dict[str, Any] | None = None, iterations: int = 2000,
              view: bool = True) -> ToolResult:
        _blocks.refuse_mask("stalign_lddmm", mask, "sorts the section's pixels itself")
        _require(options, "python")
        return _lddmm(ctx, options, sections, section_image, atlas_image, mask,
                      smoothness_um, affine, displacement_um, dict(sorting or {}),
                      iterations, view)

    def plane(ctx: ToolContext, sections: list[str], section_image: str = "preprocessed",
              atlas_image: str = "nissl", view: bool = True) -> ToolResult:
        _require(options, "python")
        return _plane_search(ctx, options, sections, section_image, atlas_image, view)

    return [
        ToolSpec("stalign_lddmm", stalign_lddmm, lddmm, kind="write", group="Nonlinear",
                 task="nonlinear", writes=("affine", "deformation"), long=True,
                 limits={"sections": 20}, version="1", meta=meta),
        ToolSpec("stalign_plane_search", stalign_plane_search, plane, kind="read",
                 group="Positioning", task="angles", limits={"sections": 8}, version="1",
                 meta=meta),
    ]


__all__ = ["tools"]
