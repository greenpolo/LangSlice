"""The readable files and pictures a call keeps: the engine's sheets, its energy, its settings."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from langslice.engines.emlddmm import engine
from langslice.engines.emlddmm.settings import PLOT, SHEET, Array
from langslice.engines.emlddmm.stack import Placed, Stack


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _sheet_picture(
    stack: Stack, i: int, placed: Placed, result: engine.FitResult, path: Path
) -> None:
    """One section as the engine saw it, its final sorting (white compared, black ignored) and
    the atlas under the fit; a file in the job, not sent."""
    k = placed.slot
    data = stack.weights[k] > 0
    seen = np.clip(stack.images[k] / SHEET.display_max, 0, 1)
    sort = np.clip(np.asarray(result.match_weights[k], np.float32), 0, 1) * data
    atl = np.asarray(result.atlas_cut[i], np.float32)
    lo, hi = np.percentile(atl[data], SHEET.percentiles) if data.any() else (0, 1)
    atl = np.clip((atl - lo) / max(hi - lo, 1e-9), 0, 1)
    tiles = [(t * 255).astype(np.uint8) for t in (seen, sort, atl)]
    gap = np.full((tiles[0].shape[0], SHEET.gap), 255, np.uint8)
    row = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], 1)
    image = Image.fromarray(row).convert("RGB")
    size = (image.width * SHEET.enlarge, image.height * SHEET.enlarge)
    image = image.resize(size, Image.Resampling.LANCZOS)
    band = Image.new("RGB", (image.width, image.height + SHEET.band), "white")
    band.paste(image, (0, SHEET.band))
    ImageDraw.Draw(band).text(
        (6, 5),
        f"{placed.id}  slot {k}: the section as sent | the final "
        "sorting (white compared) | the atlas under the fit",
        fill="black",
        font=_font(SHEET.font),
    )
    band.save(path)


def _energy_files(energy: Array, folder: Path) -> None:
    """emlddmm's own energy record as ``energy.csv`` and a small line plot (files in the job)."""
    with open(folder / "energy.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scale", "iteration", "total", "mismatch", "regularization"])
        for row in energy:
            writer.writerow([int(row[0]), int(row[1]), *[f"{v:.6g}" for v in row[2:]]])
    h, w, pad = PLOT.height, PLOT.width, PLOT.pad
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    if len(energy) > 1:
        for col, shade, name in (
            (2, (0, 0, 0), "total"),
            (3, (110, 110, 110), "mismatch"),
            (4, (190, 190, 190), "regularization"),
        ):
            y = energy[:, col]
            lo, hi = float(y.min()), float(y.max())
            span = max(hi - lo, 1e-12)
            pts = [
                (pad + (w - 2 * pad) * j / (len(y) - 1), h - pad - (h - 2 * pad) * (v - lo) / span)
                for j, v in enumerate(y)
            ]
            draw.line(pts, fill=shade, width=1)
            draw.text(
                (pad, 4 + 10 * (col - 2)),
                f"{name}: min {lo:.4g}, max {hi:.4g} (each scaled to its own range)",
                fill=shade,
                font=_font(PLOT.font),
            )
    image.save(folder / "energy.png")


@dataclass(frozen=True)
class FileRecord:
    """Hold what the call's files keep besides the fit's own results.

    Parameters
    ----------
    asked
        What the call asked.
    chosen
        What the plugin picked from the stack.
    plugin
        The plugin's options.
    extra
        What ``A.json`` adds to the matrices.
    """

    asked: Mapping[str, Any]
    chosen: Mapping[str, Any]
    plugin: Mapping[str, Any]
    extra: Mapping[str, Any]


def write_files(stack: Stack, result: engine.FitResult, folder: Path, record: FileRecord) -> None:
    """The call's readable files (see the module text); nothing here goes into a reply."""
    _energy_files(result.energy, folder)
    (folder / "images").mkdir(exist_ok=True)
    for i, placed in enumerate(stack.sections):
        _sheet_picture(
            stack, i, placed, result, folder / "images" / f"{i:02d}_{Path(placed.id).stem}.png"
        )
    np.savez_compressed(
        folder / "weights.npz",
        WM=result.match_weights,
        WB=result.background_weights,
        WA=result.artifact_weights,
        slot=np.asarray(stack.slots),
    )
    matrices = {
        "A": result.affine.tolist(),
        "A_start": result.start_affine.tolist(),
        "A2d": result.slice_affines.tolist(),
        **record.extra,
    }
    (folder / "A.json").write_text(
        json.dumps(matrices, indent=1, default=float), encoding="utf-8", newline="\n"
    )
    (folder / "settings.json").write_text(
        json.dumps(
            {
                "asked": record.asked,
                "chosen": record.chosen,
                "plugin": record.plugin,
                "emlddmm": result.chosen,
                "device": result.device,
                "energy": "captured" if result.energy_captured else "not captured",
                "peak_gpu_gb": result.peak_gpu_gb,
                "seconds": result.seconds,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )
