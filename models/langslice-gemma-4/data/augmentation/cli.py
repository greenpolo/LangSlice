"""CLI for the augmentation pipeline.

Subcommands:
    render-grid  — generate a visual inspection grid per modality
    generate     — batch-generate augmented images for one modality
    inspect      — single side-by-side original vs augmented for debugging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

__all__: list[str] = []

_MODALITIES = ("dapi", "nissl", "brightfield", "fluorescence", "ish")


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------


def _load_atlas(atlas_name: str) -> object:
    from langslice_harness.atlas.core import load_atlas

    return load_atlas(atlas_name)


def _get_position_range(atlas: object) -> tuple[float, float]:
    from langslice_harness.atlas.core import get_position_range_mm

    return get_position_range_mm(atlas)  # type: ignore[arg-type]


def _atlas_slice_float32(atlas: object, ap_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference_grayscale_image, annotation_slice).

    reference_grayscale_image: HWC float32 [0, 1] — Allen CCFv3 average grayscale
        anatomy template.
    annotation_slice: HW int32 — region IDs (used by density-aware transforms).
    """
    from langslice_harness.atlas.core import get_reference_slice, position_mm_to_index
    from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

    img_pil = get_reference_slice(atlas, ap_mm).convert("RGB")  # type: ignore[arg-type]
    img = np.array(img_pil, dtype=np.float32) / 255.0

    ctx = atlas_space_context(atlas)  # type: ignore[arg-type]
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)  # type: ignore[arg-type]
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]

    return img, ann


def _random_ap_positions(atlas: object, count: int, rng: np.random.Generator) -> list[float]:
    min_mm, max_mm = _get_position_range(atlas)
    return [float(rng.uniform(min_mm, max_mm)) for _ in range(count)]


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def _save_float32(image: np.ndarray, path: Path) -> None:
    """Save HWC float32 [0,1] as PNG."""
    from PIL import Image

    uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if uint8.ndim == 2:
        Image.fromarray(uint8, mode="L").save(path)
    else:
        Image.fromarray(uint8, mode="RGB").save(path)


def _make_grid(images: list[np.ndarray], ncols: int) -> np.ndarray:
    """Stack HWC float32 images into a grid; pads with zeros if incomplete."""
    n = len(images)
    if n == 0:
        return np.zeros((1, 1, 3), dtype=np.float32)
    h, w, c = images[0].shape
    nrows = (n + ncols - 1) // ncols
    grid = np.zeros((nrows * h, ncols * w, c), dtype=np.float32)
    for idx, img in enumerate(images):
        r, col = divmod(idx, ncols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = img[:h, :w]
    return grid


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def _progress_iter(iterable: list, desc: str):  # type: ignore[type-arg]
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, ncols=80)
    except ImportError:
        print(f"{desc} ({len(iterable)} items)...", flush=True)
        return iterable


# ---------------------------------------------------------------------------
# Augmentation dispatch
# ---------------------------------------------------------------------------


def _augment(
    img: np.ndarray,
    ann: np.ndarray,
    atlas: object,
    modality: str,
    seed: int,
    position_mm: float,
    plane: str = "coronal",
) -> np.ndarray:
    from augmentation.brightfield_pipeline import render_brightfield_section
    from augmentation.dapi_pipeline import render_dapi_section
    from augmentation.fluorescence_pipeline import render_fluorescence_section
    from augmentation.ish_pipeline import render_ish_section
    from augmentation.nissl_pipeline import render_nissl_section

    pixel_size_um = float(min(atlas.resolution[1], atlas.resolution[2]))  # type: ignore[index]

    if modality == "dapi":
        return render_dapi_section(
            img,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            plane=plane,
            position_mm=position_mm,
        )
    if modality == "nissl":
        return render_nissl_section(
            img,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            plane=plane,
            position_mm=position_mm,
        )
    if modality == "brightfield":
        return render_brightfield_section(
            img,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            plane=plane,
            position_mm=position_mm,
        )
    if modality == "fluorescence":
        return render_fluorescence_section(
            img,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            plane=plane,
            position_mm=position_mm,
        )
    if modality == "ish":
        return render_ish_section(
            img,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            plane=plane,
            position_mm=position_mm,
        )
    raise ValueError(f"Unsupported modality {modality!r}")


# ---------------------------------------------------------------------------
# Subcommand: render-grid
# ---------------------------------------------------------------------------


def _cmd_render_grid(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas = _load_atlas(args.atlas)
    rng = np.random.default_rng(args.seed)
    count: int = args.count
    ncols = min(10, count)
    modalities: list[str] = args.modalities

    for modality in modalities:
        print(f"render-grid: modality={modality}, count={count}", flush=True)
        ap_positions = _random_ap_positions(atlas, count, rng)
        images: list[np.ndarray] = []
        manifest_cells: list[dict] = []

        for i, ap_mm in enumerate(_progress_iter(ap_positions, f"  {modality}")):
            cell_seed = int(rng.integers(0, 2**31))
            img, ann = _atlas_slice_float32(atlas, ap_mm)
            aug = _augment(img, ann, atlas, modality, cell_seed, position_mm=ap_mm)
            images.append(aug)
            manifest_cells.append({"index": i, "ap_mm": round(ap_mm, 4), "seed": cell_seed})

        grid = _make_grid(images, ncols)
        grid_path = out_dir / f"{modality}.png"
        _save_float32(grid, grid_path)

        manifest = {
            "modality": modality,
            "atlas": args.atlas,
            "count": count,
            "seed": args.seed,
            "ncols": ncols,
            "cells": manifest_cells,
        }
        manifest_path = out_dir / f"{modality}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  -> {grid_path}", flush=True)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas = _load_atlas(args.atlas)
    rng = np.random.default_rng(args.seed)
    count: int = args.count
    modality: str = args.modality

    ap_positions = _random_ap_positions(atlas, count, rng)
    items = list(enumerate(ap_positions))

    for idx, ap_mm in _progress_iter(items, f"generate/{modality}"):
        cell_seed = int(rng.integers(0, 2**31))
        img, ann = _atlas_slice_float32(atlas, ap_mm)
        aug = _augment(img, ann, atlas, modality, cell_seed, position_mm=ap_mm)
        out_path = out_dir / f"{modality}_{idx:05d}.png"
        _save_float32(aug, out_path)

    print(f"Generated {count} images in {out_dir}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------


def _cmd_inspect(args: argparse.Namespace) -> int:
    atlas = _load_atlas(args.atlas)
    ap_mm: float = args.ap_mm
    modality: str = args.modality
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img, ann = _atlas_slice_float32(atlas, ap_mm)
    aug = _augment(img, ann, atlas, modality, args.seed, position_mm=ap_mm)

    side_by_side = np.concatenate([img, aug], axis=1)
    _save_float32(side_by_side, out_path)
    print(
        f"Inspect -> {out_path}  (AP={ap_mm:.3f}mm, modality={modality})",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parsing + dispatch
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m models.langslice_gemma_4.data.augmentation.cli",
        description="LangSlice augmentation pipeline CLI",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # render-grid
    rg = subparsers.add_parser(
        "render-grid",
        help="Generate a visual inspection grid per modality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rg.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    rg.add_argument(
        "--modalities",
        nargs="+",
        default=list(_MODALITIES),
        choices=list(_MODALITIES),
        metavar="MODALITY",
        help="Modalities to render",
    )
    rg.add_argument("--count", type=int, default=50, help="Images per modality")
    rg.add_argument("--out", required=True, help="Output directory")

    # generate
    gen = subparsers.add_parser(
        "generate",
        help="Batch-generate augmented images for one modality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gen.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    gen.add_argument("--modality", required=True, choices=list(_MODALITIES), help="Target modality")
    gen.add_argument("--count", type=int, default=1000, help="Number of images to generate")
    gen.add_argument("--out", required=True, help="Output directory")

    # inspect
    ins = subparsers.add_parser(
        "inspect",
        help="Single side-by-side original vs augmented for debugging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ins.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    ins.add_argument("--ap-mm", type=float, required=True, help="AP position in mm")
    ins.add_argument("--modality", required=True, choices=list(_MODALITIES), help="Target modality")
    ins.add_argument("--out", required=True, help="Output PNG path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "render-grid":
            return _cmd_render_grid(args)
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
