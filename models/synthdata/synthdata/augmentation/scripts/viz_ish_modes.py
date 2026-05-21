"""Per-mode ISH comparison grid — 5 columns, 2 rows.

Row 0: real reference image (cropped to ~mid-AP coronal section) or a
       placeholder when no real-microscopy reference is available on disk.
Row 1: procedural rendering at the same atlas AP with that mode pinned.

Column labels show mode name + counterstain + signal.

Output: tmp/outputs/ish/modes.png

Real reference sources
----------------------
allen_style  : Allen Mouse Brain Atlas CC0 ISH image (101329701.jpg, gene Kctd4)
               Path: data/datasets/allen_ish_coronal/100/101329701.jpg
               Cropped from tmp/outputs/ish/real_gm_crop.png (existing).
nbt_nfr      : Allen Mouse Brain Atlas CC0 ISH section with Nuclear Fast Red
               counterstain.  Pre-cropped at tmp/outputs/nfr/real_section.png.
dab_hematoxylin : Hematoxylin-stained mouse brain section (Bmouzon, CC BY-SA 4.0)
               Pre-cropped at tmp/outputs/hematoxylin/real_section.png.
fish         : No suitable CC-licensed fluorescent ISH image available on disk
               or via public API.  Placeholder generated from DAPI pipeline.
               Gap documented in task B1 report.
nissl_nbt    : Cresyl-violet Nissl, Mouse10 P120 dataset (CC0, EBRAINS).
               Pre-cropped at tmp/outputs/nissl/real_gm_crop.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from augmentation.ish_pipeline import render_ish_section
from augmentation.modes import ISH_MODES
from PIL import Image, ImageDraw, ImageFont

from langslice_harness.atlas.core import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEED_BASE = 42

REAL_REFS: dict[str, Path | None] = {
    "allen_style": Path("tmp/outputs/ish/real_gm_crop.png"),
    "nbt_nfr": Path("tmp/outputs/nfr/real_section.png"),
    "dab_hematoxylin": Path("tmp/outputs/hematoxylin/real_section.png"),
    "fish": None,  # no CC-licensed FISH image available on disk
    "nissl_nbt": Path("tmp/outputs/nissl/real_gm_crop.png"),
}

PANEL_W = 320
PANEL_H = 240
LABEL_H = 28
HEADER_H = 22
GUTTER = 4

OUT = Path("tmp/outputs/ish/modes.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label_panel(arr: np.ndarray, text: str, *, bg=(20, 20, 20)) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = _try_font(12)
    draw.rectangle([(0, 0), (arr.shape[1], LABEL_H - 2)], fill=bg)
    draw.text((4, 4), text, fill=(240, 240, 240), font=font)
    return np.asarray(img)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    img = Image.fromarray(arr).resize((w, h), Image.LANCZOS)
    return np.asarray(img)


def _make_placeholder(h: int, w: int, text: str) -> np.ndarray:
    """Gray placeholder when real reference is unavailable."""
    img = Image.new("RGB", (w, h), (50, 50, 60))
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    lines = text.split("\n")
    y = max(4, h // 2 - len(lines) * 8)
    for line in lines:
        try:
            tw = draw.textlength(line, font=font)
        except AttributeError:
            tw = len(line) * 7
        x = max(0, (w - tw) // 2)
        draw.text((x, y), line, fill=(180, 180, 200), font=font)
        y += 18
    return np.asarray(img)


def _load_real(path: Path | None, h: int, w: int, mode_name: str) -> np.ndarray:
    if path is None or not path.exists():
        note = f"[No real reference]\n{mode_name}\nFISH/RNAscope\n(see report)"
        return _make_placeholder(h, w, note)
    img = Image.open(path).convert("RGB")
    return _resize_to(np.asarray(img), h, w)


def upsampled_inputs(atlas, ap_mm: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)
    ann = np.take(
        np.asarray(atlas.annotation),
        idx,
        axis=axis,
    ).astype(np.int32)

    src_um = atlas.resolution[0]
    scale = src_um / TARGET_PX_UM
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    ref_arr = np.asarray(pil.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )
    return ref_arr, ann_up


def _col_header(text: str, w: int) -> np.ndarray:
    img = Image.new("RGB", (w, HEADER_H), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    try:
        tw = draw.textlength(text, font=font)
    except AttributeError:
        tw = len(text) * 7
    x = max(0, (w - tw) // 2)
    draw.text((x, 4), text, fill=(200, 220, 255), font=font)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Loading atlas …")
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)
    print(f"  atlas slice: {ref.shape} @ {TARGET_PX_UM} µm/px")

    columns: list[np.ndarray] = []  # each entry = one full-column (header+row0+row1)

    for i, mode in enumerate(ISH_MODES):
        print(f"  mode {i}: {mode.name} …", flush=True)

        # Row 0 — real reference
        real_arr = _load_real(REAL_REFS.get(mode.name), PANEL_H, PANEL_W, mode.name)
        real_arr = _label_panel(real_arr, f"REAL: {mode.name}")

        # Row 1 — procedural rendering (mode pinned)
        out = render_ish_section(
            ref,
            ann,
            atlas,
            seed=SEED_BASE + i,
            pixel_size_um=TARGET_PX_UM,
            mode=mode,
        )
        proc_arr = _to_uint8(out)
        proc_arr = _resize_to(proc_arr, PANEL_H, PANEL_W)
        proc_arr = _label_panel(proc_arr, f"PROC: {mode.counterstain}+{mode.signal}")

        # Column header
        header_text = f"{mode.name}  [{mode.counterstain}+{mode.signal}]  w={mode.weight:.0%}"
        header = _col_header(header_text, PANEL_W)

        col = np.concatenate([header, real_arr, proc_arr], axis=0)
        columns.append(col)

    # Stitch columns horizontally
    max_h = max(c.shape[0] for c in columns)
    padded = []
    for c in columns:
        if c.shape[0] < max_h:
            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.concatenate([c, pad], axis=0)
        padded.append(c)

    # Add thin vertical gutter between columns
    gutter_col = np.full((max_h, GUTTER, 3), 10, dtype=np.uint8)
    grid = padded[0]
    for c in padded[1:]:
        grid = np.concatenate([grid, gutter_col, c], axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
