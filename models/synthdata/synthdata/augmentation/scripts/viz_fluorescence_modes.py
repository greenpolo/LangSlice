"""Per-mode fluorescence comparison grid — 6 columns, 2 rows.

Row 0: real reference image (from Allen Connectivity CC0, or [no real ref] placeholder).
Row 1: procedural rendering with that mode pinned.

Columns (6 representative modes):
  0: dapi_only         — dark canvas, blue DAPI only
  1: dapi_gfp          — dark canvas, DAPI + green GFP
  2: dapi_tdtom        — dark canvas, DAPI + red tdTomato
  3: dapi_pv_som_vip   — dark canvas, DAPI + sparse triple (PV/SOM/VIP)
  4: tract_aav_gfp_neurotrace — NeuroTrace substrate + sparse AAV-GFP
  5: generic_dapi_2if  — dark canvas, DAPI + 2 random channels

Output: tmp/outputs/fluorescence/modes.png

Real reference sources
----------------------
dapi_only      : Allen Connectivity CRE experiment 286483411, CC0 (DAPI + autofluorescence)
                 Gap note: not a pure DAPI image; using closest available (DAPI visible in B channel)
dapi_gfp       : Allen Connectivity CRE experiment 301540850, CC0 (GFP-expressing Cre neurons)
dapi_tdtom     : Allen Connectivity CRE experiment 300687607, CC0 (tdTomato Cre neurons)
dapi_pv_som_vip: No suitable CC-licensed three-color PV+SOM+VIP triple confocal image available
                 on disk. Placeholder generated. Gap documented.
tract_aav_gfp  : Allen Connectivity CRE experiment 294481346, CC0 (AAV-tdTom sparse tracts)
generic_dapi_2if: No fixed real reference for generic mode (by design — picks random channels).
                 Placeholder generated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.fluorescence_pipeline import render_fluorescence_section
from augmentation.modes import FLUORESCENCE_MODES
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEED_BASE = 42

# 6 representative modes (index into FLUORESCENCE_MODES)
COLUMN_MODES: list[str] = [
    "dapi_only",
    "dapi_gfp",
    "dapi_tdtom",
    "dapi_pv_som_vip",
    "tract_aav_gfp_neurotrace",
    "generic_dapi_2if",
]

# Real reference paths — None means placeholder
REAL_REFS: dict[str, Path | None] = {
    "dapi_only":               Path("tmp/outputs/fluorescence/real_dapi_gfp_section.png"),  # closest available
    "dapi_gfp":                Path("tmp/outputs/fluorescence/real_dapi_gfp_section.png"),
    "dapi_tdtom":              Path("tmp/outputs/fluorescence/real_dapi_tdtom_section.png"),
    "dapi_pv_som_vip":         None,   # gap: no CC triple-color PV+SOM+VIP on disk
    "tract_aav_gfp_neurotrace": Path("tmp/outputs/fluorescence/real_tract_section.png"),
    "generic_dapi_2if":        None,   # gap: generic mode has no fixed real reference
}

PANEL_W = 320
PANEL_H = 240
LABEL_H = 28
HEADER_H = 22
GUTTER = 4

OUT = Path("tmp/outputs/fluorescence/modes.png")


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


def _label_panel(
    arr: np.ndarray, text: str, *, bg: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
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
    y = max(4, h // 2 - len(lines) * 9)
    for line in lines:
        try:
            tw = draw.textlength(line, font=font)
        except AttributeError:
            tw = len(line) * 7
        x = max(0, (w - tw) // 2)
        draw.text((x, y), line, fill=(180, 180, 200), font=font)
        y += 18
    return np.asarray(img)


def _load_real(
    path: Path | None, h: int, w: int, mode_name: str,
) -> np.ndarray:
    if path is None or not path.exists():
        note = f"[No real reference]\n{mode_name}\n(gap documented\nin report)"
        return _make_placeholder(h, w, note)
    img = Image.open(path).convert("RGB")
    return _resize_to(np.asarray(img), h, w)


def upsampled_inputs(atlas: object, ap_mm: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)
    ann = np.take(
        np.asarray(atlas.annotation), idx, axis=axis,
    ).astype(np.int32)

    src_um = atlas.resolution[0]  # type: ignore[attr-defined]
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

    # Build name→mode lookup
    mode_by_name = {m.name: m for m in FLUORESCENCE_MODES}

    columns: list[np.ndarray] = []

    for i, mode_name in enumerate(COLUMN_MODES):
        mode = mode_by_name[mode_name]
        print(f"  mode {i}: {mode_name} …", flush=True)

        # Row 0 — real reference
        real_arr = _load_real(REAL_REFS.get(mode_name), PANEL_H, PANEL_W, mode_name)
        real_label = "REAL (Allen CC0)" if REAL_REFS.get(mode_name) else "[No real ref]"
        real_arr = _label_panel(real_arr, f"{real_label}: {mode_name}")

        # Row 1 — procedural (mode pinned)
        out = render_fluorescence_section(
            ref, ann, atlas,
            seed=SEED_BASE + i,
            pixel_size_um=TARGET_PX_UM,
            mode=mode,
        )
        proc_arr = _to_uint8(out)
        proc_arr = _resize_to(proc_arr, PANEL_H, PANEL_W)
        proc_arr = _label_panel(proc_arr, f"PROC: {mode.counterstain}+{len(mode.channels)}ch")

        # Column header
        weight_pct = f"{mode.weight:.0%}"
        header_text = f"{mode_name}  [w={weight_pct}]"
        header = _col_header(header_text, PANEL_W)

        col = np.concatenate([header, real_arr, proc_arr], axis=0)
        columns.append(col)

    # Stitch columns horizontally with gutters
    max_h = max(c.shape[0] for c in columns)
    padded = []
    for c in columns:
        if c.shape[0] < max_h:
            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.concatenate([c, pad], axis=0)
        padded.append(c)

    gutter_col = np.full((max_h, GUTTER, 3), 10, dtype=np.uint8)
    grid = padded[0]
    for c in padded[1:]:
        grid = np.concatenate([grid, gutter_col, c], axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT)
    print(f"\n-> {OUT}  ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
