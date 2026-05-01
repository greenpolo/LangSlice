"""Oblique slice visualization — 3×3 grid showing tilted coronal slices.

Grid layout (same AP position, same seed, different tilt angles):

    (0,0) Canonical          (0,1) Random tilt A     (0,2) Random tilt B
    (1,0) Random tilt C      (1,1) Random tilt D     (1,2) Random tilt E
    (2,0) Random tilt F      (2,1) Random tilt G     (2,2) Random tilt H

Each panel shows the DAPI-rendered section at the tilted angle. The panel
header shows the (yaw, pitch, roll) used. The canonical panel (top-left)
uses yaw=pitch=roll=0.

Output: tmp/outputs/oblique.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.dapi_pipeline import render_dapi_section
from augmentation.oblique import get_oblique_slice, sample_oblique_angles
from langslice_harness.atlas.core import load_atlas

AP_MM = 5.335
TARGET_PX_UM = 5.0
DAPI_SEED = 42
MASTER_SEED = 20260429
OUT_PATH = Path("tmp/outputs/oblique.png")

# Panel dimensions
THUMB_H = 260
THUMB_W = 340
LABEL_H = 32


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------


def _try_font(size: int = 12) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Panel rendering helpers
# ---------------------------------------------------------------------------


def _to_thumb(arr: np.ndarray) -> np.ndarray:
    """Float32 HWC [0,1] → uint8 (THUMB_H, THUMB_W, 3)."""
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    return np.asarray(img)


def _label_panel(thumb: np.ndarray, label: str) -> np.ndarray:
    """Return (LABEL_H + THUMB_H) × THUMB_W panel with labelled header bar."""
    bar = np.full((LABEL_H, THUMB_W, 3), 18, dtype=np.uint8)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    draw.text((5, 7), label, fill=(220, 220, 220), font=font)
    bar = np.asarray(img)
    return np.concatenate([bar, thumb], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Loading atlas...", flush=True)
    atlas = load_atlas("allen_mouse_25um")

    rng = np.random.default_rng(MASTER_SEED)

    # Build 9 tilt configs: first is canonical, next 8 are random.
    tilt_configs: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for _ in range(8):
        yaw, pitch, roll = sample_oblique_angles(
            rng,
            max_yaw_deg=12.0,
            max_pitch_deg=12.0,
            max_roll_deg=8.0,
        )
        tilt_configs.append((yaw, pitch, roll))

    panels: list[np.ndarray] = []
    cell_h = LABEL_H + THUMB_H

    for i, (yaw, pitch, roll) in enumerate(tilt_configs):
        row_num = i // 3
        col_num = i % 3
        label_prefix = "Canonical" if i == 0 else f"Panel {i}"
        label = f"{label_prefix}  yaw={yaw:+.1f}° pitch={pitch:+.1f}° roll={roll:+.1f}°"
        print(f"  [{i+1}/9] {label}", flush=True)

        # Extract oblique reference + annotation
        ref_slice, ann_slice = get_oblique_slice(
            atlas,
            base_position_mm=AP_MM,
            plane="coronal",
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

        # Upsample to target resolution before rendering
        src_um = float(atlas.resolution[0])  # type: ignore[attr-defined]
        scale = src_um / TARGET_PX_UM
        h0, w0 = ann_slice.shape
        h_up = int(round(h0 * scale))
        w_up = int(round(w0 * scale))

        ref_up = np.asarray(
            Image.fromarray(ref_slice).resize((w_up, h_up), Image.LANCZOS),
            dtype=np.uint8,
        )
        ann_up = np.asarray(
            Image.fromarray(ann_slice.astype(np.int32), "I").resize(
                (w_up, h_up), Image.NEAREST
            ),
            dtype=np.int32,
        )

        # Render DAPI
        rendered = render_dapi_section(
            ref_up,
            ann_up,
            atlas,
            seed=DAPI_SEED,
            pixel_size_um=TARGET_PX_UM,
        )

        thumb = _to_thumb(rendered)
        panel = _label_panel(thumb, label)
        panels.append(panel)

    # Assemble 3×3 grid
    rows_rendered = []
    for row_idx in range(3):
        row_panels = panels[row_idx * 3 : row_idx * 3 + 3]
        row_img = np.concatenate(row_panels, axis=1)
        rows_rendered.append(row_img)

    grid = np.concatenate(rows_rendered, axis=0)

    # Add title bar at top
    title_h = 36
    title_bar = np.full((title_h, grid.shape[1], 3), 10, dtype=np.uint8)
    title_img = Image.fromarray(title_bar)
    draw = ImageDraw.Draw(title_img)
    font_title = _try_font(14)
    draw.text(
        (10, 9),
        f"Oblique reslicing demo  —  AP={AP_MM} mm, allen_mouse_25um, {TARGET_PX_UM} µm/px",
        fill=(230, 230, 230),
        font=font_title,
    )
    title_arr = np.asarray(title_img)

    final = np.concatenate([title_arr, grid], axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}  ({final.shape[1]}x{final.shape[0]} px)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
