"""Show how gamma randomization diversifies the DAPI texture.

One row, four columns: same atlas slice (AP=5.335) rendered at gamma=
0.9 / 1.2 / 1.6 / 2.0. Each gets its own seed so blob placements differ too.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

from viz_dapi_textures import render_on
from langslice_harness.atlas.core import load_atlas

GAMMAS = (0.9, 1.2, 1.6, 2.0)
AP_MM = 5.335


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    cells: list[np.ndarray] = []
    max_h = max_w = 0
    for i, g in enumerate(GAMMAS):
        ref, _gm, _wm, combined = render_on(atlas, AP_MM, gamma=g, seed=42 + i)
        arr = np.clip(combined * 255, 0, 255).astype(np.uint8)
        cells.append(arr)
        h, w = arr.shape[:2]
        max_h, max_w = max(max_h, h), max(max_w, w)
        print(f"gamma={g}  shape={arr.shape}", flush=True)

    grid = np.zeros((max_h, len(GAMMAS) * max_w, 3), dtype=np.uint8)
    for c, img in enumerate(cells):
        h, w = img.shape[:2]
        grid[:h, c * max_w:c * max_w + w] = img

    out = Path("tmp/outputs/dapi/gamma_sweep.png")
    Image.fromarray(grid).save(out)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
