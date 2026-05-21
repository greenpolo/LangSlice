"""Show the WM extreme-stretch + density-randomization diversification.

Single AP (5.335 mm), six runs at the new defaults — each call's per-image
density and per-blob aspect-ratio draws produce visibly different WM looks.
The model trained on these should learn to recognize white-matter tracts even
when they're faint or when nuclei are unphysically streaky.
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

AP_MM = 5.335
N_PANELS = 6


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    cells: list[np.ndarray] = []
    max_h = max_w = 0
    for i in range(N_PANELS):
        ref, _gm, _wm, combined = render_on(atlas, AP_MM, gamma=1.4, seed=100 + i * 17)
        arr = np.clip(combined * 255, 0, 255).astype(np.uint8)
        cells.append(arr)
        h, w = arr.shape[:2]
        max_h, max_w = max(max_h, h), max(max_w, w)
        print(f"panel {i}  shape={arr.shape}", flush=True)

    cols = 3
    rows = (N_PANELS + cols - 1) // cols
    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
    for idx, img in enumerate(cells):
        r, c = divmod(idx, cols)
        h, w = img.shape[:2]
        grid[r * max_h : r * max_h + h, c * max_w : c * max_w + w] = img

    out = Path("tmp/outputs/dapi/wm_diversity.png")
    Image.fromarray(grid).save(out)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
