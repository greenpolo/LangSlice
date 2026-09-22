"""Recolor the shipped README figures to the LangSlice logo palette.

The figure generators in this folder were written against Windows paths and
inputs that no longer live in this repo, so the branding pass works on the
finished PNGs instead: the blue chrome (arrows, plus signs, brackets, the
chosen-candidate box) becomes the logo accent pink, and the cyan atlas
borders drawn on the tissue become the logo wordmark off-white. Tissue,
atlas fills and grey captions are untouched.

    python recolor_brand.py IN.png OUT.png [x0,y0,x1,y1]

The optional box limits the border step to one panel (the registration
pipeline figure keeps its atlas fills untouched that way).
"""

import sys

import numpy as np
from PIL import Image

ACCENT = np.array([255, 61, 170], float)   # logo accent, #FF3DAA
WORD = np.array([220, 227, 255], float)    # logo wordmark, #DCE3FF
CHROME_BLUES = [np.array([57, 162, 250], float), np.array([64, 150, 255], float)]


def recolor_chrome(p: np.ndarray) -> np.ndarray:
    """Pixels that are a blend of black and one of the chrome blues -> pink."""
    out = p.copy()
    for blue in CHROME_BLUES:
        t = (p @ blue) / (blue @ blue)
        resid = np.linalg.norm(p - t[..., None] * blue, axis=-1)
        m = (resid < 16) & (t > 0.04)
        out[m] = np.clip(t[m, None] * ACCENT, 0, 255)
    return out


def recolor_borders(p: np.ndarray) -> np.ndarray:
    """Cyan-ness (G and B high, R low) fades each pixel toward the wordmark color."""
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    t = np.clip((np.minimum(g, b) - r) / 255, 0, 1)
    t = np.clip((t - 0.25) / 0.6, 0, 1)[..., None]
    return (1 - t) * p + t * WORD


def main(src: str, dst: str, box: str | None = None) -> None:
    p = recolor_chrome(np.array(Image.open(src).convert("RGB")).astype(float))
    if box is None:
        p = recolor_borders(p)
    else:
        x0, y0, x1, y1 = (int(v) for v in box.split(","))
        p[y0:y1, x0:x1] = recolor_borders(p[y0:y1, x0:x1])
    Image.fromarray(p.astype(np.uint8)).save(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
