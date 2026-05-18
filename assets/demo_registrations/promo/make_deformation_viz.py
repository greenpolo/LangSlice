"""Visualize the elastix B-spline deformation as a grid of arrows.

Reads the visualign_markers (atlas_x, atlas_y, slice_x, slice_y) from the
slice_19_clahe registration.json and draws arrows from each atlas grid point
to its deformed slice-space target. Arrows are blue, on a faintly dimmed
atlas-colored-regions backdrop so the user can see *what* is being deformed.

Output: deformation_viz.png in this directory (and also a no-bg variant).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

BG = (0, 0, 0)
BLUE = (57, 162, 250)
WHITE = (255, 255, 255)

ROOT = Path(__file__).resolve().parent
REGISTRATION_JSON = (
    ROOT.parent / "slice_19_clahe" / "registration" / "registration.json"
)
COLORED_REGIONS = (
    ROOT.parent / "slice_19_clahe" / "registration"
    / "candidate-ae9899978a28" / "input_colored_regions.png"
)

# Subsample the 60x36 grid so arrows are readable.
SUBSAMPLE = 3

# Visual amplification factor for the displacement arrows. The raw
# deformation is small relative to the grid spacing; multiply it so the
# field reads visually. 1.0 = true scale.
ARROW_AMP = 3.0

# Pixel sizing on the rendered canvas
CANVAS_W = 1600
ARROW_STROKE = 4
HEAD_LEN = 18
HEAD_HW = 9
DOT_R = 4
# 0.0 = full atlas color, 1.0 = pure black. Keep enough detail that the
# anatomy reads, but not so much that blue arrows lose contrast.
DARKEN = 0.25


def load_markers() -> list[tuple[float, float, float, float]]:
    with REGISTRATION_JSON.open() as f:
        data = json.load(f)
    return data["annotation_session"]["metadata"]["visualign_markers"]


def build() -> Path:
    markers = load_markers()

    # Atlas grid extents (markers are in atlas pixel coords)
    axs = [m[0] for m in markers]
    ays = [m[1] for m in markers]
    atlas_w = max(axs) - min(axs)
    atlas_h = max(ays) - min(ays)
    ax_min, ay_min = min(axs), min(ays)

    # Render canvas sized to atlas aspect (with pad)
    pad = 20
    inner_w = CANVAS_W - 2 * pad
    inner_h = int(round(inner_w * atlas_h / atlas_w))
    canvas_h = inner_h + 2 * pad

    canvas = Image.new("RGB", (CANVAS_W, canvas_h), BG)

    import os
    underlay = os.environ.get("DEFORM_UNDERLAY", "1") == "1"
    if underlay:
        bg_img = Image.open(COLORED_REGIONS).convert("RGB")
        bg_resized = bg_img.resize((inner_w, inner_h), Image.LANCZOS)
        dark = Image.new("RGB", bg_resized.size, BG)
        bg_dim = Image.blend(bg_resized, dark, DARKEN)
        canvas.paste(bg_dim, (pad, pad))

    draw = ImageDraw.Draw(canvas, "RGBA")

    def atlas_to_canvas(ax: float, ay: float) -> tuple[float, float]:
        x = pad + (ax - ax_min) / atlas_w * inner_w
        y = pad + (ay - ay_min) / atlas_h * inner_h
        return x, y

    # Detect grid layout (sorted rows by unique y)
    uniq_ys = sorted({round(m[1], 2) for m in markers})
    uniq_xs = sorted({round(m[0], 2) for m in markers})
    ncols = len(uniq_xs)
    nrows = len(uniq_ys)
    print(f"grid: {ncols} cols x {nrows} rows = {ncols*nrows} pts, {len(markers)} actual")

    # Subsample by row/col index
    selected = []
    for m in markers:
        ax, ay = m[0], m[1]
        try:
            ci = uniq_xs.index(round(ax, 2))
            ri = uniq_ys.index(round(ay, 2))
        except ValueError:
            continue
        if ci % SUBSAMPLE == 0 and ri % SUBSAMPLE == 0:
            selected.append(m)

    print(f"drawing {len(selected)} arrows")

    for ax, ay, sx, sy in selected:
        # The slice positions are in slice-pixel space (different scale).
        # Convert slice-space displacement back to atlas-space by mapping the
        # slice marker through the same affine: slice_x / slice_w * atlas_w
        # ...but actually both grids are stored as proportional coords already
        # since they share min/max ranges. Treat raw values as atlas-space.
        x0, y0 = atlas_to_canvas(ax, ay)
        x1, y1 = atlas_to_canvas(sx, sy)

        # amplify displacement
        dx = (x1 - x0) * ARROW_AMP
        dy = (y1 - y0) * ARROW_AMP
        x1, y1 = x0 + dx, y0 + dy

        # draw the start dot in white
        draw.ellipse([x0 - DOT_R, y0 - DOT_R, x0 + DOT_R, y0 + DOT_R],
                     fill=WHITE)

        # short displacements: no arrow head
        mag = math.hypot(dx, dy)
        if mag < 2:
            continue

        draw.line([(x0, y0), (x1, y1)], fill=BLUE, width=ARROW_STROKE)

        # arrowhead
        if mag >= HEAD_LEN:
            ux, uy = dx / mag, dy / mag
            # perpendicular
            px, py = -uy, ux
            base_x = x1 - ux * HEAD_LEN
            base_y = y1 - uy * HEAD_LEN
            p1 = (base_x + px * HEAD_HW, base_y + py * HEAD_HW)
            p2 = (base_x - px * HEAD_HW, base_y - py * HEAD_HW)
            draw.polygon([(x1, y1), p1, p2], fill=BLUE)

    out = ROOT / ("deformation_viz_v4.png" if underlay else "deformation_viz_v4_nobg.png")
    canvas.save(out)
    print(f"wrote {out} ({CANVAS_W}x{canvas_h})")
    return out


if __name__ == "__main__":
    build()
