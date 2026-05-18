"""Synthesize a Gemma-style 4-point star logo (used as a fallback when the
exact reference image isn't on disk). Output: assets/gemma_logo_synth.png."""
from pathlib import Path
import math
from PIL import Image, ImageDraw

SIZE = 600
BG = (0, 0, 0)
BLUE = (57, 162, 250)
LINE = 6

img = Image.new("RGB", (SIZE, SIZE), BG)
d = ImageDraw.Draw(img)

cx = cy = SIZE // 2
# Outer circle
r_outer = int(SIZE * 0.42)
d.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
          outline=BLUE, width=LINE)

# Grid: 2 horizontal + 2 vertical lines forming a 3x3 split of the circle bbox
g_in = int(SIZE * 0.20)   # inset from center to inner grid lines
g_out = int(SIZE * 0.42)  # extension from center beyond circle
for off in (-g_in, g_in):
    d.line([(cx + off, cy - g_out - 20), (cx + off, cy + g_out + 20)],
           fill=BLUE, width=LINE)
    d.line([(cx - g_out - 20, cy + off), (cx + g_out + 20, cy + off)],
           fill=BLUE, width=LINE)

# 4-point concave star (Gemini-style) built from a polygon with quadratic-ish curves.
# Approximate concave sides with many segments going through a control point near
# the center between consecutive tips.
r_tip = int(SIZE * 0.40)   # tip distance from center (longer rays)
r_ctrl = int(SIZE * 0.04)  # how far the side bows inward (smaller = more concave)

tips = []
for i in range(4):
    a = math.pi / 2 - i * math.pi / 2  # top, right, bottom, left
    tips.append((cx + r_tip * math.cos(a), cy - r_tip * math.sin(a)))

# Build a smooth concave-side polygon by sampling quadratic bezier curves
# between consecutive tips, with the control point pulled toward the center.
def quad_bezier(p0, p1, p2, n=24):
    pts = []
    for k in range(n + 1):
        t = k / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts

poly: list[tuple[float, float]] = []
for i in range(4):
    p0 = tips[i]
    p2 = tips[(i + 1) % 4]
    # Control point near center, between p0 and p2.
    mx = (p0[0] + p2[0]) / 2
    my = (p0[1] + p2[1]) / 2
    # Pull inward along (mx,my) -> center direction.
    dx = cx - mx
    dy = cy - my
    norm = math.hypot(dx, dy) or 1.0
    scale = (r_tip - r_ctrl) / norm
    ctrl = (mx + dx * scale, my + dy * scale)
    poly.extend(quad_bezier(p0, ctrl, p2)[:-1])  # drop last; next iter adds it

d.polygon(poly, fill=BLUE)

out = Path(__file__).resolve().parent.parent.parent / "gemma_logo_synth.png"
img.save(out)
print(f"wrote {out} ({SIZE}x{SIZE})")
