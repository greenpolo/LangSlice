"""Build the base-vs-langslice slicebench comparison figure for the submission.

Two-panel chart in the LangSlice aesthetic (black background, blue accent):
left = mean accuracy, right = mean absolute error. Source numbers pulled
directly from the small-coronal slicebench summary JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("C:/LabSoftware/LangSlice")
OUT = REPO / "assets" / "demo_registrations" / "promo" / "slicebench_compare.png"

BASE = REPO / "slicebench/runs/small/gemma-4-e4b-it-base-parityfix/summary.json"
FT = REPO / "out/releases/langslice-gemma-4-E4B-v1.0/eval_small_coronal_2026-05-17/summary.json"

BG = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (57, 162, 250)
DIM = (165, 175, 190)
BASE_FILL = (110, 120, 135)   # neutral gray for baseline
FT_FILL = BLUE                 # branded blue for langslice

FONT_BOLD = "C:/Windows/Fonts/consolab.ttf"
FONT_REG = "C:/Windows/Fonts/consola.ttf"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def text_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                   f: ImageFont.FreeTypeFont, fill) -> None:
    w = f.getlength(text)
    bbox = f.getbbox("Mg")
    h = bbox[3] - bbox[1]
    draw.text((xy[0] - w // 2, xy[1] - h // 2 - bbox[1]), text, font=f, fill=fill)


def main() -> None:
    base = json.loads(BASE.read_text())
    ft = json.loads(FT.read_text())

    base_acc = base["overall"]["mean_accuracy_pct"]
    ft_acc = ft["overall"]["mean_accuracy_pct"]
    base_mae = base["overall"]["mae_mm"]
    ft_mae = ft["overall"]["mae_mm"]

    W, H = 1600, 950
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # title
    title_f = font(48, bold=True)
    title_parts = [("{ ", BLUE), ("SliceBench", WHITE), ("  base vs LangSlice-Gemma-4", BLUE), (" }", BLUE)]
    total = sum(title_f.getlength(s) for s, _ in title_parts)
    x = (W - total) // 2
    bbox = title_f.getbbox("Mg")
    y_text = 35
    for s, col in title_parts:
        draw.text((x, y_text), s, font=title_f, fill=col)
        x += title_f.getlength(s)

    # subtitle
    sub = font(22, bold=False)
    text_centered(draw, (W // 2, 105),
                  "small coronal eval set  ·  n=172  ·  mouse + rat", sub, DIM)

    # ----- panels -----
    panel_y0 = 170
    panel_y1 = 810
    panel_h = panel_y1 - panel_y0
    panel_w = (W - 3 * 60) // 2  # 60px outer margins + 60 gap
    left_x0 = 60
    right_x0 = left_x0 + panel_w + 60

    # left panel: accuracy
    draw_panel(canvas, draw, left_x0, panel_y0, panel_w, panel_h,
               title="Mean accuracy (%)",
               base_val=base_acc, ft_val=ft_acc,
               y_max=100.0, fmt="{:.1f}%", higher_is_better=True)

    # right panel: MAE
    draw_panel(canvas, draw, right_x0, panel_y0, panel_w, panel_h,
               title="Mean absolute error (mm)",
               base_val=base_mae, ft_val=ft_mae,
               y_max=max(base_mae, ft_mae) * 1.25,
               fmt="{:.2f} mm", higher_is_better=False)

    # footer: delta callout
    foot = font(26, bold=True)
    delta_mae_pct = (base_mae - ft_mae) / base_mae * 100
    delta_acc = ft_acc - base_acc
    parts = [
        ("LangSlice-Gemma-4-E4B v1: ", WHITE),
        (f"+{delta_acc:.1f} pp accuracy", BLUE),
        ("   ·   ", DIM),
        (f"−{delta_mae_pct:.0f}% mean error", BLUE),
        ("   ·   ", DIM),
        (f"{base_mae / ft_mae:.2f}× MAE reduction", BLUE),
    ]
    total = sum(foot.getlength(s) for s, _ in parts)
    x = (W - total) // 2
    bbox = foot.getbbox("Mg")
    y = 860
    for s, col in parts:
        draw.text((x, y), s, font=foot, fill=col)
        x += foot.getlength(s)

    canvas.save(OUT)
    print(f"wrote {OUT} ({W}x{H})")


def draw_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw,
                x0: int, y0: int, w: int, h: int, *,
                title: str, base_val: float, ft_val: float,
                y_max: float, fmt: str, higher_is_better: bool) -> None:
    title_f = font(30, bold=True)
    text_centered(draw, (x0 + w // 2, y0 + 22), title, title_f, WHITE)

    # plot area
    plot_pad_top = 70
    plot_pad_bot = 90
    plot_pad_lr = 80
    px0 = x0 + plot_pad_lr
    px1 = x0 + w - plot_pad_lr
    py0 = y0 + plot_pad_top
    py1 = y0 + h - plot_pad_bot
    plot_h = py1 - py0

    # axis line
    draw.line([(px0, py1), (px1, py1)], fill=WHITE, width=3)
    draw.line([(px0, py0), (px0, py1)], fill=WHITE, width=3)

    # y-axis ticks (5 nice steps)
    tick_f = font(20, bold=False)
    if y_max >= 50:
        steps = [0, 25, 50, 75, 100]
    else:
        # round y_max to nice step
        steps = [round(y_max * i / 4, 1) for i in range(5)]
    for v in steps:
        if v > y_max:
            continue
        ty = py1 - int(plot_h * v / y_max)
        draw.line([(px0 - 8, ty), (px0, ty)], fill=WHITE, width=2)
        label = f"{int(v)}" if v == int(v) else f"{v:.1f}"
        lw = tick_f.getlength(label)
        draw.text((px0 - 14 - lw, ty - 12), label, font=tick_f, fill=DIM)

    # two bars
    n_bars = 2
    bar_band = (px1 - px0) // n_bars
    bar_w = int(bar_band * 0.55)
    label_f = font(22, bold=True)
    val_f = font(28, bold=True)

    bars = [("Gemma 4 E4B-IT", base_val, BASE_FILL),
            ("LangSlice-Gemma-4", ft_val, FT_FILL)]
    for i, (label, val, fill) in enumerate(bars):
        cx = px0 + bar_band * i + bar_band // 2
        bx0 = cx - bar_w // 2
        bx1 = cx + bar_w // 2
        bar_h = int(plot_h * val / y_max)
        by0 = py1 - bar_h
        draw.rectangle([bx0, by0, bx1, py1], fill=fill, outline=WHITE, width=2)
        # value label above bar
        text_centered(draw, (cx, by0 - 24), fmt.format(val), val_f, WHITE)
        # x-axis label (two lines)
        text_centered(draw, (cx, py1 + 30), label, label_f, WHITE)
        sub_label = "(base)" if i == 0 else "(v1.0, ours)"
        sub_f = font(18, bold=False)
        text_centered(draw, (cx, py1 + 58), sub_label, sub_f, DIM)

    # "higher/lower is better" hint
    hint_f = font(16, bold=False)
    hint = "↑ higher is better" if higher_is_better else "↓ lower is better"
    text_centered(draw, (x0 + w // 2, y0 + h - 14), hint, hint_f, DIM)


if __name__ == "__main__":
    main()
