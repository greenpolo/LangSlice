#!/usr/bin/env python3
"""LangSlice logo generator (final). Run `python build4.py` from this folder to
regenerate logomark/lockup/banner SVGs. build3.py holds the shared row geometry.

Textblock v4: no aura; banner ASCII at SUB sub-rows per bar so internal
structure of the real horizontal section shows. Row rectangles are unchanged,
so the silhouette is still the mark's."""
import os, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build3 as b3
from build3 import (font, row_geometry, row_color, wordmark, svg, render,
                    HI_ROW, BULB_GAP, CURSOR_W, CURSOR_H, CURSOR_GAP, UM, OUT)
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Transform

for p in b3.PAL.values():
    p["glow"] = 0.0                       # 1. no aura anywhere
PAL = b3.PAL

SUB = int(os.environ.get("SUB", 2))       # character sub-rows per bar
RAMP = os.environ.get("RAMP", "-=+*o0%8#@")
LO, HI_ = 0.08, 0.55                      # section-intensity window -> ramp


def sec():
    return b3.section()


def sample(ap_mm, lr_mm, ap_half, lr_half):
    v = sec()
    ai, li = ap_mm / UM, lr_mm / UM
    a0, a1 = int(max(0, ai - ap_half)), int(min(v.shape[0], ai + ap_half + 1))
    l0, l1 = int(max(0, li - lr_half)), int(min(v.shape[1], li + lr_half + 1))
    blk = v[a0:a1, l0:l1]
    return float(blk.mean()) if blk.size else 0.0


def ascii_parts(S, pal, uid):
    cx, pitch, bar, rows = row_geometry(S)
    cell_h = bar / SUB                    # sub-rows fill exactly the bar height
    fsize = cell_h * float(os.environ.get("FS", 1.3))
    adv = fsize * 0.6
    f = font("jb_b")
    scale = fsize / f["head"].unitsPerEm
    cmap, gs = f.getBestCmap(), f.getGlyphSet()
    defs, gid = [], {}
    for k, ch in enumerate(RAMP + "█"):
        sp = SVGPathPen(gs)
        gs[cmap[ord(ch)]].draw(TransformPen(sp, Transform(scale, 0, 0, -scale, 0, 0)))
        d = sp.getCommands()
        if d:
            gid[ch] = f"{uid}c{k}"
            defs.append(f'<path id="{uid}c{k}" d="{d}"/>')
    acc = pal["accent"]
    # AP extent of one bar row in mm (rows are ~1 mm apart; bar covers BAR_FRAC)
    ap_mm_all = [r[3] for r in rows]
    row_ap_pitch = float(np.median(np.diff(ap_mm_all)))
    bar_ap = row_ap_pitch * b3.BAR_FRAC
    sub_ap = bar_ap / SUB

    def run(x0, w, y_bar, col, i, boost=0.0):
        n = max(1, int(round(w / adv)))
        sx = w / (n * adv)
        _, _, _, ap_mm, left_mm, right_mm = rows[i]
        out = [f'<g transform="translate({x0:.2f},0) scale({sx:.4f},1)">']
        for s in range(SUB):
            ap_s = ap_mm - bar_ap / 2 + (s + 0.5) * sub_ap
            y = y_bar - bar / 2 + (s + 0.5) * cell_h
            for j in range(n):
                fx = (j + 0.5) / n
                lr = left_mm + fx * (right_mm - left_mm)
                raw = sample(ap_s, lr, max(1, sub_ap / UM / 2), max(1, (w / n) / (S * (1 - 2 * b3.PAD)) * 13.2 / UM / 2))
                val = float(np.clip((raw - LO) / (HI_ - LO), 0, 1))
                val = 0.22 + 0.78 * val
                k = int(round(min(1.0, val + boost) * (len(RAMP) - 1)))
                ch = RAMP[max(0, k)]
                out.append(f'<use href="#{gid[ch]}" xlink:href="#{gid[ch]}" '
                           f'x="{j*adv:.2f}" y="{y + fsize*0.36:.2f}" fill="{col}"/>')
        out.append("</g>")
        return out

    body, hi = [], None
    for i, (y, hw, kind, *_r) in enumerate(rows):
        if i == HI_ROW:
            hi = (y, hw); continue
        col = row_color(pal, i)
        if kind == "bulb":
            gp = bar * BULB_GAP; lw = hw - gp / 2
            body += run(cx - hw, lw, y, col, i) + run(cx + gp / 2, lw, y, col, i)
        else:
            body += run(cx - hw, 2 * hw, y, col, i)
    y, hw = hi
    body += run(cx - hw, 2 * hw, y, acc, HI_ROW, boost=0.25)
    cw, ch_ = bar * CURSOR_W, bar * CURSOR_H
    x0 = cx + hw + bar * CURSOR_GAP
    body.append(f'<rect x="{x0:.2f}" y="{y-ch_/2:.2f}" width="{cw:.2f}" '
                f'height="{ch_:.2f}" rx="{cw*0.28:.2f}" fill="{acc}"/>')
    return defs, body


RATIO = float(os.environ.get("RATIO", 3))


def write_banner(path, pal, uid, W=1600, bg=None):
    H = int(W / RATIO)
    S = int(H * 0.99)
    mx, my = H * 0.26, (H - S) / 2
    defs, chars = ascii_parts(S, pal, uid)
    body = [f'<g transform="translate({mx:.1f},{my:.1f})">'] + chars + ["</g>"]
    x_text = mx + S + H * 0.12
    ts = H * 0.355
    _, tw = wordmark(ts, 0, 0, pal["word"], pal["accent"])
    avail = W - x_text - H * 0.18
    if tw > avail:
        ts *= avail / tw
    wm, _ = wordmark(ts, x_text, H / 2 + ts * 0.36, pal["word"], pal["accent"])
    body.append(wm)
    svg(path, W, H, body, bg, defs="\n".join(defs))


def write_ascii_square(path, pal, uid, S=512, bg=None):
    defs, chars = ascii_parts(S, pal, uid)
    svg(path, S, S, chars, bg, defs="\n".join(defs))


def main():
    o = OUT
    names = {"mark": "logomark", "lockup": "lockup", "banner_ascii": "banner"}
    for tag, pal in (("", PAL["main"]), ("_dark", PAL["dark"]), ("_light", PAL["light"])):
        b3.write_mark(f"{o}/{names['mark']}{tag}.svg", pal)
        b3.write_lockup(f"{o}/{names['lockup']}{tag}.svg", pal)
        write_banner(f"{o}/{names['banner_ascii']}{tag}.svg", pal, uid="b" + (tag or "m"))
    for n, bg in (("logomark_dark", "#0d1117"), ("logomark_light", "#ffffff")):
        render(f"{o}/{n}.svg", f"{o}/{n}_512.png", 512, bg)
    print("done")


if __name__ == "__main__":
    main()
