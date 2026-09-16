#!/usr/bin/env python3
"""LangSlice logo -- Textblock v3.

ONE geometry.  `row_geometry()` returns the 15 rows (slot, half-width, kind)
derived from the real Allen dorsal silhouette.  The mark draws each row as a
rounded rect; the ASCII banner draws the SAME row as a run of monospace
characters spanning exactly the same length, at a cell height equal to the bar
pitch.  Nothing else differs.
"""
import json, os, glob
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Transform

OUT = os.path.dirname(os.path.abspath(__file__))
SIL = os.path.join(OUT, "data")
FONTS = {
    "mont_m": "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Medium.otf",
    "mont_b": "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Bold.otf",
    "jb_b": "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Bold.otf",
}
_f = {}
UM = 0.025          # mm per atlas voxel


def font(k):
    if k not in _f:
        _f[k] = TTFont(FONTS[k])
    return _f[k]


def text_paths(s, fkey, size, x=0.0, y=0.0, tracking=0.0):
    f = font(fkey)
    scale = size / f["head"].unitsPerEm
    cmap, gs, hmtx = f.getBestCmap(), f.getGlyphSet(), f["hmtx"]
    px, out = x, []
    for ch in s:
        gn = cmap[ord(ch)]
        sp = SVGPathPen(gs)
        gs[gn].draw(TransformPen(sp, Transform(scale, 0, 0, -scale, px, y)))
        d = sp.getCommands()
        if d:
            out.append(d)
        px += hmtx[gn][0] * scale + tracking
    return out, px - x - (tracking if s else 0)


def wordmark(size, x, y, colA, colB):
    frag = []
    pa, wa = text_paths("Lang", "mont_m", size, x, y)
    frag += [f'<path fill="{colA}" d="{d}"/>' for d in pa]
    pb, wb = text_paths("Slice", "mont_b", size, x + wa, y)
    frag += [f'<path fill="{colB}" d="{d}"/>' for d in pb]
    return "\n".join(frag), wa + wb


# ------------------------------------------------------------------ palette
PAL = {
    "main": dict(tissue="#5F6FD1", tissue2="#7E63CF", accent="#FF3DAA",
                 glow=0.26, word="#5F6FD1"),
    "dark": dict(tissue="#8FA0F5", tissue2="#A98BF0", accent="#FF3DAA",
                 glow=0.85, word="#DCE3FF"),
    "light": dict(tissue="#3A47A4", tissue2="#5A3FA0", accent="#D9127F",
                  glow=0.0, word="#222A55"),
}

# ------------------------------------------------------------------ geometry
ROWS_AP = ([(0.3, "bulb"), (1.2, "bulb")]
           + [(ap, "ctx") for ap in
              (2.2, 3.2, 4.2, 5.2, 6.2, 7.3, 8.4, 9.4, 10.25)]
           + [(ap, "cb") for ap in (10.85, 11.7, 12.5)])
N_ROWS = len(ROWS_AP)
HI_ROW = 6
GROUP_GAP = 0.70
PAD = 0.07
HALF_FRAC = 0.395       # widest half-bar, as a fraction of the drawn span
BAR_FRAC = 0.66         # bar height, as a fraction of the row pitch


def _silhouette():
    rows = json.load(open(os.path.join(SIL, "dorsal_rows.json")))
    ap = np.array([r["ap_mm"] for r in rows])
    left = np.array([r["left_mm"] for r in rows])
    right = np.array([r["right_mm"] for r in rows])
    return ap, left, right


def row_geometry(S):
    """The single source of truth: (cx, pitch, bar, rows) for an S x S box.

    rows := list of (y, half_width_px, kind, ap_mm, left_mm, right_mm)
    """
    ap, left, right = _silhouette()
    half = (right - left) / 2
    hmax = half.max()
    slots, slot, prev = [], 0.0, ROWS_AP[0][1]
    for a_mm, kind in ROWS_AP:
        if kind != prev:
            slot += GROUP_GAP
            prev = kind
        slots.append(slot)
        slot += 1.0
    span = S * (1 - 2 * PAD)
    halfmax = span * HALF_FRAC
    total = slots[-1] - slots[0]
    pitch = span / (total + 1.0)
    bar = pitch * BAR_FRAC
    top = S * PAD + pitch * 0.5
    out = []
    for (a_mm, kind), sl in zip(ROWS_AP, slots):
        hw = float(np.interp(a_mm, ap, half) / hmax) * halfmax
        out.append((top + sl * pitch, hw, kind,
                    a_mm,
                    float(np.interp(a_mm, ap, left)),
                    float(np.interp(a_mm, ap, right))))
    return S / 2, pitch, bar, out


def lerp_hex(c1, c2, t):
    f = lambda c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    a_, b_ = f(c1), f(c2)
    return "#" + "".join(f"{int(round(a_[i] + (b_[i] - a_[i]) * t)):02x}"
                         for i in range(3))


def row_color(pal, i):
    return lerp_hex(pal["tissue"], pal["tissue2"], i / (N_ROWS - 1))


def halo(x, y, w, h, col, strength, steps=6, grow=0.45, xgrow=None):
    xg = grow if xgrow is None else xgrow
    out = []
    for k in range(steps, 0, -1):
        ey, ex = h * grow * k, h * xg * k
        op = strength * 0.135 * (1 - (k - 1) / steps) ** 1.5
        if op < 0.010:
            continue
        out.append(
            f'<rect x="{x-ex:.2f}" y="{y-ey:.2f}" width="{w+2*ex:.2f}" '
            f'height="{h+2*ey:.2f}" rx="{(h+2*ey)/2:.2f}" fill="{col}" '
            f'opacity="{op:.3f}"/>')
    return out


BULB_GAP = 0.55          # midline opening, in units of bar height
CURSOR_W = 1.15
CURSOR_H = 2.90
CURSOR_GAP = 1.35


# ---------------------------------------------------------------- the mark
def mark_parts(S, pal):
    cx, pitch, bar, rows = row_geometry(S)
    r = bar / 2
    parts, hi = [], None
    acc = pal["accent"]

    def bar_shape(x0, w, y, h, col, i):
        return [f'<rect x="{x0:.2f}" y="{y-h/2:.2f}" width="{w:.2f}" '
                f'height="{h:.2f}" rx="{h/2:.2f}" fill="{col}"/>']

    for i, (y, hw, kind, *_rest) in enumerate(rows):
        if i == HI_ROW:
            hi = (y, hw)
            continue
        col = row_color(pal, i)
        if kind == "bulb":
            gp = bar * BULB_GAP
            lw = hw - gp / 2
            parts += bar_shape(cx - hw, lw, y, bar, col, i)
            parts += bar_shape(cx + gp / 2, lw, y, bar, col, i)
        else:
            parts += bar_shape(cx - hw, 2 * hw, y, bar, col, i)

    y, hw = hi
    h = bar * 1.34
    if pal["glow"] > 0:
        parts += halo(cx - hw, y - h / 2, 2 * hw, h, acc, pal["glow"])
    parts += bar_shape(cx - hw, 2 * hw, y, h, acc, HI_ROW)
    cw, ch = bar * CURSOR_W, bar * CURSOR_H
    x0 = cx + hw + bar * CURSOR_GAP
    if pal["glow"] > 0:
        parts += halo(x0, y - ch / 2, cw, ch, acc, pal["glow"], steps=4,
                      grow=0.30)
    parts.append(f'<rect x="{x0:.2f}" y="{y-ch/2:.2f}" width="{cw:.2f}" '
                 f'height="{ch:.2f}" rx="{cw*0.28:.2f}" fill="{acc}"/>')
    return parts


# --------------------------------------------------- the same brain in ASCII
# Dense ramp: every character has enough ink that a row of them
# collapses to a solid bar when the banner is shrunk to icon size.
RAMP = "cxo0%8#@"
SECTION = None


def section():
    global SECTION
    if SECTION is None:
        v = np.load(os.path.join(OUT, "data", "horiz_110.npy")).astype(np.float32)
        SECTION = v / max(v.max(), 1)
    return SECTION


def sample_density(ap_mm, lr_mm, ap_win):
    """Mean template intensity in a small box around (ap_mm, lr_mm)."""
    v = section()
    ai = ap_mm / UM
    li = lr_mm / UM
    a0, a1 = int(max(0, ai - ap_win)), int(min(v.shape[0], ai + ap_win + 1))
    l0, l1 = int(max(0, li - 3)), int(min(v.shape[1], li + 4))
    blk = v[a0:a1, l0:l1]
    return float(blk.mean()) if blk.size else 0.0


def ascii_parts(S, pal, uid):
    """Draw the SAME rows as runs of characters.  Cell height = row pitch."""
    cx, pitch, bar, rows = row_geometry(S)
    fsize = pitch                      # character cell height == bar pitch
    adv = fsize * 0.6
    f = font("jb_b")
    scale = fsize / f["head"].unitsPerEm
    cmap, gs = f.getBestCmap(), f.getGlyphSet()
    defs, gid = [], {}
    for k, ch in enumerate(RAMP + "█"):
        if ch == " ":
            continue
        sp = SVGPathPen(gs)
        gs[cmap[ord(ch)]].draw(
            TransformPen(sp, Transform(scale, 0, 0, -scale, 0, 0)))
        d = sp.getCommands()
        if d:
            gid[ch] = f"{uid}c{k}"
            defs.append(f'<path id="{uid}c{k}" d="{d}"/>')
    block = "█" if "█" in gid else "#"

    acc = pal["accent"]
    body, glow_pre = [], []
    ap_win = max(1, int(bar / 2 / (UM * (S * (1 - 2 * PAD) / 13.2)) * 0.5)) \
        if False else 6

    def run(x0, w, y, col, i, boost=0.0, op_floor=0.88):
        """A run of characters spanning EXACTLY [x0, x0+w] on row y.

        The run is wrapped in one horizontal scale so the first and last
        glyphs land on the bar's ends: shrink the banner and the row fills
        the same rectangle the mark draws as a solid bar.
        """
        n = max(1, int(round(w / adv)))
        sx = w / (n * adv)
        _, _, _, ap_mm, left_mm, right_mm = rows[i]
        out = [f'<g transform="translate({x0:.2f},0) scale({sx:.4f},1)">']
        for j in range(n):
            fx = (j + 0.5) / n
            lr = left_mm + fx * (right_mm - left_mm)
            val = float(np.clip(
                (sample_density(ap_mm, lr, ap_win) - 0.05) / 0.55, 0, 1))
            v2 = 0.34 + 0.66 * val          # never fall to a sparse glyph
            k = int(round(min(1.0, v2 + boost) * (len(RAMP) - 1)))
            ch = RAMP[max(0, k)]
            if ch not in gid:
                continue
            out.append(
                f'<use href="#{gid[ch]}" xlink:href="#{gid[ch]}" '
                f'x="{j*adv:.2f}" y="{y + fsize*0.35:.2f}" '
                f'fill="{col}" opacity="{op_floor + (1-op_floor)*val:.2f}"/>')
        out.append("</g>")
        return out

    hi = None
    for i, (y, hw, kind, *_rest) in enumerate(rows):
        if i == HI_ROW:
            hi = (y, hw)
            continue
        col = row_color(pal, i)
        if kind == "bulb":
            gp = bar * BULB_GAP
            lw = hw - gp / 2
            body += run(cx - hw, lw, y, col, i)
            body += run(cx + gp / 2, lw, y, col, i)
        else:
            body += run(cx - hw, 2 * hw, y, col, i)

    y, hw = hi
    h = bar * 1.34
    if pal["glow"] > 0:
        glow_pre += halo(cx - hw, y - h / 2, 2 * hw, h, acc, pal["glow"],
                         grow=0.40, xgrow=0.12)
    body += run(cx - hw, 2 * hw, y, acc, HI_ROW, boost=0.22, op_floor=1.0)
    # the cursor, as a block glyph at the same place and size as the mark's
    cw, ch = bar * CURSOR_W, bar * CURSOR_H
    x0 = cx + hw + bar * CURSOR_GAP
    if pal["glow"] > 0:
        glow_pre += halo(x0, y - ch / 2, cw, ch, acc, pal["glow"], steps=4,
                         grow=0.30)
    # the cursor, built from block glyphs scaled to the mark's cursor box
    nb = max(2, int(round(ch / pitch)))
    bw = adv                      # block glyph advance
    bh = fsize * 0.74             # block glyph height (cap to baseline)
    sxc, syc = cw / bw, (ch / nb) / bh
    body.append(f'<g transform="translate({x0:.2f},{y - ch/2:.2f}) '
                f'scale({sxc:.4f},{syc:.4f})">')
    for k in range(nb):
        body.append(f'<use href="#{gid[block]}" xlink:href="#{gid[block]}" '
                    f'x="0" y="{(k + 1) * bh:.2f}" fill="{acc}"/>')
    body.append("</g>")
    return defs, glow_pre, body


# ------------------------------------------------------------------ writers
def svg(path, w, h, body, bg=None, defs=""):
    pre = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    d = f"<defs>{defs}</defs>" if defs else ""
    open(path, "w").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">\n{pre}\n{d}\n' + "\n".join(body) +
        "\n</svg>\n")


def write_mark(path, pal, S=512, bg=None):
    svg(path, S, S, mark_parts(S, pal), bg)


def write_lockup(path, pal, W=1600, bg=None):
    H = W // 4
    S = int(H * 0.88)
    mx, my = H * 0.28, (H - S) / 2
    body = [f'<g transform="translate({mx:.1f},{my:.1f})">']
    body += mark_parts(S, pal)
    body.append("</g>")
    ts = H * 0.40
    wm, _ = wordmark(ts, mx + S + H * 0.30, H * 0.635, pal["word"],
                     pal["accent"])
    body.append(wm)
    svg(path, W, H, body, bg)


def write_banner(path, pal, uid, W=1600, bg=None):
    H = W // 4
    S = int(H * 0.99)
    mx, my = H * 0.26, (H - S) / 2
    defs, glow_pre, chars = ascii_parts(S, pal, uid)
    body = [f'<g transform="translate({mx:.1f},{my:.1f})">']
    body += glow_pre + chars
    body.append("</g>")
    ts = H * 0.355
    wm, _ = wordmark(ts, mx + S + H * 0.12, H * 0.615, pal["word"],
                     pal["accent"])
    body.append(wm)
    svg(path, W, H, body, bg, defs="\n".join(defs))


def write_ascii_square(path, pal, uid, S=512, bg=None):
    defs, glow_pre, chars = ascii_parts(S, pal, uid)
    svg(path, S, S, glow_pre + chars, bg, defs="\n".join(defs))


def render(s, p, width, bg):
    import cairosvg
    cairosvg.svg2png(url=s, write_to=p, output_width=width,
                     background_color=bg)


def main():
    o = OUT
    for tag, pal in (("", PAL["main"]), ("_dark", PAL["dark"]),
                     ("_light", PAL["light"])):
        write_mark(f"{o}/textblock_v3_mark{tag}.svg", pal)
        write_lockup(f"{o}/textblock_v3_lockup{tag}.svg", pal)
        write_banner(f"{o}/textblock_v3_banner_ascii{tag}.svg", pal,
                     uid="b" + (tag or "m"))
    write_ascii_square(f"{o}/_v3_ascii_square.svg", PAL["dark"], "sq")
    write_mark(f"{o}/_v3_mark_square.svg", PAL["dark"])

    os.makedirs(f"{o}/png3", exist_ok=True)
    for s in sorted(glob.glob(f"{o}/textblock_v3_*.svg")):
        b = os.path.splitext(os.path.basename(s))[0]
        render(s, f"{o}/png3/{b}_on_dark.png", 1024, "#0d1117")
        render(s, f"{o}/png3/{b}_on_light.png", 1024, "#ffffff")
    for b, g in (("textblock_v3_mark", "#0d1117"),
                 ("textblock_v3_mark", "#ffffff"),
                 ("textblock_v3_mark_dark", "#0d1117"),
                 ("textblock_v3_mark_light", "#ffffff"),
                 ("textblock_v3_mark", "#0d1117")):
        tagg = "dark" if g == "#0d1117" else "light"
        render(f"{o}/{b}.svg", f"{o}/png3/_{b}_32_on_{tagg}.png", 32, g)

    from PIL import Image
    tiles = sorted(glob.glob(f"{o}/png3/_*_32_on_*.png"))
    ims = [Image.open(t).convert("RGB").resize((300, 300), Image.NEAREST)
           for t in tiles]
    sheet = Image.new("RGB", (310 * len(ims) + 10, 320), "#555555")
    for i, im in enumerate(ims):
        sheet.paste(im, (10 + 310 * i, 10))
    sheet.save(f"{o}/png3/_32px_check.png")

    # ---- continuity check: mark | ascii | overlay
    render(f"{o}/_v3_mark_square.svg", f"{o}/png3/_sq_mark.png", 700, None)
    render(f"{o}/_v3_ascii_square.svg", f"{o}/png3/_sq_ascii.png", 700, None)
    a = Image.open(f"{o}/png3/_sq_mark.png").convert("RGBA")
    b = Image.open(f"{o}/png3/_sq_ascii.png").convert("RGBA")
    ov = Image.new("RGBA", a.size, (13, 17, 23, 255))
    fade = a.copy()
    fade.putalpha(a.split()[3].point(lambda v: int(v * 0.45)))
    ov.alpha_composite(fade)
    ov.alpha_composite(b)
    # the shrink test: the SAME ascii brain rendered tiny, beside the tiny mark
    render(f"{o}/_v3_mark_square.svg", f"{o}/png3/_tiny_mark.png", 40, "#0d1117")
    render(f"{o}/_v3_ascii_square.svg", f"{o}/png3/_tiny_ascii.png", 40, "#0d1117")
    t1 = Image.open(f"{o}/png3/_tiny_mark.png").convert("RGB").resize(
        (700, 700), Image.NEAREST)
    t2 = Image.open(f"{o}/png3/_tiny_ascii.png").convert("RGB").resize(
        (700, 700), Image.NEAREST)
    panels = (a, b, ov, t1, t2)
    labels = ("mark 512", "ascii 512", "overlay", "mark @40px",
              "ascii @40px")
    sheet = Image.new("RGB", (700 * 5 + 60, 700 + 60), "#0d1117")
    for i, im in enumerate(panels):
        if im.mode == "RGBA":
            bgt = Image.new("RGBA", a.size, (13, 17, 23, 255))
            bgt.alpha_composite(im)
            im = bgt.convert("RGB")
        sheet.paste(im, (10 + 710 * i, 10))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, lb in enumerate(labels):
        d.text((20 + 710 * i, 715), lb, fill="#9aa4b8")
    sheet.save(f"{o}/v3_continuity_check.png")
    print("done")


if __name__ == "__main__":
    main()
