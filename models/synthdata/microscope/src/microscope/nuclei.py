"""Realistic DAPI nucleus ground-truth generator (the shape + chromatin layer).

microsim has no nucleus generator -- nucleus SHAPE and internal chromatin
TEXTURE live entirely in the ground-truth fluorophore-count volume. This module
builds that volume; microsim's physics (PSF, spectral superposition, detector
noise) then acts on it.

Per nucleus we model:
  - cell-type variety: dense small glia, large pale neurons, tiny bright cells
  - irregular lobed boundary (low-frequency surface perturbation, not a smooth
    ellipsoid) + random in-plane orientation + anisotropy
  - chromatin texture: granular nucleoplasm, bright heterochromatin chromocenters,
    a brighter peripheral rim, and DAPI-dark nucleolus hole(s) in neurons
  - depth attenuation through the slab (deeper = dimmer)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# cell-type table: weight, radius_um(range), nucleoplasm base, rim boost,
# n chromocenters(range), chromocenter brightness, n nucleoli, axis-ratio range
# base = solid nucleoplasm fill; rim = soft peripheral boost; chromo_amp is a
# fraction OF base added by each (broad, gentle) heterochromatin focus.
#
# chromo (speck) RANGES grounded by the 10x marker-labeled DAPI extraction
# (build_dapi_texture_bank.py, out/dapi_ref/dapi_texture_stats.json): real
# NeuN+ neurons showed MORE discrete chromocenters than glia (median 3 vs 2) --
# pale neuroplasm with a few bright chips, vs glia's uniform brightness. So
# neuron -> (2,4) [median 3]; glial specs -> (1,3) [median 2]. brightness (base)
# stays from EM/biology (glia denser->brighter); 10x widefield blur could not
# measure brightness contrast (came out flat 1.04x) so it's NOT sourced there.
CELL_TYPES = {
    "glia_dense": dict(weight=0.50, radius=(2.2, 3.6), base=0.80, rim=0.35,
                       chromo=(1, 3), chromo_amp=0.55, nucleoli=0, axr=(0.7, 1.0)),
    "neuron": dict(weight=0.22, radius=(4.2, 7.0), base=0.52, rim=0.45,
                   chromo=(2, 4), chromo_amp=0.45, nucleoli=1, axr=(0.72, 1.0)),
    "small_bright": dict(weight=0.20, radius=(1.9, 2.8), base=0.95, rim=0.2,
                         chromo=(1, 3), chromo_amp=0.45, nucleoli=0, axr=(0.8, 1.0)),
    "astro_pale": dict(weight=0.08, radius=(3.2, 4.6), base=0.55, rim=0.3,
                       chromo=(1, 3), chromo_amp=0.4, nucleoli=0, axr=(0.65, 1.0)),
}
BASE_COUNT = 140.0


def _pick_type(rng):
    keys = list(CELL_TYPES)
    w = np.array([CELL_TYPES[k]["weight"] for k in keys])
    return keys[rng.choice(len(keys), p=w / w.sum())]


def _stamp(truth, scale, center_um, ctype, depth_factor, rng):
    zsc, ysc, xsc = scale
    nz, ny, nx = truth.shape
    spec = CELL_TYPES[ctype]
    r = rng.uniform(*spec["radius"])
    rx = r * rng.uniform(*spec["axr"])
    ry = r * rng.uniform(*spec["axr"])
    rz = r * rng.uniform(0.7, 1.0)
    theta = rng.uniform(0, np.pi)              # in-plane orientation
    rmax = max(rx, ry, rz) * 1.45              # pad for boundary lobes
    cz, cy, cx = center_um
    z0, z1 = max(0, int((cz - rmax) / zsc)), min(nz, int((cz + rmax) / zsc) + 1)
    y0, y1 = max(0, int((cy - rmax) / ysc)), min(ny, int((cy + rmax) / ysc) + 1)
    x0, x1 = max(0, int((cx - rmax) / xsc)), min(nx, int((cx + rmax) / xsc) + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float32)
    dz = zz * zsc - cz
    dy = yy * ysc - cy
    dx = xx * xsc - cx
    ct, st = np.cos(theta), np.sin(theta)
    xr = (ct * dx + st * dy) / rx
    yr = (-st * dx + ct * dy) / ry
    zr = dz / rz
    d = np.sqrt(xr * xr + yr * yr + zr * zr)
    # lobed boundary: perturb the unit surface with a low-frequency smooth field
    bnoise = gaussian_filter(rng.random(d.shape).astype(np.float32), sigma=3.5)
    bnoise = (bnoise - bnoise.mean()) / (bnoise.std() + 1e-6)
    surface = 1.0 + 0.09 * bnoise
    # SOFT envelope (smoothstep) -> smooth nucleus edge, not a jagged hard cutoff
    env = np.clip((surface - d) / 0.16, 0.0, 1.0)
    env = env * env * (3.0 - 2.0 * env)
    if not (env > 1e-3).any():
        return
    # solidly-filled nucleoplasm with gentle low-frequency mottle
    fine = gaussian_filter(rng.random(d.shape).astype(np.float32), sigma=2.2)
    fine = (fine - fine.min()) / max(fine.ptp(), 1e-6)
    val = spec["base"] * (0.82 + 0.30 * fine)
    # soft peripheral heterochromatin rim (d near surface)
    rim = np.clip((d - 0.6) / 0.4, 0, 1)
    val *= 1.0 + spec["rim"] * (rim ** 2.0)
    # a few BROAD, gentle heterochromatin foci (fraction of base, not sharp dots)
    n_ch = int(rng.integers(spec["chromo"][0], spec["chromo"][1] + 1))
    for _ in range(n_ch):
        p = rng.normal(size=3)
        p = p / (np.linalg.norm(p) + 1e-6) * rng.uniform(0, 0.7)
        cr = rng.uniform(0.25, 0.45)
        g = np.exp(-((xr - p[0]) ** 2 + (yr - p[1]) ** 2 + (zr - p[2]) ** 2)
                   / (2 * cr ** 2))
        val += spec["chromo_amp"] * spec["base"] * g
    # DAPI-dark nucleolus hole(s) for neurons (soft-edged)
    for _ in range(spec["nucleoli"]):
        p = rng.normal(size=3)
        p = p / (np.linalg.norm(p) + 1e-6) * rng.uniform(0, 0.35)
        nr = rng.uniform(0.18, 0.34)
        nd = np.sqrt((xr - p[0]) ** 2 + (yr - p[1]) ** 2 + (zr - p[2]) ** 2)
        hole = np.clip((nd - nr) / 0.1, 0.0, 1.0)        # 0 inside -> 1 outside
        val *= 0.10 + 0.90 * hole
    val *= env * BASE_COUNT * depth_factor * rng.uniform(0.7, 1.3)
    truth[z0:z1, y0:y1, x0:x1] += val


def build_truth(rng, shape, scale, density_per_mm3, atten_um=18.0, bg_count=1.2):
    zsc, ysc, xsc = scale
    nz, ny, nx = shape
    ext = (nz * zsc, ny * ysc, nx * xsc)
    vol_mm3 = ext[0] * ext[1] * ext[2] / 1e9
    n = max(1, int(density_per_mm3 * vol_mm3))
    truth = np.zeros(shape, dtype=np.float32)
    counts: dict[str, int] = {}
    for _ in range(n):
        ctype = _pick_type(rng)
        counts[ctype] = counts.get(ctype, 0) + 1
        cz = rng.uniform(-2, ext[0] + 2)
        center = (cz, rng.uniform(-2, ext[1] + 2), rng.uniform(-2, ext[2] + 2))
        depth = float(np.exp(-max(cz, 0.0) / atten_um))
        _stamp(truth, scale, center, ctype, depth, rng)
    if bg_count > 0:
        lf = gaussian_filter(rng.random(shape).astype(np.float32), sigma=(3, 14, 14))
        lf = (lf - lf.min()) / max(lf.ptp(), 1e-6)
        truth += bg_count * (0.5 + lf)
    return truth, n, counts
