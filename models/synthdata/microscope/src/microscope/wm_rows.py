"""Research-grounded white-matter placement (Poisson segment-cluster model).

Supersedes ``white_matter.stamp_wm`` (previous-session crystalline lattice) AND the
earlier center-seeded version of this module (which needed a re-entry hack and still
left a faint central seed-line). Grounded in the 2026-05-29 anisotropic-point-process
+ rat-fimbria research (Suzuki & Raisman 1992; Poisson line/segment-cluster processes,
research cluster 3; see project_wm_nuclei_spatial_model).

Model = a 3-D Poisson process of finite oligodendrocyte CHAINS:

  * each chain is SEEDED at a uniform-random 3-D point in the slab (frame + margin
    laterally, [0, slab] + z-margin in depth) -- no central seed line, and chains
    that straddle the section's z-faces appear as truncated fragments (the
    biological "sliced by the 50um cut").
  * each chain has a FINITE length L ~ Exponential(mean) clipped to [min, max], so
    the field is a MIX of short stubs and long rows (some traceable, most not).
  * orientation = base azimuth + per-chain in-plane dispersion (Watson-style) + an
    out-of-plane DIP ~ Normal(0, dip_sd_deg). Marching the chain, z ramps by
    gap*tan(dip); when z leaves [0, slab] the chain ENDS (sliced at the face) --
    this + finite L is what makes real axons hard to trace (truncation, defocus,
    crossings), with NO re-entry trickery.
  * along each chain: SEGMENTED rows -- runs of N_seg ~ Poisson(8)[3,12] oligos,
    each run ended by a longer gap holding a solitary interfascicular ASTROCYTE;
    within-row gaps from a shifted Gamma renewal (gap = r_min + Gamma); gentle
    in-plane waviness via an OU heading walk.
  * microglia: sparse INDEPENDENT scatter (added in stamp_wm_rows, not row-bound).

Between-row spacing is EMERGENT here (Poisson over seed density), not a fixed
lattice -- more realistic; add a Matern hard-core on seeds if striations wash out.
Produces nucleus CENTROIDS (y_um, x_um, z_um); COSEM shapes + microsim optics
turn them into an image unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import white_matter as wm
from .gpu_truth import _gaussian_filter


def wm_sites(
    fov_um: tuple[float, float],
    scale_yx_um: tuple[float, float],
    slab_um: float,
    rng: np.random.Generator,
    *,
    base_deg: float,
    n_oligo_target: int,
    mu_within_um: float = 8.0,       # within-row (oligo) spacing -- 60um/8 cells
    r_min_um: float = 5.0,           # hard floor on within-row separation
    k_reg: float = 5.0,              # Gamma renewal shape (higher = more regular)
    seg_mean: float = 8.0,           # oligos per segment (Poisson mean)
    seg_lo: int = 3,
    seg_hi: int = 12,
    astro_gap_mult: float = 2.0,     # inter-segment gap (holds the astrocyte)
    sigma_perp_um: float = 1.5,      # transverse jitter (finite row width)
    wobble_sd_deg: float = 5.0,      # per-step in-plane heading noise (waviness)
    dip_sd_deg: float = 0.0,         # per-chain OUT-OF-PLANE tilt SD (0 = flat/traceable)
    inplane_disp_deg: float = 10.0,  # per-chain in-plane heading dispersion (fanning)
    chain_len_mean_um: float = 180.0,  # mean chain arc length (Exponential)
    chain_len_min_um: float = 25.0,
    chain_len_max_um: float = 800.0,
    revert: float = 0.08,            # OU mean-reversion toward base azimuth
    margin_um: float = 30.0,         # lateral seed/region margin
    z_margin_um: float = 15.0,       # depth seed margin (chains straddling the faces)
    max_chains: int = 2_000_000,     # runaway guard
) -> tuple[np.ndarray, np.ndarray]:
    """Seed finite oligodendrocyte chains uniformly in the slab and trace them.

    Returns ``(oligo_sites, astro_sites)``, each ``(M, 3)`` of ``(y_um, x_um,
    z_um)`` (absolute slab depth). Beads outside the frame or the slab are not
    emitted (truncation at the section faces / frame edges).
    """
    fy, fx = fov_um
    if fy <= 0 or fx <= 0 or n_oligo_target <= 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

    base = math.radians(base_deg)
    sd = math.radians(wobble_sd_deg)
    lo_y, hi_y, lo_x, hi_x = -margin_um, fy + margin_um, -margin_um, fx + margin_um
    max_steps = int(chain_len_max_um / max(r_min_um, 1e-3)) + 2

    oligo: list[tuple[float, float, float]] = []
    astro: list[tuple[float, float, float]] = []

    def _seg() -> int:
        return int(np.clip(rng.poisson(seg_mean), seg_lo, seg_hi))

    def _tdip() -> float:
        if dip_sd_deg <= 0:
            return 0.0
        return math.tan(math.radians(float(np.clip(rng.normal(0.0, dip_sd_deg), -70.0, 70.0))))

    def _march(y: float, x: float, z: float, h: float, tdip: float,
               direction: float, half_len: float) -> None:
        seg_left = _seg()
        arc = 0.0
        for _ in range(max_steps):
            if arc >= half_len:
                return
            place_astro = seg_left <= 0
            gap = r_min_um + rng.gamma(k_reg, max(mu_within_um - r_min_um, 0.1) / k_reg)
            if place_astro:
                gap *= astro_gap_mult
            h += revert * (base - h) + sd * float(rng.standard_normal())
            y += direction * gap * math.sin(h)
            x += direction * gap * math.cos(h)
            z += direction * gap * tdip
            arc += gap
            if not (lo_y <= y <= hi_y and lo_x <= x <= hi_x):
                return
            if not (0.0 <= z <= slab_um):
                return                       # sliced at the section face -> chain ends
            j = sigma_perp_um * float(rng.standard_normal())
            yy = y + j * math.sin(h + math.pi / 2.0)
            xx = x + j * math.cos(h + math.pi / 2.0)
            if place_astro:
                astro.append((yy, xx, z))
                seg_left = _seg()
            else:
                oligo.append((yy, xx, z))
                seg_left -= 1

    n_chains = 0
    while len(oligo) < n_oligo_target and n_chains < max_chains:
        n_chains += 1
        y0 = float(rng.uniform(lo_y, hi_y))
        x0 = float(rng.uniform(lo_x, hi_x))
        z0 = float(rng.uniform(-z_margin_um, slab_um + z_margin_um))
        h0 = base + math.radians(float(rng.normal(0.0, inplane_disp_deg)))
        tdip = _tdip()
        length = float(np.clip(rng.exponential(chain_len_mean_um), chain_len_min_um, chain_len_max_um))
        if (lo_y <= y0 <= hi_y and lo_x <= x0 <= hi_x and 0.0 <= z0 <= slab_um):
            oligo.append((y0, x0, z0))       # seed bead (if it lands in the section)
        _march(y0, x0, z0, h0, tdip, 1.0, length / 2.0)
        _march(y0, x0, z0, h0, tdip, -1.0, length / 2.0)

    o = np.asarray(oligo, np.float32) if oligo else np.empty((0, 3), np.float32)
    a = np.asarray(astro, np.float32) if astro else np.empty((0, 3), np.float32)
    return o, a


def _put_sites(truth, sites, bank, scale, atten_um, rng) -> int:
    """Stamp a template from ``bank`` at each in-bounds (y_um, x_um, z_um) site."""
    nz, ny, nx = truth.shape
    zsc, ysc, xsc = scale
    fy, fx = ny * ysc, nx * xsc
    z_jit = max(1, int(round(1.0 / zsc)))
    placed = 0
    for y_um, x_um, z_um in sites:
        if not (0.0 <= y_um < fy and 0.0 <= x_um < fx):
            continue
        cy = int(round(y_um / ysc))
        cx = int(round(x_um / xsc))
        cz = int(np.clip(int(round(z_um / zsc)) + rng.integers(-z_jit, z_jit + 1), 0, nz - 1))
        placed += wm._place_one(truth, bank, cy, cx, cz, zsc, atten_um, rng)
    return placed


def stamp_wm_rows(
    xp: Any,
    shape: tuple[int, int, int],
    scale: tuple[float, float, float],
    density_per_mm3: float,
    oligo_bank: list[np.ndarray],
    astro_bank: list[np.ndarray],
    micro_bank: list[np.ndarray],
    rng: np.random.Generator,
    *,
    base_deg: float | None = None,
    oligo_frac: float = 0.70,
    micro_frac: float = 0.15,
    atten_um: float = 18.0,
    bg_count: float = 0.10,
    **site_kw: Any,
) -> tuple[Any, int, dict]:
    """WM truth volume: uniform-seeded finite oligo chains + astro spacers + micro
    scatter. Returns ``(truth, n_placed, counts)``."""
    nz, ny, nx = shape
    zsc, ysc, xsc = scale
    if base_deg is None:
        base_deg = float(rng.uniform(0.0, 180.0))

    slab_um = nz * zsc
    vol_mm3 = slab_um * (ny * ysc) * (nx * xsc) / 1e9
    n_total = max(1, int(density_per_mm3 * vol_mm3))
    n_oligo = max(1, int(oligo_frac * n_total))
    n_micro = int(micro_frac * n_total)

    oligo_sites, astro_sites = wm_sites(
        (ny * ysc, nx * xsc), (ysc, xsc), slab_um, rng,
        base_deg=base_deg, n_oligo_target=n_oligo, **site_kw,
    )

    truth = xp.zeros(shape, dtype=xp.float32)
    oligo_t = [xp.asarray(t) for t in oligo_bank]
    astro_t = [xp.asarray(t) for t in astro_bank]
    micro_t = [xp.asarray(t) for t in micro_bank]

    placed = 0
    placed += _put_sites(truth, oligo_sites, oligo_t, scale, atten_um, rng)
    placed += _put_sites(truth, astro_sites, astro_t, scale, atten_um, rng)

    for _ in range(n_micro):                 # microglia: sparse, independent
        cy = int(rng.integers(0, ny))
        cx = int(rng.integers(0, nx))
        cz = int(rng.integers(0, nz))
        placed += wm._place_one(truth, micro_t, cy, cx, cz, zsc, atten_um, rng)

    if bg_count > 0:
        lf = _gaussian_filter(xp, xp.asarray(rng.random(shape).astype(np.float32)), (3, 14, 14))
        lo = lf.min()
        lf = (lf - lo) / xp.maximum(lf.max() - lo, xp.float32(1e-6))
        truth += bg_count * (0.5 + lf)

    counts = {"oligo": int(len(oligo_sites)), "astro": int(len(astro_sites)), "micro": int(n_micro)}
    return truth, placed, counts
