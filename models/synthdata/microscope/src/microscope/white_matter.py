"""White-matter (fiber-tract) DAPI architecture for the microsim truth builder.

The whole model is PLACEMENT, stated in biological terms. A white-matter tract in
DAPI is three glial populations:

  * OLIGODENDROCYTES -- small, ROUND, DAPI-bright. They line up ALONG the
    (invisible, unstained) axons, so they read as chains of small round nuclei
    strung head-to-tail down each fiber. These chains ARE the directional "fiber"
    structure. (The nuclei are NOT stretched -- the line comes from arrangement.)
  * ASTROCYTES -- larger, paler/darker, randomly scattered BETWEEN the axons.
  * MICROGLIA -- small, irregular, randomly scattered between the axons.

So a tract = (1) oligodendrocytes laid in dense chains along the fiber direction,
plus (2) a random scatter of astrocytes + microglia in the spaces between. We
place these nucleus templates into microsim's 3-D ground-truth fluorophore volume
(``truth_space``); its PSF/detector then image it.

The fiber direction comes from :func:`fiber_orientation_field` (a constant azimuth
here; later it will be driven by the tract's regional shape for macro-scale fiber
structure). Backend-agnostic (numpy or cupy). Leaves ``gpu_truth.stamp_nuclei``
untouched so the existing render scripts keep working.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import nuclei_sizes as nsz  # strict nucleus-size table (size_priors.csv)
from .gpu_truth import _gaussian_filter  # xp-aware gaussian (numpy or cupy)

# ---------------------------------------------------------------------------
# Fiber direction
# ---------------------------------------------------------------------------

def fiber_orientation_field(
    ny: int,
    nx: int,
    rng: np.random.Generator,
    *,
    base_deg: float | None = None,
    wobble_deg: float = 0.0,
    corr_px: float | None = None,
) -> np.ndarray:
    """Per-pixel fiber tangent angle (radians from +x).

    A single dominant azimuth (``base_deg``, random if None). ``wobble_deg`` adds
    smooth low-frequency curvature and is OFF by default -- it exists only to
    follow a real tract that bends (driven later by the tract's regional shape),
    not to manufacture the WM look (which is pure placement).
    """
    if base_deg is None:
        base_deg = float(rng.uniform(0.0, 180.0))
    base = math.radians(base_deg)
    if wobble_deg <= 0.0:
        return np.full((ny, nx), base, dtype=np.float32)
    if corr_px is None:
        corr_px = max(8.0, 0.25 * min(ny, nx))
    pert = rng.standard_normal((ny, nx)).astype(np.float32)
    pert = _gaussian_filter(np, pert, corr_px)
    sd = float(pert.std())
    if sd > 1e-6:
        pert = pert / sd
    pert = np.clip(pert, -1.5, 1.5)
    return (base + math.radians(wobble_deg) * pert).astype(np.float32)


# ---------------------------------------------------------------------------
# Cell banks (real COSEM shape geometry + synthesized chromatin)
# ---------------------------------------------------------------------------

def _shape_radius_um(occ: np.ndarray, vox: float) -> float:
    nvox = float((occ > 0.4).sum())
    return 0.5 * (6.0 * nvox * vox / np.pi) ** (1.0 / 3.0) if nvox > 0 else 1.0


def oligo_template_bank(
    shapes: list[np.ndarray],
    rng: np.random.Generator,
    scale: tuple[float, float, float],
    *,
    n_templates: int = 300,
    radius_um: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Bank of ROUND oligodendrocyte nuclei (the cells that form the chains).

    Real COSEM shape geometry rescaled to the STRICT oligodendrocyte nucleus size
    from ``nuclei_sizes`` (6.5um diam, 5-8um range) and textured as dense, bright
    glia. Round -- the directional line comes from ARRANGEMENT (``stamp_wm`` lines
    them up along the axon), not from cell shape.
    """
    from .gpu_truth import SHAPE_SOURCE_SCALE, _random_orient, _resample_occ, _texture_for_shape

    if radius_um is None:
        radius_um = nsz.radius_range_um("oligodendrocyte")
    vox = SHAPE_SOURCE_SCALE[0] * SHAPE_SOURCE_SCALE[1] * SHAPE_SOURCE_SCALE[2]
    bank: list[np.ndarray] = []
    for _ in range(n_templates):
        occ = _random_orient(shapes[rng.integers(len(shapes))], rng)
        tgt = float(rng.uniform(*radius_um))
        occ = _resample_occ(occ, scale, size_scale=tgt / max(_shape_radius_um(occ, vox), 1e-3))
        bank.append(_texture_for_shape(occ, rng, scale, cell_type="glia"))
    return bank


def wm_scatter_bank(
    shapes: list[np.ndarray],
    rng: np.random.Generator,
    scale: tuple[float, float, float],
    *,
    n_templates: int = 200,
    astro_frac: float = 0.6,
    astro_radius_um: tuple[float, float] | None = None,
    micro_radius_um: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Bank of the interstitial cells scattered BETWEEN the axons.

    A mix of ASTROCYTES (7.0um diam, dimmer -- size routes the texture to the pale
    ``astro_pale`` spec) and MICROGLIA (6.5um diam, irregular -- random-oriented
    COSEM geometry gives the irregular look). Sizes from the strict
    ``nuclei_sizes`` table. Placed at random by :func:`stamp_wm`, not in chains.
    """
    from .gpu_truth import SHAPE_SOURCE_SCALE, _random_orient, _resample_occ, _texture_for_shape

    if astro_radius_um is None:
        astro_radius_um = nsz.radius_range_um("astrocyte")
    if micro_radius_um is None:
        micro_radius_um = nsz.radius_range_um("microglia")
    vox = SHAPE_SOURCE_SCALE[0] * SHAPE_SOURCE_SCALE[1] * SHAPE_SOURCE_SCALE[2]
    bank: list[np.ndarray] = []
    for _ in range(n_templates):
        rad_um = (float(rng.uniform(*astro_radius_um)) if rng.random() < astro_frac
                  else float(rng.uniform(*micro_radius_um)))
        occ = _random_orient(shapes[rng.integers(len(shapes))], rng)
        occ = _resample_occ(occ, scale, size_scale=rad_um / max(_shape_radius_um(occ, vox), 1e-3))
        bank.append(_texture_for_shape(occ, rng, scale, cell_type="glia"))
    return bank


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _orientation_mean(theta: np.ndarray) -> float:
    """Circular mean of an orientation field (period pi)."""
    return 0.5 * math.atan2(float(np.sin(2 * theta).mean()), float(np.cos(2 * theta).mean()))


def _theta_at(theta: np.ndarray, y_um: float, x_um: float, sy: float, sx: float) -> float:
    """Nearest-pixel sample of the orientation field at a um position."""
    ny, nx = theta.shape
    yi = min(max(int(round(y_um / sy)), 0), ny - 1)
    xi = min(max(int(round(x_um / sx)), 0), nx - 1)
    return float(theta[yi, xi])


def row_sites_um(
    fov_um: tuple[float, float],
    scale_yx_um: tuple[float, float],
    rng: np.random.Generator,
    *,
    theta: np.ndarray,
    row_pitch_um: float = 7.0,
    bead_pitch_um: float = 4.5,
    row_jitter_um: float = 1.5,
    bead_jitter_um: float = 1.5,
    margin_um: float = 30.0,
) -> np.ndarray:
    """Trace axon lines along the fibers; return oligodendrocyte (y, x) um sites.

    Seeds spaced ``row_pitch_um`` (axon spacing) along the line perpendicular to
    the mean fiber direction; each seed is marched both ways through ``theta``,
    dropping a site every ``bead_pitch_um`` (oligo spacing along the axon, ~one
    nucleus -> touching). Each chain gets a coherent depth. Returns ``(M, 3)``:
    ``(y_um, x_um, z_frac)`` where ``z_frac`` in [0,1] is that chain's slab depth.
    """
    fov_y, fov_x = fov_um
    sy, sx = scale_yx_um
    if fov_y <= 0 or fov_x <= 0:
        return np.empty((0, 2), np.float32)

    th0 = _orientation_mean(theta)
    perp = th0 + math.pi / 2.0
    cy, cx = fov_y / 2.0, fov_x / 2.0
    diag = math.hypot(fov_y, fov_x)
    n_rows = int(diag / max(row_pitch_um, 1e-3)) + 2
    n_steps = int(diag / max(bead_pitch_um, 1e-3)) + 2

    sites: list[tuple[float, float, float]] = []
    lo_y, hi_y = -margin_um, fov_y + margin_um
    lo_x, hi_x = -margin_um, fov_x + margin_um

    def _emit(y: float, x: float, zf: float) -> None:
        if lo_y <= y <= hi_y and lo_x <= x <= hi_x:
            sites.append(
                (y + float(rng.uniform(-bead_jitter_um, bead_jitter_um)),
                 x + float(rng.uniform(-bead_jitter_um, bead_jitter_um)),
                 zf)
            )

    for k in range(-n_rows, n_rows + 1):
        off = k * row_pitch_um + float(rng.uniform(-row_jitter_um, row_jitter_um))
        zf = float(rng.uniform(0.0, 1.0))   # this chain's COHERENT depth in the slab
        y0 = cy + off * math.sin(perp)
        x0 = cx + off * math.cos(perp)
        _emit(y0, x0, zf)
        for direction in (1.0, -1.0):
            y, x = y0, x0
            for _ in range(n_steps):
                th = _theta_at(theta, y, x, sy, sx)
                y += direction * bead_pitch_um * math.sin(th)
                x += direction * bead_pitch_um * math.cos(th)
                if not (lo_y <= y <= hi_y and lo_x <= x <= hi_x):
                    break
                _emit(y, x, zf)
    if not sites:
        return np.empty((0, 3), np.float32)
    return np.asarray(sites, dtype=np.float32)


def _place_one(truth: Any, templates: list, cy: int, cx: int, cz: int,
               zsc: float, atten_um: float, rng: np.random.Generator) -> int:
    """Add one random template from ``templates`` centered at voxel (cz, cy, cx)."""
    nz, ny, nx = truth.shape
    t = templates[rng.integers(len(templates))]
    tz, ty, tx = t.shape
    z0, y0, x0 = cz - tz // 2, cy - ty // 2, cx - tx // 2
    zz0, zz1 = max(z0, 0), min(z0 + tz, nz)
    yy0, yy1 = max(y0, 0), min(y0 + ty, ny)
    xx0, xx1 = max(x0, 0), min(x0 + tx, nx)
    if zz0 >= zz1 or yy0 >= yy1 or xx0 >= xx1:
        return 0
    tzs, tys, txs = zz0 - z0, yy0 - y0, xx0 - x0
    depth = float(np.exp(-max(zz0 * zsc, 0.0) / atten_um))
    bright = float(rng.uniform(0.7, 1.3))
    truth[zz0:zz1, yy0:yy1, xx0:xx1] += (
        t[tzs:tzs + (zz1 - zz0), tys:tys + (yy1 - yy0), txs:txs + (xx1 - xx0)] * (depth * bright)
    )
    return 1


def stamp_wm(
    xp: Any,
    shape: tuple[int, int, int],
    scale: tuple[float, float, float],
    density_per_mm3: float,
    oligo_bank: list[np.ndarray],
    scatter_bank: list[np.ndarray],
    rng: np.random.Generator,
    *,
    row_pitch_um: float = 7.0,
    bead_pitch_um: float = 4.5,
    row_jitter_um: float = 1.5,
    bead_jitter_um: float = 1.5,
    scatter_frac: float = 0.28,
    base_deg: float | None = None,
    wobble_deg: float = 0.0,
    theta: np.ndarray | None = None,
    atten_um: float = 18.0,
    bg_count: float = 0.10,
) -> tuple[Any, int, np.ndarray]:
    """Synthesize a white-matter truth volume: oligo chains + astro/microglia scatter.

    ``density_per_mm3`` x slab volume sets the total cell count; ``scatter_frac`` of
    those are interstitial astrocytes/microglia placed at random, the rest are
    oligodendrocytes laid in chains along the fibers (one per axon-site, repeated
    across random z to fill the slab). Returns ``(truth, n_placed, theta)``.
    """
    nz, ny, nx = shape
    zsc, ysc, xsc = scale

    if theta is None:
        theta = fiber_orientation_field(ny, nx, rng, base_deg=base_deg, wobble_deg=wobble_deg)

    vol_mm3 = (nz * zsc) * (ny * ysc) * (nx * xsc) / 1e9
    n_total = max(1, int(density_per_mm3 * vol_mm3))
    n_scatter = int(scatter_frac * n_total)
    n_oligo = max(0, n_total - n_scatter)

    truth = xp.zeros(shape, dtype=xp.float32)
    oligo_t = [xp.asarray(t) for t in oligo_bank]
    scatter_t = [xp.asarray(t) for t in scatter_bank]
    placed = 0

    # (1) oligodendrocyte chains along the axons
    sites = row_sites_um(
        (ny * ysc, nx * xsc), (ysc, xsc), rng, theta=theta,
        row_pitch_um=row_pitch_um, bead_pitch_um=bead_pitch_um,
        row_jitter_um=row_jitter_um, bead_jitter_um=bead_jitter_um,
    )
    if len(sites):
        fy, fx = ny * ysc, nx * xsc
        inb = (sites[:, 0] >= 0) & (sites[:, 0] < fy) & (sites[:, 1] >= 0) & (sites[:, 1] < fx)
        sites = sites[inb]
    if len(sites) and n_oligo > 0:
        z_layers = max(1, int(round(n_oligo / len(sites))))
        z_jit = max(1, int(round(1.5 / zsc)))  # ~1.5um chain thickness in z
        for y_um, x_um, zf in sites:
            cy = int(round(y_um / ysc))
            cx = int(round(x_um / xsc))
            base_z = int(zf * (nz - 1))
            for _ in range(z_layers):
                cz = int(np.clip(base_z + rng.integers(-z_jit, z_jit + 1), 0, nz - 1))
                placed += _place_one(truth, oligo_t, cy, cx, cz, zsc, atten_um, rng)

    # (2) astrocytes + microglia scattered at random between the axons
    for _ in range(n_scatter):
        cy = int(rng.integers(0, ny))
        cx = int(rng.integers(0, nx))
        cz = int(rng.integers(0, nz))
        placed += _place_one(truth, scatter_t, cy, cx, cz, zsc, atten_um, rng)

    # (3) faint dark floor (DAPI background is dark, not pure black)
    if bg_count > 0:
        lf = _gaussian_filter(xp, xp.asarray(rng.random(shape).astype(np.float32)), (3, 14, 14))
        lo = lf.min()
        lf = (lf - lo) / xp.maximum(lf.max() - lo, xp.float32(1e-6))
        truth += bg_count * (0.5 + lf)

    return truth, placed, theta
