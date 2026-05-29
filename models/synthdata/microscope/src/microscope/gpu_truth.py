"""Backend-agnostic nucleus-stamping truth generator (numpy or cupy).

nuclei.py builds the truth by computing, *per nucleus*, several gaussian-filtered
noise fields + a distance field + chromatin foci. That per-nucleus math is the
CPU bottleneck at region scale. Here we do that expensive math ONCE per template
to build a small bank of nucleus volumes (with shape/lobing/chromatin/nucleoli
baked in), then synthesize the truth by *placing* templates as cheap slice-adds
on the chosen backend -- so the heavy volume is born on the GPU.

Per-instance variety preserved at placement time: depth attenuation + brightness
jitter. (Finite template bank => some shape repetition; raise n_templates to
trade memory for variety. The next step for whole-slice scale is a RawKernel
atomicAdd scatter to replace the Python placement loop.)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .lab_optics import LAB_DZ_UM, LAB_PX_UM
from .nuclei import BASE_COUNT, CELL_TYPES

# Grid the COSEM occupancy masks were saved at (cosem_shapes.py resamples to the
# 20x lab grid). Templates are resampled FROM this grid TO the render scale so a
# nucleus keeps its physical size at any magnification's pixel size (a 10x frame
# has 2x the um/px, so the same nucleus must span half the pixels).
SHAPE_SOURCE_SCALE = (LAB_DZ_UM, LAB_PX_UM, LAB_PX_UM)

# neuron:glia diameter ratio, grounded by the 10x marker-labeled DAPI extraction
# (build_dapi_texture_bank.py: neuron 9.0um vs glia 7.8um = 1.145x; agrees with
# our COSEM EM neuron>glia, contradicts the 2-D size prior). Applied as a per-
# type scale split around the geometric mean so it sets the neuron/glia size
# DIFFERENCE without moving the (still-unresolved) absolute-size dial.
NEURON_GLIA_SIZE_RATIO = 1.145


def _gaussian_filter(xp: Any, arr: Any, sigma: Any) -> Any:
    if getattr(xp, "__name__", "") == "cupy":
        from cupyx.scipy.ndimage import gaussian_filter
    else:
        from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma)


def _make_template(rng: np.random.Generator, scale, ctype: str) -> np.ndarray:
    """Render one nucleus into a tight local volume (counts, peak ~ base*BASE).

    Mirrors nuclei._stamp's texture model, minus the per-instance depth and
    brightness jitter (those are applied at placement). Built on CPU (numpy);
    templates are small and few, so this stays cheap and is uploaded once.
    """
    zsc, ysc, xsc = scale
    spec = CELL_TYPES[ctype]
    r = rng.uniform(*spec["radius"])
    rx = r * rng.uniform(*spec["axr"])
    ry = r * rng.uniform(*spec["axr"])
    rz = r * rng.uniform(0.7, 1.0)
    theta = rng.uniform(0, np.pi)
    rmax = max(rx, ry, rz) * 1.45

    nzl = max(1, int(2 * rmax / zsc) + 1)
    nyl = max(1, int(2 * rmax / ysc) + 1)
    nxl = max(1, int(2 * rmax / xsc) + 1)
    zz, yy, xx = np.mgrid[0:nzl, 0:nyl, 0:nxl].astype(np.float32)
    dz = (zz - (nzl - 1) / 2) * zsc
    dy = (yy - (nyl - 1) / 2) * ysc
    dx = (xx - (nxl - 1) / 2) * xsc
    ct, st = np.cos(theta), np.sin(theta)
    xr = (ct * dx + st * dy) / rx
    yr = (-st * dx + ct * dy) / ry
    zr = dz / rz
    d = np.sqrt(xr * xr + yr * yr + zr * zr)

    bnoise = _gaussian_filter(np, rng.random(d.shape).astype(np.float32), 3.5)
    bnoise = (bnoise - bnoise.mean()) / (bnoise.std() + 1e-6)
    surface = 1.0 + 0.09 * bnoise
    env = np.clip((surface - d) / 0.16, 0.0, 1.0)
    env = env * env * (3.0 - 2.0 * env)

    fine = _gaussian_filter(np, rng.random(d.shape).astype(np.float32), 2.2)
    fine = (fine - fine.min()) / max(np.ptp(fine), 1e-6)
    val = spec["base"] * (0.82 + 0.30 * fine)

    rim = np.clip((d - 0.6) / 0.4, 0, 1)
    val *= 1.0 + spec["rim"] * (rim ** 2.0)

    n_ch = int(rng.integers(spec["chromo"][0], spec["chromo"][1] + 1))
    for _ in range(n_ch):
        p = rng.normal(size=3)
        p = p / (np.linalg.norm(p) + 1e-6) * rng.uniform(0, 0.7)
        cr = rng.uniform(0.25, 0.45)
        g = np.exp(-((xr - p[0]) ** 2 + (yr - p[1]) ** 2 + (zr - p[2]) ** 2)
                   / (2 * cr ** 2))
        val += spec["chromo_amp"] * spec["base"] * g

    for _ in range(spec["nucleoli"]):
        p = rng.normal(size=3)
        p = p / (np.linalg.norm(p) + 1e-6) * rng.uniform(0, 0.35)
        nr = rng.uniform(0.18, 0.34)
        nd = np.sqrt((xr - p[0]) ** 2 + (yr - p[1]) ** 2 + (zr - p[2]) ** 2)
        hole = np.clip((nd - nr) / 0.1, 0.0, 1.0)
        val *= 0.10 + 0.90 * hole

    val = (val * env * BASE_COUNT).astype(np.float32)
    # crop to the nonzero bounding box so placement is tight
    nz = np.argwhere(val > 1e-3)
    if nz.size == 0:
        return val
    lo = nz.min(0)
    hi = nz.max(0) + 1
    return val[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def make_template_bank(rng, scale, n_templates: int = 64) -> list[np.ndarray]:
    """Build a bank of nucleus templates, cell types drawn by their weights."""
    keys = list(CELL_TYPES)
    w = np.array([CELL_TYPES[k]["weight"] for k in keys], dtype=float)
    w /= w.sum()
    bank = []
    for _ in range(n_templates):
        ctype = keys[rng.choice(len(keys), p=w)]
        bank.append(_make_template(rng, scale, ctype))
    return bank


# ---------------------------------------------------------------------------
# Real COSEM shapes (geometry) + synthesized interior texture
# ---------------------------------------------------------------------------

def load_shape_bank(path) -> list[np.ndarray]:
    """Load soft-occupancy shape masks saved by cosem_shapes.py (object array)."""
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(a, dtype=np.float32) for a in arr]


def _pick_type_for_radius(rng, rad_um: float, keys=None) -> str:
    """Choose a cell-type texture spec whose typical radius matches a real shape.

    Soft, weighted-by-proximity draw so a given size still yields some variety
    (a 5 um nucleus is usually glia/astro but occasionally a small neuron).
    `keys` restricts the candidate specs (e.g. glial-only).
    """
    keys = list(CELL_TYPES) if keys is None else list(keys)
    mids = np.array([np.mean(CELL_TYPES[k]["radius"]) for k in keys])
    base_w = np.array([CELL_TYPES[k]["weight"] for k in keys])
    w = base_w * np.exp(-((mids - rad_um) / 1.2) ** 2)
    if w.sum() <= 0:
        w = base_w
    w = w / w.sum()
    return keys[int(rng.choice(len(keys), p=w))]


# the COSEM two-pool split is neuron vs glial; "glia" texture draws from the
# small/dense glial specs (the 3 subtypes the density agent collapses).
_GLIAL_SPECS = ["glia_dense", "small_bright", "astro_pale"]


def _resolve_spec(rng, rad_um: float, cell_type: str | None) -> str:
    if cell_type == "neuron":
        return "neuron"
    if cell_type == "glia":
        return _pick_type_for_radius(rng, rad_um, keys=_GLIAL_SPECS)
    return _pick_type_for_radius(rng, rad_um)


def _phys_dist(shape, center, scale) -> np.ndarray:
    """Anisotropic physical distance (um) from `center` voxel over a volume."""
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dz = (zz - center[0]) * scale[0]
    dy = (yy - center[1]) * scale[1]
    dx = (xx - center[2]) * scale[2]
    return np.sqrt(dz * dz + dy * dy + dx * dx).astype(np.float32)


def _texture_for_shape(occ: np.ndarray, rng, scale, cell_type=None) -> np.ndarray:
    """Fill a real soft-occupancy shape with synthesized DAPI/chromatin texture.

    Geometry (boundary, lobing, size) comes entirely from `occ` (a real COSEM
    nucleus). The interior pattern -- nucleoplasm mottle, peripheral rim,
    heterochromatin foci, nucleolar hole -- is synthesized with the same
    vocabulary as _make_template, but driven by the shape's own distance
    transform instead of an analytic ellipsoid. `cell_type` ("neuron"/"glia"/
    None) selects the texture spec; None picks by size. Returns counts.
    """
    from scipy.ndimage import distance_transform_edt

    occ = np.clip(occ.astype(np.float32), 0.0, 1.0)
    binm = occ > 0.4
    nvox = int(binm.sum())
    vox_vol = scale[0] * scale[1] * scale[2]
    if nvox < 8:
        return (occ * 0.7 * BASE_COUNT).astype(np.float32)

    rad_um = 0.5 * (6.0 * nvox * vox_vol / np.pi) ** (1 / 3)
    spec = CELL_TYPES[_resolve_spec(rng, rad_um, cell_type)]

    # interior depth from the boundary (voxels): 0 at edge, max at the core
    din = distance_transform_edt(binm).astype(np.float32)
    dmax = max(float(din.max()), 1e-6)
    depth_n = din / dmax                              # 0 edge -> 1 core

    # nucleoplasm base + fine mottle
    fine = _gaussian_filter(np, rng.random(occ.shape).astype(np.float32), 2.2)
    fine = (fine - fine.min()) / max(np.ptp(fine), 1e-6)
    val = spec["base"] * (0.82 + 0.30 * fine)

    # peripheral rim: gentle boost in the outer shell (small depth_n)
    rim_w = np.clip((0.35 - depth_n) / 0.35, 0.0, 1.0)
    val *= 1.0 + spec["rim"] * (rim_w ** 2.0)

    interior = np.argwhere(binm)
    # heterochromatin foci at random interior voxels
    n_ch = int(rng.integers(spec["chromo"][0], spec["chromo"][1] + 1))
    for _ in range(n_ch):
        c = interior[rng.integers(len(interior))]
        cr_um = rng.uniform(0.25, 0.45) * rad_um
        g = np.exp(-(_phys_dist(occ.shape, c, scale) ** 2) / (2 * cr_um ** 2))
        val += spec["chromo_amp"] * spec["base"] * g

    # nucleolar hole(s) carved near the core (high depth_n)
    deep = interior[depth_n[binm] > 0.6]
    if len(deep) == 0:
        deep = interior
    for _ in range(spec["nucleoli"]):
        c = deep[rng.integers(len(deep))]
        nr_um = rng.uniform(0.7, 1.5)
        nd = _phys_dist(occ.shape, c, scale)
        hole = np.clip((nd - nr_um) / (0.4), 0.0, 1.0)
        val *= 0.10 + 0.90 * hole

    val = (val * occ * BASE_COUNT).astype(np.float32)  # confine to the soft shape
    return val


def _random_orient(occ: np.ndarray, rng) -> np.ndarray:
    """Random in-plane rotation + axis flips of a soft-occupancy shape.

    The COSEM source block carries a region's real tissue co-alignment (nuclei
    co-stretched in one direction). Randomizing orientation makes the bank
    isotropic -- the right default when nuclei are stamped at arbitrary
    positions without modeling layer architecture.
    """
    from scipy.ndimage import rotate

    ang = float(rng.uniform(0.0, 180.0))
    o = rotate(occ, ang, axes=(1, 2), reshape=True, order=1,
               mode="constant", cval=0.0)
    for ax in range(3):  # cheap out-of-plane variety
        if rng.random() < 0.5:
            o = np.flip(o, ax)
    return np.ascontiguousarray(np.clip(o, 0.0, 1.0), dtype=np.float32)


def _resample_occ(occ: np.ndarray, target_scale, size_scale: float = 1.0):
    """Resample a source-grid occupancy to `target_scale`, optionally rescaling
    its physical size by `size_scale` (isotropic). No-op when already on grid."""
    fz = SHAPE_SOURCE_SCALE[0] / target_scale[0] * size_scale
    fy = SHAPE_SOURCE_SCALE[1] / target_scale[1] * size_scale
    fx = SHAPE_SOURCE_SCALE[2] / target_scale[2] * size_scale
    if max(abs(fz - 1), abs(fy - 1), abs(fx - 1)) < 1e-3:
        return occ
    from scipy.ndimage import zoom
    o = zoom(occ, (fz, fy, fx), order=1, mode="constant", cval=0.0)
    return np.ascontiguousarray(np.clip(o, 0.0, 1.0), dtype=np.float32)


def make_real_template_bank(shapes, rng, scale, n_templates=None, rotate=True,
                            labels=None, neuron_frac=None,
                            size_ratio: float = NEURON_GLIA_SIZE_RATIO
                            ) -> list[np.ndarray]:
    """Texture real COSEM shapes into a stamping bank.

    `shapes` = soft-occupancy masks from load_shape_bank. With n_templates=None
    every shape is textured once; pass a larger n_templates to resample shapes
    (with replacement) so the same geometry recurs with fresh texture/orientation.
    `rotate` randomizes each entry's orientation (breaks the source region's
    co-alignment).

    TYPING (neuron vs glia): preferred path is `neuron_frac` ALONE -- each
    template is independently neuron with that probability (from the density
    table's neuron/glia split), shapes drawn size-agnostically. `size_ratio`
    then makes neurons 1.145x glia in diameter (10x-grounded population stat),
    split around the geometric mean so the absolute-size dial is untouched.
    `labels` (1=neuron 0=glial) is the LEGACY size-classifier path, deprecated
    for typing; if given it still selects per-pool and sets ctype.

    Templates are resampled from SHAPE_SOURCE_SCALE to `scale`, so physical
    nucleus size is preserved at any magnification's pixel size.
    """
    labels = None if labels is None else np.asarray(labels)
    if n_templates is None:
        chosen = list(range(len(shapes)))
    elif labels is not None and neuron_frac is not None:
        neur = np.where(labels == 1)[0]
        glia = np.where(labels == 0)[0]
        n_neur = int(round(n_templates * float(neuron_frac)))
        chosen = ([int(rng.choice(neur)) for _ in range(n_neur)] +
                  [int(rng.choice(glia)) for _ in range(n_templates - n_neur)])
        rng.shuffle(chosen)
    else:
        chosen = [int(rng.integers(len(shapes))) for _ in range(n_templates)]

    root = float(np.sqrt(size_ratio)) if size_ratio and size_ratio > 0 else 1.0
    bank = []
    for i in chosen:
        if labels is not None:
            ctype = "neuron" if labels[i] == 1 else "glia"
        elif neuron_frac is not None:
            ctype = "neuron" if rng.random() < float(neuron_frac) else "glia"
        else:
            ctype = None
        occ = _random_orient(shapes[i], rng) if rotate else shapes[i]
        sz = root if ctype == "neuron" else (1.0 / root if ctype == "glia" else 1.0)
        occ = _resample_occ(occ, scale, size_scale=sz)
        bank.append(_texture_for_shape(occ, rng, scale, cell_type=ctype))
    return bank


def montage_bank(bank, path, scale=None, ncol=8, nrow=4, seed=0) -> None:
    """Quick mid-z montage of textured bank templates (to eyeball diversity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(bank))[: ncol * nrow]
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.4),
                           facecolor="black")
    for a, k in zip(np.atleast_1d(ax).ravel(), idx, strict=False):
        t = bank[k]
        a.imshow(t[t.shape[0] // 2], cmap="bone", interpolation="bilinear")
        a.axis("off")
    for a in np.atleast_1d(ax).ravel()[len(idx):]:
        a.axis("off")
    fig.suptitle("textured bank (real shape + synth chromatin, randomized orientation)",
                 color="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="black")


def stamp_nuclei(
    xp: Any,
    shape: tuple[int, int, int],
    scale: tuple[float, float, float],
    density_per_mm3: float,
    bank: list[np.ndarray],
    rng,
    *,
    atten_um: float = 18.0,
    bg_count: float = 1.2,
) -> tuple[Any, int]:
    """Synthesize a truth volume on `xp` by placing bank templates.

    Placement decisions (which template, where, brightness) use the numpy `rng`
    (cheap scalars); the volume and the slice-adds live on `xp`. Returns
    (truth, n_placed).
    """
    zsc, ysc, xsc = scale
    nz, ny, nx = shape
    vol_mm3 = (nz * zsc) * (ny * ysc) * (nx * xsc) / 1e9
    n = max(1, int(density_per_mm3 * vol_mm3))

    templates = [xp.asarray(t) for t in bank]  # upload once
    truth = xp.zeros(shape, dtype=xp.float32)

    for _ in range(n):
        t = templates[rng.integers(len(templates))]
        tz, ty, tx = t.shape
        # integer center voxel; allow partial overhang past each edge
        cz = int(rng.integers(-tz // 2, nz + tz // 2))
        cy = int(rng.integers(-ty // 2, ny + ty // 2))
        cx = int(rng.integers(-tx // 2, nx + tx // 2))
        z0, y0, x0 = cz - tz // 2, cy - ty // 2, cx - tx // 2
        # clip placement window to the volume, and the template to match
        zz0, zz1 = max(z0, 0), min(z0 + tz, nz)
        yy0, yy1 = max(y0, 0), min(y0 + ty, ny)
        xx0, xx1 = max(x0, 0), min(x0 + tx, nx)
        if zz0 >= zz1 or yy0 >= yy1 or xx0 >= xx1:
            continue
        tzs, tys, txs = zz0 - z0, yy0 - y0, xx0 - x0
        depth = float(np.exp(-max(zz0 * zsc, 0.0) / atten_um))
        bright = float(rng.uniform(0.7, 1.3))
        truth[zz0:zz1, yy0:yy1, xx0:xx1] += (
            t[tzs:tzs + (zz1 - zz0), tys:tys + (yy1 - yy0), txs:txs + (xx1 - xx0)]
            * (depth * bright)
        )

    if bg_count > 0:
        lf = _gaussian_filter(xp, xp.asarray(rng.random(shape).astype(np.float32)),
                              (3, 14, 14))
        lo = lf.min()
        lf = (lf - lo) / xp.maximum(lf.max() - lo, xp.float32(1e-6))
        truth += bg_count * (0.5 + lf)

    return truth, n
