"""Density-modulated point-process and Gaussian-blob texture synthesis.

Four stain-specific transforms: DAPINuclei, NisslCellBodies, FluorescenceSpeckle,
ISHPuncta.  All are HWC float32 [0,1] in/out and consume ctx.density_map plus
ctx.pixel_size_um from the TransformContext.

Performance note: we use density-weighted rejection sampling rather than strict
Poisson-disk (Bridson) because Bridson's Python loop is O(N^2) in practice at
high densities (>10K points on a 1024x1024 image).  Rejection sampling produces
a similar density-modulated distribution in O(N) vectorised numpy ops with a
fixed cap.  The spatial correlations are weaker than true Poisson-disk but still
avoid macro-scale clumping.
"""

from __future__ import annotations

__all__ = [
    "DAPIGrayMatterNuclei",
    "DAPIWhiteMatterNuclei",
    "DAPIDenseCellLayers",
    "DAPIBoundaryHighlights",
    "DAPIStainClumps",
    "NisslGrayMatterCellBodies",
    "NisslWhiteMatterCellBodies",
    "BrightfieldGrayMatterDAB",
    "BrightfieldWhiteMatterDAB",
    "FluorescenceMarker",
    "ISHExpressionPuncta",
    "HematoxylinGrayMatterNuclei",
    "HematoxylinWhiteMatterNuclei",
    "NuclearFastRedGrayMatterNuclei",
    "NuclearFastRedWhiteMatterNuclei",
    "DAPINuclei",
    "NisslCellBodies",
    "FluorescenceSpeckle",
    "ISHPuncta",
]

import numpy as np

from .base import TransformContext

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_DEFAULT_PIXEL_SIZE_UM = 10.0

# Hard cap on nuclei per transform call to keep CPU time bounded.
# At 25 µm/px, 1500/mm² on a 1024^2 image ≈ 984K points — far too many for
# per-blob kernel loops.  We cap and rely on density_map to modulate the
# spatial distribution.  Visual result is indistinguishable at 5K–15K blobs.
_MAX_BLOBS_DAPI = 12_000
# GM/WM caps were sized for ~5 µm/px rendering of full coronal sections
# (~85 mm² tissue). At realistic ~3000 nuclei/mm² density that's ~250K
# expected; we cap below that for splat-loop runtime, and the atlas-grayscale
# density modulation makes the visual sparser-where-dim.
_MAX_BLOBS_DAPI_GM = 150_000
_MAX_BLOBS_DAPI_WM = 20_000
_MAX_BLOBS_NISSL = 5_000
# Nissl cell bodies are larger than DAPI nuclei (whole cytoplasm vs nucleus
# only) so per-image blob counts are lower. WM Nissl is much fainter than WM
# DAPI — only sparse oligodendrocyte cell bodies, no axon-aligned stretching.
_MAX_BLOBS_NISSL_GM = 60_000
_MAX_BLOBS_NISSL_WM = 8_000
_MAX_BLOBS_BRIGHTFIELD = 80_000
_MAX_BLOBS_HEMATOXYLIN_GM = 120_000
_MAX_BLOBS_HEMATOXYLIN_WM = 12_000
_MAX_BLOBS_NFR_GM = 120_000
_MAX_BLOBS_NFR_WM = 12_000
_MAX_BLOBS_FLUOR_BLUE = 8_000
_MAX_BLOBS_FLUOR_GREEN = 300
_MAX_BLOBS_FLUOR_RED = 200
_MAX_BLOBS_ISH = 1_500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_density(ctx: TransformContext, shape: tuple[int, int]) -> np.ndarray:
    """Return (H, W) density map, falling back to uniform 0.5."""
    if ctx.density_map is not None:
        dm = ctx.density_map.astype(np.float32)
        if dm.ndim == 0:
            return np.full(shape, float(dm), dtype=np.float32)
        return dm
    return np.full(shape, 0.5, dtype=np.float32)


def _pixel_size(ctx: TransformContext) -> float:
    return ctx.pixel_size_um if ctx.pixel_size_um is not None else _DEFAULT_PIXEL_SIZE_UM


def _expected_count(
    density_per_mm2: float,
    pixel_size_um: float,
    h: int,
    w: int,
) -> int:
    """Number of blobs expected at density=1.0 averaged over the full image."""
    px_area_mm2 = (pixel_size_um / 1000.0) ** 2
    total_mm2 = h * w * px_area_mm2
    return int(density_per_mm2 * total_mm2)


# ---------------------------------------------------------------------------
# Vectorised density-weighted sampler
# ---------------------------------------------------------------------------


def _sample_points(
    density: np.ndarray,
    n_target: int,
    n_max: int,
    rng: np.random.Generator,
    tissue_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to min(n_target, n_max) point coordinates weighted by density.

    Returns (ys, xs) integer arrays.  Uses a two-pass approach:
    1. Generate uniform candidates over the image.
    2. Accept each with probability proportional to density[y,x].
    Repeat until enough accepted or budget exhausted.
    """
    h, w = density.shape[:2]
    n_want = min(n_target, n_max)
    if n_want <= 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    d_max = float(density.max())
    if d_max <= 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    accepted_y: list[np.ndarray] = []
    accepted_x: list[np.ndarray] = []
    total_accepted = 0

    # Oversample to hit target in one or two passes
    batch_size = max(n_want * 4, 4096)
    max_iters = 8

    for _ in range(max_iters):
        if total_accepted >= n_want:
            break
        still_need = n_want - total_accepted
        n_cand = min(batch_size, still_need * 6)

        cy = rng.integers(0, h, size=n_cand)
        cx = rng.integers(0, w, size=n_cand)

        # Tissue mask filter
        if tissue_mask is not None:
            in_tissue = tissue_mask[cy, cx].astype(bool)
            cy = cy[in_tissue]
            cx = cx[in_tissue]

        if len(cy) == 0:
            continue

        # Density-weighted acceptance
        d_vals = density[cy, cx] / d_max
        accept = rng.random(len(cy)) < d_vals
        cy = cy[accept]
        cx = cx[accept]

        if len(cy) == 0:
            continue

        remaining = n_want - total_accepted
        cy = cy[:remaining]
        cx = cx[:remaining]
        accepted_y.append(cy)
        accepted_x.append(cx)
        total_accepted += len(cy)

    if total_accepted == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    ys = np.concatenate(accepted_y).astype(np.intp)
    xs = np.concatenate(accepted_x).astype(np.intp)
    return ys, xs


# ---------------------------------------------------------------------------
# Vectorised Gaussian blob accumulation
# ---------------------------------------------------------------------------


def _splat_blobs(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    sigmas: np.ndarray,
    intensities: np.ndarray,
    channel_weights: np.ndarray,
    tissue_mask: np.ndarray | None,
) -> None:
    """Additively paint Gaussian blobs onto *canvas* (HWC, modified in-place).

    canvas: HWC float32 [0,1]
    ys, xs: 1-D integer blob centres
    sigmas: 1-D float32, per-blob sigma in pixels
    intensities: 1-D float32, peak intensity per blob
    channel_weights: (N,3) float32, per-blob RGB multiplier
    tissue_mask: if provided, added signal is masked to tissue pixels after splatting
    """
    if len(ys) == 0:
        return

    h, w = canvas.shape[:2]

    # Group blobs by quantised sigma to allow batch kernel reuse.
    # Quantise to nearest 0.5 px to reduce unique kernel count, and clamp at
    # 0.5 minimum so sub-pixel sigmas (which arise at coarse pixel sizes) don't
    # produce a divide-by-zero in the kernel.
    sig_q = np.maximum(np.round(sigmas * 2.0) / 2.0, 0.5).astype(np.float32)
    unique_sigs = np.unique(sig_q)

    for sig in unique_sigs:
        mask_sig = sig_q == sig
        b_ys = ys[mask_sig]
        b_xs = xs[mask_sig]
        b_int = intensities[mask_sig]
        b_cw = channel_weights[mask_sig]  # (M, 3)

        hw = max(1, int(np.ceil(3.0 * sig)))
        kr = np.arange(-hw, hw + 1, dtype=np.float32)
        k1d = np.exp(-0.5 * (kr / sig) ** 2)
        kernel = k1d[:, None] * k1d[None, :]  # (2hw+1, 2hw+1)

        for i in range(len(b_ys)):
            cy, cx = int(b_ys[i]), int(b_xs[i])
            y0, y1 = max(0, cy - hw), min(h, cy + hw + 1)
            x0, x1 = max(0, cx - hw), min(w, cx + hw + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            ky0 = y0 - (cy - hw)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (cx - hw)
            kx1 = kx0 + (x1 - x0)
            blob = b_int[i] * kernel[ky0:ky1, kx0:kx1]  # (ph, pw)
            canvas[y0:y1, x0:x1, :] += blob[:, :, None] * b_cw[i][None, None, :]

    if tissue_mask is not None:
        # Zero out any signal that leaked past mask boundary
        canvas[~tissue_mask.astype(bool)] = 0.0


# ---------------------------------------------------------------------------
# Public transforms
# ---------------------------------------------------------------------------


def _splat_anisotropic_blobs(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    sigmas_long: np.ndarray,
    sigmas_short: np.ndarray,
    thetas: np.ndarray,
    intensities: np.ndarray,
    channel_weights: np.ndarray,
    tissue_mask: np.ndarray | None,
) -> None:
    """Additively paint *elongated* Gaussian blobs onto canvas (HWC, in-place).

    Each blob has its own (sigma_long, sigma_short, theta). Theta = 0 stretches
    along x. Used by DAPIWhiteMatterNuclei to render oligodendrocyte nuclei
    elongated along their local tract axis.
    """
    if len(ys) == 0:
        return
    h, w = canvas.shape[:2]

    for i in range(len(ys)):
        cy, cx = int(ys[i]), int(xs[i])
        sl = max(float(sigmas_long[i]), 0.5)
        ss = max(float(sigmas_short[i]), 0.5)
        th = float(thetas[i])
        hw = max(1, int(np.ceil(3.0 * max(sl, ss))))

        yy, xx = np.mgrid[-hw : hw + 1, -hw : hw + 1].astype(np.float32)
        cos_t, sin_t = np.cos(th), np.sin(th)
        xr = cos_t * xx + sin_t * yy
        yr = -sin_t * xx + cos_t * yy
        kernel = np.exp(-0.5 * ((xr / sl) ** 2 + (yr / ss) ** 2))

        y0, y1 = max(0, cy - hw), min(h, cy + hw + 1)
        x0, x1 = max(0, cx - hw), min(w, cx + hw + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        ky0 = y0 - (cy - hw)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (cx - hw)
        kx1 = kx0 + (x1 - x0)
        blob = float(intensities[i]) * kernel[ky0:ky1, kx0:kx1]
        canvas[y0:y1, x0:x1, :] += blob[:, :, None] * channel_weights[i][None, None, :]

    if tissue_mask is not None:
        canvas[~tissue_mask.astype(bool)] = 0.0


def _splat_alpha_blobs(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    sigmas: np.ndarray,
    intensities: np.ndarray,
    cell_colors: np.ndarray,
    tissue_mask: np.ndarray | None,
) -> None:
    """Alpha-composite cell-coloured Gaussian blobs onto canvas (HWC, in-place).

    Per-blob coverage[y,x] = intensity * gaussian; per-pixel coverage uses the
    *maximum* over all blobs at that pixel (so dense overlap doesn't push past
    the cell color the way additive subtraction does — that's the saturation
    bug we hit in the additive Nissl path). Final pixel:
        canvas[y,x] = (1 - cov) * canvas[y,x] + cov * cell_color
    where cell_color is taken from the blob with the highest coverage at (y,x).
    """
    if len(ys) == 0:
        return
    h, w = canvas.shape[:2]

    coverage = np.zeros((h, w), dtype=np.float32)
    color_buf = np.zeros_like(canvas)

    sig_q = np.maximum(np.round(sigmas * 2.0) / 2.0, 0.5).astype(np.float32)
    unique_sigs = np.unique(sig_q)
    for sig in unique_sigs:
        sel = sig_q == sig
        b_ys = ys[sel]
        b_xs = xs[sel]
        b_int = intensities[sel]
        b_color = cell_colors[sel]
        hw = max(1, int(np.ceil(3.0 * sig)))
        kr = np.arange(-hw, hw + 1, dtype=np.float32)
        k1d = np.exp(-0.5 * (kr / sig) ** 2)
        kernel = k1d[:, None] * k1d[None, :]
        for i in range(len(b_ys)):
            cy, cx = int(b_ys[i]), int(b_xs[i])
            y0, y1 = max(0, cy - hw), min(h, cy + hw + 1)
            x0, x1 = max(0, cx - hw), min(w, cx + hw + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            ky0 = y0 - (cy - hw)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (cx - hw)
            kx1 = kx0 + (x1 - x0)
            blob_cov = b_int[i] * kernel[ky0:ky1, kx0:kx1]
            old_cov = coverage[y0:y1, x0:x1]
            wins = blob_cov > old_cov
            coverage[y0:y1, x0:x1] = np.where(wins, blob_cov, old_cov)
            color_buf[y0:y1, x0:x1, 0] = np.where(wins, b_color[i, 0], color_buf[y0:y1, x0:x1, 0])
            color_buf[y0:y1, x0:x1, 1] = np.where(wins, b_color[i, 1], color_buf[y0:y1, x0:x1, 1])
            color_buf[y0:y1, x0:x1, 2] = np.where(wins, b_color[i, 2], color_buf[y0:y1, x0:x1, 2])

    if tissue_mask is not None:
        out_mask = ~tissue_mask.astype(bool)
        coverage[out_mask] = 0.0

    alpha = np.clip(coverage, 0.0, 1.0)[..., None]
    canvas[:] = (1.0 - alpha) * canvas + alpha * color_buf


def _splat_alpha_anisotropic_blobs(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    sigmas_long: np.ndarray,
    sigmas_short: np.ndarray,
    thetas: np.ndarray,
    intensities: np.ndarray,
    cell_colors: np.ndarray,
    tissue_mask: np.ndarray | None,
) -> None:
    """Alpha-composite *elongated* cell-coloured Gaussian blobs onto canvas.

    Combines the per-blob anisotropic kernel of `_splat_anisotropic_blobs` with
    the max-coverage alpha compositing of `_splat_alpha_blobs`. Used by the
    Nissl WM pair where oligodendrocyte nuclei stretch along the local tract
    direction.
    """
    if len(ys) == 0:
        return
    h, w = canvas.shape[:2]
    coverage = np.zeros((h, w), dtype=np.float32)
    color_buf = np.zeros_like(canvas)

    for i in range(len(ys)):
        cy, cx = int(ys[i]), int(xs[i])
        sl = max(float(sigmas_long[i]), 0.5)
        ss = max(float(sigmas_short[i]), 0.5)
        th = float(thetas[i])
        hw = max(1, int(np.ceil(3.0 * max(sl, ss))))

        yy, xx = np.mgrid[-hw : hw + 1, -hw : hw + 1].astype(np.float32)
        cos_t, sin_t = np.cos(th), np.sin(th)
        xr = cos_t * xx + sin_t * yy
        yr = -sin_t * xx + cos_t * yy
        kernel = np.exp(-0.5 * ((xr / sl) ** 2 + (yr / ss) ** 2))

        y0, y1 = max(0, cy - hw), min(h, cy + hw + 1)
        x0, x1 = max(0, cx - hw), min(w, cx + hw + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        ky0 = y0 - (cy - hw)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (cx - hw)
        kx1 = kx0 + (x1 - x0)
        blob_cov = float(intensities[i]) * kernel[ky0:ky1, kx0:kx1]
        old_cov = coverage[y0:y1, x0:x1]
        wins = blob_cov > old_cov
        coverage[y0:y1, x0:x1] = np.where(wins, blob_cov, old_cov)
        color_buf[y0:y1, x0:x1, 0] = np.where(wins, cell_colors[i, 0], color_buf[y0:y1, x0:x1, 0])
        color_buf[y0:y1, x0:x1, 1] = np.where(wins, cell_colors[i, 1], color_buf[y0:y1, x0:x1, 1])
        color_buf[y0:y1, x0:x1, 2] = np.where(wins, cell_colors[i, 2], color_buf[y0:y1, x0:x1, 2])

    if tissue_mask is not None:
        coverage[~tissue_mask.astype(bool)] = 0.0
    alpha = np.clip(coverage, 0.0, 1.0)[..., None]
    canvas[:] = (1.0 - alpha) * canvas + alpha * color_buf


def _local_tract_orientation(
    mask: np.ndarray,
    *,
    smoothing_sigma: float = 8.0,
) -> np.ndarray:
    """Per-pixel theta giving the local tract direction inside *mask*.

    Tangent = perpendicular to the gradient of the heavily-smoothed
    distance transform of *mask*. Heavy smoothing (sigma ~ region
    half-thickness) is required so the medial-axis ridge inside thick
    regions doesn't destabilize the gradient direction.

    Behaviour falls out of the same algorithm for both shape classes:
      - Elongated tract (corpus callosum) → tangent runs along long axis.
      - Round region (anterior commissure) → circular flow tangent to the
        boundary at every point.
    """
    from .dapi_texture import _shape_aware_orientation

    return _shape_aware_orientation(mask, smoothing_sigma=smoothing_sigma)


def _get_or_compute_tract_orientation(
    ctx: TransformContext,
    wm_mask: np.ndarray,
) -> np.ndarray:
    """Return ctx.tract_orientation, computing it if not yet cached.

    Memoizes the per-pixel theta field derived from the WM mask's
    smoothed distance-transform gradient. Reused across multiple WM-nuclei
    transforms run on the same image.
    """
    if ctx.tract_orientation is not None:
        return ctx.tract_orientation
    theta = _local_tract_orientation(wm_mask)
    ctx.tract_orientation = theta
    return theta


def _resolve_class_mask(ctx: TransformContext, class_name: str) -> np.ndarray | None:
    """Return ctx.tissue_class_masks[class_name] if available; else None."""
    if ctx.tissue_class_masks is None:
        return None
    return ctx.tissue_class_masks.get(class_name)


class DAPIGrayMatterNuclei:
    """GT-calibrated DAPI gray-matter texture renderer.

    Replaces the legacy blob-splat approach with per-pixel Poisson cell-count
    sampling at production resolution. Calibrated against real DAPI screenshots
    in ``GT_Textures/DAPI/`` downsampled to 25 µm/px (mean=0.205, p95=0.40).
    See ``transforms/dapi_texture.py`` and ``tmp/dapi_iteration/`` for details.
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        from .dapi_texture import DAPI_GM_PARAMS, render_dapi_region_texture

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image
        return render_dapi_region_texture(
            image.copy(),
            gm_mask.astype(bool),
            DAPI_GM_PARAMS,
            rng=rng,
            pixel_size_um=_pixel_size(ctx),
            density_map=ctx.density_map,
            channel=2,
        )


class DAPIWhiteMatterNuclei:
    """GT-calibrated DAPI white-matter texture renderer.

    Sparser than GM (oligodendrocyte-dominated). Anisotropic blur aligned
    with the local tract direction reproduces the directional striping
    visible in real WM DAPI. Calibrated against
    ``GT_Textures/DAPI/DAPI_whitematter_*.png``.
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        from .dapi_texture import DAPI_WM_PARAMS, render_dapi_region_texture

        wm_mask = _resolve_class_mask(ctx, "white_matter")
        if wm_mask is None or not wm_mask.any():
            return image
        # Reuse cached tract-orientation field for directional blur alignment.
        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask.astype(bool))
        return render_dapi_region_texture(
            image.copy(),
            wm_mask.astype(bool),
            DAPI_WM_PARAMS,
            rng=rng,
            pixel_size_um=_pixel_size(ctx),
            density_map=ctx.density_map,
            tract_orientation=theta_field,
            channel=2,
        )


class DAPIDenseCellLayers:
    """Solid bright bands in tightly-packed cell layers.

    Real DAPI shows hippocampal pyramidal layers (CA1sp/CA2sp/CA3sp), the
    dentate gyrus granule cell layer (DG-sg), olfactory bulb granule layers,
    and the induseum griseum as visibly *solid* bright bands — the cells are
    too tightly packed to resolve as individual nuclei at typical histology
    resolution. This transform paints a high-intensity blue base over those
    masked regions, then sprinkles a denser cell texture on top so the band
    has fine grain at high resolution but still reads solid at low res.

    The mask comes from ``ctx.tissue_class_masks["dense_cell_layers"]``,
    populated by ``classify_tissue`` from name patterns like "granule cell
    layer" / "pyramidal layer" / "induseum griseum".
    """

    def __init__(
        self,
        p: float = 1.0,
        base_intensity_range: tuple[float, float] = (0.30, 0.55),
        density_range_per_mm2: tuple[float, float] = (8000.0, 14000.0),
        sigma_range_scale: tuple[float, float] = (0.40, 0.70),
        cell_intensity_range: tuple[float, float] = (0.40, 0.75),
    ) -> None:
        self.p = p
        self.base_intensity_range = base_intensity_range
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.cell_intensity_range = cell_intensity_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        from .dapi_texture import DAPI_DENSE_LAYER_PARAMS, render_dapi_region_texture

        mask = _resolve_class_mask(ctx, "dense_cell_layers")
        if mask is None or not mask.any():
            return image
        return render_dapi_region_texture(
            image.copy(),
            mask.astype(bool),
            DAPI_DENSE_LAYER_PARAMS,
            rng=rng,
            pixel_size_um=_pixel_size(ctx),
            density_map=ctx.density_map,
            channel=2,
        )


class DAPIStainClumps:
    """Discrete stain-precipitate clumps at slice edges, ventricles, and IsC.

    Real DAPI sections show bright clumps of precipitate collecting in:
      - the section perimeter (pia / sectioning crevices)
      - the ventricle lumen (stain pooling in the empty space)
      - the tissue band immediately bordering ventricles (ependymal layer)
      - Islands of Calleja within the olfactory tubercle (always)

    Each clump is a dense local cluster of *the same sub-pixel nuclei*
    used by GM/WM, packed densely inside an irregular elliptical region.
    After supersample-then-Lanczos downsample the cluster reads as a
    bright granular blob with cell-level texture inside — not a smooth
    ellipse. Clumps inherit per-cell shape from the DAPI cell renderer
    so they integrate visually with the rest of the texture.

    Sampling: per-zone Poisson density on the qualifying mask; IsC
    always emits one clump per connected component. All compositing is
    pure-blue max-blend (matches DAPI single-channel convention).
    """

    # Per-zone parameter bundles. Sizes in µm, intensities in [0, 1].
    # `cell_density_factor` multiplies the GM cell density when packing
    # nuclei into the clump — clumps are 3-6× denser than GM cortex.
    _ZONE_PARAMS = {
        "edge": dict(
            size_um_range=(20.0, 90.0),
            aspect_range=(0.30, 1.00),
            intensity_range=(0.50, 0.90),
            irregularity=0.70,
            density_per_mm_perimeter=2.0,
            cell_density_factor=4.5,
        ),
        "ventricle_lumen": dict(
            size_um_range=(30.0, 130.0),
            aspect_range=(0.45, 1.00),
            intensity_range=(0.55, 0.95),
            irregularity=0.75,
            density_per_mm2_area=3.5,
            cell_density_factor=5.0,
        ),
        "ventricle_wall": dict(
            size_um_range=(25.0, 90.0),
            aspect_range=(0.25, 0.70),
            intensity_range=(0.45, 0.85),
            irregularity=0.60,
            density_per_mm_perimeter=3.0,
            cell_density_factor=4.0,
            orient_to_local_tangent=True,
        ),
        "islands_of_calleja": dict(
            size_um_range=(20.0, 70.0),
            aspect_range=(0.60, 1.00),
            intensity_range=(0.55, 0.90),
            irregularity=0.55,
            cell_density_factor=5.5,
            one_per_component=True,
        ),
    }

    # Cell-level parameters shared with the GM renderer — clump nuclei
    # use the same sub-pixel sphere geometry as the rest of the texture.
    _CLUMP_CELL_RADIUS_UM_RANGE = (3.0, 5.0)
    _CLUMP_BASE_INTENSITY = 0.45
    _CLUMP_TAIL_SCALE = 0.40
    _SUPERSAMPLE = 4

    def __init__(
        self,
        p: float = 1.0,
        edge_band_px: int = 4,
        ventricle_band_px: int = 3,
    ) -> None:
        self.p = p
        self.edge_band_px = edge_band_px
        self.ventricle_band_px = ventricle_band_px

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        from scipy.ndimage import binary_dilation, binary_erosion, label

        h, w = image.shape[:2]
        pixel_size_um = _pixel_size(ctx)

        tissue = ctx.tissue_mask
        if tissue is None and ctx.tissue_class_masks is not None:
            tissue = ctx.tissue_class_masks.get("tissue")
        if tissue is None:
            return image
        tissue = tissue.astype(bool)
        if not tissue.any():
            return image

        masks = ctx.tissue_class_masks or {}
        ventricle = masks.get("ventricle")
        ventricle = ventricle.astype(bool) if ventricle is not None else None
        isc = masks.get("islands_of_calleja")
        isc = isc.astype(bool) if isc is not None else None

        # Phase 1: collect clump specs across all zones.
        clump_specs: list[dict] = []

        if self.edge_band_px > 0:
            inner = binary_erosion(tissue, iterations=self.edge_band_px)
            edge_band = tissue & ~inner
            if edge_band.any():
                clump_specs.extend(
                    self._sample_clumps(
                        edge_band,
                        "edge",
                        rng,
                        pixel_size_um,
                    )
                )

        if ventricle is not None and ventricle.any():
            clump_specs.extend(
                self._sample_clumps(
                    ventricle,
                    "ventricle_lumen",
                    rng,
                    pixel_size_um,
                )
            )

            if self.ventricle_band_px > 0:
                v_dilated = binary_dilation(ventricle, iterations=self.ventricle_band_px)
                wall_band = v_dilated & tissue & ~ventricle
                if wall_band.any():
                    from .dapi_texture import _shape_aware_orientation

                    theta = _shape_aware_orientation(ventricle, smoothing_sigma=3.0)
                    clump_specs.extend(
                        self._sample_clumps(
                            wall_band,
                            "ventricle_wall",
                            rng,
                            pixel_size_um,
                            orientation_field=theta,
                        )
                    )

        if isc is not None and isc.any():
            labeled, n_components = label(isc)
            for cid in range(1, n_components + 1):
                comp_mask = labeled == cid
                if comp_mask.any():
                    clump_specs.extend(
                        self._sample_clumps(
                            comp_mask,
                            "islands_of_calleja",
                            rng,
                            pixel_size_um,
                        )
                    )

        if not clump_specs:
            return image

        # Phase 2: render all clumps in a single hi-res buffer as dense
        # sub-pixel-cell clusters; downsample once at the end.
        hi_canvas = self._render_clumps_supersampled(
            clump_specs,
            h,
            w,
            pixel_size_um,
            rng,
        )

        from .dapi_texture import _downsample_lanczos

        clump_layer = _downsample_lanczos(hi_canvas, h, w)

        out = image.copy()
        out[..., 2] = np.maximum(out[..., 2], clump_layer)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Sampling helpers

    def _sample_clumps(
        self,
        zone_mask: np.ndarray,
        zone_key: str,
        rng: np.random.Generator,
        pixel_size_um: float,
        orientation_field: np.ndarray | None = None,
    ) -> list[dict]:
        params = self._ZONE_PARAMS[zone_key]
        ys, xs = np.where(zone_mask)
        if ys.size == 0:
            return []
        n_centers = self._sample_count(zone_mask, params, pixel_size_um, rng)
        if n_centers == 0:
            return []
        sel = rng.integers(0, ys.size, size=n_centers)
        cy_arr = ys[sel]
        cx_arr = xs[sel]

        specs: list[dict] = []
        for i in range(n_centers):
            cy, cx = int(cy_arr[i]), int(cx_arr[i])
            if orientation_field is not None:
                angle = float(orientation_field[cy, cx])
            else:
                angle = float(rng.uniform(0, np.pi))
            specs.append(
                dict(
                    cy=cy,
                    cx=cx,
                    size_um=float(rng.uniform(*params["size_um_range"])),
                    aspect=float(rng.uniform(*params["aspect_range"])),
                    angle=angle,
                    intensity=float(rng.uniform(*params["intensity_range"])),
                    irregularity=float(params["irregularity"]),
                    cell_density_factor=float(params["cell_density_factor"]),
                )
            )
        return specs

    def _sample_count(
        self,
        zone_mask: np.ndarray,
        params: dict,
        pixel_size_um: float,
        rng: np.random.Generator,
    ) -> int:
        if params.get("one_per_component"):
            return 1
        px_area_mm2 = (pixel_size_um / 1000.0) ** 2
        if "density_per_mm2_area" in params:
            area_mm2 = float(zone_mask.sum()) * px_area_mm2
            mean = params["density_per_mm2_area"] * area_mm2
        elif "density_per_mm_perimeter" in params:
            from scipy.ndimage import binary_erosion

            interior = binary_erosion(zone_mask, iterations=1)
            perimeter_px = float((zone_mask & ~interior).sum())
            perimeter_mm = perimeter_px * pixel_size_um / 1000.0
            mean = params["density_per_mm_perimeter"] * perimeter_mm
        else:
            mean = 0.0
        return int(rng.poisson(max(0.0, mean)))

    # ------------------------------------------------------------------
    # Hi-res splat — renders all clumps as dense nuclei clusters

    def _render_clumps_supersampled(
        self,
        specs: list[dict],
        h: int,
        w: int,
        pixel_size_um: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Render clumps as dense clusters of sub-pixel cells in a hi-res
        accumulator. Returns (H_hi, W_hi) float32. Downsample externally.
        """
        from scipy.ndimage import gaussian_filter

        s = self._SUPERSAMPLE
        H, W = h * s, w * s
        px_um_hi = pixel_size_um / s

        # Reference cell density (cells per mm² at GM cortex scale).
        ref_density_per_mm2 = 5500.0

        all_y, all_x, all_int, all_sigma = [], [], [], []

        for sp in specs:
            cy_hi = sp["cy"] * s + s / 2.0
            cx_hi = sp["cx"] * s + s / 2.0
            ra_um = sp["size_um"]
            rb_um = sp["size_um"] * sp["aspect"]
            ra_px_hi = ra_um / px_um_hi
            rb_px_hi = rb_um / px_um_hi
            angle = sp["angle"]
            intensity = sp["intensity"]
            irregularity = sp["irregularity"]
            cell_density_factor = sp["cell_density_factor"]

            # Number of cells in the clump = density × ellipse area
            ellipse_area_mm2 = np.pi * ra_um * rb_um / (1000.0**2)
            n_cells = max(8, int(ref_density_per_mm2 * cell_density_factor * ellipse_area_mm2))

            # Sample positions inside a unit disc (rejection)
            n_attempt = int(n_cells * 1.4) + 4
            u = rng.uniform(-1.0, 1.0, size=n_attempt)
            v = rng.uniform(-1.0, 1.0, size=n_attempt)
            keep = (u * u + v * v) < 1.0
            u = u[keep][:n_cells]
            v = v[keep][:n_cells]
            n_kept = u.size
            if n_kept == 0:
                continue

            # Irregular boundary: scale each cell's radial coordinate by
            # a small noise factor — produces ragged clump edges instead
            # of perfectly elliptical ones.
            if irregularity > 0:
                jitter = 1.0 + rng.normal(0.0, irregularity * 0.45, size=n_kept)
                u = u * jitter
                v = v * jitter

            # Stretch to ellipse + rotate
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            x_local = u * ra_px_hi
            y_local = v * rb_px_hi
            x_rot = cos_a * x_local - sin_a * y_local
            y_rot = sin_a * x_local + cos_a * y_local
            ys = np.round(cy_hi + y_rot).astype(np.int32)
            xs = np.round(cx_hi + x_rot).astype(np.int32)

            # Keep only in-bounds
            valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            ys = ys[valid]
            xs = xs[valid]
            if ys.size == 0:
                continue

            # Per-cell intensity — gamma tail like GM cells, scaled so
            # the brightest pixel of the densest part of the clump
            # approaches the requested peak intensity.
            base = self._CLUMP_BASE_INTENSITY * intensity
            tail = self._CLUMP_TAIL_SCALE * intensity
            intens = rng.gamma(
                shape=1.0,
                scale=base + tail,
                size=ys.size,
            ).astype(np.float32)

            # Per-cell radius in hi-res pixels
            sigmas = (
                rng.uniform(*self._CLUMP_CELL_RADIUS_UM_RANGE, size=ys.size) / px_um_hi
            ).astype(np.float32)

            all_y.append(ys)
            all_x.append(xs)
            all_int.append(intens)
            all_sigma.append(sigmas)

        if not all_y:
            return np.zeros((H, W), dtype=np.float32)

        ys = np.concatenate(all_y)
        xs = np.concatenate(all_x)
        intens = np.concatenate(all_int)
        sigmas = np.concatenate(all_sigma)

        # Bucketed Gaussian splat: bin sigmas into N buckets, splat each
        # bucket with a single convolution. Same trick the GM splatter
        # uses in dapi_texture._splat_cells_supersampled.
        canvas = np.zeros((H, W), dtype=np.float32)
        n_buckets = 4
        s_min = float(sigmas.min())
        s_max = float(sigmas.max())
        if s_max - s_min < 1e-6:
            s_max = s_min + 1e-6
        bucket_idx = np.clip(
            np.floor((sigmas - s_min) / (s_max - s_min) * n_buckets).astype(np.int32),
            0,
            n_buckets - 1,
        )
        bucket_edges = np.linspace(s_min, s_max, n_buckets + 1)
        for b in range(n_buckets):
            sel = bucket_idx == b
            if not sel.any():
                continue
            sigma = float((bucket_edges[b] + bucket_edges[b + 1]) * 0.5)
            impulses = np.zeros((H, W), dtype=np.float32)
            np.add.at(impulses, (ys[sel], xs[sel]), intens[sel])
            if sigma > 0.05:
                impulses = gaussian_filter(impulses, sigma=sigma).astype(np.float32)
            canvas += impulses

        return canvas


# Backwards-compat alias so dapi_pipeline.py and tests keep working
# without changes — the new transform is a drop-in replacement.
DAPIBoundaryHighlights = DAPIStainClumps


# ---------------------------------------------------------------------------
# Nissl gray / white matter cell bodies
# ---------------------------------------------------------------------------


def _nissl_cell_colors(
    rng: np.random.Generator,
    n: int,
    *,
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.30, 0.55),
        (0.25, 0.50),
        (0.50, 0.75),
    ),
) -> np.ndarray:
    """Per-blob target cell color (R, G, B) for alpha compositing.

    Default range covers blue-violet → purple-violet — both common Nissl
    presentations depending on stain (cresyl violet vs thionine).
    """
    (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi) = color_range
    return np.stack(
        [
            rng.uniform(r_lo, r_hi, size=n).astype(np.float32),
            rng.uniform(g_lo, g_hi, size=n).astype(np.float32),
            rng.uniform(b_lo, b_hi, size=n).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


class NisslGrayMatterCellBodies:
    """Purple-violet Nissl-stained cell bodies in gray-matter regions.

    Distinct from DAPI: stains whole cytoplasm (not just nuclei), produces
    larger blobs with bimodal size distribution — most blobs are normal-sized
    pyramidal/granule cells, a small fraction are giant layer-5-like pyramidal
    cells. Composited subtractively over a cream substrate; cytoplasmic blur
    softens the delta to give a glowing-around-cell effect.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (2400.0, 3600.0),
        sigma_range_scale_small: tuple[float, float] = (0.5, 0.9),
        sigma_range_scale_large: tuple[float, float] = (1.4, 2.2),
        large_cell_fraction: float = 0.10,
        intensity_range: tuple[float, float] = (0.45, 0.85),
        cream: tuple[float, float, float] = (0.85, 0.78, 0.65),
        cytoplasm_blur_radius_scale: float = 0.6,
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale_small = sigma_range_scale_small
        self.sigma_range_scale_large = sigma_range_scale_large
        self.large_cell_fraction = large_cell_fraction
        self.intensity_range = intensity_range
        self.cream = cream
        self.cytoplasm_blur_radius_scale = cytoplasm_blur_radius_scale

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))

        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_NISSL_GM, rng, gm_mask)
        n = len(ys)
        if n == 0:
            return image

        is_large = rng.random(n) < self.large_cell_fraction
        small_lo, small_hi = self.sigma_range_scale_small
        large_lo, large_hi = self.sigma_range_scale_large
        sigmas = np.empty(n, dtype=np.float32)
        sigmas[~is_large] = rng.uniform(
            small_lo * sigma_scale,
            small_hi * sigma_scale,
            size=int((~is_large).sum()),
        )
        sigmas[is_large] = rng.uniform(
            large_lo * sigma_scale,
            large_hi * sigma_scale,
            size=int(is_large.sum()),
        )

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _nissl_cell_colors(rng, n)

        out = image.copy()
        _splat_alpha_blobs(out, ys, xs, sigmas, intensities, cell_colors, gm_mask)

        if self.cytoplasm_blur_radius_scale > 0:
            blur_r = max(1, int(self.cytoplasm_blur_radius_scale * sigma_scale))
            # Blur only the change so the alpha-composited cells get a soft
            # cytoplasmic glow without smearing the substrate.
            delta = out - image
            delta = _box_blur_3ch(delta, radius=blur_r)
            delta[~gm_mask.astype(bool)] = 0.0
            out = np.clip(image + delta, 0.0, 1.0)
        return out


class NisslWhiteMatterCellBodies:
    """Sparse Nissl signal in white matter — oligodendrocyte cell bodies that
    appear dramatically elongated along the local axon-bundle direction.
    Much paler than gray matter. Dominantly substrate-textured.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (300.0, 550.0),
        sigma_range_scale: tuple[float, float] = (0.4, 0.7),
        aspect_ratio_range: tuple[float, float] = (8.0, 12.0),
        orientation_jitter_rad: float = 0.35,
        intensity_range: tuple[float, float] = (0.30, 0.55),
        cream: tuple[float, float, float] = (0.85, 0.78, 0.65),
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.orientation_jitter_rad = orientation_jitter_rad
        self.intensity_range = intensity_range
        self.cream = cream

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        wm_mask = _resolve_class_mask(ctx, "white_matter")
        if wm_mask is None or not wm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_NISSL_WM, rng, wm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas_short = rng.uniform(
            s_lo * sigma_scale,
            s_hi * sigma_scale,
            size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        ar = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * ar).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad,
            self.orientation_jitter_rad,
            size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _nissl_cell_colors(
            rng,
            n,
            color_range=((0.40, 0.55), (0.40, 0.55), (0.60, 0.78)),
        )

        out = image.copy()
        _splat_alpha_anisotropic_blobs(
            out,
            ys,
            xs,
            sigmas_long,
            sigmas_short,
            thetas,
            intensities,
            cell_colors,
            wm_mask,
        )
        return out


# ---------------------------------------------------------------------------
# Brightfield IHC (DAB) gray / white matter
# ---------------------------------------------------------------------------


def _dab_channel_weights(
    rng: np.random.Generator,
    n: int,
    tan: tuple[float, float, float],
) -> np.ndarray:
    """(DAB - tan) deltas for subtractive compositing.

    DAB chromogen reads as warm brown — high R, mid G, low B. Subtraction-mode
    splatting drops cream/tan substrate toward this brown.
    """
    dab_r = rng.uniform(0.40, 0.55, size=n).astype(np.float32)
    dab_g = rng.uniform(0.22, 0.34, size=n).astype(np.float32)
    dab_b = rng.uniform(0.10, 0.20, size=n).astype(np.float32)
    return np.stack(
        [dab_r - tan[0], dab_g - tan[1], dab_b - tan[2]],
        axis=1,
    ).astype(np.float32)


class BrightfieldGrayMatterDAB:
    """DAB-chromogen signal in gray matter regions; mode-conditional density.

    A single per-call ``mode`` parameter controls the staining pattern:
        - ``pan_neuronal``: dense across all GM (NeuN-like)
        - ``sparse_interneuron``: very sparse (PV/Calbindin-like)
        - ``myelin``: near-zero in GM (signal is in WM)

    The pipeline picks the mode randomly per image, so the synthetic dataset
    spans a realistic range of antibody-target presentations.
    """

    _MODE_DENSITY: dict[str, tuple[float, float]] = {
        "pan_neuronal": (1500.0, 2500.0),
        "sparse_interneuron": (60.0, 200.0),
        "myelin": (40.0, 120.0),
    }
    _MODE_INTENSITY: dict[str, tuple[float, float]] = {
        "pan_neuronal": (0.55, 0.90),
        "sparse_interneuron": (0.65, 0.95),
        "myelin": (0.20, 0.40),
    }

    def __init__(
        self,
        p: float = 1.0,
        sigma_range_scale: tuple[float, float] = (1.2, 2.4),
        tan: tuple[float, float, float] = (0.85, 0.78, 0.62),
    ) -> None:
        self.p = p
        self.sigma_range_scale = sigma_range_scale
        self.tan = tan

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
        mode: str = "pan_neuronal",
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        if mode not in self._MODE_DENSITY:
            raise ValueError(f"unknown brightfield mode {mode!r}")

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self._MODE_DENSITY[mode]
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_BRIGHTFIELD, rng, gm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(s_lo * sigma_scale, s_hi * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(*self._MODE_INTENSITY[mode], size=n).astype(np.float32)
        cw = _dab_channel_weights(rng, n, self.tan)

        delta = np.zeros_like(image)
        _splat_blobs(delta, ys, xs, sigmas, intensities, cw, gm_mask)
        return np.clip(image + delta, 0.0, 1.0)


class BrightfieldWhiteMatterDAB:
    """DAB-chromogen signal in white matter; mode-conditional density.

    Inverse of GM mode-density: ``myelin`` mode floods WM with signal (real
    myelin stains light up tracts strongly), pan-neuronal stains barely
    touch WM, and interneuron stains hit it not at all.
    """

    _MODE_DENSITY: dict[str, tuple[float, float]] = {
        "pan_neuronal": (50.0, 150.0),
        "sparse_interneuron": (0.0, 30.0),
        "myelin": (1500.0, 2500.0),
    }
    _MODE_INTENSITY: dict[str, tuple[float, float]] = {
        "pan_neuronal": (0.20, 0.40),
        "sparse_interneuron": (0.20, 0.40),
        "myelin": (0.55, 0.90),
    }

    def __init__(
        self,
        p: float = 1.0,
        sigma_range_scale: tuple[float, float] = (0.7, 1.6),
        tan: tuple[float, float, float] = (0.85, 0.78, 0.62),
    ) -> None:
        self.p = p
        self.sigma_range_scale = sigma_range_scale
        self.tan = tan

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
        mode: str = "pan_neuronal",
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image
        if mode not in self._MODE_DENSITY:
            raise ValueError(f"unknown brightfield mode {mode!r}")

        wm_mask = _resolve_class_mask(ctx, "white_matter")
        if wm_mask is None or not wm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self._MODE_DENSITY[mode]
        if d_hi <= 0.0:
            return image
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        if density_per_mm2 <= 0:
            return image

        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_BRIGHTFIELD, rng, wm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(s_lo * sigma_scale, s_hi * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(*self._MODE_INTENSITY[mode], size=n).astype(np.float32)
        cw = _dab_channel_weights(rng, n, self.tan)

        delta = np.zeros_like(image)
        _splat_blobs(delta, ys, xs, sigmas, intensities, cw, wm_mask)
        return np.clip(image + delta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Fluorescence sparse marker channel (green or red)
# ---------------------------------------------------------------------------


class FluorescenceMarker:
    """Sparse single-channel fluorescent marker (e.g. GFP, tdTomato).

    Renders dots in one chosen channel only — additive over a dark canvas, no
    cross-talk into the other two channels (real fluorescence imaging filters
    are channel-specific). Density is much lower than DAPI; markers tag a
    subpopulation (e.g. cFos-active neurons, calbindin interneurons).
    """

    def __init__(
        self,
        channel: int,  # 0=red, 1=green, 2=blue
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (40.0, 200.0),
        sigma_range_scale: tuple[float, float] = (0.6, 1.4),
        intensity_range: tuple[float, float] = (0.55, 1.0),
    ) -> None:
        if channel not in (0, 1, 2):
            raise ValueError(f"channel must be 0/1/2, got {channel}")
        self.channel = channel
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.intensity_range = intensity_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_FLUOR_BLUE, rng, gm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(s_lo * sigma_scale, s_hi * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cw = np.zeros((n, 3), dtype=np.float32)
        cw[:, self.channel] = 1.0

        canvas = np.zeros_like(image)
        _splat_blobs(canvas, ys, xs, sigmas, intensities, cw, gm_mask)
        return np.clip(image + canvas, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ISH (in situ hybridization) — gene-expression-modulated puncta
# ---------------------------------------------------------------------------


def _generate_expression_mask(
    shape: tuple[int, int],
    tissue_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    spatial_scale: float = 0.3,
    threshold_range: tuple[float, float] = (0.30, 0.65),
) -> np.ndarray:
    """Build a per-pixel "gene expression intensity" mask in [0, 1].

    Each gene has its own expression pattern — some are dorsal-only, some
    scattered, some restricted to a single nucleus. A low-frequency random
    field multiplied by tissue mask, then soft-thresholded, produces a
    plausible per-image expression spatial distribution that varies dramatically
    between calls.
    """
    field = _smooth_noise(shape, rng, scale=1.0)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    threshold = float(rng.uniform(*threshold_range))
    expr = np.clip((field - threshold) * (1.0 / (1.0 - threshold)), 0.0, 1.0)
    expr = expr * tissue_mask.astype(np.float32)
    return expr.astype(np.float32)


class ISHExpressionPuncta:
    """Brown chromogenic ISH puncta confined to a per-image expression mask.

    Combines a stochastic expression spatial pattern with the atlas-grayscale
    density map: puncta appear where (expression × density) is high. Most of
    the section stays at substrate cream — only the "expressed" regions
    accumulate signal. A faint diffuse wash precedes the puncta to mimic the
    soft chromogen accumulation seen at low magnification.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (1500.0, 3500.0),
        sigma_range_scale: tuple[float, float] = (0.5, 0.9),
        intensity_range: tuple[float, float] = (0.55, 0.95),
        wash_strength_range: tuple[float, float] = (0.04, 0.14),
        cream: tuple[float, float, float] = (0.86, 0.82, 0.92),
        wash_color: tuple[float, float, float] = (0.30, 0.25, 0.55),
        punctum_color_range: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = ((0.18, 0.40), (0.15, 0.32), (0.45, 0.70)),
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.intensity_range = intensity_range
        self.wash_strength_range = wash_strength_range
        self.cream = cream
        self.wash_color = wash_color
        self.punctum_color_range = punctum_color_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        expression = _generate_expression_mask((h, w), gm_mask, rng)
        # Couple expression to atlas density: cells barely visible at the atlas
        # level shouldn't show much ISH signal. Multiplicative coupling with a
        # small floor preserves the random expression pattern while letting
        # cell-rich regions dominate.
        expression = expression * np.clip(density / max(density.max(), 1e-6), 0.05, 1.0)
        if not (expression > 0.05).any():
            return image  # this gene didn't "express" anywhere — return unchanged

        # Diffuse stain wash via alpha compositing — coupled to the (now
        # density-modulated) expression strength. Allen ISH uses NBT/BCIP
        # detection which deposits a purple/blue-purple precipitate.
        wash_strength = float(rng.uniform(*self.wash_strength_range))
        stain_wash = np.array(self.wash_color, dtype=np.float32)
        wash_alpha = wash_strength * expression
        wash_alpha[~gm_mask.astype(bool)] = 0.0
        wash_alpha_3 = wash_alpha[..., None]
        out = (1.0 - wash_alpha_3) * image + wash_alpha_3 * stain_wash[None, None, :]
        out = np.clip(out.astype(np.float32), 0.0, 1.0)

        # Sharp puncta within expressed regions, alpha-composited so dense
        # regions don't oversaturate (same fix we applied to the Nissl pair).
        binary_expr = (expression > 0.10) & gm_mask.astype(bool)
        if not binary_expr.any():
            return out
        puncta_density = density * expression
        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(
            puncta_density,
            n_target,
            _MAX_BLOBS_ISH * 4,
            rng,
            binary_expr,
        )
        n = len(ys)
        if n == 0:
            return out

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(s_lo * sigma_scale, s_hi * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi) = self.punctum_color_range
        cell_colors = np.stack(
            [
                rng.uniform(r_lo, r_hi, size=n).astype(np.float32),
                rng.uniform(g_lo, g_hi, size=n).astype(np.float32),
                rng.uniform(b_lo, b_hi, size=n).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        _splat_alpha_blobs(out, ys, xs, sigmas, intensities, cell_colors, binary_expr)
        return out


class DAPINuclei:
    """Bright blue nuclei dots on dark background.

    Renders up to _MAX_BLOBS_DAPI nuclei weighted by ctx.density_map as small
    Gaussian blobs on the blue channel.

    NOTE: legacy uniform-density transform; new pipelines should use the
    DAPIGrayMatterNuclei + DAPIWhiteMatterNuclei pair instead, which honor
    tissue-class masks.

    DEPRECATED: use DAPIGrayMatterNuclei + DAPIWhiteMatterNuclei instead.
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)

        n_target = _expected_count(1500.0, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_DAPI, rng, ctx.tissue_mask)

        n = len(ys)
        if n == 0:
            return image

        sigma_scale = 10.0 / px_um
        sigmas = rng.uniform(1.5 * sigma_scale, 3.0 * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(0.6, 1.0, size=n).astype(np.float32)

        blue_dom = rng.uniform(0.85, 1.0, size=n).astype(np.float32)
        rg_bleed = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
        channel_weights = np.stack([rg_bleed, rg_bleed * 0.5, blue_dom], axis=1).astype(np.float32)

        out = image.copy()
        _splat_blobs(out, ys, xs, sigmas, intensities, channel_weights, ctx.tissue_mask)
        return np.clip(out, 0.0, 1.0)


class NisslCellBodies:
    """Purple/pink cytoplasmic stain cell bodies — larger blobs, fewer per mm².

    Renders up to _MAX_BLOBS_NISSL cells weighted by ctx.density_map.

    DEPRECATED: use NisslGrayMatterCellBodies + NisslWhiteMatterCellBodies instead.
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)

        n_target = _expected_count(600.0, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_NISSL, rng, ctx.tissue_mask)

        n = len(ys)
        if n == 0:
            return image

        sigma_scale = 10.0 / px_um
        sigmas = rng.uniform(2.5 * sigma_scale, 4.5 * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(0.5, 0.85, size=n).astype(np.float32)

        r_w = rng.uniform(0.6, 0.9, size=n).astype(np.float32)
        g_w = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
        b_w = rng.uniform(0.5, 0.85, size=n).astype(np.float32)
        channel_weights = np.stack([r_w, g_w, b_w], axis=1).astype(np.float32)

        out = image.copy()
        _splat_blobs(out, ys, xs, sigmas, intensities, channel_weights, ctx.tissue_mask)

        # Cytoplasmic softening: blur only the added delta
        delta = out - image
        delta_blurred = _box_blur_3ch(delta, radius=max(1, int(2.0 * sigma_scale)))
        if ctx.tissue_mask is not None:
            delta_blurred[~ctx.tissue_mask.astype(bool)] = 0.0
        return np.clip(image + delta_blurred, 0.0, 1.0)


def _box_blur_3ch(arr: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    out = np.empty_like(arr)
    size = 2 * radius + 1
    for c in range(arr.shape[2]):
        out[:, :, c] = uniform_filter(arr[:, :, c], size=size, mode="reflect")
    return out


class FluorescenceSpeckle:
    """Three-channel fluorescence puncta: DAPI (blue), sparse green, sparser red.

    Each channel is sampled independently from ctx.density_map with its own
    density target and blob count cap.

    DEPRECATED: use render_fluorescence_section with a FluorescenceMode instead.
    """

    _CHANNEL_PARAMS: list[tuple[str, int, float, int]] = [
        # (name, channel_index, density/mm2, max_blobs)
        ("blue", 2, 1500.0, _MAX_BLOBS_FLUOR_BLUE),
        ("green", 1, 50.0, _MAX_BLOBS_FLUOR_GREEN),
        ("red", 0, 30.0, _MAX_BLOBS_FLUOR_RED),
    ]

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        out = image.copy()

        for _ch_name, ci, ch_d, ch_max in self._CHANNEL_PARAMS:
            n_target = _expected_count(ch_d, px_um, h, w)
            ys, xs = _sample_points(density, n_target, ch_max, rng, ctx.tissue_mask)
            n = len(ys)
            if n == 0:
                continue

            sigmas = rng.uniform(1.0, 3.0, size=n).astype(np.float32)
            intensities = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
            cw = np.zeros((n, 3), dtype=np.float32)
            cw[:, ci] = 1.0
            _splat_blobs(out, ys, xs, sigmas, intensities, cw, ctx.tissue_mask)

        return np.clip(out, 0.0, 1.0)


class ISHPuncta:
    """In-situ hybridisation signal: small brown puncta in expressed regions.

    An expression mask is derived by soft-thresholding a random smooth modulation
    of ctx.density_map.  A faint diffuse brown wash is added first (stain bleed),
    then brown puncta are placed within the expression mask.

    DEPRECATED: use ISHExpressionPuncta via render_ish_section instead.
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)

        # Build expression mask
        threshold = rng.uniform(0.3, 0.7)
        noise = _smooth_noise((h, w), rng, scale=0.3)
        expr_mask = np.clip((density - threshold + noise) * 3.0, 0.0, 1.0)

        # Tissue mask intersection for puncta
        binary_expr = expr_mask > 0.3
        if ctx.tissue_mask is not None:
            binary_expr = binary_expr & ctx.tissue_mask.astype(bool)

        out = image.copy()

        # Diffuse brown wash (applied to full expr region, masked by tissue)
        wash_strength = rng.uniform(0.03, 0.12)
        brown = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        wash = wash_strength * expr_mask[:, :, None] * brown[None, None, :]
        if ctx.tissue_mask is not None:
            wash[~ctx.tissue_mask.astype(bool)] = 0.0
        out += wash

        # Puncta within expression mask
        puncta_density = density * expr_mask
        n_target = _expected_count(200.0, px_um, h, w)
        ys, xs = _sample_points(
            puncta_density,
            n_target,
            _MAX_BLOBS_ISH,
            rng,
            binary_expr if binary_expr.any() else None,
        )

        n = len(ys)
        if n > 0:
            sigmas = rng.uniform(1.5, 3.0, size=n).astype(np.float32)
            intensities = rng.uniform(0.4, 0.85, size=n).astype(np.float32)
            r_w = rng.uniform(0.3, 0.6, size=n).astype(np.float32)
            g_w = rng.uniform(0.1, 0.25, size=n).astype(np.float32)
            b_w = rng.uniform(0.05, 0.15, size=n).astype(np.float32)
            channel_weights = np.stack([r_w, g_w, b_w], axis=1).astype(np.float32)
            # Use binary_expr as the tissue mask so blobs stay within expression region
            _splat_blobs(
                out,
                ys,
                xs,
                sigmas,
                intensities,
                channel_weights,
                binary_expr if binary_expr.any() else ctx.tissue_mask,
            )

        return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Hematoxylin counterstain — nuclear stain, blue-violet on pale-cream substrate
# ---------------------------------------------------------------------------


def _hematoxylin_cell_colors(
    rng: np.random.Generator,
    n: int,
    *,
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.20, 0.40),
        (0.20, 0.45),
        (0.50, 0.75),
    ),
) -> np.ndarray:
    """Per-blob target cell color (R, G, B) for hematoxylin nuclear staining.

    Default range covers blue-violet → steel-blue — typical hematoxylin-stained
    nucleus appearance when used as a counterstain in IHC or chromogenic ISH.
    Hematoxylin stains chromatin (nuclei only), so color is confined to small,
    sharply delineated dots.
    """
    (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi) = color_range
    return np.stack(
        [
            rng.uniform(r_lo, r_hi, size=n).astype(np.float32),
            rng.uniform(g_lo, g_hi, size=n).astype(np.float32),
            rng.uniform(b_lo, b_hi, size=n).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


class HematoxylinGrayMatterNuclei:
    """Dense small blue-violet nuclei in gray-matter regions (hematoxylin counterstain).

    Hematoxylin stains nuclei only (not cytoplasm), so blobs are small and
    sharp with no cytoplasmic blur. Alpha-composited over a pale-cream
    substrate to avoid saturation in dense regions. Typical IHC hematoxylin
    counterstain appearance: small, evenly spaced dots on a nearly-white
    pinkish-cream background.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (3000.0, 4500.0),
        sigma_range_scale: tuple[float, float] = (0.5, 0.9),
        intensity_range: tuple[float, float] = (0.55, 0.90),
        color_range: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = ((0.20, 0.40), (0.20, 0.45), (0.50, 0.75)),
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.intensity_range = intensity_range
        self.color_range = color_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))

        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_HEMATOXYLIN_GM, rng, gm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(
            s_lo * sigma_scale,
            s_hi * sigma_scale,
            size=n,
        ).astype(np.float32)
        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _hematoxylin_cell_colors(rng, n, color_range=self.color_range)

        # Alpha-composite: dense nuclei without saturation; NO cytoplasmic blur
        # (hematoxylin stains nuclei only, not cytoplasm)
        out = image.copy()
        _splat_alpha_blobs(out, ys, xs, sigmas, intensities, cell_colors, gm_mask)
        return out


class HematoxylinWhiteMatterNuclei:
    """Sparse elongated nuclei in white matter (hematoxylin counterstain).

    Oligodendrocyte nuclei in WM are stained by hematoxylin but appear sparser
    and slightly elongated along the local tract direction. Less elongated than
    the Nissl WM variant (aspect 3–6x vs 8–12x for Nissl) because hematoxylin
    stains nuclei only (compact oval), not the cell body. Dimmer than GM.
    Alpha-composited to match the nuclear appearance.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (200.0, 450.0),
        sigma_range_scale: tuple[float, float] = (0.4, 0.7),
        aspect_ratio_range: tuple[float, float] = (3.0, 6.0),
        orientation_jitter_rad: float = 0.40,
        intensity_range: tuple[float, float] = (0.35, 0.65),
        color_range: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = ((0.25, 0.45), (0.28, 0.50), (0.52, 0.72)),
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.orientation_jitter_rad = orientation_jitter_rad
        self.intensity_range = intensity_range
        self.color_range = color_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        wm_mask = _resolve_class_mask(ctx, "white_matter")
        if wm_mask is None or not wm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_HEMATOXYLIN_WM, rng, wm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas_short = rng.uniform(
            s_lo * sigma_scale,
            s_hi * sigma_scale,
            size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        ar = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * ar).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad,
            self.orientation_jitter_rad,
            size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _hematoxylin_cell_colors(rng, n, color_range=self.color_range)

        out = image.copy()
        _splat_alpha_anisotropic_blobs(
            out,
            ys,
            xs,
            sigmas_long,
            sigmas_short,
            thetas,
            intensities,
            cell_colors,
            wm_mask,
        )
        return out


# ---------------------------------------------------------------------------
# Nuclear Fast Red counterstain — nuclear stain, red-pink on pale-pink substrate
# ---------------------------------------------------------------------------


def _nfr_cell_colors(
    rng: np.random.Generator,
    n: int,
    r_range: tuple[float, float] = (0.52, 0.72),
    g_range: tuple[float, float] = (0.48, 0.68),
    b_range: tuple[float, float] = (0.60, 0.80),
) -> np.ndarray:
    """Per-blob target cell color (R, G, B) for Nuclear Fast Red nuclei.

    Calibrated against Allen ISH P56 scans: real NFR nuclei in ISH sections
    appear as subtle lavender/blue-purple dots (not saturated red-pink).  The
    aluminium sulfate mordant combined with the thin section and tissue auto-
    absorption shifts the apparent hue toward a desaturated violet.
    """
    return np.stack(
        [
            rng.uniform(*r_range, size=n).astype(np.float32),
            rng.uniform(*g_range, size=n).astype(np.float32),
            rng.uniform(*b_range, size=n).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


class NuclearFastRedGrayMatterNuclei:
    """Dense small lavender-pink nuclei in gray matter for Nuclear Fast Red.

    Intensity calibrated to Allen ISH P56 reference scans: nuclei appear as
    subtle lavender dots, not bright red-pink blobs.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (3000.0, 4500.0),
        sigma_range_scale: tuple[float, float] = (0.5, 0.9),
        intensity_range: tuple[float, float] = (0.22, 0.52),
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.intensity_range = intensity_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        gm_mask = _resolve_class_mask(ctx, "gray_matter")
        if gm_mask is None or not gm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_NFR_GM, rng, gm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(
            s_lo * sigma_scale,
            s_hi * sigma_scale,
            size=n,
        ).astype(np.float32)
        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _nfr_cell_colors(rng, n)

        out = image.copy()
        _splat_alpha_blobs(out, ys, xs, sigmas, intensities, cell_colors, gm_mask)
        return out


class NuclearFastRedWhiteMatterNuclei:
    """Sparse elongated lavender-pink nuclei in white matter for Nuclear Fast Red.

    Intensity calibrated to Allen ISH P56 reference scans: nuclei are subtle.
    """

    def __init__(
        self,
        p: float = 1.0,
        aspect_ratio_range: tuple[float, float] = (3.0, 6.0),
        density_range_per_mm2: tuple[float, float] = (200.0, 450.0),
        sigma_range_scale: tuple[float, float] = (0.4, 0.7),
        intensity_range: tuple[float, float] = (0.18, 0.42),
        orientation_jitter_rad: float = 0.40,
    ) -> None:
        self.p = p
        self.aspect_ratio_range = aspect_ratio_range
        self.density_range_per_mm2 = density_range_per_mm2
        self.sigma_range_scale = sigma_range_scale
        self.intensity_range = intensity_range
        self.orientation_jitter_rad = orientation_jitter_rad

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        wm_mask = _resolve_class_mask(ctx, "white_matter")
        if wm_mask is None or not wm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)
        sigma_scale = 10.0 / px_um

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))
        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_NFR_WM, rng, wm_mask)
        n = len(ys)
        if n == 0:
            return image

        s_lo, s_hi = self.sigma_range_scale
        sigmas_short = rng.uniform(
            s_lo * sigma_scale,
            s_hi * sigma_scale,
            size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        aspect = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * aspect).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad,
            self.orientation_jitter_rad,
            size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _nfr_cell_colors(rng, n)

        out = image.copy()
        _splat_alpha_anisotropic_blobs(
            out,
            ys,
            xs,
            sigmas_long,
            sigmas_short,
            thetas,
            intensities,
            cell_colors,
            wm_mask,
        )
        return out


# ---------------------------------------------------------------------------
# Fast Green FCF counterstain — uniform substrate tint, no per-cell rendering
def _smooth_noise(shape: tuple[int, ...], rng: np.random.Generator, scale: float) -> np.ndarray:
    """Smooth low-frequency noise via bilinear-upsampling of a small random grid."""
    from scipy.ndimage import zoom

    h, w = shape[:2]
    small_h = max(4, h // 32)
    small_w = max(4, w // 32)
    small = rng.random((small_h, small_w)).astype(np.float32)
    big = zoom(small, (h / small_h, w / small_w), order=1)
    big = big[:h, :w]
    if big.shape[0] < h:
        big = np.pad(big, ((0, h - big.shape[0]), (0, 0)), mode="edge")
    if big.shape[1] < w:
        big = np.pad(big, ((0, 0), (0, w - big.shape[1])), mode="edge")
    return (big - 0.5) * scale
