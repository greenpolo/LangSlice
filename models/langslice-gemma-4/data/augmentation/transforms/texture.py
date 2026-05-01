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

        yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1].astype(np.float32)
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

        yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1].astype(np.float32)
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
    mask: np.ndarray, *, smoothing_sigma: float = 2.0,
) -> np.ndarray:
    """Per-pixel theta giving the local tract direction inside *mask*.

    Computes the gradient of the (smoothed) distance transform of mask. The
    gradient points perpendicular to the nearest mask boundary; the tract
    direction is perpendicular to that. Pixels outside mask get theta=0.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    dt = distance_transform_edt(mask).astype(np.float32)
    if smoothing_sigma > 0:
        dt = gaussian_filter(dt, sigma=smoothing_sigma)
    gy, gx = np.gradient(dt)
    return np.arctan2(-gx, gy).astype(np.float32)


def _get_or_compute_tract_orientation(
    ctx: "TransformContext", wm_mask: np.ndarray,
) -> np.ndarray:
    """Return ctx.tract_orientation, computing it if not yet cached.

    Memoizes the per-pixel theta field derived from the WM mask's distance-
    transform gradient.  When multiple WM-nuclei transforms run on the same
    image (e.g. Nissl WM + Hematoxylin WM in a layered ISH mode), the field
    is computed once and reused, avoiding redundant scipy calls.
    """
    if ctx.tract_orientation is not None:
        return ctx.tract_orientation
    theta = _local_tract_orientation(wm_mask, smoothing_sigma=2.0)
    ctx.tract_orientation = theta
    return theta


def _resolve_class_mask(ctx: TransformContext, class_name: str) -> np.ndarray | None:
    """Return ctx.tissue_class_masks[class_name] if available; else None."""
    if ctx.tissue_class_masks is None:
        return None
    return ctx.tissue_class_masks.get(class_name)


class DAPIGrayMatterNuclei:
    """Dense small bright-blue nuclei in gray-matter regions only.

    Real DAPI gray matter has ~10^5 neurons + glia per mm^3 — at 25 µm/px slice
    thickness the visible 2D density is ~1500-2500/mm^2. Renders only within
    ctx.tissue_class_masks["gray_matter"]; density modulated locally by
    ctx.density_map.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (2800.0, 4200.0),
        sigma_range_scale: tuple[float, float] = (0.45, 0.85),
        intensity_range: tuple[float, float] = (0.65, 1.0),
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

        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))

        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_DAPI_GM, rng, gm_mask)
        n = len(ys)
        if n == 0:
            return image

        sigma_scale = 10.0 / px_um
        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(s_lo * sigma_scale, s_hi * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)

        blue_dom = rng.uniform(0.85, 1.0, size=n).astype(np.float32)
        rg_bleed = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
        channel_weights = np.stack([rg_bleed, rg_bleed * 0.5, blue_dom], axis=1).astype(np.float32)

        canvas = np.zeros_like(image)
        _splat_blobs(canvas, ys, xs, sigmas, intensities, channel_weights, gm_mask)
        return np.clip(image + canvas, 0.0, 1.0)


class DAPIWhiteMatterNuclei:
    """Sparse, elongated, dim nuclei in white-matter regions only.

    Oligodendrocyte nuclei in white matter are noticeably elongated parallel to
    the local axon bundles they myelinate. Each blob's long axis is aligned
    with the local tract direction (perpendicular to the gradient of the WM
    mask's distance transform), with random per-blob jitter on top.
    """

    def __init__(
        self,
        p: float = 1.0,
        density_range_per_mm2: tuple[float, float] = (200.0, 600.0),
        aspect_ratio_range: tuple[float, float] = (3.0, 10.0),
        orientation_jitter_rad: float = 0.4,
        intensity_range: tuple[float, float] = (0.25, 0.7),
    ) -> None:
        self.p = p
        self.density_range_per_mm2 = density_range_per_mm2
        self.aspect_ratio_range = aspect_ratio_range
        self.orientation_jitter_rad = orientation_jitter_rad
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

        wm_mask = _resolve_class_mask(ctx, "white_matter")
        if wm_mask is None or not wm_mask.any():
            return image

        h, w = image.shape[:2]
        density = _get_density(ctx, (h, w))
        px_um = _pixel_size(ctx)

        # Per-image density draw — sometimes WM is faint, sometimes dense,
        # so the trained model has to recognize tracts even when low-contrast.
        d_lo, d_hi = self.density_range_per_mm2
        density_per_mm2 = float(rng.uniform(d_lo, d_hi))

        n_target = _expected_count(density_per_mm2, px_um, h, w)
        ys, xs = _sample_points(density, n_target, _MAX_BLOBS_DAPI_WM, rng, wm_mask)
        n = len(ys)
        if n == 0:
            return image

        sigma_scale = 10.0 / px_um
        sigmas_short = rng.uniform(
            0.5 * sigma_scale, 0.8 * sigma_scale, size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        ar = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * ar).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad, self.orientation_jitter_rad, size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        blue_dom = rng.uniform(0.75, 0.95, size=n).astype(np.float32)
        rg_bleed = rng.uniform(0.0, 0.10, size=n).astype(np.float32)
        channel_weights = np.stack(
            [rg_bleed, rg_bleed * 0.5, blue_dom], axis=1,
        ).astype(np.float32)

        canvas = np.zeros_like(image)
        _splat_anisotropic_blobs(
            canvas, ys, xs, sigmas_long, sigmas_short, thetas,
            intensities, channel_weights, wm_mask,
        )
        return np.clip(image + canvas, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Nissl gray / white matter cell bodies
# ---------------------------------------------------------------------------


def _nissl_cell_colors(
    rng: np.random.Generator, n: int,
    *,
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.30, 0.55), (0.25, 0.50), (0.50, 0.75),
    ),
) -> np.ndarray:
    """Per-blob target cell color (R, G, B) for alpha compositing.

    Default range covers blue-violet → purple-violet — both common Nissl
    presentations depending on stain (cresyl violet vs thionine).
    """
    (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi) = color_range
    return np.stack([
        rng.uniform(r_lo, r_hi, size=n).astype(np.float32),
        rng.uniform(g_lo, g_hi, size=n).astype(np.float32),
        rng.uniform(b_lo, b_hi, size=n).astype(np.float32),
    ], axis=1).astype(np.float32)


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
            small_lo * sigma_scale, small_hi * sigma_scale, size=int((~is_large).sum()),
        )
        sigmas[is_large] = rng.uniform(
            large_lo * sigma_scale, large_hi * sigma_scale, size=int(is_large.sum()),
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
            s_lo * sigma_scale, s_hi * sigma_scale, size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        ar = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * ar).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad, self.orientation_jitter_rad, size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _nissl_cell_colors(
            rng, n,
            color_range=((0.40, 0.55), (0.40, 0.55), (0.60, 0.78)),
        )

        out = image.copy()
        _splat_alpha_anisotropic_blobs(
            out, ys, xs, sigmas_long, sigmas_short, thetas,
            intensities, cell_colors, wm_mask,
        )
        return out


# ---------------------------------------------------------------------------
# Brightfield IHC (DAB) gray / white matter
# ---------------------------------------------------------------------------


def _dab_channel_weights(
    rng: np.random.Generator, n: int, tan: tuple[float, float, float],
) -> np.ndarray:
    """(DAB - tan) deltas for subtractive compositing.

    DAB chromogen reads as warm brown — high R, mid G, low B. Subtraction-mode
    splatting drops cream/tan substrate toward this brown.
    """
    dab_r = rng.uniform(0.40, 0.55, size=n).astype(np.float32)
    dab_g = rng.uniform(0.22, 0.34, size=n).astype(np.float32)
    dab_b = rng.uniform(0.10, 0.20, size=n).astype(np.float32)
    return np.stack(
        [dab_r - tan[0], dab_g - tan[1], dab_b - tan[2]], axis=1,
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
        "pan_neuronal":         (1500.0, 2500.0),
        "sparse_interneuron":   (60.0, 200.0),
        "myelin":               (40.0, 120.0),
    }
    _MODE_INTENSITY: dict[str, tuple[float, float]] = {
        "pan_neuronal":         (0.55, 0.90),
        "sparse_interneuron":   (0.65, 0.95),
        "myelin":               (0.20, 0.40),
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
        "pan_neuronal":         (50.0, 150.0),
        "sparse_interneuron":   (0.0, 30.0),
        "myelin":               (1500.0, 2500.0),
    }
    _MODE_INTENSITY: dict[str, tuple[float, float]] = {
        "pan_neuronal":         (0.20, 0.40),
        "sparse_interneuron":   (0.20, 0.40),
        "myelin":               (0.55, 0.90),
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
            tuple[float, float], tuple[float, float], tuple[float, float],
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
            puncta_density, n_target, _MAX_BLOBS_ISH * 4, rng, binary_expr,
        )
        n = len(ys)
        if n == 0:
            return out

        s_lo, s_hi = self.sigma_range_scale
        sigmas = rng.uniform(s_lo * sigma_scale, s_hi * sigma_scale, size=n).astype(np.float32)
        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi) = self.punctum_color_range
        cell_colors = np.stack([
            rng.uniform(r_lo, r_hi, size=n).astype(np.float32),
            rng.uniform(g_lo, g_hi, size=n).astype(np.float32),
            rng.uniform(b_lo, b_hi, size=n).astype(np.float32),
        ], axis=1).astype(np.float32)

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
        channel_weights = np.stack([rg_bleed, rg_bleed * 0.5, blue_dom], axis=1).astype(
            np.float32
        )

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
        (0.20, 0.40), (0.20, 0.45), (0.50, 0.75),
    ),
) -> np.ndarray:
    """Per-blob target cell color (R, G, B) for hematoxylin nuclear staining.

    Default range covers blue-violet → steel-blue — typical hematoxylin-stained
    nucleus appearance when used as a counterstain in IHC or chromogenic ISH.
    Hematoxylin stains chromatin (nuclei only), so color is confined to small,
    sharply delineated dots.
    """
    (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi) = color_range
    return np.stack([
        rng.uniform(r_lo, r_hi, size=n).astype(np.float32),
        rng.uniform(g_lo, g_hi, size=n).astype(np.float32),
        rng.uniform(b_lo, b_hi, size=n).astype(np.float32),
    ], axis=1).astype(np.float32)


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
            tuple[float, float], tuple[float, float], tuple[float, float],
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
            s_lo * sigma_scale, s_hi * sigma_scale, size=n,
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
            tuple[float, float], tuple[float, float], tuple[float, float],
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
            s_lo * sigma_scale, s_hi * sigma_scale, size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        ar = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * ar).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad, self.orientation_jitter_rad, size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _hematoxylin_cell_colors(rng, n, color_range=self.color_range)

        out = image.copy()
        _splat_alpha_anisotropic_blobs(
            out, ys, xs, sigmas_long, sigmas_short, thetas,
            intensities, cell_colors, wm_mask,
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
    return np.stack([
        rng.uniform(*r_range, size=n).astype(np.float32),
        rng.uniform(*g_range, size=n).astype(np.float32),
        rng.uniform(*b_range, size=n).astype(np.float32),
    ], axis=1).astype(np.float32)


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
            s_lo * sigma_scale, s_hi * sigma_scale, size=n,
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
            s_lo * sigma_scale, s_hi * sigma_scale, size=n,
        ).astype(np.float32)
        ar_lo, ar_hi = self.aspect_ratio_range
        aspect = rng.uniform(ar_lo, ar_hi, size=n).astype(np.float32)
        sigmas_long = (sigmas_short * aspect).astype(np.float32)

        theta_field = _get_or_compute_tract_orientation(ctx, wm_mask)
        thetas = theta_field[ys, xs] + rng.uniform(
            -self.orientation_jitter_rad, self.orientation_jitter_rad, size=n,
        ).astype(np.float32)

        intensities = rng.uniform(*self.intensity_range, size=n).astype(np.float32)
        cell_colors = _nfr_cell_colors(rng, n)

        out = image.copy()
        _splat_alpha_anisotropic_blobs(
            out, ys, xs, sigmas_long, sigmas_short, thetas,
            intensities, cell_colors, wm_mask,
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
