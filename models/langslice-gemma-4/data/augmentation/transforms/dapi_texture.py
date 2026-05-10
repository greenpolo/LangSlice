"""GT-calibrated DAPI texture renderer (supersampled).

Renders cells at 4× the production resolution (~6.25 µm/px) where individual
nuclei are 1-2 px and can be splatted as small Gaussians, then Lanczos-
downsamples to the production canvas (25 µm/px). This is mathematically
equivalent to how a real microscope camera bins photons from many sub-pixel
sources — it produces the smooth, integrated texture of real DAPI rather
than the speckle of per-pixel Poisson statistics.

Pipeline per region:
1. Sample N cell positions inside the supersampled mask, density-modulated.
2. Splat (np.add.at) each cell's intensity into a hi-res accumulator.
3. Convolve with a small Gaussian (cell-shape kernel ~1 px sigma at hi-res).
4. Lanczos-downsample to production resolution.
5. Add bright clusters (multi-cell autofluor / debris) as larger splats.
6. Add low-frequency autofluor floor + sensor noise.
7. Composite into canvas (max blend) inside the region mask.

Calibrated against GT DAPI screenshots in
``models/langslice-gemma-4/data/augmentation/GT_Textures/DAPI/`` at full
zoomout (25 µm/px target):

    Region   mean   med    p95    max
    GM       0.205  0.200  0.345  0.898
    WM       0.229  0.212  0.423  0.999

Usage
-----
``render_dapi_region_texture`` paints DAPI texture into a canvas inside a
boolean region mask. Call it once per region (GM, WM, dense layers) from
a wrapping ``Transform`` class so the existing pipeline contract is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class DAPIRegionParams:
    """Parameters for per-region DAPI texture rendered at supersampled scale.

    Cell density and intensity are specified in physical units (cells/mm²,
    [0, 1] intensity) so they're independent of the supersample factor.
    The renderer translates to hi-res pixels and downsamples to production.

    Attributes
    ----------
    cells_per_mm2:
        Nucleus density. Cortex GM ≈ 5500; mouse WM ≈ 1500-2500
        (oligodendrocyte-dominated); dense cell layers ≈ 14000-25000.
        Density is the upper bound; modulated per pixel by ``density_map``.
    cell_radius_um_range:
        (min, max) cell radius in µm. Real DAPI nuclei are 3.5-5 µm radius
        (7-10 µm diameter). Mouse cortex slightly smaller, dense layers
        slightly larger.
    base_cell_intensity:
        Mean per-cell peak brightness (in [0, 1] at hi-res). After Gaussian
        splat the integral of intensity equals base * 2π * sigma² .
    cell_tail_scale:
        Heavy-tail scale for per-cell intensity (gamma distribution).
        Larger = more variance, occasional much-brighter cells.
    bright_cluster_per_mm2:
        Density of bright multi-cell events (cell clumps / dividing cells /
        autofluor debris). Rendered as larger splats than individual cells.
    bright_cluster_radius_um_range:
        (min, max) radius in µm of bright clusters. 8-20 µm clusters.
    bright_cluster_intensity_range:
        Peak intensity range for bright clusters. Top end can saturate.
    autofluor_floor:
        Low-frequency tissue background intensity confined to the mask.
        Real DAPI has a non-zero floor inside tissue from autofluor /
        out-of-focus contributions.
    autofluor_lf_sigma_frac:
        Spatial smoothness of the autofluor floor (fraction of region's
        max dimension). 0.10 → ~10% scale variation.
    bg_noise_sigma:
        Per-pixel additive Gaussian noise (sensor read-noise mimic), at
        production resolution.
    psf_sigma_px:
        Final PSF blur applied at production resolution to soften aliasing.
    aniso_sigma_long_px:
        For white matter only: long-axis sigma of an anisotropic blur
        kernel oriented along the local tract direction (production res).
        Set 0 for GM.
    aniso_sigma_short_px:
        Short-axis sigma of the anisotropic blur kernel.
    supersample:
        Render at supersample × production resolution, then Lanczos-
        downsample. 4 = render at ~6.25 µm/px for a 25 µm/px production
        canvas — cells are 1-2 px and texture integrates correctly.
    """

    cells_per_mm2: float
    cell_radius_um_range: tuple[float, float]
    base_cell_intensity: float
    cell_tail_scale: float
    bright_cluster_per_mm2: float
    bright_cluster_radius_um_range: tuple[float, float]
    bright_cluster_intensity_range: tuple[float, float]
    autofluor_floor: float
    autofluor_lf_sigma_frac: float = 0.10
    bg_noise_sigma: float = 0.005
    psf_sigma_px: float = 0.5
    aniso_sigma_long_px: float = 0.0
    aniso_sigma_short_px: float = 0.0
    directional_noise_amp: float = 0.0
    supersample: int = 4


# Cortex GM: ~5500 nuclei/mm² at 7-10 µm diameter. Most cells overlap in
# the supersampled buffer and integrate into a smooth diffuse texture
# after Lanczos downsample — same physics as a real microscope camera
# binning sub-pixel sources.
DAPI_GM_PARAMS = DAPIRegionParams(
    cells_per_mm2=5500.0,
    cell_radius_um_range=(3.5, 5.0),
    base_cell_intensity=0.32,
    cell_tail_scale=0.28,
    bright_cluster_per_mm2=14.0,
    bright_cluster_radius_um_range=(8.0, 16.0),
    bright_cluster_intensity_range=(0.55, 0.95),
    autofluor_floor=0.04,
    autofluor_lf_sigma_frac=0.22,
)

DAPI_WM_PARAMS = DAPIRegionParams(
    # WM nucleus density and per-cell intensity match GM — supersampled
    # rendering means cells are tiny sub-pixel-after-downsample dots
    # whose density integrates into a smooth field. The "chain" character
    # comes from two oriented passes layered on top:
    #   1. Per-pixel oriented blur smears the cell field along the local
    #      tangent — preserves total brightness, redistributes peaks into
    #      streaks following the WM region's shape.
    #   2. Directional noise overlay generates visible chain striations
    #      at production resolution (sub-pixel chains otherwise average
    #      to a smooth band).
    # Slightly less dense than GM since WM is oligodendrocyte-rich but
    # not as cell-dense as cortex.
    cells_per_mm2=4500.0,
    cell_radius_um_range=(3.0, 4.5),
    base_cell_intensity=0.30,
    cell_tail_scale=0.26,
    bright_cluster_per_mm2=14.0,
    bright_cluster_radius_um_range=(6.0, 14.0),
    bright_cluster_intensity_range=(0.50, 0.90),
    autofluor_floor=0.04,
    autofluor_lf_sigma_frac=0.20,
    # Per-pixel oriented blur on the cell canvas — light touch (5:1 ratio,
    # short total spread). Goal is subtle directional bias of the cell
    # field, not a heavy smear. Heavier ratios destroy cell-level detail
    # and make WM read as out-of-focus.
    aniso_sigma_long_px=1.5,
    aniso_sigma_short_px=0.3,
    # Directional noise overlay — generates visible chain striations at
    # production resolution. Without this, sub-pixel chains average to a
    # smooth band. Amplitude tuned so streaks are visible but the WM
    # never reads brighter than surrounding GM.
    directional_noise_amp=0.45,
)

# Dense layers (DG-sg, CA pyramidal, MOB-gl, induseum griseum) — tight-packed
DAPI_DENSE_LAYER_PARAMS = DAPIRegionParams(
    cells_per_mm2=18000.0,
    cell_radius_um_range=(3.5, 5.0),
    base_cell_intensity=0.65,
    cell_tail_scale=0.55,
    bright_cluster_per_mm2=60.0,
    bright_cluster_radius_um_range=(8.0, 18.0),
    bright_cluster_intensity_range=(0.75, 1.0),
    autofluor_floor=0.18,  # raised floor — uniformly bright bands
    autofluor_lf_sigma_frac=0.15,
)


def _rotated_anisotropic_blur(
    arr: np.ndarray, sigma_long: float, sigma_short: float, angle: float
) -> np.ndarray:
    """Apply oriented anisotropic Gaussian blur."""
    s_max = max(sigma_long, sigma_short)
    rad = max(1, int(np.ceil(3 * s_max)))
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1].astype(np.float32)
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    xr = cos_t * xx + sin_t * yy
    yr = -sin_t * xx + cos_t * yy
    kernel = np.exp(-0.5 * ((xr / sigma_long) ** 2 + (yr / sigma_short) ** 2))
    kernel /= kernel.sum()
    return fftconvolve(arr, kernel, mode="same").astype(np.float32)


def _per_pixel_oriented_blur(
    arr: np.ndarray,
    theta_field: np.ndarray,
    mask: np.ndarray,
    sigma_long: float,
    sigma_short: float,
    n_bins: int = 8,
) -> np.ndarray:
    """Per-pixel anisotropic blur with long axis aligned to theta_field.

    Uses orientation-binned steered convolution: blurs the canvas at N
    discrete orientations, then per-pixel blends them weighted by
    cos²(θ - bin_angle). The cosine basis at evenly-spaced orientations
    sums to n_bins/2 everywhere — no division needed for normalization.

    Cost: N FFT convolutions on the bounding-box-cropped region. The
    bbox crop is what keeps WM blur cheap (WM is ~5-10% of image area).

    Parameters
    ----------
    arr : (H, W) float32 — canvas to blur (production resolution).
    theta_field : (H, W) float32 — per-pixel tangent angle (radians, mod π).
    mask : (H, W) bool — only blur inside this mask; outside is left alone.
    sigma_long, sigma_short : Gaussian sigmas along/across the tangent.
    n_bins : Number of discrete orientations sampled (8 = good).
    """
    if sigma_long <= 0 or not mask.any():
        return arr

    # Crop to bbox + padding for speed
    pad = max(int(np.ceil(3 * sigma_long)), 4)
    ys, xs = np.where(mask)
    y0 = max(0, ys.min() - pad)
    y1 = min(arr.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(arr.shape[1], xs.max() + 1 + pad)

    sub_arr = arr[y0:y1, x0:x1]
    sub_theta = theta_field[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]

    bin_angles = np.linspace(0.0, np.pi, n_bins, endpoint=False).astype(np.float32)
    norm = n_bins * 0.5  # sum of cos²(θ-φ_k) for evenly-spaced φ_k = n_bins/2

    out_sub = np.zeros_like(sub_arr)
    for ba in bin_angles:
        w = np.cos(sub_theta - ba) ** 2  # period π — correct for orientation
        blurred = _rotated_anisotropic_blur(sub_arr, sigma_long, sigma_short, float(ba))
        out_sub += blurred * w
    out_sub = out_sub / norm

    out = arr.copy()
    out[y0:y1, x0:x1] = np.where(sub_mask, out_sub, sub_arr)
    return out


def _shape_aware_orientation(
    mask: np.ndarray, smoothing_sigma: float = 8.0
) -> np.ndarray:
    """Per-pixel tangent angle field from a region's distance transform.

    For elongated regions (corpus callosum) the tangent runs along the
    long axis; for round regions (anterior commissure) it forms circular
    flow parallel to the boundary. Both fall out of the same algorithm:
    tangent = perpendicular to gradient of the heavily-smoothed distance-
    from-boundary transform.

    Why heavy smoothing:
    - Inside a thick region (e.g. corpus callosum ~10 px thick at
      25 µm/px), the unsmoothed distance transform has a "medial axis
      ridge" where the gradient is zero — orientation becomes unstable.
    - Smoothing with sigma ≥ region thickness / 2 averages over the
      ridge, giving a well-defined gradient direction perpendicular to
      the region's long axis.

    Default sigma=8 px (200 µm at 25 µm/px) covers most WM tracts
    cleanly. Pure thin tracts get a stable along-tract tangent; thick
    regions get the principal axis (perpendicular to the long axis).
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    dt = distance_transform_edt(mask).astype(np.float32)
    if smoothing_sigma > 0:
        dt = gaussian_filter(dt, sigma=smoothing_sigma)
    gy, gx = np.gradient(dt)
    # Tangent = perpendicular to gradient (rotate 90°).
    return np.arctan2(-gx, gy).astype(np.float32)


def _splat_cells_supersampled(
    H: int,
    W: int,
    mask_hi: np.ndarray,
    n_cells: int,
    radius_px_range: tuple[float, float],
    intensity_mean: float,
    intensity_tail: float,
    rng: np.random.Generator,
    density_map_hi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Splat n_cells small Gaussians into a hi-res accumulator.

    Returns (H, W) float32. Cells are sampled inside ``mask_hi``. Each cell
    is a Gaussian with sigma in radius_px_range and intensity from
    gamma(shape=1, scale=intensity_mean + intensity_tail). Splats are added
    via per-cell-radius 2D Gaussian deposition through impulse + convolution.
    """
    canvas = np.zeros((H, W), dtype=np.float32)
    if n_cells <= 0:
        return canvas

    # Sample positions inside mask, optionally weighted by density.
    ys_idx, xs_idx = np.where(mask_hi)
    if ys_idx.size == 0:
        return canvas
    if density_map_hi is not None:
        weights = density_map_hi[ys_idx, xs_idx].astype(np.float32)
        wsum = float(weights.sum())
        if wsum > 0:
            weights = weights / wsum
            sel = rng.choice(ys_idx.size, size=n_cells, replace=True, p=weights)
        else:
            sel = rng.integers(0, ys_idx.size, size=n_cells)
    else:
        sel = rng.integers(0, ys_idx.size, size=n_cells)
    ys = ys_idx[sel]
    xs = xs_idx[sel]

    # Per-cell intensities (heavy-tail gamma)
    intens = rng.gamma(
        shape=1.0,
        scale=intensity_mean + intensity_tail,
        size=n_cells,
    ).astype(np.float32)

    r_min, r_max = radius_px_range
    # Bin cells into a small set of radius buckets so we splat each bucket
    # once with a single convolution kernel instead of per-cell loops.
    n_buckets = 4
    edges = np.linspace(r_min, r_max, n_buckets + 1)
    bucket_idx = np.clip(
        np.floor((rng.uniform(r_min, r_max, size=n_cells) - r_min)
                 / max(r_max - r_min, 1e-6) * n_buckets).astype(np.int32),
        0, n_buckets - 1,
    )

    for b in range(n_buckets):
        sel_b = bucket_idx == b
        if not sel_b.any():
            continue
        sigma = float((edges[b] + edges[b + 1]) * 0.5)
        impulses = np.zeros((H, W), dtype=np.float32)
        # add intensities at integer positions
        np.add.at(impulses, (ys[sel_b], xs[sel_b]), intens[sel_b])
        if sigma > 0.05:
            impulses = gaussian_filter(impulses, sigma=sigma).astype(np.float32)
        canvas += impulses

    return canvas


def _downsample_lanczos(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Lanczos-downsample a single-channel float [0,1] array to target shape."""
    from PIL import Image

    a = np.clip(arr, 0.0, 1.0)
    pil = Image.fromarray((a * 65535.0).astype(np.uint16), mode="I;16")
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.asarray(pil).astype(np.float32) / 65535.0


def render_dapi_region_texture(
    canvas: np.ndarray,
    mask: np.ndarray,
    params: DAPIRegionParams,
    *,
    rng: np.random.Generator,
    pixel_size_um: float,
    density_map: Optional[np.ndarray] = None,
    tract_orientation: Optional[np.ndarray] = None,
    channel: int = 2,
) -> np.ndarray:
    """Paint DAPI texture in ``canvas[mask]`` via supersampled rendering.

    ``canvas`` is HWC float32 [0, 1]. Returns the same array with the
    region's blue channel updated (max-blended to avoid additive saturation).

    Renders at ``params.supersample`` × the production resolution, where
    individual cells are 1-2 px and can be splatted as small Gaussians.
    Then Lanczos-downsamples to the production canvas — the integration
    of overlapping sub-pixel cells produces the smooth, fine-grained
    texture of real DAPI.

    Parameters
    ----------
    canvas:
        HWC float32 [0, 1] production canvas.
    mask:
        HW bool — pixels to paint into.
    params:
        Calibrated region parameters (DAPI_GM_PARAMS, DAPI_WM_PARAMS, etc.).
    rng:
        Numpy RNG for determinism.
    pixel_size_um:
        Production render resolution. Default pipeline = 25.0.
    density_map:
        Optional HW [0, 1] modulation map. Cell density scales linearly
        with this. None = uniform (1.0 everywhere).
    tract_orientation:
        Optional HW float32 (radians) per-pixel tract angle. If supplied
        and ``params.aniso_sigma_long_px > 0``, anisotropic blur uses the
        field's mean angle (per-pixel rotation is too slow for production).
    channel:
        RGB channel index to paint into. DAPI = 2 (blue).
    """
    h, w = canvas.shape[:2]
    if not mask.any():
        return canvas

    s = max(1, int(params.supersample))
    H, W = h * s, w * s
    px_um_hi = pixel_size_um / s
    um_to_px_hi = 1.0 / px_um_hi
    px_area_mm2_hi = (px_um_hi / 1000.0) ** 2

    # Hi-res mask (nearest-neighbour zoom — preserves binary edges).
    if s == 1:
        mask_hi = mask.astype(bool)
    else:
        mask_hi = np.repeat(np.repeat(mask.astype(bool), s, axis=0), s, axis=1)

    # Hi-res density map (optional)
    density_hi = None
    if density_map is not None:
        if s == 1:
            density_hi = density_map.astype(np.float32)
        else:
            density_hi = np.repeat(
                np.repeat(density_map.astype(np.float32), s, axis=0), s, axis=1
            )

    region_mm2 = float(mask_hi.sum()) * px_area_mm2_hi

    # 1. Cell splats — many small Gaussians at hi-res.
    n_cells = int(params.cells_per_mm2 * region_mm2)
    cell_radius_px_hi = (
        params.cell_radius_um_range[0] * um_to_px_hi,
        params.cell_radius_um_range[1] * um_to_px_hi,
    )
    hi_canvas = _splat_cells_supersampled(
        H, W, mask_hi, n_cells, cell_radius_px_hi,
        params.base_cell_intensity, params.cell_tail_scale,
        rng, density_hi,
    )

    # 2. Bright cluster splats — fewer, larger, brighter.
    n_clusters = int(params.bright_cluster_per_mm2 * region_mm2)
    if n_clusters > 0:
        cluster_radius_px_hi = (
            params.bright_cluster_radius_um_range[0] * um_to_px_hi,
            params.bright_cluster_radius_um_range[1] * um_to_px_hi,
        )
        cluster_canvas = _splat_cells_supersampled(
            H, W, mask_hi, n_clusters, cluster_radius_px_hi,
            float(np.mean(params.bright_cluster_intensity_range)),
            (params.bright_cluster_intensity_range[1]
             - params.bright_cluster_intensity_range[0]) * 0.5,
            rng, density_hi,
        )
        hi_canvas = np.maximum(hi_canvas, cluster_canvas)

    # Mask hi-res canvas before downsample so the downsample doesn't smear
    # texture beyond region boundaries.
    hi_canvas = hi_canvas * mask_hi.astype(np.float32)

    # 3. Lanczos-downsample to production resolution.
    region_canvas = _downsample_lanczos(hi_canvas, h, w)

    # 4. Anisotropic blur (WM / oriented regions, production resolution).
    # Per-pixel oriented blur using a tangent field derived from the mask's
    # distance-transform gradient — gives along-tract flow for elongated
    # regions AND circular flow for round regions, automatically. Also
    # adds a directional noise overlay so the chain texture is visible at
    # production resolution (sub-pixel chains otherwise average to a
    # smooth band).
    if params.aniso_sigma_long_px > 0:
        theta_field = (
            tract_orientation
            if tract_orientation is not None
            else _shape_aware_orientation(mask, smoothing_sigma=8.0)
        )
        region_canvas = _per_pixel_oriented_blur(
            region_canvas,
            theta_field,
            mask.astype(bool),
            params.aniso_sigma_long_px,
            params.aniso_sigma_short_px,
            n_bins=8,
        )

        # Directional noise overlay — mimics the "chain" pattern at
        # production resolution. Generate isotropic noise then apply the
        # same per-pixel oriented blur to create elongated streaks of
        # noise aligned with the local tangent. Multiply by base region
        # intensity so dim regions don't suddenly look streaky-bright.
        if params.directional_noise_amp > 0:
            noise = rng.normal(0, 1, size=(h, w)).astype(np.float32)
            noise = _per_pixel_oriented_blur(
                noise, theta_field, mask.astype(bool),
                params.aniso_sigma_long_px * 1.6,
                params.aniso_sigma_short_px * 0.7,
                n_bins=8,
            )
            noise = noise / max(float(np.abs(noise[mask.astype(bool)]).std()) * 2.0, 1e-6)
            local_intensity = np.maximum(region_canvas, params.autofluor_floor * 0.5)
            region_canvas = region_canvas + noise * params.directional_noise_amp * local_intensity

    # 5. Autofluor floor (low-frequency, production resolution).
    if params.autofluor_floor > 0:
        floor_seed = rng.random((h, w)).astype(np.float32)
        floor = gaussian_filter(floor_seed, sigma=h * params.autofluor_lf_sigma_frac)
        floor /= (floor.max() + 1e-6)
        floor *= params.autofluor_floor
        region_canvas = region_canvas + floor

    # 6. Final PSF blur (sub-pixel softening).
    if params.psf_sigma_px > 0:
        region_canvas = gaussian_filter(
            region_canvas, sigma=params.psf_sigma_px
        ).astype(np.float32)

    # 7. Sensor noise.
    if params.bg_noise_sigma > 0:
        region_canvas = region_canvas + rng.normal(
            0, params.bg_noise_sigma, size=(h, w)
        ).astype(np.float32)

    region_canvas = np.clip(region_canvas, 0.0, 1.0)

    # Composite into canvas only inside mask, using max blend
    canvas[..., channel] = np.where(
        mask,
        np.maximum(canvas[..., channel], region_canvas),
        canvas[..., channel],
    )
    return canvas
