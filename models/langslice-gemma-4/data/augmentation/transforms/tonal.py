"""Per-modality color-space / tonal mapping transforms.

Each class converts an atlas slice (colored region map or grayscale template)
into a plausible background tone for one histology modality.  Texture (nuclei /
cell-body synthesis) is composed on top in a later wave.

Input convention: HWC float32 in [0, 1].
Output convention: HWC float32 in [0, 1], same spatial shape.
"""

from __future__ import annotations

__all__ = [
    "DAPITonal",
    "NisslTonal",
    "BrightfieldTonal",
    "FluorescenceTonal",
    "ISHTonal",
]

import numpy as np
from augmentation.transforms.base import TransformContext
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _luminance(image: np.ndarray) -> np.ndarray:
    """Perceptual luminance (HW float32) from HWC float32 image."""
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]


def _infer_tissue_mask(image: np.ndarray) -> np.ndarray:
    """Derive a binary tissue mask from luminance when none is supplied."""
    lum = _luminance(image)
    return lum > 0.02


def _density_or_uniform(ctx: TransformContext, h: int, w: int) -> np.ndarray:
    """Return density_map if available, else uniform 0.5."""
    if ctx.density_map is not None:
        return ctx.density_map.astype(np.float32)
    return np.full((h, w), 0.5, dtype=np.float32)


def _tissue_mask(image: np.ndarray, ctx: TransformContext) -> np.ndarray:
    if ctx.tissue_mask is not None:
        return ctx.tissue_mask
    return _infer_tissue_mask(image)


def _jitter(
    rng: np.random.Generator, target: tuple[float, float, float], spread: float = 0.05
) -> np.ndarray:
    """Apply per-call ±spread jitter to a target RGB colour."""
    delta = rng.uniform(-spread, spread, size=3).astype(np.float32)
    return np.clip(np.array(target, dtype=np.float32) + delta, 0.0, 1.0)


def _soft_mask(density_map: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    """Threshold density and smooth into a soft boundary mask."""
    hard = (density_map > threshold).astype(np.float32)
    return gaussian_filter(hard, sigma=sigma)


# ---------------------------------------------------------------------------
# DAPITonal
# ---------------------------------------------------------------------------


class DAPITonal:
    """Atlas slice → DAPI tonal layer.

    Dark background; blue ramp in tissue; density-modulated channel brightness.
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

        h, w, _ = image.shape
        lum = _luminance(image)
        mask = _tissue_mask(image, ctx)
        density = _density_or_uniform(ctx, h, w)

        bg_color = _jitter(rng, (0.02, 0.02, 0.06))
        out = np.stack([
            np.full((h, w), bg_color[0], dtype=np.float32),
            np.full((h, w), bg_color[1], dtype=np.float32),
            np.full((h, w), bg_color[2], dtype=np.float32),
        ], axis=-1)

        # Blue ramp: R/G kept low, B = luminance × 0.6 with cyan in mid-tones.
        b_channel = lum * 0.60
        # Subtle cyan: lift G slightly proportional to mid-tone luminance.
        # Luminance peaks at ~0.5 get the most G, darker/brighter get less.
        midtone_weight = 1.0 - np.abs(lum - 0.45) * 2.5
        midtone_weight = np.clip(midtone_weight, 0.0, 1.0)
        g_channel = lum * 0.10 * midtone_weight
        r_channel = lum * 0.04

        # Density brightening: high density → brighter blue; zero → near-black.
        b_channel = b_channel * (1.0 + 0.3 * density)
        # Ventricle-like zero-density: push blue toward zero.
        b_channel = b_channel * np.where(density < 0.05, density / 0.05, 1.0)

        tissue_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        tissue_rgb = np.clip(tissue_rgb, 0.0, 1.0)

        tissue_3 = mask[:, :, np.newaxis]
        out = out * (1.0 - tissue_3) + tissue_rgb * tissue_3

        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# NisslTonal
# ---------------------------------------------------------------------------


class NisslTonal:
    """Atlas slice → Nissl tonal layer.

    Cream background; inverted-intensity purple/pink tissue; density-saturated
    purple; Gaussian-softened region edges.
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

        h, w, _ = image.shape
        lum = _luminance(image)
        mask = _tissue_mask(image, ctx)
        density = _density_or_uniform(ctx, h, w)

        # Soften atlas boundaries biologically.
        soft_density = gaussian_filter(density * mask.astype(np.float32), sigma=1.0)

        bg_color = _jitter(rng, (0.95, 0.93, 0.88))
        stain_color = _jitter(rng, (0.45, 0.25, 0.55))

        # Invert intensity: darker input → more stain absorbed → darker output.
        stain_intensity = 1.0 - lum

        # Blend between background cream and purple-pink based on stain + density.
        density_weight = np.clip(soft_density * 1.5, 0.0, 1.0)
        # tissue regions start at stain_intensity blend; density enriches saturation.
        blend_weight = stain_intensity * 0.6 + density_weight * 0.4
        blend_weight = np.clip(blend_weight, 0.0, 1.0)

        out = np.stack([
            bg_color[0] * (1.0 - blend_weight) + stain_color[0] * blend_weight,
            bg_color[1] * (1.0 - blend_weight) + stain_color[1] * blend_weight,
            bg_color[2] * (1.0 - blend_weight) + stain_color[2] * blend_weight,
        ], axis=-1)

        # Background pixels snap back to cream.
        mask_3 = mask[:, :, np.newaxis].astype(np.float32)
        bg_rgb = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        out = out * mask_3 + bg_rgb * (1.0 - mask_3)

        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# BrightfieldTonal
# ---------------------------------------------------------------------------


class BrightfieldTonal:
    """Atlas slice → brightfield IHC tonal layer.

    Beige-cream tissue; brown DAB-like signal in a randomly chosen target region;
    random radial vignette at corners.
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

        h, w, _ = image.shape
        lum = _luminance(image)
        mask = _tissue_mask(image, ctx)
        density = _density_or_uniform(ctx, h, w)

        bg_color = _jitter(rng, (0.93, 0.90, 0.83))
        tissue_base = _jitter(rng, (0.87, 0.84, 0.77))
        brown_color = _jitter(rng, (0.40, 0.25, 0.15))

        # Random target region: pixels where density exceeds a random threshold.
        threshold = rng.uniform(0.35, 0.70)
        target_mask = (density > threshold) & mask
        # Small random subset of high-density region (random secondary threshold).
        if target_mask.sum() > 0:
            sub_threshold = rng.uniform(0.0, 1.0)
            random_field = rng.random((h, w)).astype(np.float32)
            target_mask = target_mask & (random_field > sub_threshold)

        # Tissue base slightly darker than background; modulated by luminance.
        tissue_rgb = np.stack([
            tissue_base[0] - lum * 0.05,
            tissue_base[1] - lum * 0.05,
            tissue_base[2] - lum * 0.05,
        ], axis=-1)

        # Brown modulation in target region.
        brown_strength = np.clip(density * 0.6, 0.0, 1.0)
        brown_rgb = np.array(brown_color, dtype=np.float32).reshape(1, 1, 3)
        brown_mod = brown_strength[:, :, np.newaxis]
        target_3 = target_mask[:, :, np.newaxis].astype(np.float32) * brown_mod
        tissue_rgb = tissue_rgb * (1.0 - target_3) + brown_rgb * target_3

        mask_3 = mask[:, :, np.newaxis].astype(np.float32)
        bg_rgb = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        out = tissue_rgb * mask_3 + bg_rgb * (1.0 - mask_3)

        # Radial vignette: ~10% darkening at corners.
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
        radius = np.sqrt(ys**2 + xs**2)
        vignette_strength = rng.uniform(0.05, 0.12)
        vignette = 1.0 - vignette_strength * np.clip(radius - 0.6, 0.0, None) / 0.4
        out = out * vignette[:, :, np.newaxis]

        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# FluorescenceTonal
# ---------------------------------------------------------------------------


class FluorescenceTonal:
    """Atlas slice → multi-channel fluorescence tonal layer.

    Very dark background; DAPI (blue) always present; optional green (p=0.5)
    and red (p=0.4) fluorescent channels driven by random expression masks.
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

        h, w, _ = image.shape
        lum = _luminance(image)
        mask = _tissue_mask(image, ctx)
        density = _density_or_uniform(ctx, h, w)

        bg_color = _jitter(rng, (0.01, 0.01, 0.02), spread=0.01)
        out = np.stack([
            np.full((h, w), bg_color[0], dtype=np.float32),
            np.full((h, w), bg_color[1], dtype=np.float32),
            np.full((h, w), bg_color[2], dtype=np.float32),
        ], axis=-1)

        # DAPI channel (blue): always present, density-modulated.
        dapi_scale = rng.uniform(0.40, 0.65)
        dapi_b = lum * dapi_scale * (1.0 + 0.25 * density) * mask.astype(np.float32)
        out[:, :, 2] = np.clip(out[:, :, 2] + dapi_b, 0.0, 1.0)

        # Green channel (e.g., GFP): p=0.5, sparse expression mask.
        if rng.random() < 0.5:
            expr_threshold = rng.uniform(0.50, 0.80)
            expr_mask = (density > expr_threshold) & mask
            if expr_mask.sum() > 0:
                # Further sparsify with a random sub-field.
                sparse_field = rng.random((h, w)).astype(np.float32)
                expr_mask = expr_mask & (sparse_field > rng.uniform(0.4, 0.7))
            green_scale = rng.uniform(0.30, 0.60)
            green_signal = expr_mask.astype(np.float32) * green_scale * density
            out[:, :, 1] = np.clip(out[:, :, 1] + green_signal, 0.0, 1.0)

        # Red channel (e.g., tdTomato): p=0.4, even sparser, different mask.
        if rng.random() < 0.4:
            expr_threshold_r = rng.uniform(0.60, 0.85)
            expr_mask_r = (density > expr_threshold_r) & mask
            if expr_mask_r.sum() > 0:
                sparse_field_r = rng.random((h, w)).astype(np.float32)
                expr_mask_r = expr_mask_r & (sparse_field_r > rng.uniform(0.55, 0.80))
            red_scale = rng.uniform(0.20, 0.50)
            red_signal = expr_mask_r.astype(np.float32) * red_scale * density
            out[:, :, 0] = np.clip(out[:, :, 0] + red_signal, 0.0, 1.0)

        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# ISHTonal
# ---------------------------------------------------------------------------


class ISHTonal:
    """Atlas slice → ISH tonal layer.

    Light gray-green background; brown stain in a random expressed region that
    falls off smoothly with distance from a randomly chosen high-density centre;
    region edges softened with Gaussian.
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

        h, w, _ = image.shape
        mask = _tissue_mask(image, ctx)
        density = _density_or_uniform(ctx, h, w)

        # Soften atlas region boundaries for ISH-like diffusion.
        soft_density = gaussian_filter(density * mask.astype(np.float32), sigma=0.7)

        bg_color = _jitter(rng, (0.86, 0.88, 0.83))
        brown_color = _jitter(rng, (0.45, 0.30, 0.15))

        bg_rgb = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        out = np.broadcast_to(bg_rgb, (h, w, 3)).copy()

        # Pick a random expressed region: locate a high-density candidate centre.
        high_density_mask = soft_density > 0.50
        if high_density_mask.sum() > 0:
            ys_hd, xs_hd = np.where(high_density_mask)
            pick_idx = int(rng.integers(0, len(ys_hd)))
            cy, cx = float(ys_hd[pick_idx]), float(xs_hd[pick_idx])
        else:
            cy, cx = h / 2.0, w / 2.0

        # Distance-based fall-off from the chosen centre.
        ys_grid = np.arange(h, dtype=np.float32)[:, np.newaxis]
        xs_grid = np.arange(w, dtype=np.float32)[np.newaxis, :]
        dist = np.sqrt((ys_grid - cy)**2 + (xs_grid - cx)**2)
        # Sigma roughly 15–35% of the shorter image dimension.
        sigma_px = rng.uniform(0.15, 0.35) * min(h, w)
        fall_off = np.exp(-0.5 * (dist / max(sigma_px, 1.0))**2).astype(np.float32)

        # Brown stain modulated by fall-off, density, and tissue mask.
        stain_weight = fall_off * np.clip(soft_density * 1.2, 0.0, 1.0) * mask.astype(np.float32)
        stain_weight = stain_weight[:, :, np.newaxis]

        brown_rgb = np.array(brown_color, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1.0 - stain_weight) + brown_rgb * stain_weight

        return np.clip(out, 0.0, 1.0).astype(np.float32)
