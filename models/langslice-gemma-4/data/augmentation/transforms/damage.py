"""Synthetic damage transforms: folds, tears, microbubbles, halos, debris, illumination.

All transforms follow the Wave 1 contract:
    __call__(image, *, rng, ctx) -> np.ndarray
    image: HWC float32 in [0, 1], returned image has the same shape and dtype.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (
    binary_closing,
    gaussian_filter,
    map_coordinates,
)

from .base import TransformContext

__all__ = [
    "Folds",
    "Tears",
    "Microbubbles",
    "EmbeddingHalos",
    "Debris",
    "IlluminationGradient",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_tissue_mask(image: np.ndarray) -> np.ndarray:
    """Return a boolean H×W mask of tissue pixels when ctx.tissue_mask is None.

    Luminance threshold + morphological close to fill small holes.
    """
    luminance = image.mean(axis=2) if image.ndim == 3 else image
    binary = luminance > 0.05
    # WHY: closing fills needle-thin holes caused by staining gaps without
    #      eroding the tissue boundary.
    return binary_closing(binary, iterations=3)


def _make_open_curve_points(
    h: int,
    w: int,
    n_pts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample N points along a random open curve across the image.

    Returns (N, 2) array of (row, col) coordinates.
    """
    # Parametric path: random start + end on opposite margins, wavy in between.
    axis = rng.integers(0, 2)  # 0 = horizontal sweep, 1 = vertical sweep
    if axis == 0:
        xs = np.linspace(0, w - 1, n_pts)
        mid_jitter = rng.uniform(-h * 0.3, h * 0.3, size=n_pts)
        ys = np.clip(h / 2 + mid_jitter, 0, h - 1)
        pts = np.stack([ys, xs], axis=1)
    else:
        ys = np.linspace(0, h - 1, n_pts)
        mid_jitter = rng.uniform(-w * 0.3, w * 0.3, size=n_pts)
        xs = np.clip(w / 2 + mid_jitter, 0, w - 1)
        pts = np.stack([ys, xs], axis=1)
    return pts.astype(np.float64)


def _build_displacement_field(
    h: int,
    w: int,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use thin-plate-spline RBF to build per-pixel (dr, dc) displacement maps.

    Returns (dr, dc), both H×W float64.
    """
    delta = dst_pts - src_pts  # (N, 2) displacements at control points

    rbf_r = RBFInterpolator(src_pts, delta[:, 0], kernel="thin_plate_spline", smoothing=0.0)
    rbf_c = RBFInterpolator(src_pts, delta[:, 1], kernel="thin_plate_spline", smoothing=0.0)

    grid_r, grid_c = np.mgrid[0:h, 0:w]
    query = np.stack([grid_r.ravel(), grid_c.ravel()], axis=1).astype(np.float64)

    dr = rbf_r(query).reshape(h, w)
    dc = rbf_c(query).reshape(h, w)
    return dr, dc


def _apply_displacement(image: np.ndarray, dr: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Remap image using per-pixel displacement. Returns HWC float32."""
    h, w = image.shape[:2]
    grid_r, grid_c = np.mgrid[0:h, 0:w]
    src_r = np.clip(grid_r + dr, 0, h - 1)
    src_c = np.clip(grid_c + dc, 0, w - 1)

    out_channels = []
    for ch in range(image.shape[2]):
        warped = map_coordinates(image[:, :, ch], [src_r, src_c], order=1, mode="nearest")
        out_channels.append(warped)
    return np.stack(out_channels, axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


class Folds:
    """Simulate a tissue fold via thin-plate-spline warping along a random curve.

    Control points near the fold are displaced perpendicular to the curve,
    creating a local compression band of ~5–15 px.
    """

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image

        h, w = image.shape[:2]
        n_ctrl = int(rng.integers(8, 13))
        src_pts = _make_open_curve_points(h, w, n_ctrl, rng)

        # Compute local perpendicular direction at each control point.
        tangents = np.diff(src_pts, axis=0, prepend=src_pts[:1], append=src_pts[-1:])
        # average forward and backward tangents for interior points
        tangents = (tangents[:-1] + tangents[1:]) / 2.0
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        tangents_unit = tangents / norms
        # perpendicular: rotate tangent 90 degrees
        perp = np.stack([-tangents_unit[:, 1], tangents_unit[:, 0]], axis=1)

        # Displace a random subset of control points perpendicular to the curve.
        n_displaced = max(2, n_ctrl // 2)
        displaced_idx = rng.choice(n_ctrl, size=n_displaced, replace=False)
        magnitudes = rng.uniform(5, 15, size=n_displaced)
        dst_pts = src_pts.copy()
        for i, idx in enumerate(displaced_idx):
            dst_pts[idx] += perp[idx] * magnitudes[i]

        dr, dc = _build_displacement_field(h, w, src_pts, dst_pts)
        return _apply_displacement(image, dr, dc)


# ---------------------------------------------------------------------------
# Tears
# ---------------------------------------------------------------------------


def _sample_bg_color(image: np.ndarray) -> np.ndarray:
    """Estimate slide-background color from the four image corners.

    Tears in real microscopy are voids that read as the slide background, not
    as black. Sampling the four corners gives a robust median-based estimate
    even when one corner happens to overlap tissue.
    """
    h, w = image.shape[:2]
    corner = max(20, min(h, w) // 16)
    samples = np.concatenate([
        image[:corner, :corner].reshape(-1, 3),
        image[:corner, -corner:].reshape(-1, 3),
        image[-corner:, :corner].reshape(-1, 3),
        image[-corner:, -corner:].reshape(-1, 3),
    ], axis=0)
    return np.median(samples, axis=0).astype(np.float32)


class Tears:
    """Tissue tears that read as slide-background voids.

    Real microtome sections tear at structurally weak points: ventricle walls
    (CSF spaces retract during processing, making ventricles look larger than
    the atlas predicts), tissue edges (jagged bites at the perimeter), and
    occasionally across the interior. This transform applies a random subset
    of those three patterns per call:

    - **ventricle_expansion** (preferred when ventricle pixels exist): dilate
      the ventricle mask outward by an irregular amount, filling the new
      pixels with slide-background color. Captures the "ventricles always
      look bigger than the atlas" reality of real microscopy.
    - **edge_bite**: bite small irregular regions out of the tissue boundary.
    - **interior_curve**: legacy open-curve tear, now filled with the slide
      background color rather than a hardcoded dark value.
    """

    def __init__(
        self,
        p: float = 0.5,
        n_tears_range: tuple[int, int] = (1, 3),
        ventricle_expansion_iter_range: tuple[int, int] = (2, 9),
        edge_bite_radius_range: tuple[int, int] = (4, 16),
        edge_bite_count_range: tuple[int, int] = (2, 6),
        interior_shift_range: tuple[int, int] = (3, 14),
    ) -> None:
        self.p = p
        self.n_tears_range = n_tears_range
        self.ventricle_expansion_iter_range = ventricle_expansion_iter_range
        self.edge_bite_radius_range = edge_bite_radius_range
        self.edge_bite_count_range = edge_bite_count_range
        self.interior_shift_range = interior_shift_range

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image

        bg_color = _sample_bg_color(image)
        out = image.copy()

        n_tears = int(rng.integers(
            self.n_tears_range[0], self.n_tears_range[1] + 1,
        ))

        # Choose modes per tear with availability-aware weights.
        ventricle_mask: np.ndarray | None = None
        if ctx.tissue_class_masks is not None:
            v = ctx.tissue_class_masks.get("ventricle")
            if v is not None and v.any():
                ventricle_mask = v.astype(bool)

        for _ in range(n_tears):
            modes: list[str] = ["interior", "edge_bite"]
            weights: list[float] = [0.25, 0.30]
            if ventricle_mask is not None:
                modes.append("ventricle")
                weights.append(0.55)
            wsum = sum(weights)
            probs = np.array([w / wsum for w in weights], dtype=np.float64)
            mode = str(rng.choice(modes, p=probs))

            if mode == "ventricle" and ventricle_mask is not None:
                out = self._apply_ventricle_expansion(out, ventricle_mask, bg_color, rng)
            elif mode == "edge_bite":
                out = self._apply_edge_bite(out, ctx, bg_color, rng)
            else:
                out = self._apply_interior_curve(out, bg_color, rng)

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _apply_ventricle_expansion(
        self,
        image: np.ndarray,
        ventricle_mask: np.ndarray,
        bg_color: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Dilate ventricle mask by N pixels, irregularly, fill with bg_color."""
        from scipy.ndimage import binary_dilation

        n_iter = int(rng.integers(*self.ventricle_expansion_iter_range))
        dilated = binary_dilation(ventricle_mask, iterations=n_iter)
        # Irregular boundary: keep ~70% of newly-dilated pixels (adds raggedness).
        keep = rng.random(ventricle_mask.shape) > 0.30
        new_torn = dilated & (~ventricle_mask) & keep
        image[new_torn] = bg_color
        return image

    def _apply_edge_bite(
        self,
        image: np.ndarray,
        ctx: TransformContext,
        bg_color: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Bite irregular discs from the tissue boundary."""
        from scipy.ndimage import binary_erosion

        if ctx.tissue_class_masks is not None:
            tissue = ctx.tissue_class_masks.get("tissue")
        else:
            tissue = None
        if tissue is None:
            tissue = _infer_tissue_mask(image)
        tissue = tissue.astype(bool)

        eroded = binary_erosion(tissue, iterations=2)
        boundary = tissue & (~eroded)
        bys, bxs = np.where(boundary)
        if len(bys) == 0:
            return image

        n_bites = int(rng.integers(*self.edge_bite_count_range))
        h, w = image.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        for _ in range(n_bites):
            i = int(rng.integers(0, len(bys)))
            cy, cx = int(bys[i]), int(bxs[i])
            radius = int(rng.integers(*self.edge_bite_radius_range))
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
            # Irregularize: drop ~15% of disc pixels for a ragged bite edge.
            ragged = rng.random((h, w)) > 0.15
            bite = disc & ragged
            image[bite] = bg_color
        return image

    def _apply_interior_curve(
        self,
        image: np.ndarray,
        bg_color: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Legacy curve-displacement tear filled with bg_color."""
        h, w = image.shape[:2]
        n_pts = int(rng.integers(6, 10))
        curve_pts = _make_open_curve_points(h, w, n_pts, rng)
        col_coords = curve_pts[:, 1]
        row_coords = curve_pts[:, 0]
        sort_idx = np.argsort(col_coords)
        col_sorted = col_coords[sort_idx]
        row_sorted = row_coords[sort_idx]
        all_cols = np.arange(w)
        tear_row = np.interp(all_cols, col_sorted, row_sorted).astype(int)
        tear_row = np.clip(tear_row, 0, h - 1)

        shift_px = int(rng.integers(*self.interior_shift_range))
        jitter = rng.integers(-2, 3, size=w)
        effective_shift = np.clip(shift_px + jitter, 1, shift_px + 2)

        out = image.copy()
        for col in range(w):
            s = int(effective_shift[col])
            t = int(tear_row[col])
            if t + s >= h:
                continue
            out[t + s:, col, :] = image[t:h - s, col, :]
            out[t:t + s, col, :] = bg_color
        return out


# ---------------------------------------------------------------------------
# Microbubbles
# ---------------------------------------------------------------------------


class Microbubbles:
    """Air bubbles trapped in mounting medium — refractive disc with dark rim.

    Real air-bubble appearance under brightfield / fluorescence microscopy:
      - Tissue under the bubble is severely **radially distorted** (the air
        pocket acts as a divergent lens — content appears compressed toward
        the center).
      - A **dark ring** at the bubble boundary from total internal reflection
        (the classic Becke line).
      - A small **specular highlight** inside the bubble where the light
        source reflects off the curved air–medium interface.

    Bubbles are NOT solid black; they show whatever's underneath, just
    distorted and rimmed. Size range is wider than the legacy implementation
    (8–50 px radius vs the old 5–26).
    """

    def __init__(
        self,
        p: float = 0.55,
        n_bubbles_range: tuple[int, int] = (1, 12),
        radius_range: tuple[float, float] = (8.0, 50.0),
        lens_pinch_strength: float = 0.85,
        rim_darkness: float = 0.55,
        rim_thickness_frac: float = 0.12,
        highlight_brightness: float = 0.45,
        highlight_size_frac: float = 0.18,
    ) -> None:
        self.p = p
        self.n_bubbles_range = n_bubbles_range
        self.radius_range = radius_range
        self.lens_pinch_strength = lens_pinch_strength
        self.rim_darkness = rim_darkness
        self.rim_thickness_frac = rim_thickness_frac
        self.highlight_brightness = highlight_brightness
        self.highlight_size_frac = highlight_size_frac

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image

        h, w = image.shape[:2]
        tissue = ctx.tissue_mask if ctx.tissue_mask is not None else _infer_tissue_mask(image)

        n_bubbles = int(rng.integers(
            self.n_bubbles_range[0], self.n_bubbles_range[1] + 1,
        ))
        if n_bubbles == 0:
            return image

        out = image.copy().astype(np.float32)

        for _ in range(n_bubbles):
            # Sample center; prefer tissue but allow off-tissue bubbles too
            # (mounting bubbles can land anywhere on the slide).
            center_r, center_c = h // 2, w // 2
            for _ in range(20):
                cr = int(rng.integers(0, h))
                cc = int(rng.integers(0, w))
                if tissue[cr, cc] or rng.random() < 0.15:
                    center_r, center_c = cr, cc
                    break

            radius = float(rng.uniform(*self.radius_range))
            box_r = int(np.ceil(radius * 1.15))
            y0 = max(0, center_r - box_r)
            y1 = min(h, center_r + box_r + 1)
            x0 = max(0, center_c - box_r)
            x1 = min(w, center_c + box_r + 1)
            if y0 >= y1 or x0 >= x1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dy = yy - center_r
            dx = xx - center_c
            r = np.sqrt(dy * dy + dx * dx)
            inside = r < radius

            # --- Refractive radial pinch (divergent-lens compression) ---
            # Sample from farther-out pixels, so the visible content under
            # the bubble looks compressed toward the center. Strength peaks
            # at the bubble center and falls off to 0 at the rim.
            with np.errstate(invalid="ignore", divide="ignore"):
                rim_norm = np.clip(r / max(radius, 1e-6), 0.0, 1.0)
            pinch = 1.0 + self.lens_pinch_strength * (1.0 - rim_norm) ** 2
            sample_r = r * pinch
            unit_dy = np.where(r > 1e-6, dy / np.maximum(r, 1e-6), 0.0)
            unit_dx = np.where(r > 1e-6, dx / np.maximum(r, 1e-6), 0.0)
            sample_yy = np.clip(center_r + sample_r * unit_dy, 0, h - 1)
            sample_xx = np.clip(center_c + sample_r * unit_dx, 0, w - 1)

            for ch in range(image.shape[2]):
                warped = map_coordinates(
                    image[:, :, ch],
                    [sample_yy, sample_xx],
                    order=1,
                    mode="nearest",
                )
                tile = out[y0:y1, x0:x1, ch]
                tile[inside] = warped[inside]

            # --- Dark refractive rim (Becke line) ---
            rim_thickness = max(1.5, radius * self.rim_thickness_frac)
            # Smooth dark ring centered on the boundary.
            rim_falloff = np.exp(-((r - radius) / (rim_thickness * 0.6)) ** 2)
            rim_factor = 1.0 - self.rim_darkness * rim_falloff
            for ch in range(image.shape[2]):
                out[y0:y1, x0:x1, ch] *= rim_factor

            # --- Specular highlight inside the bubble ---
            # Place at a random angle, ~50% of the radius from center.
            theta = float(rng.uniform(0, 2 * np.pi))
            hi_r = radius * 0.5
            hi_y = center_r + hi_r * np.sin(theta)
            hi_x = center_c + hi_r * np.cos(theta)
            hi_radius = max(1.5, radius * self.highlight_size_frac)
            hi_dist = np.sqrt((yy - hi_y) ** 2 + (xx - hi_x) ** 2)
            hi_falloff = np.exp(-(hi_dist / hi_radius) ** 2)
            hi_mask = inside & (hi_falloff > 0.05)
            for ch in range(image.shape[2]):
                tile = out[y0:y1, x0:x1, ch]
                tile[hi_mask] = np.minimum(
                    tile[hi_mask] + self.highlight_brightness * hi_falloff[hi_mask],
                    1.0,
                )

        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# EmbeddingHalos
# ---------------------------------------------------------------------------


class EmbeddingHalos:
    """Low-frequency radial warm gradient outside the tissue mask.

    Mimics the yellowed embedding medium (paraffin/OCT) visible around tissue.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image

        h, w = image.shape[:2]
        tissue = ctx.tissue_mask if ctx.tissue_mask is not None else _infer_tissue_mask(image)

        # Warm background color: slightly yellowed.
        warm_color = np.array([0.9, 0.85, 0.7], dtype=np.float32)
        if image.shape[2] == 1:
            warm_color = np.array([0.87], dtype=np.float32)

        # Distance from tissue boundary: Gaussian-blurred (~smooth falloff).
        float_tissue = tissue.astype(np.float32)
        dist_field = gaussian_filter(float_tissue, sigma=min(h, w) * 0.04)
        # Invert: 1 far from tissue, 0 inside tissue.
        outside_weight = np.clip(1.0 - dist_field * 4.0, 0.0, 1.0)

        # Randomise intensity of the halo slightly.
        strength = float(rng.uniform(0.5, 1.0))
        alpha = outside_weight[:, :, None] * strength

        out = image * (1.0 - alpha) + warm_color[None, None, :] * alpha
        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Debris
# ---------------------------------------------------------------------------


def _draw_ellipse_mask(
    h: int,
    w: int,
    cr: float,
    cc: float,
    ra: float,
    rb: float,
    angle_rad: float,
) -> np.ndarray:
    """Return a boolean H×W mask for a filled rotated ellipse."""
    rr, cc_grid = np.mgrid[0:h, 0:w]
    dr = rr - cr
    dc = cc_grid - cc
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    u = cos_a * dc + sin_a * dr
    v = -sin_a * dc + cos_a * dr
    return (u / ra) ** 2 + (v / rb) ** 2 <= 1.0


class Debris:
    """Small random opaque/translucent shapes overlaid, 0–20 per image.

    Shapes are small ellipses with random color near the dominant tissue tone.
    """

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image

        h, w = image.shape[:2]
        out = image.copy()
        n_pieces = int(rng.integers(0, 21))
        if n_pieces == 0:
            return image

        # Dominant tissue tone for color variation.
        tissue = ctx.tissue_mask if ctx.tissue_mask is not None else _infer_tissue_mask(image)
        tissue_pixels = image[tissue]
        if tissue_pixels.size > 0:
            base_color = tissue_pixels.mean(axis=0)
        else:
            base_color = np.array([0.5] * image.shape[2], dtype=np.float32)

        for _ in range(n_pieces):
            cr = float(rng.uniform(0, h))
            cc_ = float(rng.uniform(0, w))
            size = float(rng.uniform(2, 16))
            ra = size * float(rng.uniform(0.5, 2.0))
            rb = size * float(rng.uniform(0.5, 2.0))
            angle = float(rng.uniform(0, np.pi))
            alpha = float(rng.uniform(0.3, 0.9))
            # Color: jitter around dominant tissue tone.
            color_jitter = rng.uniform(-0.25, 0.25, size=image.shape[2]).astype(np.float32)
            color = np.clip(base_color + color_jitter, 0.0, 1.0)

            mask = _draw_ellipse_mask(h, w, cr, cc_, ra, rb, angle)
            out[mask] = out[mask] * (1.0 - alpha) + color * alpha

        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# IlluminationGradient
# ---------------------------------------------------------------------------


def _fbm_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Fractional Brownian Motion noise in [0, 1] via multi-scale Gaussian blobs.

    We generate low-res random arrays and upsample+sum them for each octave.
    This avoids any external dependency while producing smooth, low-frequency
    noise that mimics Perlin-like illumination variation.
    """
    noise = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    total_amp = 0.0
    # Start at a coarse 8×8 grid and double each octave.
    base = 8
    for i in range(octaves):
        grid_h = base * (2**i)
        grid_w = base * (2**i)
        raw = rng.standard_normal((grid_h, grid_w))
        # Smooth the random grid then upsample to image size.
        sigma = max(grid_h, grid_w) * 0.3
        smoothed = gaussian_filter(raw, sigma=sigma)
        # Bilinear upsample via map_coordinates.
        scale_r = grid_h / h
        scale_c = grid_w / w
        rr, cc = np.mgrid[0:h, 0:w]
        src_r = rr * scale_r
        src_c = cc * scale_c
        upsampled = map_coordinates(smoothed, [src_r, src_c], order=1, mode="nearest")
        noise += upsampled * amplitude
        total_amp += amplitude
        amplitude *= 0.5

    noise /= total_amp
    # Normalise to [-1, 1].
    max_abs = np.abs(noise).max()
    if max_abs > 1e-6:
        noise /= max_abs
    return noise


class IlluminationGradient:
    """2D FBM noise gain map, ±15% intensity variation.

    Simulates lamp uneven illumination and photobleaching gradients common
    in brightfield and fluorescence acquisitions.
    """

    def __init__(self, p: float = 0.7) -> None:
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image

        h, w = image.shape[:2]
        noise = _fbm_noise(h, w, rng)  # [-1, 1]
        gain = 1.0 + noise * 0.15  # [0.85, 1.15]
        out = image * gain[:, :, None]
        return np.clip(out, 0.0, 1.0).astype(np.float32)
