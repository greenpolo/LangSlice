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
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    label,
    map_coordinates,
    rotate,
)

from .base import TransformContext

__all__ = [
    "Folds",
    "HemibrainPreparation",
    "Tears",
    "Microbubbles",
    "EmbeddingHalos",
    "Debris",
    "IlluminationGradient",
    "PosteriorWingDamage",
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


def _shift_canvas_like(
    arr: np.ndarray,
    *,
    keep_side: str,
    shift_cols: int,
    fill_value: float | int | bool = 0,
) -> np.ndarray:
    """Keep one side of an array and shift it horizontally into a new canvas."""
    w = arr.shape[1]
    split = w // 2
    keep_cols = np.arange(w) < split if keep_side == "left" else np.arange(w) >= split
    out = np.full_like(arr, fill_value)
    src_cols = np.where(keep_cols)[0]
    dst_cols = src_cols + shift_cols
    valid = (dst_cols >= 0) & (dst_cols < w)
    if arr.ndim == 2:
        out[:, dst_cols[valid]] = arr[:, src_cols[valid]]
    else:
        out[:, dst_cols[valid], :] = arr[:, src_cols[valid], :]
    return out


def _mask_column_center(mask: np.ndarray) -> float | None:
    _, cols = np.where(mask)
    if len(cols) == 0:
        return None
    return float((cols.min() + cols.max()) / 2.0)


def _replace_masks_with_background(
    out: np.ndarray,
    masks: list[np.ndarray],
    bg_color: np.ndarray,
) -> None:
    if not masks:
        return
    combined = np.zeros(out.shape[:2], dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool)
    out[combined] = bg_color


def _fray_along_mask_edge(
    keep_mask: np.ndarray,
    cut_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    band_px: int | None = None,
    band_frac: float = 0.022,
    threshold: float = 0.62,
    smoothness_frac: float = 0.005,
) -> np.ndarray:
    """Boolean mask of `keep_mask` pixels to remove for an irregular torn edge.

    The fray band is `keep_mask` pixels within `band_px` of `cut_mask`. Within
    the band, smoothed uniform noise is thresholded to produce ragged notches.

    When `band_px` is None, it scales with image size as
    ``band_frac * min(H, W)`` (default ~1.2% of the shorter side) so fraying
    stays visible across resolutions. The Gaussian smoothing sigma scales the
    same way via ``smoothness_frac``.
    """
    if not keep_mask.any() or not cut_mask.any():
        return np.zeros_like(keep_mask, dtype=bool)
    h, w = keep_mask.shape
    short_side = min(h, w)
    if band_px is None:
        band_px = max(2, int(round(band_frac * short_side)))
    if band_px <= 0:
        return np.zeros_like(keep_mask, dtype=bool)
    sigma = max(1.0, smoothness_frac * short_side)
    band = binary_dilation(cut_mask, iterations=band_px) & keep_mask
    if not band.any():
        return np.zeros_like(keep_mask, dtype=bool)
    noise = gaussian_filter(rng.random((h, w)).astype(np.float32), sigma=sigma)
    return band & (noise < threshold)


def _posterior_wing_source_mask(
    tissue: np.ndarray,
    tissue_class_masks: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if tissue_class_masks is None:
        return tissue
    parts = [
        tissue_class_masks[key].astype(bool)
        for key in ("isocortex", "hippocampal_formation", "cortical_subplate")
        if key in tissue_class_masks and tissue_class_masks[key].any()
    ]
    if not parts:
        return tissue
    source = np.zeros_like(tissue, dtype=bool)
    for part in parts:
        source |= part
    return source & tissue


def _side_wing_masks(
    tissue: np.ndarray,
    *,
    center_mask: np.ndarray | None = None,
    include_satellites: bool = False,
) -> dict[str, np.ndarray]:
    """Approximate posterior lateral tissue wings from the tissue silhouette."""
    h, w = tissue.shape
    rows, cols = np.indices((h, w))
    tissue_cols = cols[tissue]
    center_source = tissue if center_mask is None else center_mask
    center_cols = cols[center_source]
    center = float(np.median(center_cols)) if center_cols.size else (w - 1) / 2.0
    tissue_min = int(tissue_cols.min()) if tissue_cols.size else 0
    tissue_max = int(tissue_cols.max()) if tissue_cols.size else w - 1
    lateral_edge_margin = max(4, int(round(0.12 * max(tissue_max - tissue_min, 1))))
    row_norm = rows / max(h - 1, 1)
    curve = 0.14 * w + 0.10 * w * (row_norm - 0.32) ** 2
    dorsal_gate = rows <= int(round(0.78 * h))
    source_cc, _ = label(tissue)
    candidates = {
        "left": tissue & dorsal_gate & (cols < center - curve),
        "right": tissue & dorsal_gate & (cols > center + curve),
    }
    wings: dict[str, np.ndarray] = {}
    for side, candidate in candidates.items():
        cc, n = label(candidate)
        if n == 0:
            wings[side] = candidate
            continue
        keep = np.zeros_like(candidate, dtype=bool)
        for component_id in range(1, n + 1):
            component = cc == component_id
            _, component_cols = np.where(component)
            if component_cols.size == 0:
                continue
            source_component = _source_component_for(component, source_cc) if include_satellites else component
            centroid = float(component_cols.mean())
            lateral = centroid < center if side == "left" else centroid > center
            reaches_outer_edge = (
                int(component_cols.min()) <= tissue_min + lateral_edge_margin
                if side == "left"
                else int(component_cols.max()) >= tissue_max - lateral_edge_margin
            )
            large_enough = int(component.sum()) >= max(12, int(0.003 * h * w))
            if lateral and reaches_outer_edge and large_enough:
                keep |= source_component
        if include_satellites and keep.any():
            satellite_halo = binary_dilation(keep, iterations=max(6, int(round(0.20 * w))))
            for component_id in range(1, n + 1):
                component = cc == component_id
                if (component & keep).any() or not (component & satellite_halo).any():
                    continue
                _, component_cols = np.where(component)
                if component_cols.size == 0:
                    continue
                centroid = float(component_cols.mean())
                lateral = centroid < center if side == "left" else centroid > center
                large_enough = int(component.sum()) >= max(8, int(0.001 * h * w))
                if lateral and large_enough:
                    keep |= _source_component_for(component, source_cc)
        wings[side] = keep
    return wings


def _source_component_for(component: np.ndarray, source_cc: np.ndarray) -> np.ndarray:
    ids = np.unique(source_cc[component & (source_cc > 0)])
    out = np.zeros_like(component, dtype=bool)
    for component_id in ids:
        out |= source_cc == component_id
    return out


def _translate_mask(mask: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    src_r0 = max(0, -dy)
    src_r1 = min(h, h - dy)
    dst_r0 = max(0, dy)
    dst_r1 = min(h, h + dy)
    src_c0 = max(0, -dx)
    src_c1 = min(w, w - dx)
    dst_c0 = max(0, dx)
    dst_c1 = min(w, w + dx)
    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _translate_numeric(
    arr: np.ndarray,
    *,
    dx: int,
    dy: int,
    fill_value: float | int = 0,
) -> np.ndarray:
    out = np.full_like(arr, fill_value)
    h, w = arr.shape
    src_r0 = max(0, -dy)
    src_r1 = min(h, h - dy)
    dst_r0 = max(0, dy)
    dst_r1 = min(h, h + dy)
    src_c0 = max(0, -dx)
    src_c1 = min(w, w - dx)
    dst_c0 = max(0, dx)
    dst_c1 = min(w, w + dx)
    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return mask.astype(bool)
    rotated = rotate(
        mask.astype(np.float32),
        angle=float(angle_deg),
        reshape=False,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return rotated > 0.5


def _rotate_image_patch(patch: np.ndarray, angle_deg: float, bg_color: np.ndarray) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return patch
    channels = []
    for ch in range(patch.shape[2]):
        channels.append(
            rotate(
                patch[:, :, ch],
                angle=float(angle_deg),
                reshape=False,
                order=1,
                mode="constant",
                cval=float(bg_color[ch]),
            )
        )
    return np.stack(channels, axis=2).astype(np.float32)


def _move_mask(mask: np.ndarray, *, dx: int, dy: int, angle_deg: float) -> np.ndarray:
    return _translate_mask(_rotate_mask(mask, angle_deg), dx=dx, dy=dy)


def _move_array(
    arr: np.ndarray,
    source_mask: np.ndarray,
    *,
    dx: int,
    dy: int,
    angle_deg: float,
    fill_value: float | int | bool,
) -> tuple[np.ndarray, np.ndarray]:
    moved_mask = _move_mask(source_mask, dx=dx, dy=dy, angle_deg=angle_deg)
    out = np.full_like(arr, fill_value)
    if not source_mask.any() or not moved_mask.any():
        return out, moved_mask

    values = np.where(source_mask, arr, fill_value)
    if abs(angle_deg) >= 1e-6:
        values = rotate(
            values,
            angle=float(angle_deg),
            reshape=False,
            order=0,
            mode="constant",
            cval=float(fill_value),
        ).astype(arr.dtype, copy=False)
    out[_translate_mask(source_mask, dx=dx, dy=dy)] = values[
        _translate_mask(source_mask, dx=dx, dy=dy)
    ]
    return out, moved_mask


class HemibrainPreparation:
    """Deliberate one-hemisphere preparation.

    Researchers often cut down the midline and mount only one hemisphere. This
    transform models that preparation directly: it removes one full side and
    shifts the retained hemisphere toward the center of the canvas.
    """

    def __init__(
        self,
        p: float = 0.08,
        keep_side: str | None = None,
        fray_band_px: int | None = None,
    ) -> None:
        self.p = p
        if keep_side not in {None, "left", "right"}:
            raise ValueError("keep_side must be 'left', 'right', or None")
        self.keep_side = keep_side
        self.fray_band_px = fray_band_px

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image
        if getattr(ctx, "plane", "coronal") not in {"coronal", "horizontal"}:
            return image

        keep_side = self.keep_side or str(rng.choice(["left", "right"]))
        tissue = ctx.tissue_mask if ctx.tissue_mask is not None else _infer_tissue_mask(image)
        w = image.shape[1]
        split = w // 2
        keep_cols = np.arange(w) < split if keep_side == "left" else np.arange(w) >= split
        kept_tissue = tissue & keep_cols[None, :]
        removed_half = tissue & ~keep_cols[None, :]
        center = _mask_column_center(kept_tissue)
        if center is None:
            return image

        # Fray the kept hemisphere's medial edge in source coordinates.
        bg_color = _sample_bg_color(image)
        fray = _fray_along_mask_edge(
            kept_tissue, removed_half, rng, band_px=self.fray_band_px
        )
        if fray.any():
            image = image.copy()
            image[fray] = bg_color
            kept_tissue = kept_tissue & ~fray

        shift_cols = int(round((w - 1) / 2.0 - center))
        out = np.empty_like(image)
        out[...] = bg_color
        out = _shift_canvas_like(image, keep_side=keep_side, shift_cols=shift_cols)
        empty = out.mean(axis=2) <= 0
        out[empty] = bg_color

        if ctx.tissue_mask is not None:
            ctx.tissue_mask = _shift_canvas_like(
                kept_tissue, keep_side=keep_side, shift_cols=shift_cols, fill_value=False
            ).astype(bool)
        if ctx.annotation_slice is not None:
            shifted_ann = _shift_canvas_like(
                ctx.annotation_slice, keep_side=keep_side, shift_cols=shift_cols, fill_value=0
            ).astype(ctx.annotation_slice.dtype, copy=False)
            if fray.any() and ctx.tissue_mask is not None:
                shifted_ann = np.where(ctx.tissue_mask, shifted_ann, 0).astype(
                    ctx.annotation_slice.dtype, copy=False
                )
            ctx.annotation_slice = shifted_ann
        if ctx.density_map is not None:
            shifted_density = _shift_canvas_like(
                ctx.density_map, keep_side=keep_side, shift_cols=shift_cols, fill_value=0.0
            ).astype(ctx.density_map.dtype, copy=False)
            if fray.any() and ctx.tissue_mask is not None:
                shifted_density = np.where(ctx.tissue_mask, shifted_density, 0.0).astype(
                    ctx.density_map.dtype, copy=False
                )
            ctx.density_map = shifted_density
        if ctx.tissue_class_masks is not None:
            updated: dict[str, np.ndarray] = {}
            for key, mask in ctx.tissue_class_masks.items():
                source = mask.astype(bool) & keep_cols[None, :] & ~fray
                updated[key] = _shift_canvas_like(
                    source, keep_side=keep_side, shift_cols=shift_cols, fill_value=False
                ).astype(bool)
            if "background" in updated:
                updated["background"] = ~updated.get("tissue", ctx.tissue_mask).astype(bool)
            ctx.tissue_class_masks = updated

        return np.clip(out, 0.0, 1.0).astype(np.float32)


class PosteriorWingDamage:
    """Posterior coronal lateral-wing loss or detachment.

    In posterior sections, the lateral "wings" can peel away as physical tissue
    slabs. This targets the whole lateral slab from the tissue silhouette, not
    only atlas isocortex pixels.
    """

    _MODES = (
        "left_missing",
        "right_missing",
        "both_missing",
        "both_detached",
        "left_missing_right_detached",
        "right_missing_left_detached",
    )

    def __init__(
        self,
        p: float = 0.18,
        posterior_min_position_mm: float = 8.5,
        mode: str | None = None,
        detach_shift_px: tuple[int, int] | None = None,
        detach_angle_deg: tuple[float, float] = (-8.0, 8.0),
        fray_band_px: int | None = None,
        fray_wing_band_px: int | None = None,
    ) -> None:
        self.p = p
        self.posterior_min_position_mm = posterior_min_position_mm
        if mode is not None and mode not in {*self._MODES, "left_detached", "right_detached"}:
            raise ValueError(f"Unsupported posterior wing damage mode: {mode!r}")
        self.mode = mode
        self.detach_shift_px = detach_shift_px
        self.detach_angle_deg = detach_angle_deg
        self.fray_band_px = fray_band_px
        self.fray_wing_band_px = fray_wing_band_px

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() > self.p:
            return image
        if getattr(ctx, "plane", "coronal") != "coronal":
            return image
        if ctx.position_mm is None or ctx.position_mm < self.posterior_min_position_mm:
            return image
        tissue = ctx.tissue_mask if ctx.tissue_mask is not None else _infer_tissue_mask(image)
        thalamus = None if ctx.tissue_class_masks is None else ctx.tissue_class_masks.get("thalamus")
        if thalamus is not None and thalamus.any():
            return image

        bg_color = _sample_bg_color(image)
        out = image.copy()
        wing_source = _posterior_wing_source_mask(tissue.astype(bool), ctx.tissue_class_masks)
        has_specific_wing_masks = (
            ctx.tissue_class_masks is not None
            and any(
                key in ctx.tissue_class_masks and ctx.tissue_class_masks[key].any()
                for key in ("isocortex", "hippocampal_formation", "cortical_subplate")
            )
        )
        wings = _side_wing_masks(
            wing_source,
            center_mask=tissue.astype(bool),
            include_satellites=has_specific_wing_masks,
        )
        if not wings["left"].any() and not wings["right"].any():
            return image

        mode = self.mode or str(rng.choice(self._MODES, p=[0.20, 0.20, 0.15, 0.25, 0.10, 0.10]))
        actions: dict[str, str] = {}
        if mode == "left_missing":
            actions["left"] = "missing"
        elif mode == "right_missing":
            actions["right"] = "missing"
        elif mode == "both_missing":
            actions["left"] = actions["right"] = "missing"
        elif mode == "both_detached":
            actions["left"] = actions["right"] = "detached"
        elif mode == "left_detached":
            actions["left"] = "detached"
        elif mode == "right_detached":
            actions["right"] = "detached"
        elif mode == "left_missing_right_detached":
            actions["left"] = "missing"
            actions["right"] = "detached"
        elif mode == "right_missing_left_detached":
            actions["right"] = "missing"
            actions["left"] = "detached"

        removed_masks: list[np.ndarray] = []
        moved_masks: list[np.ndarray] = []

        full_tissue = tissue.astype(bool)
        for side, action in actions.items():
            mask = wings[side]
            if not mask.any():
                continue

            # Tear the whole lateral slab as a unit: include any enclosed tissue
            # (e.g. fiber tracts inside the hippocampal C-shape) so the moved
            # patch isn't punched through with background pixels.
            filled = binary_fill_holes(mask)
            if filled is not None:
                mask = filled & full_tissue
            central_brain = full_tissue & ~mask
            keep_fray = _fray_along_mask_edge(
                central_brain, mask, rng, band_px=self.fray_band_px
            )
            removed_masks.append(mask | keep_fray)
            out[mask] = bg_color
            if keep_fray.any():
                out[keep_fray] = bg_color

            if action != "detached":
                continue

            wing_fray = _fray_along_mask_edge(
                mask, central_brain, rng, band_px=self.fray_wing_band_px
            )
            moved_wing = mask & ~wing_fray
            if not moved_wing.any():
                continue

            dx, dy = self._sample_shift(side, rng)
            angle = float(rng.uniform(*self.detach_angle_deg))
            moved_mask = _move_mask(moved_wing, dx=dx, dy=dy, angle_deg=angle)
            patch = np.empty_like(image)
            patch[...] = bg_color
            patch[moved_wing] = image[moved_wing]
            moved_patch = np.empty_like(image)
            moved_patch[...] = bg_color
            if abs(angle) >= 1e-6:
                patch = _rotate_image_patch(patch, angle, bg_color)
            for ch in range(image.shape[2]):
                moved_patch[:, :, ch] = _translate_numeric(
                    patch[:, :, ch], dx=dx, dy=dy, fill_value=float(bg_color[ch])
                )
            out[moved_mask] = moved_patch[moved_mask]
            moved_masks.append(moved_mask)

        self._update_context(ctx, removed_masks=removed_masks, moved_masks=moved_masks)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _sample_shift(self, side: str, rng: np.random.Generator) -> tuple[int, int]:
        if self.detach_shift_px is not None:
            return self.detach_shift_px
        lateral = -1 if side == "left" else 1
        dx = int(lateral * rng.integers(5, 15))
        dy = int(rng.integers(-8, 9))
        return dx, dy

    def _update_context(
        self,
        ctx: TransformContext,
        *,
        removed_masks: list[np.ndarray],
        moved_masks: list[np.ndarray],
    ) -> None:
        if not removed_masks:
            return
        removed = np.zeros_like(removed_masks[0], dtype=bool)
        for mask in removed_masks:
            removed |= mask
        moved = np.zeros_like(removed, dtype=bool)
        for mask in moved_masks:
            moved |= mask

        if ctx.tissue_mask is not None:
            ctx.tissue_mask = (ctx.tissue_mask.astype(bool) & ~removed) | moved
        if ctx.tissue_class_masks is not None:
            for key, mask in list(ctx.tissue_class_masks.items()):
                if key == "background":
                    continue
                updated = mask.astype(bool) & ~removed
                if key == "tissue":
                    updated |= moved
                ctx.tissue_class_masks[key] = updated
            if "background" in ctx.tissue_class_masks:
                tissue = (
                    ctx.tissue_class_masks.get("tissue")
                    if "tissue" in ctx.tissue_class_masks
                    else ctx.tissue_mask
                )
                if tissue is not None:
                    ctx.tissue_class_masks["background"] = ~tissue.astype(bool)
        if ctx.annotation_slice is not None:
            ctx.annotation_slice[removed] = 0
        if ctx.density_map is not None:
            ctx.density_map[removed] = 0.0


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
