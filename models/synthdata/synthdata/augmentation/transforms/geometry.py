"""Geometry jitter: affine, crop, resolution-shift, blade-stretch.

All transforms expect and return HWC float32 arrays in [0, 1].
torch ops run on GPU only when torch.cuda.is_available() AND the env var
LANGSLICE_AUG_GPU=1 is set; otherwise CPU.
"""

from __future__ import annotations

__all__ = ["AffineJitter", "BladeStretchHorizontal", "RandomCrop", "ResolutionShift"]

import math
import os

import numpy as np
import torch
import torch.nn.functional as TF
import torchvision.transforms.v2.functional as TVF
from torchvision.transforms.v2 import InterpolationMode

from .base import TransformContext

_GPU_ENABLED = os.environ.get("LANGSLICE_AUG_GPU", "0") == "1"


def _device() -> torch.device:
    if _GPU_ENABLED and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _np2d_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _tensor2d_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).squeeze(0).cpu().numpy()


class AffineJitter:
    """Small random affine perturbation (rotation, translation, scale).

    Parameters
    ----------
    p:
        Probability of applying the transform.
    fill:
        'reflect' (default) or 'zeros'.  Controls edge fill when the affine
        moves pixels outside the image boundary.
    """

    def __init__(self, p: float = 0.7, fill: str = "reflect") -> None:
        if fill not in {"reflect", "zeros"}:
            raise ValueError(f"fill must be 'reflect' or 'zeros', got {fill!r}")
        self.p = p
        self.fill = fill

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        angle_deg = float(rng.uniform(-5.0, 5.0))
        tx_frac = float(rng.uniform(-0.03, 0.03))
        ty_frac = float(rng.uniform(-0.03, 0.03))
        scale = float(rng.uniform(0.97, 1.03))

        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # affine_grid uses normalized coords [-1, 1]; translation in that space.
        tx_norm = tx_frac * 2.0
        ty_norm = ty_frac * 2.0

        # Theta maps output coords -> input coords (inverse warp convention).
        theta = torch.tensor(
            [
                [cos_a / scale, -sin_a / scale, tx_norm],
                [sin_a / scale, cos_a / scale, ty_norm],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        device = _device()
        t = _to_tensor(image, device)
        theta = theta.to(device)
        grid = torch.nn.functional.affine_grid(theta, list(t.shape), align_corners=False)
        padding_mode = "reflection" if self.fill == "reflect" else "zeros"
        out = TF.grid_sample(
            t, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False
        )
        result = _to_numpy(out).astype(np.float32)
        return np.clip(result, 0.0, 1.0)


class RandomCrop:
    """Crop a random 90–100% sub-area and resize back to original H×W.

    Parameters
    ----------
    p:
        Probability of applying the transform.
    min_scale:
        Lower bound of the crop area fraction (default 0.90).
    """

    def __init__(self, p: float = 0.5, min_scale: float = 0.90) -> None:
        self.p = p
        self.min_scale = min_scale

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
        crop_frac = float(rng.uniform(self.min_scale, 1.0))
        crop_h = max(1, int(round(h * crop_frac)))
        crop_w = max(1, int(round(w * crop_frac)))

        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))

        device = _device()
        t = _to_tensor(image, device)
        cropped = TVF.resized_crop(
            t,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        result = _to_numpy(cropped).astype(np.float32)
        return np.clip(result, 0.0, 1.0)


_RESOLUTION_CHOICES = (10, 20, 25, 35, 50)


class ResolutionShift:
    """Introduce realistic blur by down-then-up sampling to a different µm rate.

    The atlas is rendered at ``ctx.pixel_size_um`` (typically 25 µm/pixel).
    This transform picks a target µm rate, resizes to simulate that resolution,
    then resizes back to the original H×W.  The round-trip introduces
    anti-aliasing blur that mimics the mismatch between atlas rendering resolution
    and the agent's query resolution at deployment time.

    Side-effects on ctx
    -------------------
    ``ctx.density_map``, ``ctx.tissue_mask``, and ``ctx.annotation_slice`` are
    resized to the intermediate resolution and back, keeping them consistent with
    the image.  ``ctx.pixel_size_um`` is *not* mutated: by the time this
    transform runs the value may already be consumed by upstream stages, and the
    round-trip restores the original shape anyway.

    Parameters
    ----------
    p:
        Probability of applying the transform.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    @staticmethod
    def _resize2d(
        arr: np.ndarray, h: int, w: int, nearest: bool, device: torch.device
    ) -> np.ndarray:
        t = _np2d_to_tensor(arr.astype(np.float32), device)
        mode = InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR
        antialias = False if nearest else True
        resized = TVF.resize(t, [h, w], interpolation=mode, antialias=antialias)
        return _tensor2d_to_numpy(resized)

    def __call__(
        self,
        image: np.ndarray,
        *,
        rng: np.random.Generator,
        ctx: TransformContext,
    ) -> np.ndarray:
        if rng.random() >= self.p:
            return image

        current_um = ctx.pixel_size_um
        choices = [c for c in _RESOLUTION_CHOICES if c != current_um]
        if not choices:
            choices = list(_RESOLUTION_CHOICES)
        target_um = float(rng.choice(choices))  # type: ignore[arg-type]

        h, w = image.shape[:2]
        scale = current_um / target_um
        mid_h = max(1, int(round(h * scale)))
        mid_w = max(1, int(round(w * scale)))

        device = _device()

        t = _to_tensor(image, device)
        down = TVF.resize(
            t, [mid_h, mid_w], interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        up = TVF.resize(down, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        result = _to_numpy(up).astype(np.float32)

        if ctx.density_map is not None:
            d = self._resize2d(ctx.density_map, mid_h, mid_w, nearest=False, device=device)
            d = self._resize2d(d, h, w, nearest=False, device=device)
            ctx.density_map = d.astype(np.float32)

        if ctx.tissue_mask is not None:
            m = self._resize2d(ctx.tissue_mask, mid_h, mid_w, nearest=True, device=device)
            m = self._resize2d(m, h, w, nearest=True, device=device)
            ctx.tissue_mask = m.astype(ctx.tissue_mask.dtype)

        if ctx.annotation_slice is not None:
            ann_f32 = ctx.annotation_slice.astype(np.float32)
            a = self._resize2d(ann_f32, mid_h, mid_w, nearest=True, device=device)
            a = self._resize2d(a, h, w, nearest=True, device=device)
            ctx.annotation_slice = np.round(a).astype(ctx.annotation_slice.dtype)

        return np.clip(result, 0.0, 1.0)


class BladeStretchHorizontal:
    """Horizontal-biased anisotropic stretch mimicking blade-friction warp.

    Real microtome sections show 1–1.15x horizontal stretch (parallel to blade
    travel) with 1.0–1.05x vertical scale.  This transform applies an affine
    scale with horizontal stretch sampled from a wider range than vertical, plus
    an optional small shear that mimics the slight oblique distortion a blade
    can introduce when it is not perfectly perpendicular.

    The output is always resized back to the original H×W so the shape contract
    is preserved.

    Parameters
    ----------
    p:
        Probability of applying the transform.
    horizontal_stretch_range:
        (lo, hi) for the x-axis scale factor.
    vertical_stretch_range:
        (lo, hi) for the y-axis scale factor.
    shear_range_deg:
        (lo, hi) for a random shear angle added to the affine matrix (degrees).
    """

    def __init__(
        self,
        p: float = 1.0,
        horizontal_stretch_range: tuple[float, float] = (1.00, 1.15),
        vertical_stretch_range: tuple[float, float] = (1.00, 1.05),
        shear_range_deg: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        self.p = p
        self.horizontal_stretch_range = horizontal_stretch_range
        self.vertical_stretch_range = vertical_stretch_range
        self.shear_range_deg = shear_range_deg

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
        sx = float(rng.uniform(*self.horizontal_stretch_range))
        sy = float(rng.uniform(*self.vertical_stretch_range))
        shear_deg = float(rng.uniform(*self.shear_range_deg))
        shear_rad = math.radians(shear_deg)

        # Build 2x3 affine matrix in torch normalized-coord convention
        # (output → input mapping, i.e. inverse warp).
        # Scale: sx > 1 → stretched output maps to a narrower source region
        # so the content appears wider in the result after resize-back.
        # We stretch *content* by sx/sy, which means the warp source coords
        # are compressed by 1/sx, 1/sy relative to center.
        # Shear: small skew along x-axis.
        theta = torch.tensor(
            [
                [1.0 / sx, math.tan(shear_rad) / sx, 0.0],
                [0.0, 1.0 / sy, 0.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        device = _device()
        t = _to_tensor(image, device)
        theta = theta.to(device)
        grid = torch.nn.functional.affine_grid(theta, list(t.shape), align_corners=False)
        out = TF.grid_sample(
            t, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )
        result = _to_numpy(out).astype(np.float32)
        return np.clip(result, 0.0, 1.0)
