"""End-to-end procedural DAPI section synthesis.

Stage A architecture: from a BrainGlobe atlas reference grayscale slice + the
matching annotation slice, produce a full HWC float32 DAPI section. Each call
draws fresh per-image randomization for diversification:

    - atlas-grayscale density modulation gamma     ∈ [0.9, 2.0]
    - atlas-grayscale floor                        ∈ [0.10, 0.20]
    - DAPI tone shift (pure-blue ↔ cyan ↔ purple) — applied to final canvas
    - global brightness multiplier                  ∈ [0.80, 1.15]
    - global contrast multiplier                    ∈ [0.85, 1.15]

The two texture transforms (gray and white matter) carry their own per-call
randomization (density, sigma, aspect ratio, intensity) — see texture.py.
"""

from __future__ import annotations

import numpy as np

from .damage_pipeline import apply_damage_layer
from .density import atlas_grayscale_density_map
from .transforms.base import TransformContext
from .transforms.texture import DAPIGrayMatterNuclei, DAPIWhiteMatterNuclei
from .transforms.tissue_class import classify_tissue

__all__ = ["render_dapi_section"]


def _apply_tone_shift(canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random DAPI hue: small shift between pure-blue, cyan, and slight purple.

    DAPI emission peaks ~460 nm but real microscopes display it across a small
    palette (deep blue, blue-cyan, blue-purple) depending on filter cube and
    LUT. Drawing a per-image shift mimics that variation.
    """
    cyan_pull = rng.uniform(-0.10, 0.20)
    purple_pull = rng.uniform(-0.10, 0.15)
    out = canvas.copy()
    # cyan_pull > 0  → bleed blue into green
    # purple_pull > 0 → bleed blue into red
    out[..., 1] += cyan_pull * canvas[..., 2]
    out[..., 0] += purple_pull * canvas[..., 2]
    return np.clip(out, 0.0, 1.0)


def _apply_brightness_contrast(
    canvas: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    brightness = rng.uniform(0.80, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    mean = canvas.mean(axis=(0, 1), keepdims=True)
    out = (canvas - mean) * contrast + mean
    out *= brightness
    return np.clip(out, 0.0, 1.0)


def render_dapi_section(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    seed: int,
    pixel_size_um: float,
    gamma_range: tuple[float, float] = (0.9, 2.0),
    floor_range: tuple[float, float] = (0.10, 0.20),
    apply_damage: bool = True,
    damage_intensity: str = "medium",
) -> np.ndarray:
    """Procedurally render one synthetic DAPI section.

    Args:
        reference_slice: HW grayscale uint8 or float32 [0, 1] — atlas template.
        annotation_slice: HW int32 — region IDs from the same atlas/AP.
        atlas: BrainGlobe atlas object (for tissue-class hierarchy lookup).
        seed: RNG seed; determinism is per-image.
        pixel_size_um: Render resolution in µm/pixel.
        gamma_range: Range from which to sample the atlas-grayscale gamma.
        floor_range: Range from which to sample the density floor.
        apply_damage: When True (default), apply the damage/realism layer as a
            final post-processing stage after all signal rendering.
        damage_intensity: ``"light"`` / ``"medium"`` / ``"heavy"``; passed to
            ``apply_damage_layer``.  Only used when ``apply_damage=True``.

    Returns:
        HWC float32 in [0, 1].
    """
    rng = np.random.default_rng(seed)
    h, w = annotation_slice.shape[:2]

    # Ensure reference_slice is HW grayscale as required by atlas_grayscale_density_map.
    ref = reference_slice
    if ref.ndim == 3 and ref.shape[2] == 3:
        ref = ref[..., 0] * 0.299 + ref[..., 1] * 0.587 + ref[..., 2] * 0.114
        ref = ref.astype(np.float32)

    masks = classify_tissue(annotation_slice, atlas)
    gamma = float(rng.uniform(*gamma_range))
    floor = float(rng.uniform(*floor_range))
    density = atlas_grayscale_density_map(
        ref, masks["tissue"], gamma=gamma, floor=floor,
    )

    ctx = TransformContext(
        modality="dapi",
        annotation_slice=annotation_slice,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=pixel_size_um,
        tissue_class_masks=masks,
    )

    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas = DAPIGrayMatterNuclei(p=1.0)(canvas, rng=rng, ctx=ctx)
    canvas = DAPIWhiteMatterNuclei(p=1.0)(canvas, rng=rng, ctx=ctx)

    canvas = _apply_tone_shift(canvas, rng)
    canvas = _apply_brightness_contrast(canvas, rng)
    if apply_damage:
        canvas = apply_damage_layer(
            canvas, rng=rng, ctx=ctx, modality="dapi", intensity=damage_intensity,
        )
    return canvas
