"""End-to-end procedural Nissl section synthesis.

Distinct from DAPI in three structural ways:
1. Cream/tan brightfield background (vs DAPI's near-black). Cells subtract
   from the substrate toward purple, never add to a black canvas.
2. Cell bodies — bigger blobs, bimodal size distribution (most normal-cell
   sized, ~20% giant pyramidal-cell-sized).
3. Cytoplasmic glow — each cell-body blob's delta gets blurred slightly so
   the dark-purple core fades smoothly into the cream substrate, mimicking
   the glow that real Nissl-stained cytoplasm produces under brightfield.

Per-image randomization mirrors the DAPI pipeline (gamma, density, intensity
ranges per-call; brightness/contrast at the end), but tone shift is into the
warm-purple direction rather than blue-cyan.
"""

from __future__ import annotations

import numpy as np

from .damage_pipeline import apply_damage_layer
from .density import atlas_grayscale_density_map, shade_substrate_by_atlas
from .transforms.base import TransformContext
from .transforms.texture import (
    NisslGrayMatterCellBodies,
    NisslWhiteMatterCellBodies,
)
from .transforms.tissue_class import classify_tissue

__all__ = ["render_nissl_section"]


def _build_cream_canvas(
    shape: tuple[int, int],
    substrate_mask: np.ndarray,
    cream: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Light-cream substrate where ``substrate_mask`` is True; slide-white
    elsewhere. ``substrate_mask`` should cover only gray + white matter so
    ventricles fall through to the slide background tone.
    """
    h, w = shape
    canvas = np.empty((h, w, 3), dtype=np.float32)
    bg_color = rng.uniform(0.93, 0.98, size=3).astype(np.float32)
    canvas[..., 0] = bg_color[0]
    canvas[..., 1] = bg_color[1]
    canvas[..., 2] = bg_color[2]

    cream_jitter = rng.uniform(-0.04, 0.04, size=3).astype(np.float32)
    cream_arr = np.clip(np.array(cream, dtype=np.float32) + cream_jitter, 0.0, 1.0)
    canvas[substrate_mask] = cream_arr
    return canvas


def _apply_nissl_tone_shift(
    canvas: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Warm/cool drift in cream tone — mimics scanner white balance variation."""
    warm_pull = rng.uniform(-0.05, 0.10)
    cool_pull = rng.uniform(-0.05, 0.05)
    out = canvas.copy()
    out[..., 0] += warm_pull * 0.5
    out[..., 1] += warm_pull * 0.3
    out[..., 2] += cool_pull * 0.5
    return np.clip(out, 0.0, 1.0)


def _apply_brightness_contrast(
    canvas: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    brightness = rng.uniform(0.85, 1.10)
    contrast = rng.uniform(0.85, 1.15)
    mean = canvas.mean(axis=(0, 1), keepdims=True)
    out = (canvas - mean) * contrast + mean
    out *= brightness
    return np.clip(out, 0.0, 1.0)


_CREAM_PRESETS: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.78, 0.65),  # classic warm cream-tan (thionine-like)
    (0.78, 0.78, 0.85),  # neutral cream
    (0.74, 0.82, 0.95),  # blue-cream cresyl-violet (Mouse10-like)
    (0.82, 0.75, 0.88),  # purple-cream
)


def render_nissl_section(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    seed: int,
    pixel_size_um: float,
    gamma_range: tuple[float, float] = (0.9, 1.8),
    floor_range: tuple[float, float] = (0.10, 0.20),
    cream_base: tuple[float, float, float] | None = None,
    apply_damage: bool = True,
    damage_intensity: str = "medium",
    apply_geometry_warp: bool = True,
    plane: str = "coronal",
    position_mm: float | None = None,
) -> np.ndarray:
    """Procedurally render one synthetic Nissl section.

    If ``cream_base`` is None, one of four preset substrate tones is picked at
    random per image — covers warm-cream thionine, neutral, blue-cream cresyl
    violet, and purple-cream variants. Returns HWC float32 in [0, 1].

    Parameters
    ----------
    apply_damage:
        When True (default), apply the damage/realism layer as a final
        post-processing stage after all signal rendering.
    damage_intensity:
        ``"light"`` / ``"medium"`` / ``"heavy"``; passed to
        ``apply_damage_layer``.  Only used when ``apply_damage=True``.
    """
    rng = np.random.default_rng(seed)
    h, w = annotation_slice.shape[:2]

    if cream_base is None:
        cream_base = _CREAM_PRESETS[int(rng.integers(0, len(_CREAM_PRESETS)))]

    # Ensure reference_slice is HW grayscale as required by atlas_grayscale_density_map.
    ref = reference_slice
    if ref.ndim == 3 and ref.shape[2] == 3:
        ref = ref[..., 0] * 0.299 + ref[..., 1] * 0.587 + ref[..., 2] * 0.114
        ref = ref.astype(np.float32)

    masks = classify_tissue(annotation_slice, atlas)
    gamma = float(rng.uniform(*gamma_range))
    floor = float(rng.uniform(*floor_range))
    density = atlas_grayscale_density_map(
        ref,
        masks["tissue"],
        gamma=gamma,
        floor=floor,
    )

    ctx = TransformContext(
        modality="nissl",
        annotation_slice=annotation_slice,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=pixel_size_um,
        tissue_class_masks=masks,
        plane=plane,
        position_mm=position_mm,
    )

    substrate_mask = masks["gray_matter"] | masks["white_matter"]
    canvas = _build_cream_canvas((h, w), substrate_mask, cream_base, rng)
    # Atlas-grayscale substrate shading: cell-rich regions absorb more light
    # and read slightly darker even before stained cells are added.
    shading_strength = float(rng.uniform(0.12, 0.28))
    canvas = shade_substrate_by_atlas(
        canvas,
        reference_slice,
        substrate_mask,
        strength=shading_strength,
        gamma=1.0,
    )
    canvas = NisslGrayMatterCellBodies(p=1.0, cream=cream_base)(canvas, rng=rng, ctx=ctx)
    canvas = NisslWhiteMatterCellBodies(p=1.0, cream=cream_base)(canvas, rng=rng, ctx=ctx)
    canvas = _apply_nissl_tone_shift(canvas, rng)
    canvas = _apply_brightness_contrast(canvas, rng)
    if apply_damage:
        canvas = apply_damage_layer(
            canvas,
            rng=rng,
            ctx=ctx,
            modality="nissl",
            intensity=damage_intensity,
            geometry=apply_geometry_warp,
        )
    return canvas
