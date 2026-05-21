"""End-to-end procedural ISH section synthesis with mode-sampled counterstain.

Architecture
------------
Each image is rendered in three stages:

1. **Counterstain** — a renderer from ``COUNTERSTAIN_REGISTRY`` produces the
   anatomical substrate (canvas + stained cells) and populates
   ``ctx.counterstain_signal_mask``.

2. **Signal** — a wrapper from ``SIGNAL_REGISTRY`` layers the expression
   signal on top, consulting the mask so the chromogen wash does not
   double-stain nuclei.

3. **Global jitter** — brightness/contrast tweak for per-image variation.

The ``ISHMode`` is drawn per-image from ``ISH_MODES`` according to declared
weights.  Five modes are supported:

    allen_style      (0.30) — ``none`` counterstain + ``nbt_bcip`` signal
    nbt_nfr          (0.30) — ``nfr`` counterstain  + ``nbt_bcip`` signal
    dab_hematoxylin  (0.20) — ``hematoxylin``        + ``dab`` signal
    fish             (0.15) — ``dapi``               + ``fluorescent_probe``
    nissl_nbt        (0.05) — ``nissl``              + ``nbt_bcip`` signal

Callers can pin a specific mode via ``mode=ISH_MODES[i]`` for
visualisation or deterministic testing.
"""

from __future__ import annotations

import numpy as np

from .counterstain import COUNTERSTAIN_REGISTRY
from .damage_pipeline import apply_damage_layer
from .density import atlas_grayscale_density_map
from .modes import ISH_MODES, ISHMode, sample_mode
from .signals import SIGNAL_REGISTRY
from .transforms.base import TransformContext
from .transforms.tissue_class import classify_tissue

__all__ = ["render_ish_section"]


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


def render_ish_section(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    seed: int,
    pixel_size_um: float,
    gamma_range: tuple[float, float] = (0.9, 1.7),
    floor_range: tuple[float, float] = (0.10, 0.20),
    mode: ISHMode | None = None,
    apply_damage: bool = True,
    damage_intensity: str = "medium",
    apply_geometry_warp: bool = True,
    plane: str = "coronal",
    position_mm: float | None = None,
) -> np.ndarray:
    """Procedurally render one synthetic ISH section.

    Parameters
    ----------
    reference_slice:
        HW grayscale uint8 or float32 — atlas reference template used for
        atlas-grayscale density shading.
    annotation_slice:
        HW int32 BrainGlobe annotation IDs used to build tissue-class masks.
    atlas:
        BrainGlobeAtlas instance (passed through to ``classify_tissue``).
    seed:
        Integer seed for ``np.random.default_rng``; same seed → identical output.
    pixel_size_um:
        Rendering resolution in microns per pixel.  Controls blob sigmas and
        expected-count calculations.
    gamma_range:
        Range for atlas-grayscale density-map gamma exponent.
    floor_range:
        Range for density-map floor (minimum tissue density).
    mode:
        If given, use this ``ISHMode`` instead of sampling one.  Pass
        ``ISH_MODES[i]`` to pin a specific mode for testing or visualisation.
    apply_damage:
        When True (default), apply the damage/realism layer as a final
        post-processing stage after all signal rendering.
    damage_intensity:
        ``"light"`` / ``"medium"`` / ``"heavy"``; passed to
        ``apply_damage_layer``.  Only used when ``apply_damage=True``.

    Returns
    -------
    np.ndarray
        HWC float32 in [0, 1].
    """
    rng = np.random.default_rng(seed)

    masks = classify_tissue(annotation_slice, atlas)
    gamma = float(rng.uniform(*gamma_range))
    floor = float(rng.uniform(*floor_range))
    density = atlas_grayscale_density_map(
        reference_slice,
        masks["tissue"],
        gamma=gamma,
        floor=floor,
    )

    ctx = TransformContext(
        modality="ish",
        annotation_slice=annotation_slice,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=pixel_size_um,
        tissue_class_masks=masks,
        plane=plane,
        position_mm=position_mm,
    )

    chosen_mode: ISHMode = mode if mode is not None else sample_mode(rng, ISH_MODES)

    # --- Stage 1: counterstain (substrate + anatomical cells) ---
    counterstain_fn = COUNTERSTAIN_REGISTRY[chosen_mode.counterstain]
    canvas = counterstain_fn(
        reference_slice,
        annotation_slice,
        atlas,
        masks=masks,
        density_map=density,
        ctx=ctx,
        rng=rng,
        pixel_size_um=pixel_size_um,
        **chosen_mode.counterstain_kwargs,
    )

    # --- Stage 2: expression signal ---
    signal_fn = SIGNAL_REGISTRY[chosen_mode.signal]
    canvas = signal_fn(canvas, ctx=ctx, rng=rng, **chosen_mode.signal_kwargs)

    # --- Stage 3: global brightness/contrast jitter ---
    canvas = _apply_brightness_contrast(canvas, rng)

    # --- Stage 4 (optional): damage / realism layer ---
    if apply_damage:
        canvas = apply_damage_layer(
            canvas,
            rng=rng,
            ctx=ctx,
            modality="ish",
            intensity=damage_intensity,
            geometry=apply_geometry_warp,
        )
    return canvas
