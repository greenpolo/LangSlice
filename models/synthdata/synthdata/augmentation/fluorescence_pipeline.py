"""End-to-end procedural multi-channel fluorescence section synthesis.

Architecture
------------
Each image is rendered in three stages:

1. **Counterstain** — a renderer from ``COUNTERSTAIN_REGISTRY`` produces the
   anatomical substrate (canvas + stained cells).  For DAPI-based modes this
   is a dark near-black canvas with blue nuclear dots.  For tract-tracing
   modes the Nissl counterstain is used with a cool blue-cream NeuroTrace-like
   substrate.

2. **Marker channels** — one or more ``FluorescenceMarker`` transforms layer
   fluorescent signal on top.  Named combos have pre-defined
   ``IFChannelSpec`` parameters; generic modes sample random presets.

3. **Global jitter** — brightness/contrast tweak for per-image variation.

The ``FluorescenceMode`` is drawn per-image from ``FLUORESCENCE_MODES``
according to declared weights.  Thirteen modes are supported spanning:

    DAPI-only, single-IF, dual/triple multiplex, tract-tracing, and generic
    1–3 channel modes for breadth.

Callers can pin a specific mode via ``mode=FLUORESCENCE_MODES[i]`` for
visualisation or deterministic testing.

Backwards compatibility
-----------------------
The old ``p_green`` / ``p_red`` float parameters are still accepted when
``mode`` is not given.  Passing them triggers a DEPRECATED code path that
maps probability flags to a simple DAPI + optional-channel rendering.  Pass
an explicit mode instead.
"""

from __future__ import annotations

import warnings

import numpy as np

from .counterstain import COUNTERSTAIN_REGISTRY
from .damage_pipeline import apply_damage_layer
from .density import atlas_grayscale_density_map
from .modes import FLUORESCENCE_MODES, FluorescenceMode, IFChannelSpec, sample_mode
from .transforms.base import TransformContext
from .transforms.texture import FluorescenceMarker
from .transforms.tissue_class import classify_tissue

__all__ = ["render_fluorescence_section"]


# ---------------------------------------------------------------------------
# Generic channel presets
# ---------------------------------------------------------------------------

_GENERIC_IF_PRESETS: tuple[IFChannelSpec, ...] = (
    IFChannelSpec(
        name="generic_red_sparse", channel=0, distribution="sparse",
        density_range_per_mm2=(30.0, 200.0), intensity_range=(0.55, 0.95),
        sigma_range_scale=(0.6, 1.4),
    ),
    IFChannelSpec(
        name="generic_red_moderate", channel=0, distribution="moderate",
        density_range_per_mm2=(150.0, 600.0), intensity_range=(0.50, 0.90),
        sigma_range_scale=(0.7, 1.5),
    ),
    IFChannelSpec(
        name="generic_green_sparse", channel=1, distribution="sparse",
        density_range_per_mm2=(30.0, 200.0), intensity_range=(0.55, 0.95),
        sigma_range_scale=(0.6, 1.4),
    ),
    IFChannelSpec(
        name="generic_green_moderate", channel=1, distribution="moderate",
        density_range_per_mm2=(150.0, 600.0), intensity_range=(0.50, 0.90),
        sigma_range_scale=(0.7, 1.5),
    ),
)


def _sample_generic_if_channels(rng: np.random.Generator, n: int) -> list[IFChannelSpec]:
    """Pick n random IFChannelSpec instances for generic multiplex modes."""
    chosen = []
    for _ in range(n):
        chosen.append(_GENERIC_IF_PRESETS[int(rng.integers(0, len(_GENERIC_IF_PRESETS)))])
    return chosen


# ---------------------------------------------------------------------------
# Global jitter
# ---------------------------------------------------------------------------


def _apply_brightness_contrast(
    canvas: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    mean = canvas.mean(axis=(0, 1), keepdims=True)
    out = (canvas - mean) * contrast + mean
    out *= brightness
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Backwards-compat mode builder
# ---------------------------------------------------------------------------


_LEGACY_GFP = IFChannelSpec(
    name="legacy_gfp", channel=1, distribution="moderate",
    density_range_per_mm2=(150.0, 600.0), intensity_range=(0.55, 0.95),
    sigma_range_scale=(0.7, 1.5),
)
_LEGACY_TDTOM = IFChannelSpec(
    name="legacy_tdtom", channel=0, distribution="moderate",
    density_range_per_mm2=(120.0, 500.0), intensity_range=(0.60, 0.95),
    sigma_range_scale=(0.7, 1.5),
)


def _legacy_mode(
    rng: np.random.Generator, p_green: float, p_red: float,
) -> tuple[FluorescenceMode, list[IFChannelSpec]]:
    """Build a synthetic DAPI mode from legacy p_green/p_red flags.

    DEPRECATED — kept for backwards compatibility only.
    """
    channels: list[IFChannelSpec] = []
    if rng.random() < p_green:
        channels.append(_LEGACY_GFP)
    if rng.random() < p_red:
        channels.append(_LEGACY_TDTOM)
    legacy_mode = FluorescenceMode("legacy_compat", 1.0, "dapi", tuple(channels))
    return legacy_mode, channels


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_fluorescence_section(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    seed: int,
    pixel_size_um: float,
    gamma_range: tuple[float, float] = (0.9, 1.7),
    floor_range: tuple[float, float] = (0.10, 0.20),
    mode: FluorescenceMode | None = None,
    # Backwards-compat shims (DEPRECATED; only honoured when mode=None and
    # both p_green/p_red are explicitly passed as non-None floats).
    p_green: float | None = None,
    p_red: float | None = None,
    apply_damage: bool = True,
    damage_intensity: str = "medium",
    apply_geometry_warp: bool = True,
    plane: str = "coronal",
    position_mm: float | None = None,
) -> np.ndarray:
    """Procedurally render one synthetic multiplex fluorescence section.

    Parameters
    ----------
    reference_slice:
        HW grayscale uint8 or float32 — atlas reference template used for
        atlas-grayscale density shading.
    annotation_slice:
        HW int32 BrainGlobe annotation IDs used to build tissue-class masks.
    atlas:
        BrainGlobeAtlas instance passed through to ``classify_tissue``.
    seed:
        Integer seed for ``np.random.default_rng``; same seed → identical output.
    pixel_size_um:
        Rendering resolution in microns per pixel.
    gamma_range:
        (lo, hi) for atlas-grayscale density shaping gamma.
    floor_range:
        (lo, hi) for minimum density floor.
    mode:
        Pin a specific ``FluorescenceMode``; if ``None`` (default) one is
        sampled from ``FLUORESCENCE_MODES`` according to its weight.
    p_green, p_red:
        DEPRECATED.  Legacy probability floats.  Honoured only when ``mode``
        is ``None`` AND both are passed explicitly.  Use ``mode`` instead.
    apply_damage:
        When True (default), apply the damage/realism layer as a final
        post-processing stage after all signal rendering.
    damage_intensity:
        ``"light"`` / ``"medium"`` / ``"heavy"``; passed to
        ``apply_damage_layer``.  Only used when ``apply_damage=True``.

    Returns
    -------
    HWC float32 in [0, 1].
    """
    rng = np.random.default_rng(seed)
    h, w = annotation_slice.shape[:2]

    masks = classify_tissue(annotation_slice, atlas)
    gamma = float(rng.uniform(*gamma_range))
    floor = float(rng.uniform(*floor_range))
    density = atlas_grayscale_density_map(
        reference_slice, masks["tissue"], gamma=gamma, floor=floor,
    )

    ctx = TransformContext(
        modality="fluorescence",
        annotation_slice=annotation_slice,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=pixel_size_um,
        tissue_class_masks=masks,
        plane=plane,
        position_mm=position_mm,
    )

    # -------------------------------------------------------------------
    # Mode selection
    # -------------------------------------------------------------------
    use_legacy = (mode is None and p_green is not None and p_red is not None)

    if use_legacy:
        warnings.warn(
            "render_fluorescence_section: p_green/p_red are deprecated; "
            "pass an explicit mode= instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        chosen_mode, extra_channels = _legacy_mode(rng, p_green, p_red)  # type: ignore[arg-type]
        channels: list[IFChannelSpec] = extra_channels
    elif mode is not None:
        chosen_mode = mode
        channels = list(chosen_mode.channels)
    else:
        chosen_mode = sample_mode(rng, FLUORESCENCE_MODES)
        channels = list(chosen_mode.channels)

    # -------------------------------------------------------------------
    # Generic mode: pick channels at runtime
    # -------------------------------------------------------------------
    if not use_legacy and chosen_mode.name.startswith("generic_"):
        if "1if" in chosen_mode.name:
            n_if = 1
        elif "2if" in chosen_mode.name:
            n_if = 2
        elif "3if" in chosen_mode.name:
            n_if = 3
        else:
            n_if = 1
        channels = _sample_generic_if_channels(rng, n_if)

    # -------------------------------------------------------------------
    # Stage 1: counterstain
    # -------------------------------------------------------------------
    counterstain_fn = COUNTERSTAIN_REGISTRY[chosen_mode.counterstain]
    counterstain_kwargs: dict = {}
    if chosen_mode.counterstain == "nissl":
        # NeuroTrace look — cool blue-cream substrate for tract-tracing modes
        counterstain_kwargs["cream_base"] = (0.74, 0.82, 0.95)
    canvas = counterstain_fn(
        reference_slice, annotation_slice, atlas,
        masks=masks,
        density_map=density,
        ctx=ctx,
        rng=rng,
        pixel_size_um=pixel_size_um,
        **counterstain_kwargs,
    )

    # -------------------------------------------------------------------
    # Stage 2: marker channels
    # -------------------------------------------------------------------
    for spec in channels:
        marker = FluorescenceMarker(
            channel=spec.channel,
            p=1.0,
            density_range_per_mm2=spec.density_range_per_mm2,
            intensity_range=spec.intensity_range,
            sigma_range_scale=spec.sigma_range_scale,
        )
        canvas = marker(canvas, rng=rng, ctx=ctx)

    # -------------------------------------------------------------------
    # Stage 3: brightness/contrast jitter
    # -------------------------------------------------------------------
    canvas = _apply_brightness_contrast(canvas, rng)

    # -------------------------------------------------------------------
    # Stage 4 (optional): damage / realism layer
    # -------------------------------------------------------------------
    if apply_damage:
        canvas = apply_damage_layer(
            canvas, rng=rng, ctx=ctx, modality="fluorescence", intensity=damage_intensity,
            geometry=apply_geometry_warp,
        )
    return canvas
