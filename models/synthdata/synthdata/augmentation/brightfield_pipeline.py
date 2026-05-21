"""End-to-end procedural brightfield IHC section synthesis.

Distinct from Nissl in three structural ways:
1. Stain target varies — different antibodies produce dramatically different
   patterns. Each call picks one of three modes:
       - pan_neuronal       (NeuN-like; dense in GM)
       - sparse_interneuron (PV/Calbindin-like; ~5-10% of GM cells, no WM)
       - myelin             (Woelcke/MBP-like; dense in WM, faint GM)
   The chosen mode is recorded in the rng-derived seed and surfaces in the
   diversity grid.
2. DAB chromogen color (warm brown, R>G>B) — different from Nissl purple.
3. No cytoplasmic blur — DAB precipitate produces sharper edges than the
   diffuse glow of Nissl cytoplasmic stain.

Counterstain support:
    ``counterstain="none"``        — cream tan substrate only (legacy behaviour)
    ``counterstain="hematoxylin"`` — render hematoxylin nuclei first, layer
                                     DAB signal on top at reduced intensity
    ``counterstain="auto"``        — 70% none / 30% hematoxylin per call
"""

from __future__ import annotations

import numpy as np

from .counterstain import render_hematoxylin_counterstain
from .damage_pipeline import apply_damage_layer
from .density import atlas_grayscale_density_map, shade_substrate_by_atlas
from .signals import apply_dab_signal
from .transforms.base import TransformContext
from .transforms.texture import (
    BrightfieldGrayMatterDAB,
    BrightfieldWhiteMatterDAB,
)
from .transforms.tissue_class import classify_tissue

__all__ = ["render_brightfield_section", "BRIGHTFIELD_MODES"]


BRIGHTFIELD_MODES: tuple[str, ...] = (
    "pan_neuronal",
    "sparse_interneuron",
    "myelin",
)

# Fraction of "auto" calls that get a hematoxylin counterstain.
# 30 % — most chromogenic-IHC labs ship without counterstain because
# the brown signal is the focal feature; hematoxylin is added when
# anatomical context matters (Allen Brain Atlas style papers, clinical
# sections, etc.).
_HEMATOXYLIN_AUTO_PROB = 0.30


def _build_tan_canvas(
    shape: tuple[int, int],
    substrate_mask: np.ndarray,
    tan: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Beige tan substrate where ``substrate_mask`` is True; slide-white
    elsewhere. ``substrate_mask`` should be gray + white matter only so
    ventricles fall through to the slide background tone.
    """
    h, w = shape
    canvas = np.empty((h, w, 3), dtype=np.float32)
    bg_color = rng.uniform(0.92, 0.98, size=3).astype(np.float32)
    canvas[..., 0] = bg_color[0]
    canvas[..., 1] = bg_color[1]
    canvas[..., 2] = bg_color[2]
    tan_jitter = rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
    tan_arr = np.clip(np.array(tan, dtype=np.float32) + tan_jitter, 0.0, 1.0)
    canvas[substrate_mask] = tan_arr
    return canvas


def _apply_brightfield_tone_shift(
    canvas: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tan/yellow drift mimicking microscope white-balance variation."""
    warm = rng.uniform(-0.05, 0.10)
    yellow = rng.uniform(-0.04, 0.06)
    out = canvas.copy()
    out[..., 0] += warm * 0.4
    out[..., 1] += warm * 0.3 + yellow * 0.3
    out[..., 2] += -yellow * 0.3
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


def render_brightfield_section(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    seed: int,
    pixel_size_um: float,
    mode: str | None = None,
    gamma_range: tuple[float, float] = (0.9, 1.7),
    floor_range: tuple[float, float] = (0.10, 0.20),
    tan_base: tuple[float, float, float] = (0.85, 0.78, 0.62),
    counterstain: str = "auto",
    apply_damage: bool = True,
    damage_intensity: str = "medium",
    apply_geometry_warp: bool = True,
    plane: str = "coronal",
    position_mm: float | None = None,
) -> np.ndarray:
    """Procedurally render one synthetic brightfield IHC section.

    Parameters
    ----------
    reference_slice:
        Grayscale reference atlas slice (H × W uint8).
    annotation_slice:
        Annotation atlas slice (H × W int32).
    atlas:
        BrainGlobe atlas object.
    seed:
        RNG seed for reproducibility.
    pixel_size_um:
        Physical pixel size in micrometres.
    mode:
        Staining mode: ``"pan_neuronal"``, ``"sparse_interneuron"``, or
        ``"myelin"``.  When ``None``, one of ``BRIGHTFIELD_MODES`` is
        chosen at random.
    gamma_range:
        Range for atlas-shading gamma exponent (``counterstain="none"`` path).
    floor_range:
        Range for density-map floor (``counterstain="none"`` path).
    tan_base:
        Substrate RGB triple for the ``counterstain="none"`` path.
    counterstain:
        ``"none"``        — cream tan substrate only (legacy behaviour).
        ``"hematoxylin"`` — render hematoxylin nuclei first, layer DAB on top.
        ``"auto"``        — 70 % none / 30 % hematoxylin, chosen per call.
    apply_damage:
        When True (default), apply the damage/realism layer as a final
        post-processing stage after all signal rendering.
    damage_intensity:
        ``"light"`` / ``"medium"`` / ``"heavy"``; passed to
        ``apply_damage_layer``.  Only used when ``apply_damage=True``.
    """
    rng = np.random.default_rng(seed)
    h, w = annotation_slice.shape[:2]

    chosen_mode = mode if mode is not None else str(rng.choice(BRIGHTFIELD_MODES))

    # Resolve counterstain choice before consuming any more RNG state so that
    # pinning counterstain="none" / "hematoxylin" gives deterministic images
    # regardless of the auto-sampling branch.
    if counterstain == "auto":
        use_hematoxylin = rng.random() < _HEMATOXYLIN_AUTO_PROB
    elif counterstain == "hematoxylin":
        use_hematoxylin = True
    elif counterstain == "none":
        use_hematoxylin = False
    else:
        raise ValueError(
            f"Unknown counterstain {counterstain!r}; expected 'none', 'hematoxylin', or 'auto'."
        )

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
        modality="brightfield",
        annotation_slice=annotation_slice,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=pixel_size_um,
        tissue_class_masks=masks,
        plane=plane,
        position_mm=position_mm,
    )

    if use_hematoxylin:
        # ------------------------------------------------------------------ #
        # Hematoxylin path: build cellular-anatomy layer first, then DAB on top.
        #
        # Two tuning knobs preserve both stains visually:
        #   1. Reduced hematoxylin density (1000–2000 GM, 100–250 WM per mm²)
        #      vs. the standalone hematoxylin preset (3000–4500 GM, 200–450 WM).
        #      Counterstain-mode hematoxylin should be light — nuclei visible
        #      but not a solid blue field — so the brown DAB can stand out.
        #   2. Reduced DAB intensity scale (0.55) so brown signal does not
        #      overwhelm the blue-violet nuclei already on the canvas.
        # ------------------------------------------------------------------ #
        canvas = render_hematoxylin_counterstain(
            reference_slice,
            annotation_slice,
            atlas,
            masks=masks,
            density_map=density,
            ctx=ctx,
            rng=rng,
            pixel_size_um=pixel_size_um,
            # Lighter nuclear density for counterstain role
            density_range_per_mm2_gm=(1000.0, 2000.0),
            density_range_per_mm2_wm=(80.0, 200.0),
        )
        # DAB signal on top — reduced intensity when counterstain present so
        # both stains remain visually distinct (blue nuclei not overwhelmed).
        canvas = apply_dab_signal(
            canvas,
            ctx=ctx,
            rng=rng,
            mode=chosen_mode,
            dab_intensity_scale=0.55,  # ~45% reduction vs full-strength
        )
        canvas = _apply_brightness_contrast(canvas, rng)
    else:
        # ------------------------------------------------------------------ #
        # No-counterstain path (legacy behaviour, unchanged).
        # ------------------------------------------------------------------ #
        substrate_mask = masks["gray_matter"] | masks["white_matter"]
        canvas = _build_tan_canvas((h, w), substrate_mask, tan_base, rng)
        shading_strength = float(rng.uniform(0.10, 0.24))
        canvas = shade_substrate_by_atlas(
            canvas,
            reference_slice,
            substrate_mask,
            strength=shading_strength,
            gamma=1.0,
        )
        canvas = BrightfieldGrayMatterDAB(p=1.0, tan=tan_base)(
            canvas,
            rng=rng,
            ctx=ctx,
            mode=chosen_mode,
        )
        canvas = BrightfieldWhiteMatterDAB(p=1.0, tan=tan_base)(
            canvas,
            rng=rng,
            ctx=ctx,
            mode=chosen_mode,
        )
        canvas = _apply_brightfield_tone_shift(canvas, rng)
        canvas = _apply_brightness_contrast(canvas, rng)

    if apply_damage:
        canvas = apply_damage_layer(
            canvas,
            rng=rng,
            ctx=ctx,
            modality="brightfield",
            intensity=damage_intensity,
            geometry=apply_geometry_warp,
        )
    return canvas
