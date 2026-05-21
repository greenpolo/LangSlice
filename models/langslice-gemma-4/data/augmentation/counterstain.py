"""Counterstain renderers — anatomical-context layer for layered modalities.

Each `render_*_counterstain` function produces a fully-rendered HWC float32 [0, 1]
canvas containing substrate + counterstain cells. Higher-level pipelines (ISH,
Brightfield, Fluorescence) pick a counterstain via ``COUNTERSTAIN_REGISTRY``,
call its renderer to produce the substrate, then layer signal-specific marks
on top via the wrappers in ``signals.py``.

Renderer signature (uniform across all):

    render_<name>_counterstain(
        reference_slice: np.ndarray,
        annotation_slice: np.ndarray,
        atlas: object,
        *,
        masks: dict[str, np.ndarray],
        density_map: np.ndarray,
        ctx: TransformContext,
        rng: np.random.Generator,
        pixel_size_um: float,
        **kwargs,
    ) -> np.ndarray
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .density import shade_substrate_by_atlas
from .transforms.base import TransformContext
from .transforms.texture import (
    DAPIGrayMatterNuclei,
    DAPIWhiteMatterNuclei,
    HematoxylinGrayMatterNuclei,
    HematoxylinWhiteMatterNuclei,
    NisslGrayMatterCellBodies,
    NisslWhiteMatterCellBodies,
    NuclearFastRedGrayMatterNuclei,
    NuclearFastRedWhiteMatterNuclei,
)

__all__ = [
    "render_dapi_counterstain",
    "render_nissl_counterstain",
    "render_no_counterstain",
    "render_hematoxylin_counterstain",
    "render_nuclear_fast_red_counterstain",
    "COUNTERSTAIN_REGISTRY",
]


# ---------------------------------------------------------------------------
# Substrate constructors
# ---------------------------------------------------------------------------


def _build_substrate(
    shape: tuple[int, int],
    substrate_mask: np.ndarray,
    substrate_color: tuple[float, float, float],
    rng: np.random.Generator,
    *,
    bg_range: tuple[float, float] = (0.93, 0.98),
    color_jitter: float = 0.04,
) -> np.ndarray:
    """Slide-white outside ``substrate_mask``; jittered ``substrate_color`` inside."""
    h, w = shape
    canvas = np.empty((h, w, 3), dtype=np.float32)
    bg = rng.uniform(bg_range[0], bg_range[1], size=3).astype(np.float32)
    canvas[..., 0] = bg[0]
    canvas[..., 1] = bg[1]
    canvas[..., 2] = bg[2]
    color_arr = np.clip(
        np.array(substrate_color, dtype=np.float32)
        + rng.uniform(-color_jitter, color_jitter, size=3).astype(np.float32),
        0.0,
        1.0,
    )
    canvas[substrate_mask] = color_arr
    return canvas


# ---------------------------------------------------------------------------
# DAPI counterstain — fluorescent, dark canvas, blue nuclear dots
# ---------------------------------------------------------------------------


def render_dapi_counterstain(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    masks: dict[str, np.ndarray],
    density_map: np.ndarray,
    ctx: TransformContext,
    rng: np.random.Generator,
    pixel_size_um: float,
    density_range_per_mm2: tuple[float, float] = (2200.0, 3500.0),  # noqa: ARG001 — legacy
    intensity_range: tuple[float, float] = (0.55, 0.90),  # noqa: ARG001 — legacy
) -> np.ndarray:
    """DAPI counterstain — additive blue blobs on near-black canvas.

    Uses the GT-calibrated DAPI texture renderers (see ``transforms/dapi_texture.py``).
    The ``density_range_per_mm2`` and ``intensity_range`` kwargs are kept for
    API compatibility but no longer have effect (calibrated parameters live
    in ``DAPI_GM_PARAMS`` / ``DAPI_WM_PARAMS``).
    """
    h, w = annotation_slice.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas = DAPIGrayMatterNuclei(p=1.0)(canvas, rng=rng, ctx=ctx)
    canvas = DAPIWhiteMatterNuclei(p=1.0)(canvas, rng=rng, ctx=ctx)
    # Coverage map = where blue brightness landed; used by signal layers to
    # avoid double-staining nuclei.
    coverage = canvas[..., 2]  # blue channel = nucleus indicator
    if ctx.counterstain_signal_mask is None:
        ctx.counterstain_signal_mask = (coverage > 0.05).astype(np.float32)
    else:
        ctx.counterstain_signal_mask = np.maximum(
            ctx.counterstain_signal_mask,
            (coverage > 0.05).astype(np.float32),
        )
    return canvas


# ---------------------------------------------------------------------------
# Nissl counterstain — cream substrate, purple cell bodies
# ---------------------------------------------------------------------------


_NISSL_CREAM_PRESETS: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.78, 0.65),  # warm cream-tan thionine
    (0.78, 0.78, 0.85),  # neutral cream
    (0.74, 0.82, 0.95),  # blue-cream cresyl-violet (Mouse10-like)
    (0.82, 0.75, 0.88),  # purple-cream
)


def render_nissl_counterstain(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    masks: dict[str, np.ndarray],
    density_map: np.ndarray,
    ctx: TransformContext,
    rng: np.random.Generator,
    pixel_size_um: float,
    cream_base: tuple[float, float, float] | None = None,
    shading_strength_range: tuple[float, float] = (0.12, 0.28),
) -> np.ndarray:
    """Nissl counterstain — cream substrate, atlas-shaded, purple Nissl cells.

    Picks a cream tone preset per call when ``cream_base`` is not given.
    Covers warm-cream thionine, neutral, blue-cream cresyl-violet, and
    purple-cream variants.
    """
    h, w = annotation_slice.shape[:2]
    if cream_base is None:
        cream_base = _NISSL_CREAM_PRESETS[int(rng.integers(0, len(_NISSL_CREAM_PRESETS)))]
    substrate_mask = masks["gray_matter"] | masks["white_matter"]
    canvas = _build_substrate((h, w), substrate_mask, cream_base, rng)
    shading_strength = float(rng.uniform(*shading_strength_range))
    canvas = shade_substrate_by_atlas(
        canvas,
        reference_slice,
        substrate_mask,
        strength=shading_strength,
        gamma=1.0,
    )
    canvas = NisslGrayMatterCellBodies(p=1.0, cream=cream_base)(canvas, rng=rng, ctx=ctx)
    canvas = NisslWhiteMatterCellBodies(p=1.0, cream=cream_base)(canvas, rng=rng, ctx=ctx)
    # Coverage: any pixel meaningfully darker than substrate cream is a Nissl
    # cell. Approximated via luminance drop from cream baseline.
    cream_lum = 0.2126 * cream_base[0] + 0.7152 * cream_base[1] + 0.0722 * cream_base[2]
    canvas_lum = 0.2126 * canvas[..., 0] + 0.7152 * canvas[..., 1] + 0.0722 * canvas[..., 2]
    coverage = np.clip((cream_lum - canvas_lum) / max(cream_lum * 0.5, 1e-6), 0.0, 1.0)
    if ctx.counterstain_signal_mask is None:
        ctx.counterstain_signal_mask = coverage.astype(np.float32)
    else:
        ctx.counterstain_signal_mask = np.maximum(
            ctx.counterstain_signal_mask,
            coverage.astype(np.float32),
        )
    return canvas


# ---------------------------------------------------------------------------
# No-counterstain — Allen-style: lavender-tinted substrate, atlas shading only
# ---------------------------------------------------------------------------


def render_no_counterstain(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    masks: dict[str, np.ndarray],
    density_map: np.ndarray,
    ctx: TransformContext,
    rng: np.random.Generator,
    pixel_size_um: float,
    substrate_color: tuple[float, float, float] = (0.88, 0.86, 0.92),
    shading_strength_range: tuple[float, float] = (0.10, 0.22),
) -> np.ndarray:
    """Allen-style: faint lavender-cream substrate, no rendered cells.

    The pale anatomical structure visible in real Allen ISH images comes from
    intrinsic tissue scattering + atlas-grayscale shading, not a counterstain.
    """
    h, w = annotation_slice.shape[:2]
    substrate_mask = masks["gray_matter"] | masks["white_matter"]
    canvas = _build_substrate((h, w), substrate_mask, substrate_color, rng)
    shading_strength = float(rng.uniform(*shading_strength_range))
    canvas = shade_substrate_by_atlas(
        canvas,
        reference_slice,
        substrate_mask,
        strength=shading_strength,
        gamma=1.0,
    )
    # No cellular signal mask — substrate is uniform within tissue.
    if ctx.counterstain_signal_mask is None:
        ctx.counterstain_signal_mask = np.zeros((h, w), dtype=np.float32)
    return canvas


# ---------------------------------------------------------------------------
# Registry — populated by other counterstain modules as they're added
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hematoxylin counterstain — brightfield, pale cream substrate, blue-violet nuclei
# ---------------------------------------------------------------------------


_HEMATOXYLIN_SUBSTRATE_PRESETS: tuple[tuple[float, float, float], ...] = (
    (0.92, 0.88, 0.90),  # classic pale pink-cream (alone-as-counterstain)
    (0.93, 0.86, 0.85),  # slightly warmer cream
    (0.88, 0.88, 0.92),  # cooler / blueish neutral
    (0.88, 0.85, 0.92),  # pale violet-cream
)


def render_hematoxylin_counterstain(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    masks: dict[str, np.ndarray],
    density_map: np.ndarray,
    ctx: TransformContext,
    rng: np.random.Generator,
    pixel_size_um: float,
    substrate_base: tuple[float, float, float] | None = None,
    shading_strength_range: tuple[float, float] = (0.08, 0.20),
    gamma_range: tuple[float, float] = (0.8, 1.6),
    density_range_per_mm2_gm: tuple[float, float] = (3000.0, 4500.0),
    density_range_per_mm2_wm: tuple[float, float] = (200.0, 450.0),
) -> np.ndarray:
    """Hematoxylin counterstain — pale pink-cream substrate, blue-violet nuclear dots.

    Used as the anatomical context layer in IHC (DAB+H) and chromogenic ISH
    (NBT+H, RNAscope+H). Picks a substrate tone preset per call when
    ``substrate_base`` is not provided. Covers classic pale pink-cream,
    slightly warmer cream, cool/neutral, and pale violet-cream variants.

    Key difference from Nissl: nuclei only (no cytoplasmic blur), smaller
    blobs, and the substrate is closer to white/near-white rather than the
    warm cream-tan of Nissl thionine.
    """
    h, w = annotation_slice.shape[:2]
    if substrate_base is None:
        substrate_base = _HEMATOXYLIN_SUBSTRATE_PRESETS[
            int(rng.integers(0, len(_HEMATOXYLIN_SUBSTRATE_PRESETS)))
        ]

    substrate_mask = masks["gray_matter"] | masks["white_matter"]
    canvas = _build_substrate((h, w), substrate_mask, substrate_base, rng)

    shading_strength = float(rng.uniform(*shading_strength_range))
    gamma = float(rng.uniform(*gamma_range))
    # Use gamma-adjusted reference for shading so cell-dense cortical layers
    # get a mild darkening without over-darkening WM tracts.
    canvas = shade_substrate_by_atlas(
        canvas,
        reference_slice,
        substrate_mask,
        strength=shading_strength,
        gamma=gamma,
    )

    canvas = HematoxylinGrayMatterNuclei(
        p=1.0,
        density_range_per_mm2=density_range_per_mm2_gm,
    )(canvas, rng=rng, ctx=ctx)
    canvas = HematoxylinWhiteMatterNuclei(
        p=1.0,
        density_range_per_mm2=density_range_per_mm2_wm,
    )(canvas, rng=rng, ctx=ctx)

    # Coverage: any pixel meaningfully darker than substrate is a stained nucleus.
    sub_lum = 0.2126 * substrate_base[0] + 0.7152 * substrate_base[1] + 0.0722 * substrate_base[2]
    canvas_lum = 0.2126 * canvas[..., 0] + 0.7152 * canvas[..., 1] + 0.0722 * canvas[..., 2]
    coverage = np.clip(
        (sub_lum - canvas_lum) / max(sub_lum * 0.5, 1e-6),
        0.0,
        1.0,
    )
    if ctx.counterstain_signal_mask is None:
        ctx.counterstain_signal_mask = coverage.astype(np.float32)
    else:
        ctx.counterstain_signal_mask = np.maximum(
            ctx.counterstain_signal_mask,
            coverage.astype(np.float32),
        )
    return canvas


# ---------------------------------------------------------------------------
# Nuclear Fast Red counterstain — brightfield, pale pink substrate, red nuclei
# ---------------------------------------------------------------------------


_NUCLEAR_FAST_RED_SUBSTRATE_PRESETS: dict[str, tuple[float, float, float]] = {
    # Calibrated against Allen ISH P56 mouse brain NFR counterstain (CC0).
    # Real ISH scans show a very pale, nearly-white substrate with a subtle
    # pink/lavender tint — not the saturated rose often seen in textbook photos.
    "pale_blush": (0.96, 0.94, 0.95),
    "pale_lavender": (0.95, 0.94, 0.96),
    "cool_white": (0.96, 0.95, 0.96),
    "warm_white": (0.97, 0.95, 0.94),
}


def render_nuclear_fast_red_counterstain(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    masks: dict[str, np.ndarray],
    density_map: np.ndarray,
    ctx: TransformContext,
    rng: np.random.Generator,
    pixel_size_um: float,
    substrate_base: tuple[float, float, float] | None = None,
    shading_strength_range: tuple[float, float] = (0.08, 0.20),
    gamma_range: tuple[float, float] = (0.8, 1.6),
    density_range_per_mm2_gm: tuple[float, float] = (3000.0, 4500.0),
    density_range_per_mm2_wm: tuple[float, float] = (200.0, 450.0),
) -> np.ndarray:
    """Nuclear Fast Red counterstain on a pale pink brightfield substrate."""
    h, w = annotation_slice.shape[:2]
    if substrate_base is None:
        presets = tuple(_NUCLEAR_FAST_RED_SUBSTRATE_PRESETS.values())
        substrate_base = presets[int(rng.integers(0, len(presets)))]

    substrate_mask = masks["gray_matter"] | masks["white_matter"]
    canvas = _build_substrate((h, w), substrate_mask, substrate_base, rng)

    shading_strength = float(rng.uniform(*shading_strength_range))
    gamma = float(rng.uniform(*gamma_range))
    canvas = shade_substrate_by_atlas(
        canvas,
        reference_slice,
        substrate_mask,
        strength=shading_strength,
        gamma=gamma,
    )

    canvas = NuclearFastRedGrayMatterNuclei(
        p=1.0,
        density_range_per_mm2=density_range_per_mm2_gm,
    )(canvas, rng=rng, ctx=ctx)
    canvas = NuclearFastRedWhiteMatterNuclei(
        p=1.0,
        density_range_per_mm2=density_range_per_mm2_wm,
    )(canvas, rng=rng, ctx=ctx)

    sub_lum = 0.2126 * substrate_base[0] + 0.7152 * substrate_base[1] + 0.0722 * substrate_base[2]
    canvas_lum = 0.2126 * canvas[..., 0] + 0.7152 * canvas[..., 1] + 0.0722 * canvas[..., 2]
    coverage = np.clip(
        (sub_lum - canvas_lum) / max(sub_lum * 0.5, 1e-6),
        0.0,
        1.0,
    )
    if ctx.counterstain_signal_mask is None:
        ctx.counterstain_signal_mask = coverage.astype(np.float32)
    else:
        ctx.counterstain_signal_mask = np.maximum(
            ctx.counterstain_signal_mask,
            coverage.astype(np.float32),
        )
    return canvas


CounterstainFn = Callable[..., np.ndarray]

COUNTERSTAIN_REGISTRY: dict[str, CounterstainFn] = {
    "dapi": render_dapi_counterstain,
    "nissl": render_nissl_counterstain,
    "none": render_no_counterstain,
    "hematoxylin": render_hematoxylin_counterstain,
    "nfr": render_nuclear_fast_red_counterstain,
}
