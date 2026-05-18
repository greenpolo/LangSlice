"""Signal-renderer wrappers for layered ISH/IHC modality pipelines.

Each ``apply_*_signal`` function is a **signal layer** — it takes a
canvas that has already been counterstain-rendered and adds the
modality-specific expression signal on top.

Renderer signature (uniform across all entries in SIGNAL_REGISTRY):

    apply_<name>_signal(
        canvas: np.ndarray,            # HWC float32 [0,1] — pre-rendered counterstain
        *,
        ctx: TransformContext,
        rng: np.random.Generator,
        **kwargs,                       # mode-specific overrides forwarded from ISHMode.signal_kwargs
    ) -> np.ndarray                    # HWC float32 [0,1]

The renderers respect ``ctx.counterstain_signal_mask`` when it is not
None: pixels where the counterstain has already deposited pigment
(value > 0) are excluded or softened so that wash/chromogen does not
double-stain already-coloured nuclei.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .transforms.base import TransformContext
from .transforms.texture import (
    BrightfieldGrayMatterDAB,
    BrightfieldWhiteMatterDAB,
    FluorescenceMarker,
    ISHExpressionPuncta,
)

__all__ = [
    "apply_nbt_bcip_signal",
    "apply_dab_signal",
    "apply_fluorescent_probe_signal",
    "SIGNAL_REGISTRY",
]


# ---------------------------------------------------------------------------
# NBT/BCIP — purple chromogenic ISH signal
# ---------------------------------------------------------------------------


def apply_nbt_bcip_signal(
    canvas: np.ndarray,
    *,
    ctx: TransformContext,
    rng: np.random.Generator,
    wash_color: tuple[float, float, float] = (0.30, 0.25, 0.55),
    punctum_color_range: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float],
    ] = ((0.18, 0.40), (0.15, 0.32), (0.45, 0.70)),
    density_range_per_mm2: tuple[float, float] = (1500.0, 3500.0),
    wash_strength_range: tuple[float, float] = (0.04, 0.14),
    **kwargs,
) -> np.ndarray:
    """NBT/BCIP purple ISH signal layered on top of a pre-rendered counterstain.

    When ``ctx.counterstain_signal_mask`` is populated the diffuse wash
    alpha is dampened on nucleus-stained pixels.  This prevents the purple
    wash from visibly tinting already-coloured hematoxylin / NFR / Nissl
    nuclei while leaving the background substrate free to accumulate colour.

    The puncta are rendered with ``ISHExpressionPuncta`` at its default
    parameters but ``cream`` is inferred from the mean tissue-mask colour
    of the supplied canvas so the substrate reference point is always
    contextually appropriate.
    """
    h, w = canvas.shape[:2]

    # Infer a substrate "cream" reference from the canvas itself so that
    # ISHExpressionPuncta's internal wash calculation is anchored to the
    # actual background rather than a hard-coded lavender preset.
    if ctx.tissue_mask is not None and ctx.tissue_mask.any():
        cream_rgb = canvas[ctx.tissue_mask].mean(axis=0)
        cream = (float(cream_rgb[0]), float(cream_rgb[1]), float(cream_rgb[2]))
    else:
        cream = (0.88, 0.86, 0.92)  # fallback lavender-cream

    # Build a scratch context copy so ISHExpressionPuncta uses the same
    # counterstain_signal_mask and density_map from ctx.
    transform = ISHExpressionPuncta(
        p=1.0,
        density_range_per_mm2=density_range_per_mm2,
        wash_strength_range=wash_strength_range,
        cream=cream,
        wash_color=wash_color,
        punctum_color_range=punctum_color_range,
    )
    result = transform(canvas, rng=rng, ctx=ctx)

    # Dampen wash on counterstain-nucleus pixels: where the mask value is
    # high (= nucleus was stained) we alpha-blend back toward the canvas
    # *before* wash so the nucleus colour dominates.
    mask = ctx.counterstain_signal_mask
    if mask is not None and mask.any():
        # mask: HW float32 in [0,1] — 1 = nucleus pixel
        # Blend: result_protected = canvas * mask + result * (1 - mask)
        # This is equivalent to rolling back the wash proportionally to the
        # fraction of counterstain coverage.
        dampen = mask[..., None]
        result = np.clip(canvas * dampen + result * (1.0 - dampen), 0.0, 1.0)

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# DAB — brown chromogenic IHC/ISH signal
# ---------------------------------------------------------------------------


def apply_dab_signal(
    canvas: np.ndarray,
    *,
    ctx: TransformContext,
    rng: np.random.Generator,
    mode: str | None = None,
    dab_intensity_scale: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """DAB-brown chromogenic signal (IHC antibody or chromogenic ISH).

    Applies BrightfieldGrayMatterDAB and BrightfieldWhiteMatterDAB.  The
    ``mode`` parameter (``"pan_neuronal"``, ``"sparse_interneuron"``,
    ``"myelin"``) is picked randomly when not specified.  Callers that want
    a deterministic mode can pass it via ``ISHMode.signal_kwargs``.

    The existing DAB transforms absorb nuclear-protein overlap automatically:
    since DAB is an additive darkening of the substrate it composites
    gracefully on top of any brightfield counterstain without special
    mask handling.

    Parameters
    ----------
    dab_intensity_scale:
        Scalar in (0, 1] applied to the DAB darkening delta before it is
        composited onto ``canvas``.  Pass a value < 1.0 when a counterstain
        is present so the brown signal does not overwhelm the nuclear colour
        (e.g. 0.65 for hematoxylin counterstain).
    """
    if mode is None:
        modes = ("pan_neuronal", "sparse_interneuron", "myelin")
        mode = modes[int(rng.integers(0, len(modes)))]

    if dab_intensity_scale == 1.0:
        # Fast path — no scaling needed
        out = BrightfieldGrayMatterDAB(p=1.0)(canvas, rng=rng, ctx=ctx, mode=mode)
        out = BrightfieldWhiteMatterDAB(p=1.0)(out, rng=rng, ctx=ctx, mode=mode)
    else:
        # Render DAB onto a white reference canvas to isolate the delta, then
        # scale and add to the real canvas.  This preserves hue of the brown
        # while reducing its contribution so counterstain nuclei stay visible.
        white = np.ones_like(canvas)
        dab_on_white = BrightfieldGrayMatterDAB(p=1.0)(white, rng=rng, ctx=ctx, mode=mode)
        dab_on_white = BrightfieldWhiteMatterDAB(p=1.0)(dab_on_white, rng=rng, ctx=ctx, mode=mode)
        delta = dab_on_white - white  # negative (darkening), shape HWC
        out = np.clip(canvas + delta * float(dab_intensity_scale), 0.0, 1.0)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Fluorescent probe — FISH / RNAscope signal
# ---------------------------------------------------------------------------


def apply_fluorescent_probe_signal(
    canvas: np.ndarray,
    *,
    ctx: TransformContext,
    rng: np.random.Generator,
    channel: int = 1,
    density_range_per_mm2: tuple[float, float] = (60.0, 300.0),
    sigma_range_scale: tuple[float, float] = (0.5, 1.0),
    intensity_range: tuple[float, float] = (0.65, 1.0),
    **kwargs,
) -> np.ndarray:
    """Fluorescent ISH probe (FISH / RNAscope) on a DAPI-counterstained canvas.

    Renders a single sparse channel of bright spots using ``FluorescenceMarker``
    at lower density than a cell-fill marker (these are individual mRNA transcripts
    or small clusters, not whole-cell fills).

    Default ``channel=1`` (green) mimics Alexa-488 / FITC probe appearance.
    Use ``channel=0`` for red (Cy3 / Alexa-555) or ``channel=2`` for a second
    blue probe (unusual but valid).

    The DAPI canvas underneath is already on a near-black background so the
    probe dots are additive and clearly visible without special compositing.
    The counterstain_signal_mask is not used here because the fluorescent probe
    signal and DAPI nuclear dots occupy different axial channels.
    """
    transform = FluorescenceMarker(
        channel=channel,
        p=1.0,
        density_range_per_mm2=density_range_per_mm2,
        sigma_range_scale=sigma_range_scale,
        intensity_range=intensity_range,
    )
    return transform(canvas, rng=rng, ctx=ctx).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SignalFn = Callable[..., np.ndarray]

SIGNAL_REGISTRY: dict[str, SignalFn] = {
    "nbt_bcip": apply_nbt_bcip_signal,
    "dab": apply_dab_signal,
    "fluorescent_probe": apply_fluorescent_probe_signal,
}
