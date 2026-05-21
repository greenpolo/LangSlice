"""Damage / realism layer orchestrator for procedural section pipelines.

``apply_damage_layer`` applies a randomised subset of damage and geometry
transforms to a rendered canvas as a final post-processing stage.  Callers
pass ``modality`` so light-background vs dark-background rules can be applied
(e.g. ``EmbeddingHalos`` is only meaningful on brightfield/Nissl/ISH, not on
DAPI or fluorescence dark canvases).

Intensity levels
----------------
``"light"``  — subtle perturbations; good for borderline-clean training examples.
``"medium"`` — default; visibly realistic but anatomical content fully legible.
``"heavy"``  — pushed params; clearly degraded but still recognisable.

Probability schedule (medium)
-----------------------------
ALWAYS     BladeStretchHorizontal              — horizontal-biased anisotropic stretch
                                               (~22% max H stretch)
85%        IlluminationGradient                — lamp / photobleaching gradient
75%        AffineJitter                        — small rotation + translation
70%        EmbeddingHalos                      — warm halo outside tissue (light-bg only)
55%        VentricleExpansion                  — CSF retraction (pre-warp so it follows the warp)
45%        Folds                               — sinusoidal tissue-fold warp
40%        AnteriorOlfactoryBulbDetachment     — drop OB in anterior coronal
                                               sections, recenter tissue
30%        AnteriorIsocortexDetachment         — drop emerging isocortex in narrow AP 1.5-3.5 window
35%        Microbubbles                        — sparse refractive bubbles (toned down)
25%        Debris                              — small dark dust specks on tissue

Disabled: ``ResolutionShift`` (real microscopy is in focus) and ``Tears`` (the
edge-bite mode produced grainy black discs that don't match real damage).
Both classes still live in the codebase for the deprecated ``pipeline.py``
path but are no longer wired into ``apply_damage_layer``.

Light-background modalities that accept EmbeddingHalos:
    nissl, brightfield, ish

Dark-background modalities (EmbeddingHalos skipped):
    dapi, fluorescence
"""

from __future__ import annotations

import numpy as np

from .transforms.base import TransformContext
from .transforms.damage import (
    AnteriorIsocortexDetachment,
    AnteriorOlfactoryBulbDetachment,
    Debris,
    EmbeddingHalos,
    Folds,
    HemibrainPreparation,
    IlluminationGradient,
    Microbubbles,
    PosteriorWingDamage,
    VentricleExpansion,
)
from .transforms.geometry import AffineJitter, BladeStretchHorizontal

__all__ = ["apply_damage_layer"]

# Modalities with a light (bright) background substrate.
_LIGHT_BG_MODALITIES = {"nissl", "brightfield", "ish"}

# ---------------------------------------------------------------------------
# Intensity multipliers: (probability_scale, param_scale)
# ---------------------------------------------------------------------------
_INTENSITY_PROB_SCALE = {
    "light": 0.65,
    "medium": 1.00,
    "heavy": 1.45,
}
_INTENSITY_PARAM_SCALE = {
    "light": 0.65,
    "medium": 1.00,
    "heavy": 1.75,
}


def apply_damage_layer(
    canvas: np.ndarray,
    *,
    rng: np.random.Generator,
    ctx: TransformContext,
    modality: str,
    intensity: str = "medium",
    geometry: bool = True,
) -> np.ndarray:
    """Apply a randomised damage and geometry layer to a rendered canvas.

    Parameters
    ----------
    canvas:
        HWC float32 in [0, 1] — the clean rendered section.
    rng:
        Seeded numpy Generator shared with the calling pipeline.
    ctx:
        TransformContext from the calling pipeline.
    modality:
        One of ``"dapi"``, ``"nissl"``, ``"brightfield"``, ``"fluorescence"``,
        ``"ish"``.  Controls which transforms are eligible (e.g. EmbeddingHalos
        is skipped for dark-background modalities).
    intensity:
        ``"light"`` / ``"medium"`` / ``"heavy"``.  Scales both probabilities
        and transform magnitude parameters.
    geometry:
        When True (default), apply pixel-displacing transforms
        (``BladeStretchHorizontal``, ``AffineJitter``, ``Folds``, ``Tears``,
        ``Microbubbles``).  When False, those are skipped while non-coord
        transforms (illumination, halos, debris, resolution-shift) still run.
        Set False for the bbox-grounding bucket so the saved image stays
        coord-aligned with the bbox computed from the unwarped annotation.

    Returns
    -------
    np.ndarray
        HWC float32 in [0, 1].
    """
    if intensity not in _INTENSITY_PROB_SCALE:
        raise ValueError(f"intensity must be 'light', 'medium', or 'heavy'; got {intensity!r}")

    prob_scale = _INTENSITY_PROB_SCALE[intensity]
    param_scale = _INTENSITY_PARAM_SCALE[intensity]
    light_bg = modality in _LIGHT_BG_MODALITIES

    def _p(base: float) -> float:
        """Clamp scaled probability to [0, 1]."""
        return min(1.0, base * prob_scale)

    out = canvas

    # --- 0. Deliberate hemibrain preparation --------------------------------
    # Researchers often mount a single hemisphere on purpose. Model this as a
    # clean preparation variant before adding microtome/slide artifacts.
    if geometry:
        hemibrain = HemibrainPreparation(p=_p(0.08))
        out = hemibrain(out, rng=rng, ctx=ctx)

    # --- 0a2. AnteriorOlfactoryBulbDetachment (pre-warp) --------------------
    # Anterior coronal sections frequently lose one or both olfactory bulbs
    # during cryosectioning; the FOV recenters on the remaining cortex. Must
    # run before VentricleExpansion + geometry warps so all downstream stages
    # see the recentered tissue.
    if geometry:
        ob_drop = AnteriorOlfactoryBulbDetachment(p=_p(0.40))
        out = ob_drop(out, rng=rng, ctx=ctx)

    # --- 0a3. AnteriorIsocortexDetachment (pre-warp) ------------------------
    # In the narrow AP window where isocortex is just emerging at the
    # anterior pole, the thin dorsal cortical wedge can detach during
    # sectioning while the OB and other structures stay intact.
    if geometry:
        iso_drop = AnteriorIsocortexDetachment(p=_p(0.30))
        out = iso_drop(out, rng=rng, ctx=ctx)

    # --- 0b. VentricleExpansion (pre-warp) ----------------------------------
    # CSF retraction makes ventricles read larger than the atlas predicts.
    # MUST run before BladeStretchHorizontal / AffineJitter / Folds so the
    # expanded ventricle warps along with the canvas — running it after the
    # warps would leave the cavity aligned with the pre-warp ventricle
    # position, producing a stale outline that doesn't match the visible
    # ventricle.
    if geometry:
        ventricle = VentricleExpansion(p=_p(0.55))
        out = ventricle(out, rng=rng, ctx=ctx)

    # --- 1. BladeStretchHorizontal (ALWAYS applied) -------------------------
    # H stretch is the defining microtome artifact — bumped to up to 22% at
    # medium, capped at 30% even at heavy. V stretch up to 10%. Shear up to 4°.
    if geometry:
        h_max = 1.00 + (0.22 * param_scale)
        v_max = 1.00 + (0.10 * param_scale)
        shear_max = 4.0 * param_scale
        blade = BladeStretchHorizontal(
            p=1.0,
            horizontal_stretch_range=(1.00, min(h_max, 1.30)),
            vertical_stretch_range=(1.00, min(v_max, 1.18)),
            shear_range_deg=(-min(shear_max, 7.0), min(shear_max, 7.0)),
        )
        out = blade(out, rng=rng, ctx=ctx)

    # --- 2. IlluminationGradient (85% medium) --------------------------------
    illum = IlluminationGradient(p=_p(0.85))
    out = illum(out, rng=rng, ctx=ctx)

    # --- 3. EmbeddingHalos (70% medium, light-bg only) -----------------------
    if light_bg:
        halos = EmbeddingHalos(p=_p(0.70))
        out = halos(out, rng=rng, ctx=ctx)

    # --- 4. AffineJitter (75% medium) ----------------------------------------
    if geometry:
        jitter = AffineJitter(p=_p(0.75))
        out = jitter(out, rng=rng, ctx=ctx)

    # --- 5. Microbubbles (35% medium) ----------------------------------------
    if geometry:
        bubbles = Microbubbles(p=_p(0.35))
        out = bubbles(out, rng=rng, ctx=ctx)

    # --- 6. Debris (25% medium) ----------------------------------------------
    debris = Debris(p=_p(0.25))
    out = debris(out, rng=rng, ctx=ctx)

    # --- 7. Folds (45% medium) -----------------------------------------------
    if geometry:
        folds = Folds(p=_p(0.45))
        out = folds(out, rng=rng, ctx=ctx)

    # ResolutionShift and Tears intentionally omitted — see module docstring.

    # --- 10. Posterior wing detachment / loss --------------------------------
    # In posterior coronal sections, lateral tissue wings shed as whole slabs
    # once the thalamus is no longer present. This is the dominant damage mode
    # for posterior coronal sections — bumped to p=0.95 with mode weights
    # heavily biased toward "both_missing" (default 0.55) so missing-both is
    # the most common outcome among posterior-damaged slices.
    if geometry:
        posterior_wing = PosteriorWingDamage(p=_p(0.95))
        out = posterior_wing(out, rng=rng, ctx=ctx)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
