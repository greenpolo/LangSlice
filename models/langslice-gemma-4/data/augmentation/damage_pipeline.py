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

Probability schedule (medium — bumped 2026-04-30 to match real-microscopy realism)
----------------------------------------------------------------------------------
ALWAYS     BladeStretchHorizontal  — horizontal-biased anisotropic stretch (~22% max H stretch)
85%        IlluminationGradient    — lamp / photobleaching gradient
75%        AffineJitter            — small rotation + translation
70%        EmbeddingHalos          — warm halo outside tissue (light-bg only)
55%        Microbubbles            — circular bright mounting-medium artifacts
55%        Debris                  — small dark spots / dust
55%        Tears                   — bg-colored voids; ventricle expansion + edge bites
45%        Folds                   — sinusoidal tissue-fold warp
35%        ResolutionShift         — down-then-up blur (resolution variation)

Light-background modalities that accept EmbeddingHalos:
    nissl, brightfield, ish

Dark-background modalities (EmbeddingHalos skipped):
    dapi, fluorescence
"""

from __future__ import annotations

import numpy as np

from .transforms.base import TransformContext
from .transforms.damage import (
    Debris,
    EmbeddingHalos,
    Folds,
    HemibrainPreparation,
    IlluminationGradient,
    Microbubbles,
    PosteriorWingDamage,
    Tears,
)
from .transforms.geometry import AffineJitter, BladeStretchHorizontal, ResolutionShift

__all__ = ["apply_damage_layer"]

# Modalities with a light (bright) background substrate.
_LIGHT_BG_MODALITIES = {"nissl", "brightfield", "ish"}

# ---------------------------------------------------------------------------
# Intensity multipliers: (probability_scale, param_scale)
# ---------------------------------------------------------------------------
_INTENSITY_PROB_SCALE = {
    "light":  0.65,
    "medium": 1.00,
    "heavy":  1.45,
}
_INTENSITY_PARAM_SCALE = {
    "light":  0.65,
    "medium": 1.00,
    "heavy":  1.75,
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
        raise ValueError(
            f"intensity must be 'light', 'medium', or 'heavy'; got {intensity!r}"
        )

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

    # --- 5. Microbubbles (55% medium) ----------------------------------------
    if geometry:
        bubbles = Microbubbles(p=_p(0.55))
        out = bubbles(out, rng=rng, ctx=ctx)

    # --- 6. Debris (55% medium) ----------------------------------------------
    debris = Debris(p=_p(0.55))
    out = debris(out, rng=rng, ctx=ctx)

    # --- 7. Folds (45% medium) -----------------------------------------------
    if geometry:
        folds = Folds(p=_p(0.45))
        out = folds(out, rng=rng, ctx=ctx)

    # --- 8. ResolutionShift (35% medium) -------------------------------------
    res = ResolutionShift(p=_p(0.35))
    out = res(out, rng=rng, ctx=ctx)

    # --- 9. Tears (55% medium) -----------------------------------------------
    # Tears now use bg-colored fill (not black) and bias toward ventricle
    # expansion + edge bites. Ventricles in real sections almost always look
    # larger than the atlas predicts due to CSF retraction; this captures that.
    if geometry:
        tears = Tears(p=_p(0.55))
        out = tears(out, rng=rng, ctx=ctx)

    # --- 10. Posterior wing detachment / loss --------------------------------
    # In posterior coronal sections, lateral tissue wings often detach as whole
    # slabs once the thalamus is no longer present.
    if geometry:
        posterior_wing = PosteriorWingDamage(p=_p(0.30))
        out = posterior_wing(out, rng=rng, ctx=ctx)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
