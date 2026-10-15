"""Where each image counts: the masks every fit is given.

Two kinds of fit, two kinds of mask:

- FOOTPRINT masks (:func:`footprint_masks`): the Elastix affine, every
  region-restricted step and every fit of traced lines. Section (fixed)
  mask = placed atlas footprint widened by :data:`MASK_MARGIN_MM`; atlas
  (moving) mask = the same footprint minus the excluded regions and an
  :data:`EXCLUSION_MARGIN_MM` margin around them (the damage zone).
- EM masks (:func:`em_masks`): a whole-section stain fit. It is fitted once
  with the footprint masks; :func:`em_weights` then asks, pixel by pixel,
  whether the section matches the warped atlas image, is an artifact or is
  background, and the fit is run again from the same start with section mask
  = P(match) >= :data:`EM_MATCH_THRESHOLD` over the whole grid and atlas mask
  = everything but the damage zone. Holes, tears, folds and debris drop out
  of the metric; the footprint no longer limits where tissue may go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

#: Widening of the placed atlas footprint so its outline sits inside the metric.
MASK_MARGIN_MM = 0.2
#: Margin around excluded regions also kept out of the moving mask. Excluded
#: pixels are blanked to background, which leaves an edge; without this margin
#: that artificial edge still pulls the section's tissue (measured on a
#: synthetic shifted region: Elastix moved it 92% as far as with no exclusion).
EXCLUSION_MARGIN_MM = 0.1


def _disk(radius_px: float) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _disk(radius_px)) > 0


@dataclass(frozen=True)
class Masks:
    """A fit's masks on the working grid; *damage_zone* is what no atlas mask
    covers (the excluded regions and their margin)."""

    fixed: np.ndarray
    moving: np.ndarray
    damage_zone: np.ndarray


def footprint_masks(placed: np.ndarray, excluded: np.ndarray, spacing_mm: float) -> Masks:
    """The footprint masks of a fit whose placed atlas labels are *placed* and
    whose excluded pixels are *excluded* (both on the working grid)."""
    footprint = dilate(placed != 0, MASK_MARGIN_MM / spacing_mm)
    damage_zone = dilate(excluded, EXCLUSION_MARGIN_MM / spacing_mm)
    return Masks(fixed=footprint, moving=footprint & ~damage_zone, damage_zone=damage_zone)


# --- the EM section mask ---------------------------------------------------
#
# Three-class EM pixel weights as emlddmm implements them (Tward et al. 2025,
# Nat Commun, doi 10.1038/s41467-025-65317-7; emlddmm.py `emlddmm()`), with
# the warp held fixed (the first fit's) while E and M steps alternate:
#
#   matching   J ~ N(f(I_warped), sigmaM)     f: a contrast map, linear per block
#   artifact   J ~ N(muA, sigmaA)
#   background J ~ N(muB, sigmaB)
#
# J is the fit's section image, I_warped the atlas image through the first
# fit. Where this departs from emlddmm's defaults it says so below.

#: The EM runs on a coarse grid, as emlddmm runs on downsampled images
#: (area-averaged; the weights are upsampled back).
EM_GRID_UM = 50.0
#: emlddmm's ``local_contrast`` block, in EM-grid pixels (16 x 50 um = 0.8 mm):
#: its serial-section example's [1, 16, 16]. Order 1 (an offset and a gain).
EM_BLOCK_PX = 16
#: E/M alternations.
EM_ITERATIONS = 20
#: emlddmm's ridge on each block's normal equations (x the largest entry).
EM_RIDGE = 1e-7
#: Class widths (matching, background, artifact) as multiples of
#: EM_SIGMA_SCALE x std(J), held fixed (emlddmm does not update them in its
#: default path). DEVIATION: emlddmm's code default is (1, 2, 5) x std; on
#: Carey (brightfield) and PL_Projections (DAPI) that flagged nothing, not
#: even a 0.3 mm hole. Its published runs use near-equal widths; equal widths
#: at 0.5 x std catch holes, tears and debris and leave brightfield anatomy
#: alone; 0.75 x std missed the hole again.
EM_SIGMA_RATIOS = (1.0, 1.0, 1.0)
EM_SIGMA_SCALE = 0.5
#: Class priors (matching, artifact, background), held fixed. DEVIATION:
#: emlddmm's code updates them; updated priors ran away on DAPI, where dim
#: tissue has the background's intensity (background prior 0.1 -> 0.35, the
#: brainstem left out). These are its README example's values.
EM_PRIORS = (0.9, 0.05, 0.05)
#: Background is sampled this far outside the placed footprint.
EM_BACKGROUND_DISTANCE_UM = 500.0
#: Section pixels whose P(match) is below this leave the second fit's metric.
EM_MATCH_THRESHOLD = 0.5
#: Everything above that changes the mask; part of the deformable cache key.
EM_PARAMETERS: dict[str, Any] = {
    "method": "three-class EM after emlddmm (Tward et al. 2025)",
    "grid_um": EM_GRID_UM, "block_px": EM_BLOCK_PX, "iterations": EM_ITERATIONS,
    "ridge": EM_RIDGE, "sigma_ratios": list(EM_SIGMA_RATIOS), "sigma_scale": EM_SIGMA_SCALE,
    "priors": list(EM_PRIORS), "background_distance_um": EM_BACKGROUND_DISTANCE_UM,
    "match_threshold": EM_MATCH_THRESHOLD,
}


def _block_linear(J: np.ndarray, image: np.ndarray, weight: np.ndarray, block: int) -> np.ndarray:
    """emlddmm's local contrast: per *block* x *block* tile, the weighted least
    squares J ~ c0 + c1 * image, evaluated on the tile."""
    height, width = J.shape
    ph, pw = (-height) % block, (-width) % block

    def pad(a: np.ndarray) -> np.ndarray:
        return np.pad(a, ((0, ph), (0, pw)))

    Jp, Ip, Wp = pad(J), pad(image), pad(weight)
    nh, nw = Jp.shape[0] // block, Jp.shape[1] // block

    def tiles(a: np.ndarray) -> np.ndarray:
        return a.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(nh, nw, block * block)

    Jb, Ib, Wb = tiles(Jp), tiles(Ip), tiles(Wp)
    s0, s1, s2 = Wb.sum(-1), (Wb * Ib).sum(-1), (Wb * Ib * Ib).sum(-1)
    t0, t1 = (Wb * Jb).sum(-1), (Wb * Ib * Jb).sum(-1)
    normal = np.stack([np.stack([s0, s1], -1), np.stack([s1, s2], -1)], -2)
    normal = normal + np.eye(2) * (normal.max(axis=(-1, -2), keepdims=True) + 1) * EM_RIDGE
    c = np.linalg.solve(normal, np.stack([t0, t1], -1)[..., None])[..., 0]
    f = c[..., :1] + c[..., 1:] * Ib
    f = f.reshape(nh, nw, block, block).transpose(0, 2, 1, 3).reshape(nh * block, nw * block)
    return f[:height, :width]


def em_weights(
    J: np.ndarray, warped_atlas: np.ndarray, footprint: np.ndarray, spacing_mm: float,
) -> dict[str, Any]:
    """Per-pixel class weights on the working grid.

    *J* is the fit's section image ([0, 1]), *warped_atlas* the atlas image
    through the first fit, *footprint* the placed atlas footprint (with its
    margin), all on the working grid of *spacing_mm*. Returns ``WM`` (P(match)),
    ``WA`` (artifact), ``WB`` (background), ``predicted`` (the contrast map's
    prediction of the section) on that grid, and ``params`` (the class means,
    widths and priors used).
    """
    full = J.shape
    k = max(spacing_mm * 1000.0 / EM_GRID_UM, 1e-3)
    small = (max(2, round(full[1] * k)), max(2, round(full[0] * k)))
    Jc = cv2.resize(J.astype(np.float32), small, interpolation=cv2.INTER_AREA).astype(np.float64)
    Ic = cv2.resize(warped_atlas.astype(np.float32), small,
                    interpolation=cv2.INTER_AREA).astype(np.float64)
    fp = cv2.resize(footprint.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0.5
    lo, hi = np.percentile(Ic, (0.5, 99.5))
    Ic = (Ic - lo) / max(float(hi - lo), 1e-6)
    sd = float(Jc.std())
    rM, rB, rA = EM_SIGMA_RATIOS
    sM, sB, sA = (r * sd * EM_SIGMA_SCALE for r in (rM, rB, rA))
    r = max(1, round(EM_BACKGROUND_DISTANCE_UM / EM_GRID_UM))
    far = ~(cv2.dilate(fp.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0)
    inside = Jc[fp] if fp.any() else Jc.ravel()
    # Means held fixed (emlddmm: update_muA/update_muB False). Its default is
    # muB = 0.1% quantile, muA = 99.9% (a dark background); its Nissl
    # serial-section example sets muB = white, muA = black. Here muB is this
    # image's own background (far from the tissue) and muA the opposite
    # extreme quantile of the tissue, which reproduces both.
    if far.sum() > 200:
        muB = float(np.median(Jc[far]))
    else:
        muB = float(np.percentile(Jc, 0.1 if np.median(inside) > Jc.mean() else 99.9))
    qlo, qhi = np.percentile(inside, (0.1, 99.9))
    muA = float(qlo if abs(qlo - muB) > abs(qhi - muB) else qhi)
    pM, pA, pB = EM_PRIORS
    WM = np.full(Jc.shape, pM)
    f = Jc
    for _ in range(EM_ITERATIONS):
        f = _block_linear(Jc, Ic, WM, EM_BLOCK_PX)  # M step: the contrast map
        lM = pM * np.exp(-(Jc - f) ** 2 / 2 / sM ** 2) / sM  # E step
        lA = pA * np.exp(-(Jc - muA) ** 2 / 2 / sA ** 2) / sA
        lB = pB * np.exp(-(Jc - muB) ** 2 / 2 / sB ** 2) / sB
        total = lM + lA + lB
        total = total + total.max() * 1e-12
        WM, WA, WB = lM / total, lA / total, lB / total

    def up(a: np.ndarray) -> np.ndarray:
        return cv2.resize(a.astype(np.float32), (full[1], full[0]), interpolation=cv2.INTER_LINEAR)

    return {
        "WM": up(WM), "WA": up(WA), "WB": up(WB), "predicted": up(f),
        "params": {"muA": muA, "muB": muB, "sigmaM": sM, "sigmaA": sA, "sigmaB": sB,
                   "priors": list(EM_PRIORS)},
    }


def em_masks(
    J: np.ndarray, warped_atlas: np.ndarray, first: Masks, spacing_mm: float,
) -> tuple[Masks, dict[str, Any]]:
    """The second fit's masks from the first fit's warped atlas image, and the
    EM's facts (``left_out``: the section pixels the mask drops; ``params``;
    ``left_out_in_footprint``: their fraction of the placed footprint)."""
    weights = em_weights(J, warped_atlas, first.fixed, spacing_mm)
    fixed = weights["WM"] >= EM_MATCH_THRESHOLD
    left_out = ~fixed
    facts = {
        "left_out": left_out, "params": weights["params"],
        "left_out_in_footprint": float(left_out[first.fixed].mean()) if first.fixed.any() else 0.0,
    }
    return Masks(fixed=fixed, moving=~first.damage_zone, damage_zone=first.damage_zone), facts
