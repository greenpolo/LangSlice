"""Atlas annotation → per-pixel cell-density map.

Output is a normalized float32 array in [0, 1] where 1.0 represents the highest
relative cell density.  texture.py applies absolute nuclei/cell-per-um2 scaling at
render time.

Modality multipliers are all 1.0 in v1; the hook below is the intended extension
point for per-modality calibration (e.g. DAPI nuclei vs. Nissl soma counts).
"""

from __future__ import annotations

import importlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

_MODALITY_MULTIPLIER: dict[str, float] = {
    "dapi": 1.0,
    "nissl": 1.0,
    "brightfield": 1.0,
    "fluorescence": 1.0,
    "ish": 1.0,
}

_WHITE_MATTER_KEYWORDS = frozenset(
    ["tract", "fiber", "fibre", "callosum", "capsule", "commissure", "funiculus", "lemniscus"]
)
_VENTRICLE_KEYWORDS = frozenset(["ventricle", "vl", "v3", "v4", "aq", "aqueduct"])

_DENSITY_HIGH = 1.0
_DENSITY_WHITE_MATTER = 0.15
_DENSITY_VENTRICLE = 0.0


def _name_tier(name: str, acronym: str) -> float:
    lower_name = name.lower()
    lower_acronym = acronym.lower()

    if any(kw in lower_name for kw in _VENTRICLE_KEYWORDS) or any(
        kw in lower_acronym for kw in _VENTRICLE_KEYWORDS
    ):
        return _DENSITY_VENTRICLE

    if any(kw in lower_name for kw in _WHITE_MATTER_KEYWORDS):
        return _DENSITY_WHITE_MATTER

    return _DENSITY_HIGH


def _load_structures(atlas_name: str) -> object | None:
    try:
        module = importlib.import_module("brainglobe_atlasapi")
        atlas = module.BrainGlobeAtlas(atlas_name)
        return getattr(atlas, "structures", None)
    except Exception:
        return None


def _lookup_tier_by_id(structure_id: int, structures: object | None) -> float:
    if structures is None:
        return _DENSITY_HIGH

    try:
        data = structures[structure_id]  # type: ignore[index]
        if isinstance(data, dict):
            name = str(data.get("name", ""))
            acronym = str(data.get("acronym", ""))
            return _name_tier(name, acronym)
    except Exception:
        pass

    return _DENSITY_HIGH


def shade_substrate_by_atlas(
    canvas: np.ndarray,
    reference_slice: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    strength: float = 0.20,
    gamma: float = 1.0,
) -> np.ndarray:
    """Multiplicatively darken substrate by atlas grayscale.

    Mimics the optical-absorption effect in real brightfield histology where
    cell-rich tissue absorbs more light, so the substrate itself reads slightly
    darker in dense regions even before any stained cells are added.

    Args:
        canvas: HWC float32 in [0, 1] — cream substrate to be shaded.
        reference_slice: HW grayscale uint8 or float32 in [0, 1] — atlas template.
        tissue_mask: HW bool mask; off-tissue pixels are not shaded.
        strength: max darkening fraction in [0, 1]; 0.0 = no shading,
            0.30 = up to 30% darker in the brightest atlas regions.
        gamma: shape applied to normalized atlas brightness before scaling.
            >1 sharpens contrast (only the very brightest regions darken),
            <1 softens (most tissue gets some shading).

    Returns HWC float32 in [0, 1].
    """
    ref = reference_slice
    if ref.ndim == 3 and ref.shape[2] == 3:
        ref = ref[..., 0] * 0.299 + ref[..., 1] * 0.587 + ref[..., 2] * 0.114
    if ref.dtype == np.uint8:
        ref = ref.astype(np.float32) / 255.0
    else:
        ref = ref.astype(np.float32)
    ref = np.clip(ref, 0.0, 1.0)
    if gamma != 1.0:
        ref = np.power(ref, gamma)

    darken = (1.0 - strength * ref).astype(np.float32)
    out = canvas.copy().astype(np.float32)
    mask3 = tissue_mask[..., None] if canvas.ndim == 3 else tissue_mask
    factor = np.where(mask3, darken[..., None] if canvas.ndim == 3 else darken, 1.0)
    return np.clip(out * factor, 0.0, 1.0)


def atlas_grayscale_density_map(
    reference_slice: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    gamma: float = 1.2,
    floor: float = 0.15,
) -> np.ndarray:
    """Use the atlas reference template's grayscale as a density-modulation map.

    The Allen CCFv3 average template already encodes per-pixel cell-density
    information: cortical laminae appear as bright bands, the hippocampus
    pyramidal layer is conspicuously bright, fiber tracts are dark, etc.
    Treating the template as the modulation source gives intra- and
    inter-region density variation for free, without per-region tuning.

    Args:
        reference_slice: HW grayscale array (uint8 or float32 in [0, 1]).
        tissue_mask: HW bool mask; pixels outside get density 0.
        gamma: exponent applied to normalized brightness; >1 darkens midtones,
            sharpening the contrast between cell-rich and cell-poor regions.
        floor: minimum density inside tissue, so dim regions still produce
            some texture (otherwise sparse tracts would be empty).

    Returns:
        HW float32 in [0, 1].
    """
    ref = reference_slice
    # Convert RGB to grayscale if a 3-channel reference is passed.
    if ref.ndim == 3 and ref.shape[2] == 3:
        ref = ref[..., 0] * 0.299 + ref[..., 1] * 0.587 + ref[..., 2] * 0.114
    if ref.dtype == np.uint8:
        ref = ref.astype(np.float32) / 255.0
    else:
        ref = ref.astype(np.float32)
    density = np.power(np.clip(ref, 0.0, 1.0), gamma)
    density = np.where(tissue_mask, np.maximum(density, floor), 0.0)
    return density.astype(np.float32)


def apply_region_density_variance(
    density_map: np.ndarray,
    annotation_slice: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: float,
    clamp_sigmas: float = 3.0,
) -> np.ndarray:
    """Per-image, per-region multiplicative jitter of a density map.

    Borrowed mechanism from SiDoLa-NS's ``FIVCellDensityVarianceFactor`` (npj
    Syst Biol Appl 2026): each anatomical region's relative cell density is
    rolled independently on every rendered frame, so a network cannot shortcut
    "this region ⇒ exactly this density" — it has to actually read the tissue.
    SiDoLa rolls the per-region factor over a deliberately extreme [0.2, 4.0]
    (≈20× swing) because they train *segmentation* detectors; for photoreal
    AP-estimation tissue we want a small fraction of that — natural section-to-
    section variation, not chaos. ``strength`` is the log-normal sigma of the
    per-region factor, e.g. ``0.14`` ⇒ ~68% of regions land within ±15%.

    Label-safe: density is only scaled *within* existing region masks — region
    boundaries, the annotation, and the AP label are never touched. The map is
    renormalized to preserve its pre-jitter tissue mean, so this is a pure
    cross-region *redistribution* with no global brightness/exposure drift
    (cell count is fixed upstream by cells-per-mm²; this only shifts where
    cells are relatively concentrated).

    Args:
        density_map: HW float32 in [0, 1]; tissue pixels are > 0
            (see :func:`atlas_grayscale_density_map`).
        annotation_slice: HW integer region-ID array aligned to ``density_map``.
        rng: NumPy Generator; one draw per region ID in sorted-ID order
            (deterministic for a given seed). ``strength <= 0`` draws nothing.
        strength: log-normal sigma of the per-region factor. ``<= 0`` is a
            no-op returning the input unchanged (default pipeline behavior).
        clamp_sigmas: clamp each factor to ``exp(±clamp_sigmas * strength)`` to
            reject rare extreme draws.

    Returns:
        HW float32 in [0, 1].
    """
    if strength <= 0.0:
        return density_map

    out = density_map.astype(np.float32, copy=True)
    tissue = out > 0.0
    if not tissue.any():
        return out

    pre_mean = float(out[tissue].mean())
    lo = float(np.exp(-clamp_sigmas * strength))
    hi = float(np.exp(clamp_sigmas * strength))

    region_ids = np.unique(annotation_slice[tissue])
    region_ids = region_ids[region_ids > 0]
    for rid in region_ids:
        factor = float(np.clip(np.exp(rng.normal(0.0, strength)), lo, hi))
        out[annotation_slice == int(rid)] *= factor

    # Preserve the pre-jitter tissue mean → pure redistribution, no global drift.
    post_mean = float(out[tissue].mean())
    if post_mean > 0.0:
        out[tissue] *= pre_mean / post_mean

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def region_density_map(
    annotation_slice: np.ndarray,
    modality: str,
    render_resolution_um: float,  # noqa: ARG001 — reserved for v2 absolute-density scaling
    *,
    atlas_name: str = "allen_mouse_25um",
) -> np.ndarray:
    """Compute a per-pixel relative cell-density map from an annotation slice.

    Args:
        annotation_slice: HW integer array of BrainGlobe region IDs (background = 0).
        modality: Target histology modality (one of dapi/nissl/brightfield/fluorescence/ish).
        render_resolution_um: Atlas render resolution in microns per pixel.
        atlas_name: BrainGlobe atlas identifier; used to resolve region names.

    Returns:
        HW float32 array in [0, 1] where background and ventricles are 0.
    """
    h, w = annotation_slice.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    unique_ids = np.unique(annotation_slice)
    non_background = unique_ids[unique_ids > 0]

    if len(non_background) == 0:
        return density

    structures = _load_structures(atlas_name)

    for uid in non_background:
        uid_int = int(uid)
        tier = _lookup_tier_by_id(uid_int, structures)
        density[annotation_slice == uid] = tier

    multiplier = _MODALITY_MULTIPLIER.get(modality, 1.0)
    if multiplier != 1.0:
        density *= multiplier

    return density
