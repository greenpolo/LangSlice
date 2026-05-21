"""Mode dataclasses and weighted sampling for layered modality pipelines.

Per-image rendering picks ONE mode from the relevant table; the mode names a
counterstain (background anatomy layer) and a signal (gene-expression / IHC /
fluorescent-marker layer). Mode weights are calibrated against published
mouse-brain imaging prevalence (Frontiers 2023 systematic review, BICCN
methodological surveys).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ISHMode",
    "FluorescenceMode",
    "IFChannelSpec",
    "ISH_MODES",
    "FLUORESCENCE_MODES",
    "sample_mode",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ISHMode:
    """One ISH protocol variant: counterstain + signal layer."""

    name: str
    weight: float
    counterstain: (
        str  # key into COUNTERSTAIN_REGISTRY: "none" | "nfr" | "hematoxylin" | "dapi" | "nissl"
    )
    signal: str  # "nbt_bcip" | "dab" | "fluorescent_probe"
    counterstain_kwargs: dict = field(default_factory=dict)
    signal_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IFChannelSpec:
    """One immunofluorescence channel — distribution + appearance."""

    name: str
    channel: int  # 0=R, 1=G, 2=B (DAPI usually 2; markers usually 0 or 1)
    distribution: str  # "sparse" | "moderate" | "scattered" | "tract_sparse"
    density_range_per_mm2: tuple[float, float]
    intensity_range: tuple[float, float]
    sigma_range_scale: tuple[float, float]


@dataclass(frozen=True)
class FluorescenceMode:
    """One multiplex IF / FISH / tract-tracing variant."""

    name: str
    weight: float
    counterstain: (
        str  # "dapi" | "nissl" | "none"  (NeuroTrace is rendered like Nissl with cooler palette)
    )
    channels: tuple[IFChannelSpec, ...] = ()


# ---------------------------------------------------------------------------
# ISH mode table — weights match field-prevalence research
# ---------------------------------------------------------------------------


ISH_MODES: tuple[ISHMode, ...] = (
    ISHMode("allen_style", 0.30, "none", "nbt_bcip"),
    ISHMode("nbt_nfr", 0.30, "nfr", "nbt_bcip"),
    ISHMode("dab_hematoxylin", 0.20, "hematoxylin", "dab"),
    ISHMode("fish", 0.15, "dapi", "fluorescent_probe"),
    ISHMode("nissl_nbt", 0.05, "nissl", "nbt_bcip"),
)


# ---------------------------------------------------------------------------
# Common IF channel specs
# ---------------------------------------------------------------------------


_GFP_LIKE_MODERATE = IFChannelSpec(
    name="gfp_moderate",
    channel=1,
    distribution="moderate",
    density_range_per_mm2=(150.0, 600.0),
    intensity_range=(0.55, 0.95),
    sigma_range_scale=(0.7, 1.5),
)
_TDTOM_LIKE_MODERATE = IFChannelSpec(
    name="tdtom_moderate",
    channel=0,
    distribution="moderate",
    density_range_per_mm2=(120.0, 500.0),
    intensity_range=(0.60, 0.95),
    sigma_range_scale=(0.7, 1.5),
)
_PV_SPARSE_RED = IFChannelSpec(
    name="pv_sparse_red",
    channel=0,
    distribution="sparse",
    density_range_per_mm2=(30.0, 100.0),
    intensity_range=(0.65, 0.95),
    sigma_range_scale=(0.6, 1.3),
)
_SOM_SPARSE_GREEN = IFChannelSpec(
    name="som_sparse_green",
    channel=1,
    distribution="sparse",
    density_range_per_mm2=(25.0, 90.0),
    intensity_range=(0.60, 0.95),
    sigma_range_scale=(0.6, 1.3),
)
_VIP_SPARSE_RED = IFChannelSpec(
    name="vip_sparse_red",
    channel=0,
    distribution="sparse",
    density_range_per_mm2=(20.0, 80.0),
    intensity_range=(0.60, 0.90),
    sigma_range_scale=(0.6, 1.2),
)
_NEUN_PAN_GREEN = IFChannelSpec(
    name="neun_pan_green",
    channel=1,
    distribution="moderate",
    density_range_per_mm2=(800.0, 2000.0),
    intensity_range=(0.45, 0.85),
    sigma_range_scale=(0.5, 1.0),
)
_GFAP_SPARSE_RED = IFChannelSpec(
    name="gfap_sparse_red",
    channel=0,
    distribution="scattered",
    density_range_per_mm2=(80.0, 250.0),
    intensity_range=(0.40, 0.80),
    sigma_range_scale=(0.6, 1.4),
)
_IBA1_SCATTERED = IFChannelSpec(
    name="iba1_scattered",
    channel=0,
    distribution="scattered",
    density_range_per_mm2=(60.0, 180.0),
    intensity_range=(0.50, 0.85),
    sigma_range_scale=(0.6, 1.3),
)
_CFOS_SCATTERED_GREEN = IFChannelSpec(
    name="cfos_scattered_green",
    channel=1,
    distribution="scattered",
    density_range_per_mm2=(20.0, 200.0),
    intensity_range=(0.55, 0.95),
    sigma_range_scale=(0.5, 1.1),
)
_AAV_GFP_TRACT_SPARSE = IFChannelSpec(
    name="aav_gfp_tract_sparse",
    channel=1,
    distribution="tract_sparse",
    density_range_per_mm2=(15.0, 80.0),
    intensity_range=(0.70, 1.00),
    sigma_range_scale=(0.7, 1.5),
)
_AAV_TDTOM_TRACT_SPARSE = IFChannelSpec(
    name="aav_tdtom_tract_sparse",
    channel=0,
    distribution="tract_sparse",
    density_range_per_mm2=(15.0, 80.0),
    intensity_range=(0.70, 1.00),
    sigma_range_scale=(0.7, 1.5),
)


# ---------------------------------------------------------------------------
# Fluorescence mode table — weights match field-prevalence research
# ---------------------------------------------------------------------------


FLUORESCENCE_MODES: tuple[FluorescenceMode, ...] = (
    # Common named pairings (60% combined weight)
    FluorescenceMode("dapi_only", 0.05, "dapi", ()),
    FluorescenceMode("dapi_gfp", 0.18, "dapi", (_GFP_LIKE_MODERATE,)),
    FluorescenceMode("dapi_tdtom", 0.13, "dapi", (_TDTOM_LIKE_MODERATE,)),
    FluorescenceMode(
        "dapi_pv_som_vip",
        0.07,
        "dapi",
        (_PV_SPARSE_RED, _SOM_SPARSE_GREEN, _VIP_SPARSE_RED),
    ),
    FluorescenceMode(
        "dapi_neun_gfap_iba1",
        0.06,
        "dapi",
        (_NEUN_PAN_GREEN, _GFAP_SPARSE_RED, _IBA1_SCATTERED),
    ),
    FluorescenceMode("dapi_cfos", 0.05, "dapi", (_CFOS_SCATTERED_GREEN,)),
    FluorescenceMode(
        "dapi_gfp_tdtom",
        0.06,
        "dapi",
        (_GFP_LIKE_MODERATE, _TDTOM_LIKE_MODERATE),
    ),
    # Tract tracing — Nissl/NeuroTrace counterstain + sparse AAV signals
    FluorescenceMode(
        "tract_aav_gfp_neurotrace",
        0.05,
        "nissl",
        (_AAV_GFP_TRACT_SPARSE,),
    ),
    FluorescenceMode(
        "tract_aav_tdtom_neurotrace",
        0.05,
        "nissl",
        (_AAV_TDTOM_TRACT_SPARSE,),
    ),
    FluorescenceMode(
        "tract_dual_aav_neurotrace",
        0.04,
        "nissl",
        (_AAV_GFP_TRACT_SPARSE, _AAV_TDTOM_TRACT_SPARSE),
    ),
    # Generic fallbacks for breadth (26% combined weight)
    FluorescenceMode("generic_dapi_1if", 0.16, "dapi", ()),  # channels picked at runtime
    FluorescenceMode("generic_dapi_2if", 0.07, "dapi", ()),
    FluorescenceMode("generic_dapi_3if", 0.03, "dapi", ()),
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_mode(rng: np.random.Generator, modes: tuple) -> object:
    """Pick one mode from the table according to its weight."""
    weights = np.array([m.weight for m in modes], dtype=np.float64)
    weights /= weights.sum()
    return modes[int(rng.choice(len(modes), p=weights))]
