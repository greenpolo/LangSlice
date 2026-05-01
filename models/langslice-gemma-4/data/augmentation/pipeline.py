"""Per-modality Stage A pipeline composition.

.. deprecated::
    ``MODALITY_PRESETS`` in this module is superseded by the per-modality
    renderers introduced in the layered counterstain architecture. Use the
    appropriate modality-specific entry point instead:

    - DAPI:         ``augmentation.dapi_pipeline.render_dapi_section``
    - Nissl:        ``augmentation.nissl_pipeline.render_nissl_section``
    - Brightfield:  ``augmentation.brightfield_pipeline.render_brightfield_section``
    - Fluorescence: ``augmentation.fluorescence_pipeline.render_fluorescence_section``
    - ISH:          ``augmentation.ish_pipeline.render_ish_section``

    These entry points compose the counterstain registry
    (``augmentation.counterstain.COUNTERSTAIN_REGISTRY``) with the signal
    registry (``augmentation.signals.SIGNAL_REGISTRY``) and mode tables
    (``augmentation.modes``), providing accurate layered rendering that
    ``run_pipeline`` cannot produce.

    ``run_pipeline`` and ``Augmenter`` have been removed. Use the modality-
    specific renderers listed above.
"""

from __future__ import annotations

import numpy as np
from augmentation.transforms.base import Transform, TransformContext
from augmentation.transforms.damage import (
    Debris,
    EmbeddingHalos,
    Folds,
    IlluminationGradient,
    Microbubbles,
    Tears,
)
from augmentation.transforms.geometry import AffineJitter, RandomCrop, ResolutionShift
from augmentation.transforms.texture import (
    DAPINuclei,
    FluorescenceSpeckle,
    ISHPuncta,
    NisslCellBodies,
)
from augmentation.transforms.tonal import (
    BrightfieldTonal,
    DAPITonal,
    FluorescenceTonal,
    ISHTonal,
    NisslTonal,
)

__all__ = [
    "MODALITY_PRESETS",
    "build_modality_pipeline",
]

_MODALITIES = ("dapi", "nissl", "brightfield", "fluorescence", "ish")

# EmbeddingHalos paints a warm gradient outside the tissue mask — biologically
# plausible only on light-background modalities. Dark-field fluorescence (DAPI,
# multi-channel fluorescence) has near-black off-tissue and a warm halo there
# would look obviously wrong.
_LIGHT_BACKGROUND = {"nissl", "brightfield", "ish"}


def _damage_instances(modality: str) -> list[Transform]:
    transforms: list[Transform] = [
        Folds(),
        Tears(),
        Microbubbles(),
    ]
    if modality in _LIGHT_BACKGROUND:
        transforms.append(EmbeddingHalos())
    transforms.extend([Debris(), IlluminationGradient()])
    return transforms


MODALITY_PRESETS: dict[str, list[Transform]] = {
    "dapi": [
        DAPITonal(),
        DAPINuclei(),
        *_damage_instances("dapi"),
        AffineJitter(),
        RandomCrop(),
        ResolutionShift(),
    ],
    "nissl": [
        NisslTonal(),
        NisslCellBodies(),
        *_damage_instances("nissl"),
        AffineJitter(),
        RandomCrop(),
        ResolutionShift(),
    ],
    "brightfield": [
        BrightfieldTonal(),
        *_damage_instances("brightfield"),
        AffineJitter(),
        RandomCrop(),
        ResolutionShift(),
    ],
    "fluorescence": [
        FluorescenceTonal(),
        FluorescenceSpeckle(),
        *_damage_instances("fluorescence"),
        AffineJitter(),
        RandomCrop(),
        ResolutionShift(),
    ],
    "ish": [
        ISHTonal(),
        ISHPuncta(),
        *_damage_instances("ish"),
        AffineJitter(),
        RandomCrop(),
        ResolutionShift(),
    ],
}


def build_modality_pipeline(modality: str, *, seed: int) -> list[Transform]:  # noqa: ARG001
    """Return a fresh list of Transform instances for *modality*.

    Each call returns new instances so per-call RNG state is independent.
    """
    if modality not in _MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}. Choose from {_MODALITIES}.")
    return list(MODALITY_PRESETS[modality])


def _infer_mask_from_luminance(image: np.ndarray) -> np.ndarray:
    lum = (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )
    return (lum > 0.02)
