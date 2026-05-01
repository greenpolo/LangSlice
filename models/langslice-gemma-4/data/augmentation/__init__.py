"""Synthetic histology augmentation package."""

from __future__ import annotations

import numpy as np

from .brightfield_pipeline import render_brightfield_section
from .core import Modality
from .dapi_pipeline import render_dapi_section
from .fluorescence_pipeline import render_fluorescence_section
from .ish_pipeline import render_ish_section
from .nissl_pipeline import render_nissl_section


def augment_atlas_slice(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    modality: Modality,
    seed: int,
    pixel_size_um: float,
    mode: object | None = None,
    counterstain: str | None = None,
    apply_damage: bool = True,
    damage_intensity: str = "medium",
) -> np.ndarray:
    if modality == "dapi":
        return render_dapi_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            apply_damage=apply_damage,
            damage_intensity=damage_intensity,
        )
    if modality == "nissl":
        return render_nissl_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            apply_damage=apply_damage,
            damage_intensity=damage_intensity,
        )
    if modality == "brightfield":
        return render_brightfield_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            mode=None if mode is None else str(mode),
            counterstain="auto" if counterstain is None else str(counterstain),
            apply_damage=apply_damage,
            damage_intensity=damage_intensity,
        )
    if modality == "fluorescence":
        return render_fluorescence_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            mode=mode,
            apply_damage=apply_damage,
            damage_intensity=damage_intensity,
        )
    if modality == "ish":
        return render_ish_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            mode=mode,
            apply_damage=apply_damage,
            damage_intensity=damage_intensity,
        )
    raise ValueError(f"Unsupported modality: {modality!r}")


__all__ = ["Modality", "augment_atlas_slice"]
