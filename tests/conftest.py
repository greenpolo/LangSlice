from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice_harness.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")


@pytest.fixture(scope="module")
def atlas_slice_inputs(atlas: object) -> tuple[np.ndarray, np.ndarray]:
    """(ref_uint8_HW, ann_int32_HW) at AP=5.335mm for the 25um Allen atlas."""
    from langslice_harness.atlas.core import get_reference_slice, position_mm_to_index
    from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

    pil = get_reference_slice(atlas, 5.335).convert("L")
    ref = np.asarray(pil, dtype=np.uint8)

    ctx_space = atlas_space_context(atlas)
    axis = slice_axis_index(ctx_space, "coronal")
    idx = position_mm_to_index(atlas, 5.335)  # type: ignore[arg-type]
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]
    return ref, ann


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def hwc_image() -> np.ndarray:
    """Small random HWC float32 image in [0,1]."""
    return np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)


@pytest.fixture
def hw_reference_uint8() -> np.ndarray:
    """Small synthetic HW uint8 reference slice (grayscale)."""
    return np.random.default_rng(0).integers(0, 255, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def hw_annotation_int32() -> np.ndarray:
    """Small synthetic HW int32 annotation with GM/WM-style ids."""
    h, w = 64, 64
    ann = np.zeros((h, w), dtype=np.int32)
    ann[: h // 2, :] = 315  # grey-like
    ann[3 * h // 4 :, :] = 1009  # fiber-tract-like
    return ann


@pytest.fixture
def transform_ctx():
    from augmentation.transforms.base import TransformContext

    return TransformContext(
        modality="dapi",
        annotation_slice=None,
        density_map=None,
        tissue_mask=None,
        pixel_size_um=25.0,
    )


def ctx_with(
    ctx,
    *,
    modality: str | None = None,
    annotation_slice: np.ndarray | None | object = None,
    density_map: np.ndarray | None | object = None,
    tissue_mask: np.ndarray | None | object = None,
    pixel_size_um: float | None = None,
    tissue_class_masks: dict[str, np.ndarray] | None | object = None,
):
    """Small helper to derive a TransformContext with targeted overrides."""
    fields = {}
    if modality is not None:
        fields["modality"] = modality
    if annotation_slice is not None:
        fields["annotation_slice"] = annotation_slice  # type: ignore[assignment]
    if density_map is not None:
        fields["density_map"] = density_map  # type: ignore[assignment]
    if tissue_mask is not None:
        fields["tissue_mask"] = tissue_mask  # type: ignore[assignment]
    if pixel_size_um is not None:
        fields["pixel_size_um"] = pixel_size_um
    if tissue_class_masks is not None:
        fields["tissue_class_masks"] = tissue_class_masks  # type: ignore[assignment]
    return replace(ctx, **fields)

