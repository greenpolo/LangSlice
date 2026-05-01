"""Consolidated contract tests for augmentation transform classes."""

from __future__ import annotations

import time

import numpy as np
import pytest
from augmentation.transforms.base import TransformContext
from augmentation.transforms.damage import (
    Debris,
    EmbeddingHalos,
    Folds,
    IlluminationGradient,
    Microbubbles,
    Tears,
)
from augmentation.transforms.geometry import (
    AffineJitter,
    BladeStretchHorizontal,
    RandomCrop,
    ResolutionShift,
)
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


def _img(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def _ctx(
    *,
    modality: str = "dapi",
    h: int = 64,
    w: int = 64,
    density: float | np.ndarray | None = 0.8,
    tissue_mask: np.ndarray | None = None,
    annotation_slice: np.ndarray | None = None,
    pixel_size_um: float = 25.0,
) -> TransformContext:
    if isinstance(density, (int, float)):
        dm = np.full((h, w), float(density), dtype=np.float32)
    elif density is None:
        dm = None
    else:
        dm = density.astype(np.float32)
    return TransformContext(
        modality=modality,
        annotation_slice=annotation_slice,
        density_map=dm,
        tissue_mask=tissue_mask,
        pixel_size_um=pixel_size_um,
    )


_CONTRACT_TRANSFORMS: list[type] = [
    # geometry
    AffineJitter,
    BladeStretchHorizontal,
    RandomCrop,
    ResolutionShift,
    # damage
    Folds,
    Tears,
    Microbubbles,
    EmbeddingHalos,
    Debris,
    IlluminationGradient,
    # tonal
    DAPITonal,
    NisslTonal,
    BrightfieldTonal,
    FluorescenceTonal,
    ISHTonal,
    # legacy texture (density-modulated)
    DAPINuclei,
    NisslCellBodies,
    FluorescenceSpeckle,
    ISHPuncta,
]


@pytest.mark.parametrize("cls", _CONTRACT_TRANSFORMS, ids=lambda c: c.__name__)
def test_transform_contract_shape_dtype_range(cls: type) -> None:
    t = cls(p=1.0) if "p" in cls.__init__.__code__.co_varnames else cls()
    img = _img(64, 80, seed=0)
    ctx = _ctx(h=64, w=80)
    out = t(img.copy(), rng=np.random.default_rng(1), ctx=ctx)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize("cls", _CONTRACT_TRANSFORMS, ids=lambda c: c.__name__)
def test_transform_contract_determinism_same_seed(cls: type) -> None:
    t = cls(p=1.0) if "p" in cls.__init__.__code__.co_varnames else cls()
    img = _img(seed=10)
    ctx = _ctx(h=64, w=64)
    out_a = t(img.copy(), rng=np.random.default_rng(42), ctx=ctx)
    out_b = t(img.copy(), rng=np.random.default_rng(42), ctx=ctx)
    np.testing.assert_array_equal(out_a, out_b)


@pytest.mark.parametrize("cls", _CONTRACT_TRANSFORMS, ids=lambda c: c.__name__)
def test_transform_contract_p_zero_skips(cls: type) -> None:
    if "p" not in cls.__init__.__code__.co_varnames:
        pytest.skip("no p-gate on this transform")
    t = cls(p=0.0)
    img = _img(seed=5)
    ctx = _ctx(h=64, w=64)
    out = t(img.copy(), rng=np.random.default_rng(0), ctx=ctx)
    np.testing.assert_array_equal(out, img)


def test_affine_jitter_invalid_fill_raises() -> None:
    with pytest.raises(ValueError):
        AffineJitter(fill="border")


def test_resolution_shift_updates_ctx_arrays() -> None:
    img = _img()
    density = np.random.default_rng(5).random((64, 64)).astype(np.float32)
    mask = (np.random.default_rng(6).random((64, 64)) > 0.5).astype(np.float32)
    ann = np.random.default_rng(7).integers(0, 100, (64, 64)).astype(np.int32)
    ctx = TransformContext(
        modality="dapi",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=mask,
        pixel_size_um=25.0,
    )
    t = ResolutionShift(p=1.0)
    t(img, rng=np.random.default_rng(22), ctx=ctx)
    assert ctx.density_map is not None and ctx.density_map.shape == (64, 64)
    assert ctx.tissue_mask is not None and ctx.tissue_mask.shape == (64, 64)
    assert ctx.annotation_slice is not None and ctx.annotation_slice.shape == (64, 64)
    assert ctx.annotation_slice.dtype == np.int32


def test_blade_stretch_horizontal_anisotropy_smoke() -> None:
    h, w = 64, 64
    image = np.random.default_rng(42).random((h, w, 3)).astype(np.float32)
    ctx = _ctx(modality="nissl", h=h, w=w, density=None, tissue_mask=None)

    t_h = BladeStretchHorizontal(
        p=1.0,
        horizontal_stretch_range=(1.10, 1.10),
        vertical_stretch_range=(1.0, 1.0),
        shear_range_deg=(0.0, 0.0),
    )
    t_v = BladeStretchHorizontal(
        p=1.0,
        horizontal_stretch_range=(1.0, 1.0),
        vertical_stretch_range=(1.10, 1.10),
        shear_range_deg=(0.0, 0.0),
    )

    out_h = t_h(image, rng=np.random.default_rng(0), ctx=ctx)
    out_v = t_v(image, rng=np.random.default_rng(0), ctx=ctx)
    diff = float(np.abs(out_h - out_v).mean())
    assert diff > 1e-4


@pytest.mark.parametrize(
    "cls",
    [DAPINuclei, NisslCellBodies, FluorescenceSpeckle, ISHPuncta],
    ids=lambda c: c.__name__,
)
def test_legacy_texture_zero_density_near_input(cls: type) -> None:
    t = cls(p=1.0)
    img = _img(seed=7)
    ctx = _ctx(h=64, w=64, density=0.0)
    out = t(img.copy(), rng=np.random.default_rng(0), ctx=ctx)
    max_diff = float(np.abs(out - img).max())
    assert max_diff < 0.25


@pytest.mark.parametrize(
    "cls",
    [DAPINuclei, NisslCellBodies, FluorescenceSpeckle, ISHPuncta],
    ids=lambda c: c.__name__,
)
def test_legacy_texture_high_density_modifies_image(cls: type) -> None:
    t = cls(p=1.0)
    img = np.zeros((64, 64, 3), dtype=np.float32)
    ctx = _ctx(h=64, w=64, density=1.0)
    out = t(img.copy(), rng=np.random.default_rng(0), ctx=ctx)
    assert float(np.abs(out - img).mean()) > 0.001


@pytest.mark.parametrize(
    "cls",
    [DAPINuclei, NisslCellBodies, FluorescenceSpeckle, ISHPuncta],
    ids=lambda c: c.__name__,
)
def test_legacy_texture_tissue_mask_confines_signal(cls: type) -> None:
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=bool)
    mask[: h // 2, :] = True

    img = np.zeros((h, w, 3), dtype=np.float32)
    ctx = _ctx(h=h, w=w, density=np.ones((h, w), dtype=np.float32), tissue_mask=mask)
    t = cls(p=1.0)
    out = t(img.copy(), rng=np.random.default_rng(0), ctx=ctx)
    bottom = out[h // 2 :, :, :]
    if cls is ISHPuncta:
        assert float(bottom.max()) < 0.3
    else:
        assert float(bottom.max()) < 0.05


@pytest.mark.parametrize(
    "cls",
    [DAPINuclei, NisslCellBodies, FluorescenceSpeckle, ISHPuncta],
    ids=lambda c: c.__name__,
)
def test_legacy_texture_none_density_map_does_not_raise(cls: type) -> None:
    t = cls(p=1.0)
    img = _img()
    ctx = _ctx(h=64, w=64, density=None)
    out = t(img.copy(), rng=np.random.default_rng(0), ctx=ctx)
    assert out.shape == img.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize(
    "cls",
    [DAPINuclei, NisslCellBodies, FluorescenceSpeckle, ISHPuncta],
    ids=lambda c: c.__name__,
)
def test_legacy_texture_performance_1024(cls: type) -> None:
    t = cls(p=1.0)
    img = np.zeros((1024, 1024, 3), dtype=np.float32)
    ctx = _ctx(h=1024, w=1024, density=0.8, pixel_size_um=25.0)
    t0 = time.perf_counter()
    out = t(img, rng=np.random.default_rng(0), ctx=ctx)
    elapsed = time.perf_counter() - t0
    assert out.shape == (1024, 1024, 3)
    assert elapsed < 2.0, f"{cls.__name__} took {elapsed:.2f}s on 1024x1024"

