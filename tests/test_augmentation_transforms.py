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
    HemibrainPreparation,
    IlluminationGradient,
    Microbubbles,
    PosteriorWingDamage,
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
    HemibrainPreparation,
    PosteriorWingDamage,
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


def test_hemibrain_preparation_keeps_one_side_and_recenters() -> None:
    h, w = 40, 80
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[8:32, 4:36, :] = 0.8
    image[8:32, 44:76, :] = 0.4
    tissue = image.mean(axis=2) > 0
    ctx = _ctx(modality="nissl", h=h, w=w, tissue_mask=tissue)

    out = HemibrainPreparation(p=1.0, keep_side="left")(
        image.copy(),
        rng=np.random.default_rng(0),
        ctx=ctx,
    )

    out_tissue = out.mean(axis=2) > 0.1
    _, cols = np.where(out_tissue)
    assert cols.min() >= 20
    assert cols.max() <= 56
    assert abs(float(cols.mean()) - (w - 1) / 2.0) < 1.0
    assert np.isclose(out[out_tissue].mean(), 0.8)
    assert ctx.tissue_mask is not None
    np.testing.assert_array_equal(ctx.tissue_mask, out_tissue)


def test_hemibrain_preparation_skips_sagittal() -> None:
    h, w = 40, 80
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[8:32, 4:36, :] = 0.8
    tissue = image.mean(axis=2) > 0
    ctx = _ctx(modality="nissl", h=h, w=w, tissue_mask=tissue)
    ctx.plane = "sagittal"

    out = HemibrainPreparation(p=1.0, keep_side="left")(
        image.copy(),
        rng=np.random.default_rng(0),
        ctx=ctx,
    )

    np.testing.assert_array_equal(out, image)
    np.testing.assert_array_equal(ctx.tissue_mask, tissue)


def _posterior_wing_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = 72, 96
    image = np.full((h, w, 3), 0.05, dtype=np.float32)
    rr, cc = np.ogrid[:h, :w]
    left_wing = ((rr - 25) ** 2 / 22**2 + (cc - 15) ** 2 / 15**2) <= 1.0
    right_wing = ((rr - 25) ** 2 / 22**2 + (cc - 81) ** 2 / 15**2) <= 1.0
    core = ((rr - 42) ** 2 / 18**2 + (cc - 48) ** 2 / 16**2) <= 1.0
    image[left_wing, :] = 0.8
    image[right_wing, :] = 0.7
    image[core, :] = 0.45
    return image, left_wing, right_wing, core


def test_posterior_wing_damage_removes_entire_lateral_wing() -> None:
    image, left_wing, right_wing, core = _posterior_wing_fixture()
    h, w = image.shape[:2]
    isocortex = np.zeros((h, w), dtype=bool)
    isocortex[left_wing & (np.indices((h, w))[0] < 23)] = True
    isocortex[right_wing & (np.indices((h, w))[0] < 23)] = True
    tissue = image.mean(axis=2) > 0.1
    ctx = _ctx(modality="brightfield", h=h, w=w, tissue_mask=tissue)
    ctx.position_mm = 9.5
    ctx.tissue_class_masks = {
        "isocortex": isocortex,
        "hippocampal_formation": left_wing & ~isocortex,
        "thalamus": np.zeros((h, w), dtype=bool),
        "tissue": tissue,
    }

    out = PosteriorWingDamage(p=1.0, mode="left_missing")(
        image.copy(),
        rng=np.random.default_rng(1),
        ctx=ctx,
    )

    non_isocortex_wing = left_wing & ~isocortex
    assert float(out[left_wing].mean()) < 0.25
    assert float(out[non_isocortex_wing].mean()) < 0.1
    assert float(out[right_wing].mean()) > 0.6
    assert float(out[core].mean()) > 0.4
    assert ctx.tissue_mask is not None
    assert not ctx.tissue_mask[left_wing].any()


def test_posterior_wing_damage_preserves_medial_core_shoulder() -> None:
    h, w = 80, 160
    image = np.full((h, w, 3), 0.05, dtype=np.float32)
    rr, cc = np.ogrid[:h, :w]
    left_wing = ((rr - 28) ** 2 / 24**2 + (cc - 16) ** 2 / 12**2) <= 1.0
    core = ((rr - 48) ** 2 / 20**2 + (cc - 82) ** 2 / 23**2) <= 1.0
    shoulder = np.zeros((h, w), dtype=bool)
    shoulder[12:58, 32:40] = True
    image[left_wing, :] = 0.8
    image[core, :] = 0.45
    image[shoulder, :] = 0.5
    tissue = image.mean(axis=2) > 0.1
    ctx = _ctx(modality="brightfield", h=h, w=w, tissue_mask=tissue)
    ctx.position_mm = 9.5
    ctx.tissue_class_masks = {
        "isocortex": left_wing,
        "thalamus": np.zeros((h, w), dtype=bool),
        "tissue": tissue,
    }

    out = PosteriorWingDamage(p=1.0, mode="left_missing")(
        image.copy(),
        rng=np.random.default_rng(1),
        ctx=ctx,
    )

    assert float(out[left_wing].mean()) < 0.1
    assert float(out[shoulder].mean()) > 0.45
    assert float(out[core].mean()) > 0.4


def test_posterior_wing_damage_includes_lateral_isocortex_satellite() -> None:
    h, w = 80, 160
    image = np.full((h, w, 3), 0.05, dtype=np.float32)
    rr, cc = np.ogrid[:h, :w]
    left_wing = ((rr - 30) ** 2 / 22**2 + (cc - 18) ** 2 / 14**2) <= 1.0
    satellite = ((rr - 10) ** 2 / 5**2 + (cc - 45) ** 2 / 5**2) <= 1.0
    core = ((rr - 48) ** 2 / 20**2 + (cc - 82) ** 2 / 23**2) <= 1.0
    image[left_wing, :] = 0.8
    image[satellite, :] = 0.75
    image[core, :] = 0.45
    tissue = image.mean(axis=2) > 0.1
    ctx = _ctx(modality="brightfield", h=h, w=w, tissue_mask=tissue)
    ctx.position_mm = 9.5
    ctx.tissue_class_masks = {
        "isocortex": left_wing | satellite,
        "thalamus": np.zeros((h, w), dtype=bool),
        "tissue": tissue,
    }

    out = PosteriorWingDamage(p=1.0, mode="left_missing")(
        image.copy(),
        rng=np.random.default_rng(1),
        ctx=ctx,
    )

    assert float(out[left_wing].mean()) < 0.1
    assert float(out[satellite].mean()) < 0.1
    assert float(out[core].mean()) > 0.4


def test_posterior_wing_damage_detaches_and_repositions_wing() -> None:
    image, left_wing, right_wing, core = _posterior_wing_fixture()
    h, w = image.shape[:2]
    tissue = image.mean(axis=2) > 0.1
    ctx = _ctx(modality="brightfield", h=h, w=w, tissue_mask=tissue)
    ctx.position_mm = 9.5
    ctx.tissue_class_masks = {
        "isocortex": left_wing | right_wing,
        "thalamus": np.zeros((h, w), dtype=bool),
        "tissue": tissue,
    }

    out = PosteriorWingDamage(
        p=1.0,
        mode="right_detached",
        detach_shift_px=(24, -8),
        detach_angle_deg=(0.0, 0.0),
    )(
        image.copy(),
        rng=np.random.default_rng(1),
        ctx=ctx,
    )

    shifted_right_wing = np.zeros_like(right_wing)
    shifted_right_wing[:-8, 24:] = right_wing[8:, :-24]
    source_gap = right_wing & ~shifted_right_wing
    assert float(out[source_gap].mean()) < 0.2
    assert float(out[shifted_right_wing].mean()) > 0.45
    assert float(out[left_wing].mean()) > 0.7
    assert float(out[core].mean()) > 0.4
    assert ctx.tissue_mask is not None
    assert ctx.tissue_mask[source_gap].sum() < 0.05 * source_gap.sum()
    assert ctx.tissue_mask[shifted_right_wing].any()


def test_posterior_wing_damage_skips_before_posterior_gate() -> None:
    image, _, _, _ = _posterior_wing_fixture()
    h, w = image.shape[:2]
    tissue = image.mean(axis=2) > 0.1
    ctx = _ctx(modality="brightfield", h=h, w=w, tissue_mask=tissue)
    ctx.position_mm = 5.0
    ctx.tissue_class_masks = {
        "thalamus": np.zeros((h, w), dtype=bool),
        "tissue": tissue,
    }

    out = PosteriorWingDamage(p=1.0, mode="both_missing")(
        image.copy(),
        rng=np.random.default_rng(1),
        ctx=ctx,
    )

    np.testing.assert_array_equal(out, image)


def test_posterior_wing_damage_skips_when_thalamus_present() -> None:
    image, _, _, _ = _posterior_wing_fixture()
    h, w = image.shape[:2]
    tissue = image.mean(axis=2) > 0.1
    thalamus = np.zeros((h, w), dtype=bool)
    thalamus[34:46, 42:54] = True
    ctx = _ctx(modality="brightfield", h=h, w=w, tissue_mask=tissue | thalamus)
    ctx.position_mm = 9.5
    ctx.tissue_class_masks = {
        "thalamus": thalamus,
        "tissue": tissue | thalamus,
    }

    out = PosteriorWingDamage(p=1.0, mode="both_missing")(
        image.copy(),
        rng=np.random.default_rng(1),
        ctx=ctx,
    )

    np.testing.assert_array_equal(out, image)


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
