"""Consolidated stain/counterstain and signal-layer tests."""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest
from augmentation.counterstain import (
    COUNTERSTAIN_REGISTRY,
    render_dapi_counterstain,
    render_nissl_counterstain,
    render_no_counterstain,
)
from augmentation.density import atlas_grayscale_density_map
from augmentation.transforms.base import TransformContext
from augmentation.transforms.texture import (
    DAPIGrayMatterNuclei,
    DAPIWhiteMatterNuclei,
    HematoxylinGrayMatterNuclei,
    HematoxylinWhiteMatterNuclei,
    NuclearFastRedGrayMatterNuclei,
    NuclearFastRedWhiteMatterNuclei,
)
from augmentation.transforms.tissue_class import classify_tissue


def _substrate_canvas(
    h: int,
    w: int,
    mask: np.ndarray,
    *,
    substrate: tuple[float, float, float],
) -> np.ndarray:
    canvas = np.full((h, w, 3), 0.97, dtype=np.float32)
    canvas[mask] = np.array(substrate, dtype=np.float32)
    return canvas


def _ctx_with_masks(
    *,
    modality: str,
    gm: np.ndarray,
    wm: np.ndarray,
    pixel_size_um: float = 10.0,
) -> TransformContext:
    tissue = gm | wm
    return TransformContext(
        modality=modality,
        annotation_slice=np.zeros_like(gm, dtype=np.int32),
        density_map=np.full(gm.shape, 0.8, dtype=np.float32),
        tissue_mask=tissue,
        pixel_size_um=pixel_size_um,
        tissue_class_masks={
            "gray_matter": gm,
            "white_matter": wm,
            "ventricle": np.zeros_like(gm, dtype=bool),
            "tissue": tissue,
            "background": ~tissue,
        },
    )


# ---------------------------------------------------------------------------
# Counterstain registry: contract + a few key signatures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(COUNTERSTAIN_REGISTRY))
def test_counterstain_contract_returns_hwc_float32_unit_range(
    atlas: object, atlas_slice_inputs: tuple[np.ndarray, np.ndarray], name: str
) -> None:
    ref, ann = atlas_slice_inputs
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="generic",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    out = COUNTERSTAIN_REGISTRY[name](
        ref,
        ann,
        atlas,
        masks=masks,
        density_map=density,
        ctx=ctx,
        rng=np.random.default_rng(0),
        pixel_size_um=10.0,
    )
    assert out.dtype == np.float32
    assert out.shape == (ann.shape[0], ann.shape[1], 3)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


@pytest.mark.parametrize("name", sorted(COUNTERSTAIN_REGISTRY))
def test_counterstain_populates_signal_mask(
    atlas: object, atlas_slice_inputs: tuple[np.ndarray, np.ndarray], name: str
) -> None:
    ref, ann = atlas_slice_inputs
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="generic",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    COUNTERSTAIN_REGISTRY[name](
        ref,
        ann,
        atlas,
        masks=masks,
        density_map=density,
        ctx=ctx,
        rng=np.random.default_rng(0),
        pixel_size_um=10.0,
    )
    assert ctx.counterstain_signal_mask is not None
    assert ctx.counterstain_signal_mask.shape == ann.shape
    assert ctx.counterstain_signal_mask.dtype == np.float32


def test_dapi_counterstain_blue_dominant(atlas: object, atlas_slice_inputs: tuple[np.ndarray,
    np.ndarray]) -> None:
    ref, ann = atlas_slice_inputs
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="generic",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    out = render_dapi_counterstain(
        ref, ann, atlas, masks=masks, density_map=density,
                ctx=ctx, rng=np.random.default_rng(0), pixel_size_um=10.0
    )
    r, g, b = out[..., 0].sum(), out[..., 1].sum(), out[..., 2].sum()
    assert b > r * 3
    assert b > g * 3


def test_nissl_counterstain_is_brightfieldish(atlas: object, atlas_slice_inputs: tuple[np.ndarray,
    np.ndarray]) -> None:
    ref, ann = atlas_slice_inputs
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="generic",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    out = render_nissl_counterstain(
        ref, ann, atlas, masks=masks, density_map=density,
                ctx=ctx, rng=np.random.default_rng(0), pixel_size_um=10.0
    )
    lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).mean()
    assert lum > 0.4


def test_none_counterstain_is_substrate_only(atlas: object, atlas_slice_inputs: tuple[np.ndarray,
    np.ndarray]) -> None:
    ref, ann = atlas_slice_inputs
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="generic",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    out = render_no_counterstain(
        ref, ann, atlas, masks=masks, density_map=density,
                ctx=ctx, rng=np.random.default_rng(0), pixel_size_um=10.0
    )
    assert (ctx.counterstain_signal_mask == 0.0).all()
    lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])
    tissue_lum = lum[masks["tissue"]]
    assert tissue_lum.std() < 0.15


# ---------------------------------------------------------------------------
# Texture pair tests: DAPI / Hematoxylin / NFR (confinement + color signatures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,substrate,mask_key",
    [
        (DAPIGrayMatterNuclei, (0.0, 0.0, 0.0), "gm"),
        (DAPIWhiteMatterNuclei, (0.0, 0.0, 0.0), "wm"),
        (HematoxylinGrayMatterNuclei, (0.92, 0.88, 0.90), "gm"),
        (HematoxylinWhiteMatterNuclei, (0.92, 0.88, 0.90), "wm"),
        (NuclearFastRedGrayMatterNuclei, (0.96, 0.94, 0.95), "gm"),
        (NuclearFastRedWhiteMatterNuclei, (0.96, 0.94, 0.95), "wm"),
    ],
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_texture_transforms_confined_to_class_mask(cls: type, substrate, mask_key: str) -> None:
    h, w = 160, 200
    gm = np.zeros((h, w), dtype=bool)
    gm[24:120, 32:136] = True
    wm = np.zeros((h, w), dtype=bool)
    wm[96:136, 24:176] = True

    ctx = _ctx_with_masks(modality="generic", gm=gm, wm=wm)
    mask = gm if mask_key == "gm" else wm
    img = _substrate_canvas(h, w, mask, substrate=substrate)
    out = cls(p=1.0)(img, rng=np.random.default_rng(0), ctx=ctx)
    assert np.allclose(out[~mask], img[~mask], atol=1e-5)


def test_dapi_texture_blue_dominates() -> None:
    h, w = 256, 256
    gm = np.ones((h, w), dtype=bool)
    ctx = _ctx_with_masks(modality="dapi", gm=gm, wm=np.zeros((h, w), dtype=bool))
    img = np.zeros((h, w, 3), dtype=np.float32)
    out = DAPIGrayMatterNuclei(p=1.0)(img, rng=np.random.default_rng(0), ctx=ctx)
    r_total = out[..., 0].sum()
    g_total = out[..., 1].sum()
    b_total = out[..., 2].sum()
    assert b_total > r_total * 3
    assert b_total > g_total * 3


@pytest.mark.parametrize(
    "cls,substrate,color_expectation",
    [
        (HematoxylinGrayMatterNuclei, (0.92, 0.88, 0.90), "blue"),
        (NuclearFastRedGrayMatterNuclei, (0.96, 0.94, 0.95), "lavender"),
    ],
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_counterstain_color_signatures(cls: type, substrate, color_expectation: str) -> None:
    h, w = 220, 220
    gm = np.ones((h, w), dtype=bool)
    ctx = _ctx_with_masks(modality="generic", gm=gm, wm=np.zeros((h, w), dtype=bool))
    img = _substrate_canvas(h, w, gm, substrate=substrate)
    out = cls(p=1.0)(img, rng=np.random.default_rng(1), ctx=ctx)
    delta = np.abs(out - np.array(substrate, dtype=np.float32)[None, None, :])
    cell_mask = delta.max(axis=2) > 0.02
    assert cell_mask.any()
    cell_pixels = out[cell_mask]
    if color_expectation == "blue":
        assert cell_pixels[:, 2].mean() > cell_pixels[:, 0].mean()
        assert cell_pixels[:, 2].mean() > cell_pixels[:, 1].mean()
    else:
        # NFR: subtle lavender; blue >= green and darker than substrate overall.
        sub_lum = 0.2126 * substrate[0] + 0.7152 * substrate[1] + 0.0722 * substrate[2]
        cell_lum = (
            0.2126 * cell_pixels[:, 0].mean()
            + 0.7152 * cell_pixels[:, 1].mean()
            + 0.0722 * cell_pixels[:, 2].mean()
        )
        assert cell_lum < sub_lum
        assert cell_pixels[:, 2].mean() >= cell_pixels[:, 1].mean()


def test_wm_blobs_are_elongated_for_counterstains() -> None:
    h, w = 5, 260
    wm = np.ones((h, w), dtype=bool)
    ctx_hema = _ctx_with_masks(modality="hematoxylin",
        gm=np.zeros((h, w), dtype=bool), wm=wm, pixel_size_um=5.0)
    ctx_nfr = _ctx_with_masks(modality="nfr",
        gm=np.zeros((h, w), dtype=bool), wm=wm, pixel_size_um=5.0)

    substrate = (0.92, 0.88, 0.90)
    img = _substrate_canvas(h, w, wm, substrate=substrate)

    hema = HematoxylinWhiteMatterNuclei(
        p=1.0, density_range_per_mm2=(200.0, 200.0), aspect_ratio_range=(3.0, 6.0)
    )(img, rng=np.random.default_rng(99), ctx=ctx_hema)
    nfr = NuclearFastRedWhiteMatterNuclei(
        p=1.0, density_range_per_mm2=(200.0, 200.0), aspect_ratio_range=(3.0, 6.0)
    )(img, rng=np.random.default_rng(99), ctx=ctx_nfr)

    for out in (hema, nfr):
        delta = np.abs(out - np.array(substrate, dtype=np.float32)[None, None, :])
        cell_mask = delta.max(axis=2) > 0.04
        if not cell_mask.any():
            pytest.skip("No WM cells rendered in thin strip")
        ys_cell, xs_cell = np.where(cell_mask)
        if len(xs_cell) < 4:
            pytest.skip("Too few WM cell pixels for elongation measurement")
        assert xs_cell.std() > ys_cell.std()


def test_tract_orientation_memoized_per_context() -> None:
    import augmentation.transforms.texture as tex_module

    h, w = 128, 128
    wm = np.zeros((h, w), dtype=bool)
    wm[56:72, 20:108] = True
    ctx = _ctx_with_masks(modality="dapi", gm=np.zeros((h, w), dtype=bool), wm=wm)
    assert ctx.tract_orientation is None

    img = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.default_rng(42)

    call_count = 0
    original_fn = tex_module._local_tract_orientation

    def counting(mask, *, smoothing_sigma=2.0):
        nonlocal call_count
        call_count += 1
        return original_fn(mask, smoothing_sigma=smoothing_sigma)

    with mock.patch.object(tex_module, "_local_tract_orientation", counting):
        DAPIWhiteMatterNuclei(p=1.0)(img, rng=rng, ctx=ctx)
        assert call_count == 1
        DAPIWhiteMatterNuclei(p=1.0)(img, rng=rng, ctx=ctx)
        assert call_count == 1


# ---------------------------------------------------------------------------
# Signal wrappers smoke tests (nbt_bcip, dab, fluorescent_probe)
# ---------------------------------------------------------------------------


def test_signal_registry_integrity() -> None:
    from augmentation.signals import SIGNAL_REGISTRY

    assert set(SIGNAL_REGISTRY.keys()) == {"nbt_bcip", "dab", "fluorescent_probe"}
    for key, fn in SIGNAL_REGISTRY.items():
        assert callable(fn), f"{key} is not callable"


@pytest.fixture(scope="module")
def base_canvas_and_ctx(
    atlas: object, atlas_slice_inputs: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, TransformContext]:
    from augmentation.counterstain import COUNTERSTAIN_REGISTRY

    ref, ann = atlas_slice_inputs
    rng = np.random.default_rng(0)
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="ish",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    canvas = COUNTERSTAIN_REGISTRY["none"](
        ref, ann, atlas, masks=masks, density_map=density, ctx=ctx, rng=rng, pixel_size_um=10.0
    )
    return canvas, ctx


@pytest.mark.parametrize("fn_name", ["apply_nbt_bcip_signal", "apply_dab_signal"])
def test_signal_wrappers_contract_and_determinism(
    base_canvas_and_ctx: tuple[np.ndarray, TransformContext], fn_name: str
) -> None:
    import augmentation.signals as sig

    canvas, ctx = base_canvas_and_ctx
    fn = getattr(sig, fn_name)
    out = fn(canvas, ctx=ctx, rng=np.random.default_rng(1))
    assert out.dtype == np.float32
    assert out.shape == canvas.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.array_equal(canvas, out)

    a = fn(canvas, ctx=ctx, rng=np.random.default_rng(7))
    b = fn(canvas, ctx=ctx, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_fluorescent_probe_green_channel_additive(
    atlas: object, atlas_slice_inputs: tuple[np.ndarray, np.ndarray]
) -> None:
    from augmentation.signals import apply_fluorescent_probe_signal

    ref, ann = atlas_slice_inputs
    rng = np.random.default_rng(6)
    masks = classify_tissue(ann, atlas)
    density = atlas_grayscale_density_map(ref, masks["tissue"], gamma=1.2, floor=0.15)
    ctx = TransformContext(
        modality="ish",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=10.0,
        tissue_class_masks=masks,
    )
    canvas = COUNTERSTAIN_REGISTRY["dapi"](
        ref, ann, atlas, masks=masks, density_map=density, ctx=ctx, rng=rng, pixel_size_um=10.0
    )
    out = apply_fluorescent_probe_signal(canvas, ctx=ctx, rng=rng, channel=1)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out[..., 1].sum() >= canvas[..., 1].sum() - 1e-3
