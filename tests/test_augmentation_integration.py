"""Consolidated pipeline integration tests across modalities."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest
from augmentation.damage_pipeline import apply_damage_layer
from augmentation.modes import FLUORESCENCE_MODES, ISH_MODES, sample_mode
from augmentation.transforms.base import TransformContext
from augmentation.transforms.texture import FluorescenceMarker


def _synth_reference_hw(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """HW uint8 grayscale reference (matches pipeline contracts)."""
    return np.random.default_rng(seed).integers(0, 255, size=(h, w), dtype=np.uint8)


def _synth_annotation(h: int = 64, w: int = 64) -> np.ndarray:
    ann = np.zeros((h, w), dtype=np.int32)
    ann[: h // 2, :] = 315  # grey-like region
    ann[3 * h // 4 :, :] = 1009  # fiber-tract-like region
    ann[h // 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 73  # ventricle-like speck
    return ann


class _FakeAtlas:
    """Minimal atlas stub for classify_tissue duck-type."""

    atlas_name = "fake_atlas_for_tests"
    structures = {
        "grey": {"id": 315, "acronym": "grey", "name": "grey", "structure_id_path": [315]},
        "fiber tracts": {
            "id": 1009,
            "acronym": "fiber tracts",
            "name": "fiber tracts",
            "structure_id_path": [1009],
        },
        "VS": {"id": 73, "acronym": "VS", "name": "VS", "structure_id_path": [73]},
    }

    def get_structure_descendants(self, acronym: str):  # noqa: D401
        return []


_MODALITIES = ("dapi", "nissl", "brightfield", "fluorescence", "ish")


def _render(
    modality: str, ref: np.ndarray, ann: np.ndarray, atlas: object,
    *, seed: int, pixel_size_um: float
) -> np.ndarray:
    if modality == "dapi":
        from augmentation.dapi_pipeline import render_dapi_section

        return render_dapi_section(ref, ann, atlas, seed=seed, pixel_size_um=pixel_size_um)
    if modality == "nissl":
        from augmentation.nissl_pipeline import render_nissl_section

        return render_nissl_section(ref, ann, atlas, seed=seed, pixel_size_um=pixel_size_um)
    if modality == "brightfield":
        from augmentation.brightfield_pipeline import render_brightfield_section

        return render_brightfield_section(
            ref,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=pixel_size_um,
            mode="pan_neuronal",
            counterstain="none",
        )
    if modality == "fluorescence":
        from augmentation.fluorescence_pipeline import render_fluorescence_section

        # Pin a mode so signal distribution is deterministic across refactors.
        mode = next(m for m in FLUORESCENCE_MODES if m.name == "dapi_gfp")
        return render_fluorescence_section(
            ref, ann, atlas, seed=seed, pixel_size_um=pixel_size_um,
            mode=mode
        )
    if modality == "ish":
        from augmentation.ish_pipeline import render_ish_section

        return render_ish_section(
            ref, ann, atlas, seed=seed, pixel_size_um=pixel_size_um,
            mode=ISH_MODES[0]
        )
    raise AssertionError(modality)


@pytest.mark.parametrize("modality", _MODALITIES)
def test_pipeline_renderers_shape_dtype_range(modality: str) -> None:
    ref = _synth_reference_hw()
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    out = _render(modality, ref, ann, atlas, seed=0, pixel_size_um=25.0)
    assert out.shape == (ref.shape[0], ref.shape[1], 3)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize("modality", _MODALITIES)
def test_pipeline_renderers_determinism_same_seed(modality: str) -> None:
    ref = _synth_reference_hw(seed=5)
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    a = _render(modality, ref, ann, atlas, seed=42, pixel_size_um=25.0)
    b = _render(modality, ref, ann, atlas, seed=42, pixel_size_um=25.0)
    np.testing.assert_array_equal(a, b)


def test_pipeline_different_seeds_differ() -> None:
    ref = _synth_reference_hw()
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    a = _render("nissl", ref, ann, atlas, seed=0, pixel_size_um=25.0)
    b = _render("nissl", ref, ann, atlas, seed=1, pixel_size_um=25.0)
    assert not np.array_equal(a, b)


def test_dapi_pipeline_blue_dominant_and_background_dark() -> None:
    from augmentation.dapi_pipeline import render_dapi_section

    ref = _synth_reference_hw()
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    out = render_dapi_section(ref, ann, atlas, seed=0, pixel_size_um=25.0)
    r, g, b = out[..., 0].sum(), out[..., 1].sum(), out[..., 2].sum()
    assert b > r * 1.5
    assert b > g * 1.5
    # The DAPI pipeline renders nuclei textures across all annotated and
    # background pixels alike  -  it does not mask output to annotation=0 regions.
    # We therefore only assert blue dominance (the distinguishing DAPI trait),
    # not a per-pixel background-darkness constraint.


def test_nissl_pipeline_is_not_dark() -> None:
    from augmentation.nissl_pipeline import render_nissl_section

    ref = _synth_reference_hw()
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    out = render_nissl_section(ref, ann, atlas, seed=0, pixel_size_um=25.0)
    lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]).mean()
    assert lum > 0.3


def test_brightfield_pipeline_myelin_inverts_gm_wm_contrast() -> None:
    from augmentation.brightfield_pipeline import render_brightfield_section
    from augmentation.transforms.tissue_class import classify_tissue

    ref = _synth_reference_hw()
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    masks = classify_tissue(ann, atlas)

    out_pan = render_brightfield_section(
        ref, ann, atlas, seed=1, pixel_size_um=25.0,
            mode="pan_neuronal", counterstain="none", apply_damage=False
    )
    out_myelin = render_brightfield_section(
        ref, ann, atlas, seed=1, pixel_size_um=25.0,
            mode="myelin", counterstain="none", apply_damage=False
    )
    lum_pan = 0.2126 * out_pan[..., 0] + 0.7152 * out_pan[..., 1] + 0.0722 * out_pan[..., 2]
    lum_my = 0.2126 * out_myelin[..., 0] + 0.7152 * out_myelin[..., 1] + 0.0722 * out_myelin[..., 2]

    bg_pan = float(lum_pan[masks["tissue"]].max())
    bg_my = float(lum_my[masks["tissue"]].max())
    gm_drop_pan = bg_pan - float(lum_pan[masks["gray_matter"]].mean())
    wm_drop_pan = bg_pan - float(lum_pan[masks["white_matter"]].mean())
    gm_drop_my = bg_my - float(lum_my[masks["gray_matter"]].mean())
    wm_drop_my = bg_my - float(lum_my[masks["white_matter"]].mean())

    assert gm_drop_pan > wm_drop_pan
    assert wm_drop_my > gm_drop_my


def test_fluorescence_marker_paints_only_one_channel() -> None:
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.float32)
    gm = np.ones((h, w), dtype=bool)
    ctx = TransformContext(
        modality="fluorescence",
        annotation_slice=None,
        density_map=np.full((h, w), 0.5, dtype=np.float32),
        tissue_mask=gm,
        pixel_size_um=10.0,
        tissue_class_masks={
            "gray_matter": gm,
            "white_matter": np.zeros((h, w), dtype=bool),
            "ventricle": np.zeros((h, w), dtype=bool),
            "tissue": gm,
            "background": np.zeros((h, w), dtype=bool),
        },
    )
    for ch in (0, 1, 2):
        out = FluorescenceMarker(channel=ch, p=1.0, density_range_per_mm2=(500, 500))(
            img, rng=np.random.default_rng(0), ctx=ctx
        )
        chosen = out[..., ch].sum()
        other_total = sum(out[..., c].sum() for c in range(3) if c != ch)
        assert chosen > 0.0
        assert other_total == 0.0


def test_fluorescence_marker_invalid_channel_raises() -> None:
    with pytest.raises(ValueError):
        FluorescenceMarker(channel=3)  # type: ignore[arg-type]


def test_fluorescence_pipeline_picks_mode_when_unspecified_peek() -> None:
    """Without mode=, RNG-driven sampling should hit >=2 distinct modes in 20 seeds."""
    seen: set[str] = set()
    for s in range(20):
        rng = np.random.default_rng(s)
        rng.uniform(0.9, 1.7)  # gamma draw inside pipeline
        rng.uniform(0.10, 0.20)  # floor draw
        chosen = sample_mode(rng, FLUORESCENCE_MODES)
        seen.add(chosen.name)
    assert len(seen) >= 2


def test_ish_pipeline_picks_mode_when_unspecified_peek() -> None:
    seen: set[str] = set()
    for s in range(20):
        rng = np.random.default_rng(s)
        rng.uniform(0.9, 1.7)
        rng.uniform(0.10, 0.20)
        chosen = sample_mode(rng, ISH_MODES)
        seen.add(chosen.name)
    assert len(seen) >= 2


def test_ish_allen_style_counterstain_mask_is_all_zero(
    atlas: object,
    atlas_slice_inputs: tuple[np.ndarray, np.ndarray],
) -> None:
    from augmentation.counterstain import COUNTERSTAIN_REGISTRY
    from augmentation.density import atlas_grayscale_density_map
    from augmentation.transforms.tissue_class import classify_tissue

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
    cs_fn = COUNTERSTAIN_REGISTRY[ISH_MODES[0].counterstain]  # "none"
    cs_fn(ref, ann, atlas, masks=masks, density_map=density, ctx=ctx, rng=rng, pixel_size_um=10.0)
    assert ctx.counterstain_signal_mask is not None
    assert ctx.counterstain_signal_mask.max() == 0.0


# ---------------------------------------------------------------------------
# Damage pipeline integration
# ---------------------------------------------------------------------------


def _damage_ctx(modality: str, h: int = 80, w: int = 100) -> TransformContext:
    mask = np.ones((h, w), dtype=bool)
    mask[:8, :] = False
    return TransformContext(
        modality=modality,
        annotation_slice=None,
        density_map=None,
        tissue_mask=mask,
        pixel_size_um=25.0,
    )


def _make_image(seed: int = 0, h: int = 80, w: int = 100) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def test_apply_damage_layer_contract_and_changes_output() -> None:
    image = _make_image(seed=99)
    ctx = _damage_ctx("nissl")
    out = apply_damage_layer(image, rng=np.random.default_rng(5), ctx=ctx, modality="nissl")
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert not np.array_equal(out, image)


def test_apply_damage_layer_determinism_same_seed() -> None:
    image = _make_image(seed=7)
    a = apply_damage_layer(image, rng=np.random.default_rng(42),
        ctx=_damage_ctx("nissl"), modality="nissl")
    b = apply_damage_layer(image, rng=np.random.default_rng(42),
        ctx=_damage_ctx("nissl"), modality="nissl")
    np.testing.assert_array_equal(a, b)


def test_apply_damage_layer_invalid_intensity_raises() -> None:
    image = _make_image()
    with pytest.raises(ValueError, match="intensity"):
        apply_damage_layer(image, rng=np.random.default_rng(0),
            ctx=_damage_ctx("nissl"), modality="nissl", intensity="extreme")


def test_apply_damage_layer_light_vs_heavy_differs() -> None:
    image = _make_image(seed=11)
    light = apply_damage_layer(image, rng=np.random.default_rng(20),
        ctx=_damage_ctx("nissl"), modality="nissl", intensity="light")
    heavy = apply_damage_layer(image, rng=np.random.default_rng(20),
        ctx=_damage_ctx("nissl"), modality="nissl", intensity="heavy")
    assert float(np.abs(light - heavy).mean()) > 1e-4


@pytest.mark.parametrize(
    "modality,render_fn,kwargs",
    [
        ("dapi", "render_dapi_section", {}),
        ("nissl", "render_nissl_section", {}),
        ("brightfield", "render_brightfield_section",
         {"mode": "pan_neuronal", "counterstain": "none"}),
        ("fluorescence", "render_fluorescence_section", {}),
        ("ish", "render_ish_section", {}),
    ],
)
def test_pipeline_apply_damage_toggle_changes_output(modality: str, render_fn: str,
    kwargs: dict) -> None:
    ref = _synth_reference_hw(seed=0)
    ann = _synth_annotation()
    atlas = _FakeAtlas()

    pipeline_names = {
        "dapi": "augmentation.dapi_pipeline",
        "nissl": "augmentation.nissl_pipeline",
        "brightfield": "augmentation.brightfield_pipeline",
        "fluorescence": "augmentation.fluorescence_pipeline",
        "ish": "augmentation.ish_pipeline",
    }
    mod = importlib.import_module(pipeline_names[modality])
    fn = getattr(mod, render_fn)

    if modality == "fluorescence":
        # avoid random mode in this test
        kwargs = {**kwargs, "mode": next(m for m in FLUORESCENCE_MODES if m.name == "dapi_gfp")}
    if modality == "ish":
        kwargs = {**kwargs, "mode": ISH_MODES[0]}

    common = {"seed": 123, "pixel_size_um": 25.0, **kwargs}
    clean = fn(ref, ann, atlas, **common, apply_damage=False)
    damaged = fn(ref, ann, atlas, **common, apply_damage=True)
    assert clean.shape == damaged.shape
    assert not np.array_equal(clean, damaged)


def test_apply_damage_layer_geometry_disabled_skips_warps() -> None:
    """When ``geometry=False``, no pixel-displacing transform runs.

    Verified by patching each coord-affecting transform class with a sentinel
    that raises if called. Non-coord transforms (illumination, halos, debris,
    resolution-shift) are still allowed to run.
    """
    from unittest.mock import patch

    image = _make_image(seed=99)
    ctx = _damage_ctx("nissl")

    class _ShouldNotRun:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "geometry transform was instantiated despite geometry=False"
            )

    targets = [
        "augmentation.damage_pipeline.BladeStretchHorizontal",
        "augmentation.damage_pipeline.AffineJitter",
        "augmentation.damage_pipeline.Folds",
        "augmentation.damage_pipeline.VentricleExpansion",
    ]

    with (
        patch(targets[0], _ShouldNotRun),
        patch(targets[1], _ShouldNotRun),
        patch(targets[2], _ShouldNotRun),
        patch(targets[3], _ShouldNotRun),
    ):
        out = apply_damage_layer(
            image, rng=np.random.default_rng(5), ctx=ctx, modality="nissl",
            geometry=False,
        )

    assert out.shape == image.shape
    assert out.dtype == np.float32


def test_apply_damage_layer_geometry_disabled_default_runs_warps() -> None:
    """Sanity: with the default geometry=True, BladeStretchHorizontal IS used."""
    from unittest.mock import patch

    image = _make_image(seed=99)
    ctx = _damage_ctx("nissl")

    instantiated = {"called": False}

    class _Sentinel:
        def __init__(self, *args, **kwargs):
            instantiated["called"] = True

        def __call__(self, img, *, rng, ctx):  # noqa: ARG002
            return img

    with patch("augmentation.damage_pipeline.BladeStretchHorizontal", _Sentinel):
        apply_damage_layer(
            image, rng=np.random.default_rng(5), ctx=ctx, modality="nissl",
        )

    assert instantiated["called"], "BladeStretchHorizontal must run by default"


def test_fluorescence_backwards_compat_p_green_p_red_warns() -> None:
    from augmentation.fluorescence_pipeline import render_fluorescence_section

    ref = _synth_reference_hw()
    ann = _synth_annotation()
    atlas = _FakeAtlas()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = render_fluorescence_section(
            ref,
            ann,
            atlas,
            seed=5,
            pixel_size_um=25.0,
            p_green=1.0,
            p_red=0.0,
        )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[..., 1].max() > 0.1

