"""Oblique-plane sampling and (pitch, yaw) fitting on a synthetic phantom."""

from __future__ import annotations

import numpy as np
import pytest

from langslice.core.oblique import (
    build_rotation_matrix,
    compute_similarity_metric,
    fit_oblique,
    plane_axes,
    sample_oblique_plane,
    score_oblique,
)

RES_UM = 100.0
SHAPE = (120, 96, 96)  # (AP, DV, ML) for an "asr" atlas


def _phantom(asymmetric: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A brain-ish volume with AP texture, a flat ventral edge, and one lateral rod.

    * ellipsoid, elongated ML vs DV, so in-plane moments have distinct axes
    * ventral quarter truncated, so the 180-degree pose candidate loses on IoU
    * intensity modulated along AP, so tilting the plane changes what it sees
    * a bright rod on one side only (``asymmetric``), so the volume is not
      left-right symmetric even when the plane is flat
    * a soft boundary, so an Otsu silhouette of a sampled plane lands on the
      same surface as the annotation mask. With a hard edge, linear
      interpolation puts Otsu half a voxel inside the annotation and the pose
      fit sees a systematically undersized section.
    """
    n_ap, n_dv, n_ml = SHAPE
    ap = np.arange(n_ap)[:, None, None]
    dv = np.arange(n_dv)[None, :, None]
    ml = np.arange(n_ml)[None, None, :]
    # (n - 1) / 2, not n / 2: the sampler's mirror axis is the array centre,
    # and a half-voxel offset here would make the phantom subtly asymmetric.
    c_ap, c_dv, c_ml = (n_ap - 1) / 2.0, (n_dv - 1) / 2.0, (n_ml - 1) / 2.0

    radial = np.sqrt(
        ((ap - c_ap) / 48.0) ** 2 + ((dv - c_dv) / 28.0) ** 2 + ((ml - c_ml) / 40.0) ** 2
    )
    inside = (radial < 1.0) & (dv < 0.78 * n_dv)

    # Long AP periods on purpose. Texture that repeats faster than the AP swing
    # a 15-degree tilt produces aliases into a needle-sharp score peak that no
    # coarse grid could find — an artefact of a synthetic section being a
    # pixel-exact copy of the atlas, not something real histology does.
    texture = 60.0 * np.sin(2.0 * np.pi * ap / 46.0) + 35.0 * np.cos(2.0 * np.pi * ap / 62.0)
    body = 130.0 + texture * (radial < 0.55)

    # A pair of dark "ventricles" whose height drifts with AP. Without a feature
    # that MOVES along AP, a global intensity modulation is all there is, and
    # neither the position nor the tilt is well localised.
    drift = (c_dv - 8.0) + 20.0 * (ap - c_ap) / n_ap
    ventricles = ((dv - drift) ** 2 + (np.abs(ml - c_ml) - 15.0) ** 2) < 36.0
    body = np.where(ventricles, 25.0, body)

    if asymmetric:
        rod = (np.abs(ml - (c_ml + 24)) < 5) & (np.abs(dv - (c_dv - 8)) < 5)
        body = body + 60.0 * rod

    edge = 1.0 / (1.0 + np.exp((radial - 1.0) / 0.012))
    edge = edge * (dv < 0.78 * n_dv)
    volume = np.clip(body * edge, 0.0, 255.0).astype(np.float32)
    return volume, inside


class _Phantom:
    atlas_name = "phantom"
    orientation = "asr"
    resolution = (RES_UM, RES_UM, RES_UM)
    metadata: dict[str, object] = {}

    def __init__(self, asymmetric: bool = True) -> None:
        template, inside = _phantom(asymmetric)
        self.template = template
        self.annotation = inside.astype(np.int32)


@pytest.fixture(scope="module")
def atlas() -> _Phantom:
    return _Phantom()


def test_plane_axes_coronal_asr(atlas: _Phantom) -> None:
    assert plane_axes(atlas, "coronal") == (0, 1, 2)


def test_build_rotation_matrix_identity() -> None:
    matrix = build_rotation_matrix(0.0, 0.0, row_axis=1, col_axis=2)
    assert np.allclose(matrix, np.eye(3))


def test_flat_plane_matches_the_flat_slice(atlas: _Phantom) -> None:
    """Zero angles must reproduce the plain np.take slice, voxel for voxel."""
    index = 55
    position_mm = index * RES_UM / 1000.0
    flat = atlas.template[index]
    sampled = sample_oblique_plane(atlas, position_mm, "coronal", 0.0, 0.0)

    assert sampled.shape == flat.shape
    assert np.allclose(sampled, flat, atol=1e-4)


def test_tilted_plane_differs_from_the_flat_slice(atlas: _Phantom) -> None:
    position_mm = 55 * RES_UM / 1000.0
    flat = sample_oblique_plane(atlas, position_mm, "coronal", 0.0, 0.0)
    tilted = sample_oblique_plane(atlas, position_mm, "coronal", 10.0, 0.0)

    assert not np.allclose(flat, tilted, atol=1.0)


def test_pitch_is_lr_symmetric_but_yaw_is_not() -> None:
    """The whole hypothesis in one assertion.

    On a left-right symmetric volume a pitched plane stays symmetric — so it
    can carry no hemisphere-flip signal — while a yawed plane does not.
    """
    symmetric = _Phantom(asymmetric=False)
    position_mm = 55 * RES_UM / 1000.0

    pitched = sample_oblique_plane(symmetric, position_mm, "coronal", 10.0, 0.0)
    yawed = sample_oblique_plane(symmetric, position_mm, "coronal", 0.0, 10.0)

    assert np.allclose(pitched, pitched[:, ::-1], atol=1.0)
    assert not np.allclose(yawed, yawed[:, ::-1], atol=1.0)


def test_mirrored_yawed_plane_scores_worse_unmirrored(atlas: _Phantom) -> None:
    """A mirrored section beats its as-is self once the plane is yawed."""
    position_mm = 55 * RES_UM / 1000.0
    truth_yaw = 10.0
    section = sample_oblique_plane(atlas, position_mm, "coronal", 4.0, truth_yaw)
    mirrored_section = section[:, ::-1]

    kwargs = dict(pitch_deg=4.0, yaw_deg=truth_yaw, downsample=1, metric="ncc")
    as_is = score_oblique(atlas, mirrored_section, position_mm, "coronal", **kwargs)
    remirrored = score_oblique(
        atlas, mirrored_section, position_mm, "coronal", mirror=True, **kwargs
    )
    assert remirrored > as_is


# NCC, not MI: on a 96x96 phantom plane the masked region is only ~3.4k pixels,
# and a joint-histogram MI over that few samples is noise-dominated. On the real
# 25 um atlas (~19k pixels) the two agree.
@pytest.mark.parametrize(("truth_pitch", "truth_yaw"), [(8.0, 0.0), (-6.0, 5.0)])
def test_known_rotation_is_recovered(
    atlas: _Phantom, truth_pitch: float, truth_yaw: float
) -> None:
    position_mm = 55 * RES_UM / 1000.0
    section = sample_oblique_plane(atlas, position_mm, "coronal", truth_pitch, truth_yaw)

    fit = fit_oblique(
        atlas,
        section,
        position_mm,
        "coronal",
        pitch_bounds=(-15.0, 15.0),
        yaw_bounds=(-15.0, 15.0),
        allow_mirror=False,
        downsample=1,
        metric="ncc",
    )

    assert fit["pitch_deg"] == pytest.approx(truth_pitch, abs=2.5)
    assert fit["yaw_deg"] == pytest.approx(truth_yaw, abs=2.5)
    assert fit["mirrored"] is False


def test_mirror_hypothesis_wins_for_a_mirrored_section(atlas: _Phantom) -> None:
    """With yaw constrained to the brain's true sign, the mirror branch wins.

    Bounds matter: given free yaw over a symmetric range, the as-is branch can
    always reach -yaw and tie the mirror branch, so the margin collapses.
    """
    position_mm = 55 * RES_UM / 1000.0
    section = sample_oblique_plane(atlas, position_mm, "coronal", 4.0, 9.0)
    mirrored_section = np.ascontiguousarray(section[:, ::-1])

    fit = fit_oblique(
        atlas,
        mirrored_section,
        position_mm,
        "coronal",
        pitch_bounds=(0.0, 8.0),
        yaw_bounds=(6.0, 12.0),
        allow_mirror=True,
        downsample=1,
        metric="ncc",
    )

    assert fit["mirrored"] is True
    assert fit["mirror_margin"] is not None and fit["mirror_margin"] > 0.0
    assert fit["yaw_deg"] == pytest.approx(9.0, abs=2.5)


def test_position_window_recovers_a_shifted_position(atlas: _Phantom) -> None:
    truth_mm = 62 * RES_UM / 1000.0
    section = sample_oblique_plane(atlas, truth_mm, "coronal", 0.0, 0.0)

    fit = fit_oblique(
        atlas,
        section,
        truth_mm + 0.25,
        "coronal",
        pitch_bounds=(-6.0, 6.0),
        yaw_bounds=(-6.0, 6.0),
        position_window_mm=0.5,
        allow_mirror=False,
        downsample=1,
        metric="ncc",
    )

    assert abs(fit["position_mm"] - truth_mm) < 0.15


def test_metric_dispatch_and_self_similarity(atlas: _Phantom) -> None:
    plane = sample_oblique_plane(atlas, 2.6, "coronal", 0.0, 0.0)
    other = sample_oblique_plane(atlas, 3.4, "coronal", 0.0, 0.0)

    for metric in ("mi", "ncc", "ssim", "combined"):
        same = compute_similarity_metric(plane, plane, metric)  # type: ignore[arg-type]
        different = compute_similarity_metric(plane, other, metric)  # type: ignore[arg-type]
        assert same > different, metric

    with pytest.raises(ValueError):
        compute_similarity_metric(plane, plane, "nope")  # type: ignore[arg-type]


def test_annotation_sampling_keeps_large_ids_exact() -> None:
    """Allen ids exceed float32's exact integer range; the sampler must not round."""
    from langslice.core.oblique import sample_oblique_annotation

    atlas = _Phantom()
    # 484682516 (ccb) is the id that exposed this: float32 lands it on ...528.
    atlas.annotation = np.where(atlas.annotation != 0, 484682516, 0).astype(np.int64)
    flat = sample_oblique_annotation(atlas, 55 * RES_UM / 1000.0, "coronal", 0.0, 0.0)
    tilted = sample_oblique_annotation(atlas, 55 * RES_UM / 1000.0, "coronal", 6.0, -3.0)

    assert set(np.unique(flat)) <= {0, 484682516}
    assert set(np.unique(tilted)) <= {0, 484682516}
    assert np.array_equal(flat, atlas.annotation[55])
    assert (tilted != 0).any()
