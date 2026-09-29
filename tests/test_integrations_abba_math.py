"""Tests for the pure-numpy core of the ABBA adapter (no JVM required)."""

import numpy as np
import pytest
from PIL import Image

from langslice.integrations.abba import (
    classify_to_ids,
    landmarks_from_field,
    normalize_to_rgb,
    render_atlas_at_coords,
)


class FakeAtlas:
    """Tiny synthetic BrainGlobe-shaped atlas: 10x10x10 voxels at 100um."""

    resolution = (100, 100, 100)  # um

    def __init__(self):
        self.annotation = np.zeros((10, 10, 10), dtype=np.int32)
        self.annotation[:, 2:8, 2:8] = 7
        self.annotation[:, 4:6, 4:6] = 9
        self.reference = np.zeros((10, 10, 10), dtype=np.uint16)
        self.reference[:, 2:8, 2:8] = 1000
        self.structures = {}


def test_render_atlas_at_coords_samples_and_masks_oob():
    atlas = FakeAtlas()
    # 2x2 grid: one pixel inside region 7, one inside 9, one in background,
    # one out of volume.
    coords = np.array(
        [
            [[0.5, 0.25, 0.25], [0.5, 0.45, 0.45]],
            [[0.5, 0.05, 0.05], [0.5, -1.0, 0.25]],
        ]
    )
    colored, reference, ids = render_atlas_at_coords(atlas, coords)
    assert ids[0, 0] == 7
    assert ids[0, 1] == 9
    assert ids[1, 0] == 0
    assert ids[1, 1] == 0  # out of volume
    assert reference[0, 0] == 255  # normalized peak
    assert colored.shape == (2, 2, 3)
    assert tuple(colored[1, 1]) == (0, 0, 0)


def test_classify_to_ids_nearest_color_and_background():
    lut = {7: (200, 40, 40), 9: (40, 200, 40)}
    ids_present = np.array([[0, 7], [9, 7]])
    rgb = np.array(
        [
            [[190, 50, 50], [45, 190, 45]],
            [[5, 5, 5], [210, 30, 30]],
        ],
        dtype=np.uint8,
    )
    out = classify_to_ids(rgb, ids_present, lut)
    assert out[0, 0] == 7
    assert out[0, 1] == 9
    assert out[1, 0] == 0  # near-black -> background
    assert out[1, 1] == 7


def test_landmarks_from_field_direction_and_mask():
    h = w = 50
    field = np.zeros((h, w, 2))
    field[..., 0] = 3.0  # dx
    field[..., 1] = -2.0  # dy
    mask = np.zeros((h, w), dtype=bool)
    mask[10:40, 10:40] = True
    src, tgt = landmarks_from_field(field, mask, grid=8)
    assert len(src) >= 4
    np.testing.assert_allclose(tgt - src, np.tile([3.0, -2.0], (len(src), 1)))
    # all landmarks inside the mask
    assert all(mask[int(y), int(x)] for x, y in src)


def test_landmarks_from_field_raises_when_empty():
    field = np.zeros((20, 20, 2))
    with pytest.raises(RuntimeError):
        landmarks_from_field(field, np.zeros((20, 20), dtype=bool), grid=5)


def test_normalize_to_rgb_range():
    img = np.random.default_rng(0).normal(100.0, 20.0, size=(30, 30))
    out = np.asarray(normalize_to_rgb(img))
    assert out.shape == (30, 30, 3)
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0


@pytest.mark.parametrize("varying", [False, True])
def test_border_refinement_preserves_host_grid_and_inverts_landmark_pairs(
    monkeypatch, tmp_path, varying
):
    from types import SimpleNamespace

    from langslice.atlas import core
    from langslice.integrations.abba import (
        LangSliceAbbaConfig,
        compute_registration_landmarks,
    )
    from langslice.nonlinear import border_refinement

    atlas = FakeAtlas()
    yy, xx = np.indices((40, 50))
    coords = np.stack((np.full_like(xx, 0.5, dtype=float), yy / 50, xx / 60), axis=-1)
    _, _, expected_ids = render_atlas_at_coords(atlas, coords)
    histology = (xx * 2 + yy).astype(float)
    field = np.zeros((40, 50, 2))
    field[..., 0] = 3 + (0.1 * xx if varying else 0)
    field[..., 1] = -2 + (0.05 * yy if varying else 0)
    fitted_labels = np.zeros_like(expected_ids)
    fitted_labels[8:32, 10:40] = 7
    seen = []

    def refine(image, rough_labels, passed_atlas, **kwargs):
        seen.append(kwargs)
        assert passed_atlas is atlas
        assert image.size == (50, 40)
        np.testing.assert_array_equal(rough_labels, expected_ids)
        np.testing.assert_array_equal(np.asarray(image), np.asarray(normalize_to_rgb(histology)))
        return SimpleNamespace(
            raw_model_image=Image.new("RGB", (100, 80), "yellow"),
            rough_border_overlay=image,
            model_border_overlay=Image.new("RGB", image.size, "red"),
            fitted_border_overlay=Image.new("RGB", image.size, "blue"),
            fitted_labels=fitted_labels,
            deformation_field=field,
        )

    monkeypatch.setattr(border_refinement, "refine_borders", refine)
    monkeypatch.setattr(core, "load_atlas", lambda _: atlas)
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", str(tmp_path))
    config = LangSliceAbbaConfig(provider="openai-oauth", landmark_grid=8)
    src, tgt = compute_registration_landmarks(coords, histology, config)
    expected_tissue, expected_atlas = landmarks_from_field(field, fitted_labels != 0, 8)
    np.testing.assert_array_equal(src, expected_atlas)
    np.testing.assert_array_equal(tgt, expected_tissue)
    if not varying:
        np.testing.assert_allclose(tgt - src, np.tile([-3, 2], (len(src), 1)))
    assert seen[0]["provider"] == "openai-oauth"
    debug = tmp_path / "abba/attempt_1"
    assert Image.open(debug / "raw_border_correction.png").size == (100, 80)
    assert Image.open(debug / "model_border_overlay.png").getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(debug / "fitted_border_overlay.png").getpixel((0, 0)) == (0, 0, 255)


def test_border_refinement_rejects_multiple_draws_before_loading_atlas(monkeypatch):
    from langslice.atlas import core
    from langslice.integrations.abba import (
        LangSliceAbbaConfig,
        compute_registration_landmarks,
    )
    from langslice.nonlinear import border_refinement

    monkeypatch.setattr(
        border_refinement,
        "refine_borders",
        lambda *args, **kwargs: pytest.fail("Must not call model"),
    )
    monkeypatch.setattr(core, "load_atlas", lambda _: pytest.fail("Must not load atlas"))
    with pytest.raises(ValueError, match="exactly one draw"):
        compute_registration_landmarks(
            np.zeros((2, 2, 3)), np.zeros((2, 2)), LangSliceAbbaConfig(draws=2)
        )


# --- ABBA slicing rotations <-> LangSlice pitch/yaw ---------------------------

#: Stage 0 (atlas (ML, DV, AP) mm -> ABBA world) of ABBA's own QuPath
#: ``ABBA-Transform-allen_mouse_10um_java.json`` exports (LSD_910, the lab's
#: saves), with the save's rotationX/rotationY in radians. Only the 3x3 block;
#: world z is the slicing axis, so its row is the cutting plane's normal.
#: ABBA's ASR atlas names x > 5.7 "Left", BrainGlobe asr names the high ML index
#: left, and the exported Left ROIs sit on the image's right for every
#: unflipped section (M12: 38/38 with flips), so x IS BrainGlobe ML here.
_ABBA_EXPORTS = {
    "M11_B_01": (
        0.22689280275926282,
        0.03490658503988659,
        [
            [1.0006095442988217, -0.008062094885272547, 0.0],
            [0.0, 1.0263041077933914, 0.0],
            [0.03492076949174772, -0.23100891551524302, 0.9999999999999999],
        ],
    ),
    "M12_C_01": (
        -0.148352986419518,
        0.05235987755982988,
        [
            [1.0013723459979211, 0.00783239509233458, 0.0],
            [0.0, 1.0111061278640618, 0.0],
            [0.052407779283041175, 0.14965609983271452, 1.0],
        ],
    ),
}


def _langslice_normal_ml_dv_ap(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """LangSlice's coronal cutting-plane normal on an asr atlas, as (ML, DV, AP)."""
    from langslice.oblique import build_rotation_matrix

    # asr: axis 0 = AP (the coronal normal), 1 = DV (rows), 2 = ML (columns)
    n = build_rotation_matrix(pitch_deg, yaw_deg, row_axis=1, col_axis=2) @ np.array(
        [1.0, 0.0, 0.0]
    )
    return np.array([n[2], n[1], n[0]])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = abs(float(a @ b)) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, cos))))


@pytest.mark.parametrize("section", sorted(_ABBA_EXPORTS))
def test_rotate_sign_constants_reproduce_abba_exported_plane(section):
    import math

    from langslice.integrations.abba_linear import (
        PITCH_TO_ROTATE_X_SIGN,
        YAW_TO_ROTATE_Y_SIGN,
    )

    rx, ry, stage0 = _ABBA_EXPORTS[section]
    abba_normal = np.asarray(stage0)[2]
    # the read-back direction abba_linear uses (angle = deg(rotate) / sign)
    pitch = math.degrees(rx) / PITCH_TO_ROTATE_X_SIGN
    yaw = math.degrees(ry) / YAW_TO_ROTATE_Y_SIGN
    assert _angle_deg(_langslice_normal_ml_dv_ap(pitch, yaw), abba_normal) < 1e-6
    # the opposite yaw sign is measurably wrong (2-3 deg of yaw -> 4-6 deg off)
    assert _angle_deg(_langslice_normal_ml_dv_ap(pitch, -yaw), abba_normal) > 3.0
    # and the set direction is the exact inverse of the read-back
    assert math.radians(pitch * PITCH_TO_ROTATE_X_SIGN) == pytest.approx(rx)
    assert math.radians(yaw * YAW_TO_ROTATE_Y_SIGN) == pytest.approx(ry)
