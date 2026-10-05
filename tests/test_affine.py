"""The shared silhouette-affine core and its parameter conventions.

No model calls and no Elastix: the fit is closed-form OpenCV moments against a
tiny synthetic atlas built here.
"""


import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.core.affine import (
    affine_matrix,
    decompose_affine,
    extract_slice_silhouette,
    normalized_affine,
    silhouette_affine,
    silhouette_iou,
)
from tests.fakes import EllipseAtlas, ellipse_section

# --- affine_matrix -------------------------------------------------------


def test_translation_is_a_fraction_of_the_image():
    matrix = affine_matrix(
        rotation_deg=0.0,
        scale_x=1.0,
        scale_y=1.0,
        translate_x=0.25,
        translate_y=-0.5,
        size=(200, 100),
    )
    point = np.array([10.0, 20.0, 1.0])
    moved = matrix @ point
    assert moved[0] == pytest.approx(10.0 + 0.25 * 200)
    assert moved[1] == pytest.approx(20.0 - 0.5 * 100)


def test_rotation_turns_counter_clockwise_about_the_centre():
    matrix = affine_matrix(
        rotation_deg=90.0,
        scale_x=1.0,
        scale_y=1.0,
        translate_x=0.0,
        translate_y=0.0,
        size=(100, 100),
    )
    # A marker below the centre ends up to its right: 6 o'clock -> 3 o'clock,
    # which is counter-clockwise on a screen with y pointing down.
    below = np.array([50.0, 80.0, 1.0])
    turned = matrix @ below
    assert turned[0] == pytest.approx(80.0)
    assert turned[1] == pytest.approx(50.0)


def test_scale_is_a_per_axis_multiplier_about_the_centre():
    matrix = affine_matrix(
        rotation_deg=0.0,
        scale_x=2.0,
        scale_y=0.5,
        translate_x=0.0,
        translate_y=0.0,
        size=(100, 100),
    )
    corner = matrix @ np.array([100.0, 100.0, 1.0])
    assert corner[0] == pytest.approx(50.0 + 2.0 * 50.0)
    assert corner[1] == pytest.approx(50.0 + 0.5 * 50.0)


def test_the_matrix_moves_pixels_the_way_it_moves_points():
    """cv2.warpAffine reads our 2x3 as source -> destination, as documented."""
    image = np.zeros((80, 120), dtype=np.uint8)
    image[38:42, 18:22] = 255  # a marker left of centre
    matrix = affine_matrix(
        rotation_deg=0.0,
        scale_x=1.0,
        scale_y=1.0,
        translate_x=0.25,
        translate_y=0.25,
        size=(120, 80),
    )
    warped = cv2.warpAffine(image, matrix, (120, 80))

    ys, xs = np.nonzero(warped)
    assert xs.mean() == pytest.approx(20.0 + 0.25 * 120, abs=1.0)
    assert ys.mean() == pytest.approx(40.0 + 0.25 * 80, abs=1.0)


# --- normalized_affine ---------------------------------------------------


def test_normalized_affine_is_the_same_map_in_fractional_coordinates():
    size = (200, 100)  # deliberately not square: the x/y scaling differs
    matrix = affine_matrix(
        rotation_deg=17.0,
        scale_x=1.1,
        scale_y=0.9,
        translate_x=0.05,
        translate_y=-0.02,
        size=size,
    )
    a, b, tx, c, d, ty = normalized_affine(matrix, size)

    width, height = size
    x_px, y_px = 37.0, 61.0
    expected = matrix @ np.array([x_px, y_px, 1.0])
    x_n, y_n = x_px / width, y_px / height

    assert a * x_n + b * y_n + tx == pytest.approx(expected[0] / width)
    assert c * x_n + d * y_n + ty == pytest.approx(expected[1] / height)


def test_the_identity_normalizes_to_the_identity():
    matrix = affine_matrix(
        rotation_deg=0.0,
        scale_x=1.0,
        scale_y=1.0,
        translate_x=0.0,
        translate_y=0.0,
        size=(640, 480),
    )
    assert normalized_affine(matrix, (640, 480)) == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )


# --- silhouette_affine ---------------------------------------------------


def test_silhouette_affine_aligns_a_rotated_section_to_the_atlas():
    atlas = EllipseAtlas()
    section = ellipse_section(angle=25.0)

    fit = silhouette_affine(section, atlas=atlas, position_mm=5.0, long_edge=256)

    assert fit.matrix.shape == (2, 3)
    assert fit.size == (256, int(round(256 * 160 / 200)))
    # A closed-form moments fit of one ellipse onto another should be tight.
    assert fit.iou > 0.9

    # And the reported IoU is really the overlap after warping.
    gray = cv2.cvtColor(fit.slice_rgb, cv2.COLOR_RGB2GRAY)
    warped = cv2.warpAffine(
        extract_slice_silhouette(gray),
        fit.matrix,
        fit.size,
        flags=cv2.INTER_NEAREST,
    )
    assert silhouette_iou(warped, fit.atlas_mask) == pytest.approx(fit.iou, abs=1e-6)


def test_silhouette_affine_refuses_a_blank_field():
    atlas = EllipseAtlas()
    blank = Image.new("RGB", (200, 160), (255, 255, 255))

    with pytest.raises(ValueError, match="Otsu likely failed"):
        silhouette_affine(blank, atlas=atlas, position_mm=5.0, long_edge=256)


# --- reading a transform back --------------------------------------------


def test_decompose_affine_recovers_the_knobs_it_was_built_from():
    matrix = affine_matrix(
        rotation_deg=12.0,
        scale_x=1.25,
        scale_y=0.8,
        translate_x=0.1,
        translate_y=-0.05,
        size=(200, 200),
    )
    parts = decompose_affine(matrix)
    assert parts["rotation_deg"] == pytest.approx(12.0, abs=1e-3)
    assert parts["scale_x"] == pytest.approx(1.25, abs=1e-3)
    assert parts["scale_y"] == pytest.approx(0.8, abs=1e-3)
    assert parts["shear"] == pytest.approx(0.0, abs=1e-6)
    assert parts["mirrored"] is False

    # And it reads the normalized 6-vector too, translations as fractions.
    normalized = decompose_affine(normalized_affine(matrix, (200, 200)))
    assert normalized["rotation_deg"] == pytest.approx(12.0, abs=1e-3)
    assert normalized["translate_x_frac"] == pytest.approx(
        parts["translate_x_frac"] / 200.0, abs=1e-3
    )


def test_decompose_affine_flags_a_reflection():
    mirror = decompose_affine([-1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert mirror["mirrored"] is True
    assert mirror["scale_x"] == 1.0 and mirror["scale_y"] == -1.0

    identity = decompose_affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert identity["mirrored"] is False
    assert identity["rotation_deg"] == 0.0


def test_silhouette_affine_never_reflects():
    """The fit may rotate, scale and shear, never mirror: flips are orientation."""
    import numpy as np
    from PIL import Image, ImageDraw

    from langslice.core.affine import silhouette_affine

    class _Atlas:  # minimal: get_root_mask reads only what the fit asks for
        pass

    canvas = Image.new("RGB", (200, 160), 0)
    ImageDraw.Draw(canvas).ellipse((30, 20, 170, 140), fill=(200, 200, 200))
    # Off-centre notch so the silhouette has a handedness the fit could "fix".
    ImageDraw.Draw(canvas).rectangle((30, 20, 60, 50), fill=0)

    import langslice.core.affine as affine_mod

    def fake_root_mask(atlas, position_mm, size, plane="coronal", pitch_deg=0.0, yaw_deg=0.0):
        w, h = size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((int(w * 0.2), int(h * 0.15), int(w * 0.8), int(h * 0.85)), fill=255)
        draw.rectangle((int(w * 0.65), int(h * 0.15), int(w * 0.8), int(h * 0.3)), fill=0)
        return np.asarray(mask, dtype=np.uint8)

    original = affine_mod.get_root_mask
    affine_mod.get_root_mask = fake_root_mask
    try:
        fit = silhouette_affine(canvas, atlas=_Atlas(), position_mm=1.0)
    finally:
        affine_mod.get_root_mask = original
    assert np.linalg.det(fit.matrix[:, :2]) > 0


def test_decompose_affine_reads_a_pure_rotation_on_a_wide_image_when_given_its_size():
    """The normalized frame is anisotropic on a non-square image; with the
    size the decomposition is done in pixels and a 5-degree turn reads as 5."""
    size = (768, 552)
    matrix = affine_matrix(
        rotation_deg=5.0, scale_x=1.0, scale_y=1.0, translate_x=0.0, translate_y=0.0, size=size
    )
    normalized = normalized_affine(matrix, size)
    skewed = decompose_affine(normalized)
    proper = decompose_affine(normalized, size)
    assert abs(proper["rotation_deg"] - 5.0) < 0.01
    assert abs(proper["shear"]) < 1e-3
    assert abs(proper["scale_x"] - 1.0) < 1e-3 and abs(proper["scale_y"] - 1.0) < 1e-3
    assert abs(skewed["rotation_deg"] - 5.0) > 1.0  # the frame effect it corrects


def test_the_six_numbers_go_back_to_the_pixels_they_came_from():
    """A stored transform has to be DRAWN again, shear and all."""
    from langslice.core.affine import denormalized_affine

    size = (300, 200)
    matrix = np.array([[1.03, 0.21, 12.0], [-0.07, 0.94, -5.0]])
    back = denormalized_affine(normalized_affine(matrix, size), size)
    assert back == pytest.approx(matrix, abs=1e-9)


def test_silhouette_affine_measures_against_the_mask_it_is_given():
    """A stack cut at an angle passes its oblique mask; the fit must use it,
    not the flat root mask it would otherwise build."""
    import numpy as np
    from PIL import Image

    from langslice.core.affine import silhouette_affine

    seen: dict[str, tuple[int, int]] = {}

    def mask_at(size):
        seen["size"] = size
        w, h = size
        m = np.zeros((h, w), dtype=np.uint8)
        m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
        return m

    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    canvas[60:140, 40:160] = 200
    fit = silhouette_affine(
        Image.fromarray(canvas), atlas=None, position_mm=1.0, long_edge=200, atlas_mask_at=mask_at
    )
    assert seen["size"] == (200, 200)
    assert 0.0 < fit.iou <= 1.0
