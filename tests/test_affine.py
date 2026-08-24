"""The shared silhouette-affine core and its parameter conventions.

No model calls and no Elastix: the fit is closed-form OpenCV moments against a
tiny synthetic atlas built here.
"""

import inspect

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.affine import (
    affine_matrix,
    extract_slice_silhouette,
    normalized_affine,
    silhouette_affine,
    silhouette_iou,
)
from tests.fakes import EllipseAtlas, ellipse_section

# --- the move ------------------------------------------------------------


def test_the_affine_core_is_shared_and_nonlinears_api_is_unchanged():
    from langslice.nonlinear import quick_affine

    # quick_affine now wraps the shared core rather than owning it.
    assert quick_affine.silhouette_affine is silhouette_affine
    signature = inspect.signature(quick_affine.quick_affine_register)
    assert list(signature.parameters) == [
        "image",
        "atlas_name",
        "position_mm",
        "plane",
        "out_path",
    ]
    assert signature.parameters["plane"].default == "coronal"


def test_the_atlas_root_mask_kept_its_old_name_in_nonlinear():
    from langslice.atlas.core import get_root_mask
    from langslice.nonlinear.image_gen_helpers import _build_atlas_root_mask

    assert _build_atlas_root_mask is get_root_mask


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
