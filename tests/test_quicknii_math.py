"""Script-style checks for matrix-first QUINT export math."""

import numpy as np

from langslice.export import (
    CORONAL_FRAME_PADDING_FACTOR,
    compute_anchoring,
    compute_coronal_frame_geometry,
)
from langslice.registration import AffineResult, affine_matrix_from_legacy_params

ATLAS_SHAPE = (528, 320, 456)
ATLAS_RESOLUTION = (25.0, 25.0, 25.0)
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 670


def _anchoring_array(anchoring: object) -> np.ndarray:
    return np.asarray(getattr(anchoring, "to_list")(), dtype=np.float64)


identity_anchoring = compute_anchoring(
    position_mm=1.0,
    atlas_shape=ATLAS_SHAPE,
    atlas_resolution=ATLAS_RESOLUTION,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
)
identity_matrix_anchoring = compute_anchoring(
    position_mm=1.0,
    atlas_shape=ATLAS_SHAPE,
    atlas_resolution=ATLAS_RESOLUTION,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    affine_matrix=np.eye(3, dtype=np.float64),
)
assert np.allclose(
    _anchoring_array(identity_anchoring),
    _anchoring_array(identity_matrix_anchoring),
)

identity_geometry = compute_coronal_frame_geometry(
    atlas_shape=ATLAS_SHAPE,
    frame_width=IMAGE_WIDTH,
    frame_height=IMAGE_HEIGHT,
)
assert abs(identity_geometry.atlas_width_px - (IMAGE_WIDTH / CORONAL_FRAME_PADDING_FACTOR)) < 1e-9
assert abs(identity_geometry.atlas_height_px - (IMAGE_HEIGHT / CORONAL_FRAME_PADDING_FACTOR)) < 1e-9

identity_origin = np.array(
    [identity_anchoring.ox, identity_anchoring.oy, identity_anchoring.oz],
    dtype=np.float64,
)
identity_top_right = identity_geometry.frame_to_atlas(
    px=float(IMAGE_WIDTH),
    py=0.0,
    ap_voxel=identity_anchoring.oy,
)
identity_bottom_left = identity_geometry.frame_to_atlas(
    px=0.0,
    py=float(IMAGE_HEIGHT),
    ap_voxel=identity_anchoring.oy,
)
assert np.allclose(
    identity_geometry.frame_to_atlas(px=0.0, py=0.0, ap_voxel=identity_anchoring.oy),
    identity_origin,
    atol=1e-6,
)
assert np.allclose(
    identity_top_right - identity_origin,
    np.array([identity_anchoring.ux, identity_anchoring.uy, identity_anchoring.uz], dtype=np.float64),
    atol=1e-6,
)
assert np.allclose(
    identity_bottom_left - identity_origin,
    np.array([identity_anchoring.vx, identity_anchoring.vy, identity_anchoring.vz], dtype=np.float64),
    atol=1e-6,
)

legacy_matrix = affine_matrix_from_legacy_params(
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    rotation_deg=5.0,
    translate_x_pct=1.0,
    translate_y_pct=-0.5,
)
legacy_param_anchoring = compute_anchoring(
    position_mm=1.0,
    atlas_shape=ATLAS_SHAPE,
    atlas_resolution=ATLAS_RESOLUTION,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    rotation_deg=5.0,
    translate_x_pct=1.0,
    translate_y_pct=-0.5,
)
matrix_anchoring = compute_anchoring(
    position_mm=1.0,
    atlas_shape=ATLAS_SHAPE,
    atlas_resolution=ATLAS_RESOLUTION,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    affine_matrix=legacy_matrix,
)
assert np.allclose(
    _anchoring_array(legacy_param_anchoring),
    _anchoring_array(matrix_anchoring),
)

similarity_matrix = np.array(
    [[1.08, -0.12, 14.0], [0.12, 1.08, -9.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
similarity_result = AffineResult(
    matrix=similarity_matrix,
    source_size=(1000, 800),
    output_size=(900, 700),
    backend="test",
    reasoning="synthetic",
)
expected_scale = float(np.hypot(1.08, 0.12))
assert abs(similarity_result.rotation_deg - 6.3401917) < 1e-3
assert abs(similarity_result.scale[0] - expected_scale) < 1e-6
assert abs(similarity_result.scale[1] - expected_scale) < 1e-6

shear_matrix = np.array(
    [[1.0, 0.35, 8.0], [0.0, 0.85, -6.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
shear_anchoring = compute_anchoring(
    position_mm=1.0,
    atlas_shape=ATLAS_SHAPE,
    atlas_resolution=ATLAS_RESOLUTION,
    image_width=1000,
    image_height=800,
    affine_matrix=shear_matrix,
    output_width=900,
    output_height=700,
)
u_norm = float(np.linalg.norm([shear_anchoring.ux, shear_anchoring.uy, shear_anchoring.uz]))
v_norm = float(np.linalg.norm([shear_anchoring.vx, shear_anchoring.vy, shear_anchoring.vz]))
assert abs(shear_anchoring.vx) > 1e-6
assert abs(u_norm - v_norm) > 1.0

print("Matrix-first QUINT math OK")
