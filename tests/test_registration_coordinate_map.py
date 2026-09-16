"""Composition into native atlas pixels, independent of generated palette colors."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.nonlinear import image_gen_registration as registration


def test_fit_matrix_matches_rounded_resize_pixel_centers_and_padding():
    # 7x3 -> 10x4, then 2 pixels above/below on a 10x8 canvas.
    matrix = registration._canvas_from_native_matrix((7, 3), (10, 8))
    np.testing.assert_allclose(matrix, [[10 / 7, 0, 3 / 14], [0, 4 / 3, 2 + 1 / 6], [0, 0, 1]])
    point = np.array([2.0, 1.0, 1.0])
    center = matrix @ point
    np.testing.assert_allclose(center, [(2 + 0.5) * 10 / 7 - 0.5, 2 + (1 + 0.5) * 4 / 3 - 0.5, 1])


def test_native_pullback_composes_nontrivial_prealign_and_residual_in_order():
    field = np.zeros((8, 10, 2))
    yy, xx = np.indices((8, 10))
    field[..., 0] = 1.25 + 0.03 * yy
    field[..., 1] = -0.75 + 0.02 * xx
    prealign = np.array([[1.2, 0.15, 2.3], [-0.1, 0.8, -1.7]])
    result = registration._native_coordinate_map(field, (7, 3), prealign)
    placement = np.eye(3)
    placement[:2] = prealign
    canvas = registration._canvas_from_native_matrix((7, 3), (10, 8))
    for x, y in [(0, 0), (3, 5), (9, 7)]:
        recovered = placement @ canvas @ np.r_[result[y, x], 1]
        np.testing.assert_allclose(recovered[:2], [x, y] + field[y, x])


def test_integer_gather_preserves_large_ids_and_masks_outside_and_nonfinite():
    labels = np.array([[2**60 + 1, 2**60 + 3], [7, 9]], dtype=np.int64)
    coordinates = np.array(
        [[[0, 0], [1, 0], [-1, 0], [np.nan, 1]], [[0, 1], [1, 1], [2, 0], [np.inf, 1]]]
    )
    result = registration._gather_native_labels(labels, coordinates)
    np.testing.assert_array_equal(result, [[2**60 + 1, 2**60 + 3, 0, 0], [7, 9, 0, 0]])
    assert result.dtype == np.int64


def test_public_defaults_to_border_wrapper_and_forwards_supplied_alignment(monkeypatch):
    from langslice.nonlinear import border_registration

    calls = []
    sentinel = object()

    def wrapper(image, **kwargs):
        calls.append((image, kwargs))
        return sentinel

    monkeypatch.setattr(border_registration, "generate_border_registration_candidate", wrapper)
    matrix = np.array([[1.2, 0, 5], [0, 0.8, -3], [0, 0, 1]])
    image = Image.new("RGB", (10, 8))
    result = registration.generate_registration_candidate(
        image,
        atlas_name="fake",
        position_mm=4,
        initial_atlas_to_slice=matrix,
        initial_alignment_source="host",
        atlas_mirror_lr=True,
    )
    assert result is sentinel
    assert calls[0][0] is image
    assert calls[0][1]["initial_atlas_to_slice"] is matrix
    assert calls[0][1]["initial_alignment_source"] == "host"
    assert calls[0][1]["atlas_mirror_lr"] is True
    assert not {"draws", "elastix", "max_off_palette"} & calls[0][1].keys()


def test_explicit_colormap_dispatch_and_invalid_options(monkeypatch):
    seen = []
    monkeypatch.setattr(
        registration,
        "_generate_registration_candidate",
        lambda image, **kwargs: seen.append(kwargs),
    )
    image = Image.new("RGB", (10, 8))
    registration.generate_registration_candidate(
        image,
        atlas_name="fake",
        position_mm=4,
        registration_mode="colormap",
        atlas_mirror_lr=True,
    )
    assert seen[0]["atlas_mirror_lr"] is True
    with pytest.raises(ValueError, match="exactly one draw"):
        registration.generate_registration_candidate(
            image, atlas_name="fake", position_mm=4, draws=2
        )
    with pytest.raises(ValueError, match="supplied alignment"):
        registration.generate_registration_candidate(
            image,
            atlas_name="fake",
            position_mm=4,
            registration_mode="colormap",
            initial_atlas_to_slice=np.eye(3),
        )


def test_explicit_mirror_is_applied_after_orientation():
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    result = registration._orient_pil(
        Image.fromarray(original), SimpleNamespace(), "coronal", None, atlas_mirror_lr=True
    )
    np.testing.assert_array_equal(np.asarray(result), original[:, ::-1])
