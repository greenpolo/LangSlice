"""Initial placement and residual deformation compose in explicit pixel frames."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.nonlinear import border_registration as registration
from langslice.nonlinear.border_refinement import BorderRefinementResult


def test_zero_residual_retains_initial_affine_with_rounding_and_padding():
    original_to_canvas = registration.pixel_center_map((101, 57), (37, 21), (7, 9))
    initial = np.array([[1.3, 0.2, 9.0], [-0.1, 0.8, 12.0], [0, 0, 1]])
    canonical = registration.canonical_atlas_map((31, 19), (51, 39))
    field = np.zeros((39, 51, 2))
    markers, native = registration.composed_correspondences(
        field, original_to_canvas @ initial, original_to_canvas, canonical
    )
    dense = registration.composed_native_map(field, original_to_canvas @ initial)
    expected_origin = np.linalg.inv(original_to_canvas @ initial) @ [0, 0, 1]
    np.testing.assert_allclose(dense[0, 0], expected_origin[:2])
    assert markers
    for marker, pair in zip(markers, native, strict=True):
        source = np.array([*marker[:2], 1])
        expected_native = np.linalg.inv(initial) @ source
        expected_target = np.linalg.inv(original_to_canvas) @ canonical @ expected_native
        np.testing.assert_allclose(pair[2:], expected_native[:2], atol=1e-10)
        np.testing.assert_allclose(marker[2:], expected_target[:2], atol=1e-10)
    assert not np.allclose(np.array(markers)[:, :2], np.array(markers)[:, 2:])


def test_dense_initial_map_composes_residual_and_documents_edge_clamping():
    yy, xx = np.mgrid[:41, :45]
    initial = np.stack([xx * 2.0 + 3, yy * 3.0 - 4], axis=-1)
    field = np.zeros_like(initial)
    field[..., 0] = 1.5
    field[..., 1] = -2.0
    markers, pairs = registration.composed_correspondences(
        field, None, np.eye(3), np.eye(3), initial_coordinate_map=initial
    )
    dense = registration.composed_native_map(field, None, initial)
    for marker, pair in zip(markers, pairs, strict=True):
        x, y = marker[:2]
        expected = [min(x + 1.5, 44) * 2 + 3, max(y - 2, 0) * 3 - 4]
        np.testing.assert_allclose(pair[2:], expected)
        np.testing.assert_allclose(marker[2:], expected)
        np.testing.assert_allclose(dense[int(y), int(x)], expected)


@pytest.fixture
def case(monkeypatch):
    import langslice.nonlinear.image_gen_registration as legacy

    labels = np.zeros((20, 30), dtype=np.int64)
    labels[2:18, 3:14] = 100000001
    labels[3:17, 14:27] = 100000003
    image = Image.new("RGB", (30, 20), (70, 80, 90))
    monkeypatch.setattr(registration, "load_atlas", lambda name: SimpleNamespace())
    monkeypatch.setattr(registration, "annotation_slice", lambda *a, **k: labels.copy())
    monkeypatch.setattr(legacy, "prepare_canvas", lambda image, **k: (image, image.size, 0, 0, 0))
    monkeypatch.setattr(registration, "_classified_to_rgb",
                        lambda ids, atlas: np.zeros((*ids.shape, 3), dtype=np.uint8))
    received = []

    def refine(image, rough, atlas, **kwargs):
        received.append((rough.copy(), kwargs))
        return BorderRefinementResult(
            image, image, rough > 0, image, rough.copy(), image, None,
            np.zeros((*rough.shape, 2)), 0.0, {"prompt": "test"},
        )

    monkeypatch.setattr(registration, "refine_borders", refine)
    return image, labels, received


def test_supplied_placement_precedes_automatic_and_preserves_ids(case, monkeypatch, tmp_path):
    image, labels, received = case

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit placement must not run automatic placement")

    monkeypatch.setattr(registration, "place_plane_on_tissue_with_matrix", forbidden)
    import langslice.nonlinear.image_gen_registration as legacy
    monkeypatch.setattr(legacy, "generate_registration_candidate", forbidden)
    result = registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, initial_atlas_to_slice=np.eye(3),
        initial_alignment_source="linear-agent", debug_dir=str(tmp_path), candidate_id="one",
    )
    np.testing.assert_array_equal(received[0][0], labels)
    assert result.metadata["initial_alignment_source"] == "linear-agent"
    assert result.metadata["output_kind"] == "border_overlay"
    assert result.metadata["inverse_warp_status"] == "not_computed"
    assert result.markers
    saved = np.load(tmp_path / "registration" / "one" / "warped_leaf_ids.npz")["ids"]
    np.testing.assert_array_equal(saved, labels)
    assert set(np.unique(saved)) == {0, 100000001, 100000003}


def test_mirror_is_explicit_and_applied_before_supplied_matrix(case):
    image, labels, received = case
    registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, initial_atlas_to_slice=np.eye(3),
        atlas_mirror_lr=True,
    )
    np.testing.assert_array_equal(received[0][0], np.fliplr(labels))


def test_standalone_starts_with_color_registration(case, monkeypatch):
    image, labels, received = case
    import langslice.nonlinear.image_gen_registration as legacy
    calls = []
    yy, xx = np.mgrid[:20, :30]
    first = SimpleNamespace(
        candidate_id="initial", warped_labels=labels,
        atlas_coordinate_map=np.stack([xx, yy], axis=-1).astype(float),
        metadata={"target_size": [30, 20], "unpadded_size": [30, 20],
                  "native_atlas_size": [30, 20],
                  "canvas_origin_px": [0, 0], "elastix": {"codes": [{"code": "WARP_FOLDS"}]}},
    )

    def initial(*args, **kwargs):
        calls.append(kwargs)
        return first

    monkeypatch.setattr(legacy, "generate_registration_candidate", initial)
    result = registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, provider="openai-oauth"
    )
    assert len(calls) == 1
    assert calls[0]["registration_mode"] == "colormap"
    np.testing.assert_array_equal(received[0][0], labels)
    assert result.metadata["prior"]["source"] == "colormap_registration"
    assert result.metadata["initial_stage_elastix"] == {"codes": [{"code": "WARP_FOLDS"}]}
    assert result.metadata["elastix"]["codes"] == []


def test_replay_does_not_silently_generate_initial_alignment(case):
    image, _, _ = case
    with pytest.raises(ValueError, match="replay requires"):
        registration.generate_border_registration_candidate(
            image, atlas_name="test", position_mm=1, generated_image=image
        )


def test_nonfinite_fields_and_singular_affines_fail():
    field = np.zeros((4, 5, 2))
    field[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        registration.composed_correspondences(field, np.eye(3), np.eye(3), np.eye(3))
    with pytest.raises(ValueError, match="invertible"):
        registration.composed_correspondences(np.zeros((4, 5, 2)), np.zeros((3, 3)),
                                             np.eye(3), np.eye(3))


def test_residual_report_flags_folds_and_missing_regions():
    rough = np.ones((10, 12), dtype=np.int64)
    rough[:, 6:] = 2
    fitted = rough.copy()
    fitted[:, 6:] = 1
    yy, xx = np.indices(rough.shape)
    field = np.stack([-2.0 * xx, np.zeros_like(yy)], axis=-1)
    report = registration.residual_fit_report(rough, fitted, field)
    codes = {row["code"] for row in report["codes"]}
    assert {"WARP_FOLDS", "REGION_MISSING"} <= codes
    assert "residual" in report["scope"]
    assert report["max_residual_displacement_px"] == 22.0


def test_residual_report_flags_empty_and_globally_collapsed_fit():
    rough = np.ones((10, 12), dtype=np.int64)
    fitted = np.zeros_like(rough)
    field = np.zeros((*rough.shape, 2))
    report = registration.residual_fit_report(rough, fitted, field)
    assert "EMPTY_WARP" in {row["code"] for row in report["codes"]}
    fitted[2, 3] = 1
    report = registration.residual_fit_report(rough, fitted, field)
    assert "ATLAS_COLLAPSED" in {row["code"] for row in report["codes"]}


@pytest.mark.parametrize("invalid", [np.full((10, 12), 99, dtype=np.int64),
                                     np.ones((10, 11), dtype=np.int64),
                                     np.ones((10, 12), dtype=np.float64)])
def test_residual_report_rejects_corrupted_labels(invalid):
    with pytest.raises(ValueError):
        registration.residual_fit_report(np.ones((10, 12), dtype=np.int64), invalid,
                                         np.zeros((10, 12, 2)))
