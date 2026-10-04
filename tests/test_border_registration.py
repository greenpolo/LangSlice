"""Initial placement and residual deformation compose in explicit pixel frames."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.core.affine import pixel_center_map
from langslice.core.nonlinear import border_registration as registration
from langslice.core.nonlinear.border_refinement import BorderRefinementResult


def test_zero_residual_retains_initial_affine_with_rounding_and_padding():
    original_to_canvas = pixel_center_map((101, 57), (37, 21), (7, 9))
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


class _Calls(list):
    """The correction calls a test saw; ``fitted``: the placements fitted."""

    def __init__(self) -> None:
        super().__init__()
        self.fitted: list = []


@pytest.fixture
def case(monkeypatch):
    import langslice.core.nonlinear.image_gen_registration as legacy

    labels = np.zeros((20, 30), dtype=np.int64)
    labels[2:18, 3:14] = 100000001
    labels[3:17, 14:27] = 100000003
    image = Image.new("RGB", (30, 20), (70, 80, 90))
    monkeypatch.setattr(registration, "load_atlas", lambda name: SimpleNamespace())
    monkeypatch.setattr(registration, "annotation_slice", lambda *a, **k: labels.copy())
    monkeypatch.setattr(legacy, "prepare_canvas", lambda image, **k: (image, image.size, 0, 0, 0))
    monkeypatch.setattr(registration, "_classified_to_rgb",
                        lambda ids, atlas: np.zeros((*ids.shape, 3), dtype=np.uint8))
    received = _Calls()

    def refine(image, rough, atlas, **kwargs):
        received.append((rough.copy(), kwargs))
        return BorderRefinementResult(image, image, rough > 0, image, {"prompt": "test"})

    def fit(canvas, lines, atlas, placement, **kwargs):
        """An identity fit: the deformable package has its own tests."""
        received.fitted.append(placement)
        rough = received[-1][0]
        return SimpleNamespace(
            field_px=np.zeros((*rough.shape, 2)), fitted_labels=rough.copy(),
            fitted_border_overlay=canvas, record=None, elapsed=0.5,
            metadata={"fit": "deformable", "flags": []},
        )

    monkeypatch.setattr(registration, "refine_borders", refine)
    monkeypatch.setattr(registration, "fit_border_lines", fit)
    monkeypatch.setattr(registration, "placement_on_canvas",
                        lambda atlas, matrix, **k: SimpleNamespace(atlas_to_section=matrix, **k))
    return image, labels, received


def test_supplied_placement_precedes_automatic_and_preserves_ids(case, monkeypatch, tmp_path):
    image, labels, received = case

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit placement must not run automatic placement")

    monkeypatch.setattr(registration, "place_plane_on_tissue_with_matrix", forbidden)
    import langslice.core.nonlinear.image_gen_registration as legacy
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


def _stub_silhouette_placement(monkeypatch, labels, iou=0.87):
    """Route "atlas" needs no real tissue image: stand in for both the
    tissue-outline step and the moments placement it feeds."""
    monkeypatch.setattr(
        registration, "tissue_mask", lambda *a, **k: np.ones(labels.shape, dtype=np.uint8) * 255
    )
    monkeypatch.setattr(
        registration, "place_plane_on_tissue_with_matrix",
        lambda labs, tissue: (labs.copy(), (1, 1), iou, np.eye(3)),
    )


def test_atlas_route_draws_boundaries_then_places_and_fits(case, monkeypatch):
    """No supplied placement, a real provider: route "atlas". One model call
    against the outlined atlas template; the silhouette placement is local
    (never the model), and the SAME refine_borders fit as route "supplied"
    runs on the model's output."""
    image, labels, received = case
    import langslice.core.nonlinear.image_gen_registration as legacy

    outlined = Image.new("RGB", (5, 5))
    monkeypatch.setattr(legacy, "outlined_atlas_template", lambda *a, **k: outlined)
    _stub_silhouette_placement(monkeypatch, labels)
    calls = []
    drawn = Image.new("RGB", image.size, (10, 20, 30))

    def fake_generate(request):
        calls.append(request)
        return SimpleNamespace(image=drawn, route="test")

    monkeypatch.setattr(registration, "generate_warped_segmentation_image", fake_generate)
    result = registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, provider="openai-oauth"
    )
    assert len(calls) == 1, "passes defaults to 1: exactly one model call"
    assert calls[0].reference_images == [outlined]
    np.testing.assert_array_equal(received[0][0], labels)
    assert received[0][1]["generated_image"] is drawn
    assert result.metadata["prior"]["source"] == "silhouette_moments_atlas_route"
    assert result.metadata["prior"]["silhouette_iou"] == 0.87
    assert result.metadata["prior"]["passes"] == 1
    assert result.metadata["prior"]["atlas_route_model_calls"] == 1
    assert result.metadata["model_called"] is True
    assert "replayed" not in result.metadata
    assert result.metadata["fit"]["fit"] == "deformable"
    # The silhouette placement (identity here) is what the fit starts from.
    np.testing.assert_array_equal(received.fitted[0].atlas_to_section, np.eye(3))


def test_atlas_route_passes_two_corrects_against_pass_one_and_the_template(case, monkeypatch):
    image, labels, received = case
    import langslice.core.nonlinear.image_gen_registration as legacy

    template = Image.new("RGB", (5, 5))
    monkeypatch.setattr(legacy, "outlined_atlas_template", lambda *a, **k: template)
    _stub_silhouette_placement(monkeypatch, labels)
    calls = []
    pass1_reply = image.copy()
    pass1_reply.putpixel((10, 10), (255, 255, 0))  # a drawn boundary pixel

    def fake_generate(request):
        calls.append(request)
        if len(calls) == 1:
            return SimpleNamespace(image=pass1_reply, route="test")
        return SimpleNamespace(image=image.copy(), route="test")

    monkeypatch.setattr(registration, "generate_warped_segmentation_image", fake_generate)
    registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, provider="openai-oauth", passes=2,
    )
    assert len(calls) == 2
    # Pass 2's Image 1 is the clean tissue, Image 2 pass 1's lines redrawn on
    # it (same canvas), Image 3 the outlined atlas template again.
    assert calls[1].reference_images[1] is template
    redrawn = np.asarray(calls[1].reference_images[0])
    assert redrawn[10, 10].tolist() == [255, 255, 0]
    untouched = np.ones(redrawn.shape[:2], dtype=bool)
    untouched[10, 10] = False
    np.testing.assert_array_equal(redrawn[untouched], np.asarray(image)[untouched])


@pytest.mark.parametrize("passes", [0, 3, -1])
def test_passes_must_be_one_or_two(case, passes):
    image, _labels, _received = case
    with pytest.raises(ValueError, match="passes must be 1 or 2"):
        registration.generate_border_registration_candidate(
            image, atlas_name="test", position_mm=1,
            initial_atlas_to_slice=np.eye(3), passes=passes,
        )


def test_replay_route_atlas_skips_every_model_call(case, monkeypatch):
    """A supplied generated_image with no placement replays route "atlas"
    end to end: it IS that route's final output, so no model call happens —
    this is the offline-smoke-test contract."""
    image, labels, received = case
    import langslice.core.nonlinear.image_gen_registration as legacy

    monkeypatch.setattr(
        legacy, "outlined_atlas_template", lambda *a, **k: Image.new("RGB", (5, 5))
    )
    _stub_silhouette_placement(monkeypatch, labels)

    def forbidden(*args, **kwargs):
        pytest.fail("A replayed atlas-route output must not call the model")

    monkeypatch.setattr(registration, "generate_warped_segmentation_image", forbidden)
    result = registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, provider="openai-oauth",
        generated_image=image,
    )
    assert result.metadata["prior"]["source"] == "silhouette_moments_atlas_route"
    assert result.metadata["prior"]["atlas_route_model_calls"] == 0
    assert result.metadata["model_called"] is False
    assert received[0][1]["generated_image"] is image


def test_nonfinite_fields_and_singular_affines_fail():
    field = np.zeros((4, 5, 2))
    field[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        registration.composed_correspondences(field, np.eye(3), np.eye(3), np.eye(3))
    with pytest.raises(ValueError, match="invertible"):
        registration.composed_correspondences(np.zeros((4, 5, 2)), np.zeros((3, 3)),
                                             np.eye(3), np.eye(3))


def test_supplied_placement_is_fitted_and_none_fits_nothing(case, tmp_path):
    image, labels, received = case
    initial = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    result = registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, initial_atlas_to_slice=initial,
    )
    assert result.metadata["deformation"] == "deformable"
    assert result.metadata["fit"]["fit"] == "deformable"
    np.testing.assert_array_equal(received.fitted[0].atlas_to_section, initial)
    skipped = registration.generate_border_registration_candidate(
        image, atlas_name="test", position_mm=1, initial_atlas_to_slice=initial,
        deformation="none",
    )
    assert len(received.fitted) == 1
    assert skipped.metadata["fit"] == {"fit": "none", "fit_skipped": True}
    assert skipped.metadata["fit_elapsed_s"] == 0.0


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("axes", [None, "si,rl", "lr,si"])
def test_native_to_oriented_map_reproduces_the_label_orientation(axes, mirror):
    """The fit works on the native plane, the canvas placement on the oriented
    labels: the map between them is exactly the turn/flip the labels took."""
    import cv2

    from langslice.core.space import atlas_space_context, orient_slice_to_axes
    from tests.deformable_synthetic import SyntheticAtlas

    atlas = SyntheticAtlas()
    native = np.arange(7 * 11, dtype=np.int64).reshape(7, 11) + 1
    oriented = native
    if axes:
        oriented = orient_slice_to_axes(native, atlas_space_context(atlas), "coronal", axes)
    if mirror:
        oriented = np.fliplr(oriented)
    matrix = registration.native_to_oriented_map(native.shape, atlas, "coronal", axes, mirror)
    placed = cv2.warpAffine(native.astype(np.float64), matrix[:2],
                            (oriented.shape[1], oriented.shape[0]), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(placed.astype(np.int64), oriented)
