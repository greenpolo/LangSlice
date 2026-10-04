"""Boundary corrections preserve source pixels; the deformable package fits the lines."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.nonlinear import border_refinement as borders


@pytest.fixture
def case(monkeypatch):
    labels = np.zeros((40, 60), dtype=np.int64)
    labels[6:34, 8:30] = 100000001
    labels[6:34, 30:52] = 100000003
    photo = Image.fromarray(np.full((40, 60, 3), (70, 80, 90), dtype=np.uint8))
    monkeypatch.setattr(borders, "_merge_classified", lambda ids, atlas: ids)
    return photo, labels, SimpleNamespace()


def test_two_inputs_and_corrected_lines_never_replace_the_photograph(case, monkeypatch):
    photo, labels, atlas = case
    corrected = borders._extract_borders_from_classified(np.roll(labels, 4, axis=1)) > 0
    # Simulate a model that redraws the photograph while correcting boundaries.
    reply = borders.border_overlay(Image.new("RGB", photo.size, (180, 20, 30)), corrected)
    requests = []

    def generate(request):
        requests.append(request)
        return SimpleNamespace(image=reply, route="test")

    result = borders.refine_borders(photo, labels, atlas, image_call=generate)
    assert len(requests) == 1
    request = requests[0]
    assert len(request.reference_images) == 1
    assert np.array_equal(request.reference_images[0], photo)
    rough_lines = borders._extract_borders_from_classified(labels) > 0
    assert np.array_equal(request.slice_image, borders.border_overlay(photo, rough_lines))
    assert np.array_equal(result.raw_model_image, reply)
    untouched = ~result.model_border_mask
    assert np.array_equal(np.asarray(result.model_border_overlay)[untouched],
                          np.asarray(photo)[untouched])
    assert result.model_border_mask[corrected].mean() > 0.5
    assert result.metadata["model_called"] is True


def test_without_an_injected_call_the_transport_adapter_is_used(case, monkeypatch):
    photo, labels, atlas = case
    reply = borders.border_overlay(photo, borders._extract_borders_from_classified(labels) > 0)
    calls = []
    monkeypatch.setattr(borders, "generate_warped_segmentation_image",
                        lambda request: calls.append(request) or SimpleNamespace(
                            image=reply, route="test"))
    borders.refine_borders(photo, labels, atlas)
    assert len(calls) == 1


def test_replay_crops_letterbox_before_resampling(case, monkeypatch):
    photo, labels, atlas = case
    canvas = Image.new("RGB", (80, 40))
    pixels = np.array(canvas)
    pixels[7:33, 40] = (255, 255, 0)  # x=30 after symmetric 10px crop
    raw = Image.fromarray(pixels)
    result = borders.refine_borders(photo, labels, atlas, generated_image=raw)
    assert result.raw_model_image.size == (80, 40)
    assert result.model_border_overlay.size == photo.size
    assert np.all(np.nonzero(result.model_border_mask)[1] == 30)
    assert result.metadata["replayed"] is True


def test_model_free_calls_nothing_and_keeps_the_rough_borders(case, monkeypatch):
    photo, labels, atlas = case

    def forbidden(*args, **kwargs):
        pytest.fail("Model-free placement must not generate")

    monkeypatch.setattr(borders, "generate_warped_segmentation_image", forbidden)
    result = borders.refine_borders(photo, labels, atlas, provider="none")
    assert result.metadata["model_free"] is True
    np.testing.assert_array_equal(result.model_border_mask,
                                  borders._extract_borders_from_classified(labels) > 0)


def test_no_lines_fails_before_fit(case):
    photo, labels, atlas = case
    with pytest.raises(ValueError, match="no usable yellow"):
        borders.refine_borders(photo, labels, atlas, generated_image=photo)


def test_label_frame_must_match_original(case):
    photo, labels, atlas = case
    with pytest.raises(ValueError, match="integer map on the image canvas"):
        borders.refine_borders(photo, labels[:, :-1], atlas, provider="none")


def test_thinning_keeps_connected_line_and_yellow_tolerance():
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    rgb[5:25, 12:17] = (240, 210, 20)
    mask = borders.thin(borders.yellow_mask(rgb))
    assert mask.sum() > 10
    assert np.unique(np.nonzero(mask)[1]).size == 1
    assert not borders.yellow_mask(np.full((3, 3, 3), 150, dtype=np.uint8)).any()


# --- the fit after the correction: the deformable package's ------------------


@pytest.fixture(scope="module")
def synthetic():
    from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section

    atlas = SyntheticAtlas()
    field = SMOOTH_FIELD()
    image, truth = render_section(atlas, field)
    return atlas, field, image, truth


def _family_lines(labels, atlas):
    from langslice.atlas.render import family_labels

    merged = family_labels(labels, atlas)
    lines = np.zeros(merged.shape, dtype=bool)
    lines[:, 1:] |= merged[:, 1:] != merged[:, :-1]
    lines[1:, :] |= merged[1:, :] != merged[:-1, :]
    return lines


def _mean_error(field_px, field_mm, truth):
    from scipy import ndimage as ndi

    from tests.deformable_synthetic import SECTION_MM_PER_PX

    inner = ndi.binary_erosion(truth > 0, iterations=6)
    true_px = field_mm / SECTION_MM_PER_PX
    error = np.linalg.norm(field_px - true_px, axis=-1)[inner].mean()
    flipped = np.linalg.norm(field_px + true_px, axis=-1)[inner].mean()
    return error, flipped, np.linalg.norm(true_px, axis=-1)[inner].mean()


def test_the_model_lines_are_fitted_by_the_deformable_package(synthetic):
    """Lines drawn where the warped regions meet pull the placed borders there:
    the field comes back in canvas pixels in the documented direction."""
    from langslice.nonlinear.border_fit import fit_border_lines
    from tests.deformable_synthetic import placement

    atlas, field, image, truth = synthetic
    fit = fit_border_lines(image, _family_lines(truth, atlas), atlas, placement())
    error, flipped, magnitude = _mean_error(fit.field_px, field, truth)
    assert error < 0.6 * magnitude and flipped > 1.5 * magnitude
    assert fit.record.settings.section_image == "lines"
    assert fit.record.settings.engine == "elastix"
    assert fit.metadata["fit"] == "deformable" and fit.metadata["metric"] == "mean_squares"
    linear = fit.record.placed_labels(atlas.annotation[0])
    inside = truth > 0
    assert np.mean(fit.fitted_labels[inside] == truth[inside]) > np.mean(
        linear[inside] == truth[inside])
    assert fit.fitted_border_overlay.size == image.size
    json.dumps(fit.metadata)  # plain JSON for the candidate metadata and debug files


def test_an_empty_line_mask_is_refused_before_any_fit(synthetic):
    from langslice.nonlinear.border_fit import fit_border_lines
    from tests.deformable_synthetic import placement

    atlas, _field, image, _truth = synthetic
    with pytest.raises(ValueError, match="no usable yellow"):
        fit_border_lines(image, np.zeros((image.height, image.width), bool), atlas, placement())


def test_host_labels_fit_like_the_atlas_plane_they_equal(synthetic):
    """ABBA hands labels sampled at its own coordinates (placed by an identity
    on its grid). Given the very labels the atlas plane holds, the fit is the
    plane's, bit for bit; only the volume mapping is not claimed."""
    from langslice.nonlinear.border_fit import fit_border_lines
    from tests.deformable_synthetic import placement

    atlas, _field, image, truth = synthetic
    lines = _family_lines(truth, atlas)
    plane = fit_border_lines(image, lines, atlas, placement())
    host = fit_border_lines(image, lines, atlas, placement(), native=atlas.annotation[0].copy())
    np.testing.assert_array_equal(host.field_px, plane.field_px)
    np.testing.assert_array_equal(host.fitted_labels, plane.fitted_labels)
    assert host.record.atlas["native_to_volume_index"] is None
    assert plane.record.atlas["native_to_volume_index"] is not None


def test_supplied_native_labels_refuse_a_grayscale_atlas_image(synthetic):
    from langslice.deformable import FitSettings, prepare_fit
    from tests.deformable_synthetic import placement

    atlas, _field, image, _truth = synthetic
    with pytest.raises(ValueError, match="border atlas images only"):
        prepare_fit(image, atlas, placement(), FitSettings(engine="elastix"),
                    native=atlas.annotation[0])


def test_the_canvas_pixel_size_comes_from_the_placement_scale():
    from langslice.nonlinear.border_fit import placement_on_canvas

    atlas = SimpleNamespace(resolution=(25.0, 25.0, 25.0))
    matrix = np.array([[2.5, 0.0, 3.0], [0.0, 2.5, 4.0], [0.0, 0.0, 1.0]])
    placed = placement_on_canvas(atlas, matrix, atlas_name="a", position_mm=5.0)
    assert placed.section_mm_per_px == pytest.approx(0.025 / 2.5)
    np.testing.assert_array_equal(placed.atlas_to_section, matrix)
