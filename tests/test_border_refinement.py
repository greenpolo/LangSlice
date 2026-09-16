"""Boundary corrections preserve source pixels and warp original atlas identities."""

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


def mock_fit(monkeypatch, labels, *, shift=4):
    fitted = np.roll(labels, shift, axis=1)
    monkeypatch.setattr(borders, "_run_elastix_april_borders", lambda f, m: ("fit", 1.25))
    monkeypatch.setattr(borders, "_warp_classified_labels", lambda ids, t: fitted.copy())
    field = np.zeros((*labels.shape, 2))
    field[..., 0] = -shift
    monkeypatch.setattr(borders, "_compute_deformation_field", lambda t, m: field)
    return fitted


def test_two_inputs_and_corrected_lines_never_replace_the_photograph(case, monkeypatch):
    photo, labels, atlas = case
    fitted = mock_fit(monkeypatch, labels)
    corrected = borders._extract_borders_from_classified(fitted) > 0
    # Simulate a model that redraws the photograph while correcting boundaries.
    reply = borders.border_overlay(Image.new("RGB", photo.size, (180, 20, 30)), corrected)
    requests = []

    def generate(request):
        requests.append(request)
        return SimpleNamespace(image=reply, route="test")

    monkeypatch.setattr(borders, "generate_warped_segmentation_image", generate)
    result = borders.refine_borders(photo, labels, atlas)
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
    assert set(np.unique(result.fitted_labels)) == {0, 100000001, 100000003}
    assert np.array_equal(result.fitted_labels, fitted)
    assert np.all(result.deformation_field[..., 0] == -4)


def test_replay_crops_letterbox_before_resampling(case, monkeypatch):
    photo, labels, atlas = case
    mock_fit(monkeypatch, labels)
    canvas = Image.new("RGB", (80, 40))
    pixels = np.array(canvas)
    pixels[7:33, 40] = (255, 255, 0)  # x=30 after symmetric 10px crop
    raw = Image.fromarray(pixels)
    result = borders.refine_borders(photo, labels, atlas, generated_image=raw)
    assert result.raw_model_image.size == (80, 40)
    assert result.model_border_overlay.size == photo.size
    assert np.all(np.nonzero(result.model_border_mask)[1] == 30)
    assert result.metadata["replayed"] is True


def test_model_free_is_an_exact_identity_residual(case, monkeypatch):
    photo, labels, atlas = case

    def forbidden(*args, **kwargs):
        pytest.fail("Model-free placement must not generate or fit")

    monkeypatch.setattr(borders, "generate_warped_segmentation_image", forbidden)
    monkeypatch.setattr(borders, "_run_elastix_april_borders", forbidden)
    result = borders.refine_borders(photo, labels, atlas, provider="none")
    assert np.array_equal(result.fitted_labels, labels)
    assert not result.deformation_field.any()
    assert result.result_transform is None


def test_no_lines_fails_before_fit(case, monkeypatch):
    photo, labels, atlas = case

    def forbidden(*args, **kwargs):
        pytest.fail("Empty boundary replies must not reach fitting")

    monkeypatch.setattr(borders, "_run_elastix_april_borders", forbidden)
    with pytest.raises(ValueError, match="no usable yellow"):
        borders.refine_borders(photo, labels, atlas, generated_image=photo)


def test_affine_uses_only_affine_fit(case, monkeypatch):
    photo, labels, atlas = case
    mock_fit(monkeypatch, labels)
    calls = []

    def affine(fixed, moving, **kwargs):
        calls.append(kwargs)
        return "fit", 0.2

    monkeypatch.setattr(borders, "_register_channel_stacks", affine)
    reply = borders.border_overlay(photo, borders._extract_borders_from_classified(labels) > 0)
    borders.refine_borders(photo, labels, atlas, generated_image=reply, deformation="affine")
    assert calls == [{"deformation": "affine"}]


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
