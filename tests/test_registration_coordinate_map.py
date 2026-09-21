"""The public dispatcher's routing and the explicit atlas mirror."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from langslice.nonlinear import image_gen_registration as registration


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


def test_public_forwards_passes_for_route_atlas_without_supplied_alignment(monkeypatch):
    from langslice.nonlinear import border_registration

    calls = []
    sentinel = object()

    def wrapper(image, **kwargs):
        calls.append((image, kwargs))
        return sentinel

    monkeypatch.setattr(border_registration, "generate_border_registration_candidate", wrapper)
    image = Image.new("RGB", (10, 8))
    result = registration.generate_registration_candidate(
        image, atlas_name="fake", position_mm=4, provider="openai-oauth", passes=2,
    )
    assert result is sentinel
    assert calls[0][1]["passes"] == 2
    assert calls[0][1]["initial_atlas_to_slice"] is None


def test_explicit_mirror_is_applied_after_orientation():
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    result = registration._orient_pil(
        Image.fromarray(original), SimpleNamespace(), "coronal", None, atlas_mirror_lr=True
    )
    np.testing.assert_array_equal(np.asarray(result), original[:, ::-1])
