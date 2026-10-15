"""N4 bias-field correction and denoising of one plane (SimpleITK)."""

from __future__ import annotations

import numpy as np

from langslice.core.image_prep import enhance_plane


def _tissue(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:128]
    ramp = 0.5 + xx / 128.0  # uneven illumination, left to right
    plane = 80.0 * ramp + rng.normal(0, 6, size=(96, 128))
    plane[:8] = 0  # background: left alone
    return np.clip(plane, 0, 255).astype(np.uint8)


def test_nothing_asked_returns_the_plane() -> None:
    plane = _tissue()
    assert enhance_plane(plane, n4=False, denoise=False) is plane


def test_n4_flattens_the_illumination_ramp_and_keeps_the_background() -> None:
    plane = _tissue()
    out = enhance_plane(plane, n4=True, denoise=False)
    left, right = (lambda a: (a[16:, :32].mean(), a[16:, -32:].mean()))(out.astype(float))
    before = plane[16:, -32:].mean() / plane[16:, :32].mean()
    assert right / left < before
    assert (out[:8] == 0).all()
    assert abs(out[16:].mean() - plane[16:].mean()) < 3  # the channel keeps its brightness


def test_denoise_smooths_and_is_repeatable() -> None:
    plane = _tissue()
    first = enhance_plane(plane, n4=False, denoise=True)
    again = enhance_plane(plane, n4=False, denoise=True)
    assert np.array_equal(first, again)
    assert first[16:].astype(float).std() < plane[16:].astype(float).std()


def test_a_constant_plane_is_returned_unchanged() -> None:
    plane = np.full((32, 32), 100, np.uint8)
    out = enhance_plane(plane, n4=True, denoise=True)
    assert out.shape == plane.shape and abs(int(out.mean()) - 100) <= 1
