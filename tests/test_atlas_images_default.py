"""``atlas.images.default_image``: what ``"nissl"`` stands for on each atlas."""

from __future__ import annotations

import numpy as np
import pytest

from langslice.core.atlas import images
from langslice.core.atlas.images import (
    atlas_images,
    atlas_volume,
    default_image,
    resolve_image,
)


class _Atlas:
    """A small atlas with chosen additional references and name."""

    def __init__(self, name: str = "toy_25um", extra: tuple[str, ...] = (),
                 shape: tuple[int, int, int] = (4, 5, 6), orientation: str = "asr",
                 resolution: tuple[float, float, float] = (25.0, 25.0, 25.0)) -> None:
        self.atlas_name = name
        self.orientation = orientation
        self.resolution = resolution
        self.metadata = {"orientation": orientation, "shape": list(shape)}
        self.template = np.zeros(shape, dtype=np.float32)
        self.annotation = np.ones(shape, dtype=np.uint32)
        self.additional_references = {
            key: np.full(shape, index + 1, dtype=np.float32) for index, key in enumerate(extra)}


def test_an_atlas_without_a_nissl_uses_its_template():
    atlas = _Atlas()
    assert default_image(atlas) == "template"
    assert resolve_image(atlas, "nissl") == "template"
    assert np.array_equal(atlas_volume(atlas, "nissl"), atlas.template)


def test_an_additional_reference_named_for_nissl_is_the_default():
    atlas = _Atlas(extra=("mri", "Nissl_average", "stpt"))
    assert default_image(atlas) == "Nissl_average"
    assert resolve_image(atlas, "nissl") == "Nissl_average"
    assert atlas_volume(atlas, "nissl")[0, 0, 0] == 2  # the second reference
    assert atlas_images(atlas) == ("template", "mri", "Nissl_average", "stpt", "nissl")


def test_the_augmented_atlas_is_its_own_nissl():
    atlas = _Atlas(name=images.AUGMENTED_ATLAS, extra=("mri",))
    assert default_image(atlas) == "template"


def test_a_ccfv3_atlas_takes_the_borrowed_nissl(monkeypatch):
    shape = (264, 160, 228)  # 50 um CCFv3
    atlas = _Atlas(name="allen_mouse_50um", shape=shape, resolution=(50.0, 50.0, 50.0))
    monkeypatch.setattr(images, "_takes_borrowed_nissl", lambda a: a is atlas)
    assert default_image(atlas) == images.BORROWED_NISSL
    assert resolve_image(atlas, "nissl") == images.BORROWED_NISSL
    assert atlas_images(atlas) == ("template", images.BORROWED_NISSL, "nissl")
    # A Nissl-named reference does not outrank the borrowed one.
    other = _Atlas(extra=("nissl_stack",))
    assert default_image(other) == "nissl_stack"


def test_other_names_are_checked_against_the_atlas():
    atlas = _Atlas(extra=("mri",))
    assert resolve_image(atlas, "template") == "template"
    assert resolve_image(atlas, "mri") == "mri"
    with pytest.raises(ValueError, match="has no image 'gfp'.*template, mri, nissl"):
        resolve_image(atlas, "gfp")
    with pytest.raises(ValueError, match="has no image"):
        atlas_volume(atlas, "gfp")
    # The borrowed Nissl exists only on an atlas that takes it.
    with pytest.raises(ValueError, match="has no image"):
        resolve_image(atlas, images.BORROWED_NISSL)
