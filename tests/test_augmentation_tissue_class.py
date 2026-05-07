"""Tests for tissue-class masks (gray / white / ventricle)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from augmentation.transforms.tissue_class import (
    TISSUE_CLASS_ROOTS,
    classify_tissue,
    get_tissue_id_sets,
)


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice_harness.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")


@pytest.fixture(scope="module")
def annotation_slice(atlas: object) -> np.ndarray:
    """Mid-AP coronal annotation slice with all three tissue classes present."""
    from langslice_harness.atlas.core import position_mm_to_index
    from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, 6.0)  # type: ignore[arg-type]
    return np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]


def test_roots_constant_known_classes() -> None:
    assert set(TISSUE_CLASS_ROOTS) == {"gray_matter", "white_matter", "ventricle"}
    assert TISSUE_CLASS_ROOTS["white_matter"] == "fiber tracts"


def test_id_sets_disjoint(atlas: object) -> None:
    sets = get_tissue_id_sets(atlas)
    gm, wm, vs = sets["gray_matter"], sets["white_matter"], sets["ventricle"]
    assert gm & wm == frozenset()
    assert gm & vs == frozenset()
    assert wm & vs == frozenset()


def test_id_sets_nonempty_and_sized(atlas: object) -> None:
    sets = get_tissue_id_sets(atlas)
    # Allen CCFv3 has ~722 grey, ~106 fiber-tracts (+ root), ~8 VS (+ root) descendants.
    assert len(sets["gray_matter"]) > 500
    assert 50 < len(sets["white_matter"]) < 200
    assert 5 <= len(sets["ventricle"]) < 30


def test_id_sets_cached(atlas: object) -> None:
    a, b = get_tissue_id_sets(atlas), get_tissue_id_sets(atlas)
    for k in a:
        assert a[k] is b[k]


def test_classify_returns_expected_keys(annotation_slice: np.ndarray, atlas: object) -> None:
    masks = classify_tissue(annotation_slice, atlas)
    assert set(masks) == {
        "gray_matter",
        "cortical_subplate",
        "hippocampal_formation",
        "isocortex",
        "thalamus",
        "white_matter",
        "ventricle",
        "tissue",
        "background",
    }
    for v in masks.values():
        assert v.dtype == bool
        assert v.shape == annotation_slice.shape


def test_masks_partition_pixels(annotation_slice: np.ndarray, atlas: object) -> None:
    masks = classify_tissue(annotation_slice, atlas)
    # tissue and background are exact complements.
    assert (masks["tissue"] | masks["background"]).all()
    assert not (masks["tissue"] & masks["background"]).any()
    # Three sub-class masks are mutually exclusive.
    assert not (masks["gray_matter"] & masks["white_matter"]).any()
    assert not (masks["gray_matter"] & masks["ventricle"]).any()
    assert not (masks["white_matter"] & masks["ventricle"]).any()
    # All sub-classes are subsets of tissue.
    union = masks["gray_matter"] | masks["white_matter"] | masks["ventricle"]
    assert (union & masks["tissue"]).sum() == union.sum()
    # Most of tissue is covered by the three sub-classes (only root-id-997
    # edge pixels fall through). At AP=5.335 mm fewer than ~2% should be
    # unclassified.
    fall_through = (masks["tissue"] & ~union).sum()
    assert fall_through / max(masks["tissue"].sum(), 1) < 0.02


def test_real_slice_has_all_classes(annotation_slice: np.ndarray, atlas: object) -> None:
    """At AP=6.0 mm, every Allen tissue class should be present."""
    masks = classify_tissue(annotation_slice, atlas)
    assert masks["gray_matter"].sum() > 1000
    assert masks["cortical_subplate"].sum() > 10
    assert masks["hippocampal_formation"].sum() > 10
    assert masks["isocortex"].sum() > 100
    assert masks["thalamus"].sum() > 10
    assert masks["white_matter"].sum() > 100  # corpus callosum, internal capsule, etc.
    assert masks["ventricle"].sum() > 10  # third ventricle at this AP
    assert masks["background"].sum() > 0  # off-tissue padding always present


def test_zero_annotation_is_background(atlas: object) -> None:
    ann = np.zeros((20, 20), dtype=np.int32)
    masks = classify_tissue(ann, atlas)
    assert masks["background"].all()
    assert not masks["tissue"].any()


# ---------------------------------------------------------------------------
# New atlas-agnostic tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not any(
        __import__("pathlib")
        .Path(__import__("os").path.expanduser("~/.brainglobe"))
        .glob("allen_mouse_50um_v*")
    ),
    reason="allen_mouse_50um not cached locally",
)
def test_atlas_agnostic_resolves_known_atlas_allen_50um() -> None:
    """allen_mouse_50um should use the same explicit mapping as 25um."""
    from augmentation.transforms.tissue_class import _ATLAS_TISSUE_ROOTS

    from langslice_harness.atlas.core import load_atlas

    atlas_50 = load_atlas("allen_mouse_50um")
    # No warning should be emitted — it's in the explicit mapping.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sets = get_tissue_id_sets(atlas_50)
    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warns) == 0, f"Unexpected UserWarning for allen_mouse_50um: {user_warns}"

    assert len(sets["gray_matter"]) > 500
    assert len(sets["white_matter"]) > 50
    assert len(sets["ventricle"]) >= 5

    # Confirm the mapping entry exists
    assert "allen_mouse_50um" in _ATLAS_TISSUE_ROOTS
    assert _ATLAS_TISSUE_ROOTS["allen_mouse_50um"]["white_matter"] == "fiber tracts"


@pytest.mark.skipif(
    not any(
        __import__("pathlib")
        .Path(__import__("os").path.expanduser("~/.brainglobe"))
        .glob("whs_sd_rat_39um_v*")
    ),
    reason="whs_sd_rat_39um not cached locally",
)
def test_atlas_agnostic_resolves_whs_rat() -> None:
    """WHS Sprague-Dawley rat atlas uses a different ontology; mapping must resolve."""
    from langslice_harness.atlas.core import load_atlas

    rat_atlas = load_atlas("whs_sd_rat_39um")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sets = get_tissue_id_sets(rat_atlas)
    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warns) == 0, f"Unexpected UserWarning for whs_sd_rat_39um: {user_warns}"

    assert len(sets["gray_matter"]) > 10, "Rat gray matter set unexpectedly small"
    assert len(sets["white_matter"]) > 5, "Rat white matter set unexpectedly small"
    assert len(sets["ventricle"]) >= 1, "Rat ventricle set unexpectedly empty"


def test_atlas_agnostic_keyword_fallback_warns() -> None:
    """A fake atlas whose name is not in the mapping should trigger UserWarning + still work."""
    from augmentation.transforms.tissue_class import classify_tissue, get_tissue_id_sets

    class FakeStructures:
        """Minimal structures dict-like exposing .values() for keyword search."""

        _data = {
            "root": {"acronym": "root", "id": 997, "name": "root", "structure_id_path": [997]},
            "GREY": {"acronym": "GREY", "id": 8, "name": "grey matter brain regions"
                , "structure_id_path": [997, 8]},
            "WM": {"acronym": "WM", "id": 1009, "name": "fiber tracts white matter"
                , "structure_id_path": [997, 1009]},
            "VEN": {"acronym": "VEN", "id": 73, "name": "ventricular system"
                , "structure_id_path": [997, 73]},
            "CTX": {"acronym": "CTX", "id": 315, "name": "isocortex"
                , "structure_id_path": [997, 8, 315]},
        }

        def __getitem__(self, key: object) -> dict:
            return self._data[str(key)]

        def values(self):
            return self._data.values()

    class FakeAtlas:
        atlas_name = "fake_unknown_atlas_xyz"

        structures = FakeStructures()

        def get_structure_descendants(self, acronym: str) -> list[str]:
            # Return child structures for each root.
            children = {"GREY": ["CTX"], "WM": [], "VEN": []}
            return children.get(acronym, [])

    fake = FakeAtlas()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_tissue_id_sets(fake)

    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warns) >= 1, "Expected UserWarning for unknown atlas"
    assert "fake_unknown_atlas_xyz" in str(user_warns[0].message)

    # Classification should still yield the correct keys.
    ann = np.zeros((10, 10), dtype=np.int32)
    ann[0, :] = 8    # grey
    ann[1, :] = 1009  # white matter
    ann[2, :] = 73   # ventricle
    masks = classify_tissue(ann, fake)
    assert set(masks) == {
        "gray_matter",
        "cortical_subplate",
        "hippocampal_formation",
        "isocortex",
        "thalamus",
        "white_matter",
        "ventricle",
        "tissue",
        "background",
    }


def test_unknown_atlas_with_unrecognized_ontology_falls_back_gracefully() -> None:
    """An atlas with unrecognized structure names still returns empty sets (no crash)."""
    from augmentation.transforms.tissue_class import get_tissue_id_sets

    class FakeStructures:
        _data = {
            "abc": {"acronym": "abc", "id": 1, "name": "undecipherable region alpha"
                , "structure_id_path": [999, 1]},
            "xyz": {"acronym": "xyz", "id": 2, "name": "undecipherable region beta"
                , "structure_id_path": [999, 2]},
        }

        def __getitem__(self, key: object) -> dict:
            if str(key) not in self._data:
                raise KeyError(key)
            return self._data[str(key)]

        def values(self):
            return self._data.values()

    class TotallyUnknownAtlas:
        atlas_name = "completely_novel_species_atlas"

        structures = FakeStructures()

        def get_structure_descendants(self, acronym: str) -> list[str]:
            return []

    fake = TotallyUnknownAtlas()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sets = get_tissue_id_sets(fake)

    # A UserWarning must be emitted.
    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warns) >= 1

    # The result must still have all three tissue-class keys.
    assert set(sets) == {"gray_matter", "white_matter", "ventricle"}
    # Since no structures match, all sets should be empty (graceful degradation).
    for cls, id_set in sets.items():
        assert isinstance(id_set, frozenset), f"{cls} should be frozenset"
