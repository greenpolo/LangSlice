"""Annotation-backed landmark queries, against a synthetic slab atlas.

No BrainGlobe download: ``SlabAtlas`` carries an annotation volume and a
minimal structure tree, which is everything ``landmarks`` reads.
"""

from typing import Any

import pytest

from langslice.atlas import landmarks
from tests.fakes import SlabAtlas


@pytest.fixture(autouse=True)
def _clear_caches():
    """The module caches by atlas NAME; every fake shares one."""
    landmarks._PRESENCE_CACHE.clear()
    landmarks._STRUCTURE_CACHE.clear()
    yield
    landmarks._PRESENCE_CACHE.clear()
    landmarks._STRUCTURE_CACHE.clear()


@pytest.fixture
def atlas() -> Any:
    return SlabAtlas()


# --- structures_at -------------------------------------------------------


def test_structures_at_lists_what_the_level_carries_largest_first(atlas: Any):
    found = landmarks.structures_at(atlas, 6.0, "coronal")

    # FA (rows 3-5) and FB (rows 5-9) both exist at index 6; FB is larger.
    assert [entry["acronym"] for entry in found] == ["FB", "FA"]
    assert found[0]["name"] == "Forebrain area B"
    assert found[0]["area_share"] > found[1]["area_share"]
    assert 0.0 < found[0]["area_share"] <= 1.0


def test_structures_at_drops_root_and_reports_empty_levels(atlas: Any):
    assert all(
        entry["acronym"] != "root" for entry in landmarks.structures_at(atlas, 6.0)
    )
    # Index 10 is tissue-only: root everywhere, no named structure.
    assert landmarks.structures_at(atlas, 10.0) == []


def test_structures_at_caps_the_list_and_drops_slivers(atlas: Any):
    assert landmarks.structures_at(atlas, 6.0, limit=1) == [
        landmarks.structures_at(atlas, 6.0)[0]
    ]
    assert landmarks.structures_at(atlas, 6.0, min_area_share=0.99) == []


# --- axis_range_of -------------------------------------------------------


def test_axis_range_of_spans_a_leaf_structure(atlas: Any):
    assert landmarks.axis_range_of(atlas, "FA", "coronal") == (2.0, 6.0)
    assert landmarks.axis_range_of(atlas, "HB", "coronal") == (12.0, 18.0)


def test_axis_range_of_rolls_descendants_up(atlas: Any):
    """FOR has no voxels of its own; its span is FA's plus FB's."""
    assert landmarks.axis_range_of(atlas, "FOR", "coronal") == (2.0, 9.0)


def test_axis_range_of_is_none_when_nothing_is_annotated(atlas: Any):
    empty = SlabAtlas()
    empty.atlas_name = "fake_slab_empty"  # its own cache key
    empty._annotation[:] = 0
    assert landmarks.axis_range_of(empty, "FA") is None


def test_axis_range_of_rejects_an_unknown_structure(atlas: Any):
    with pytest.raises(LookupError, match="ghost"):
        landmarks.axis_range_of(atlas, "ghost")


def test_axis_range_of_scans_the_volume_once_per_plane(atlas: Any):
    landmarks.axis_range_of(atlas, "FA", "coronal")
    reads = atlas.annotation_reads
    landmarks.axis_range_of(atlas, "FB", "coronal")
    landmarks.axis_range_of(atlas, "FOR", "coronal")
    assert atlas.annotation_reads == reads

    landmarks.axis_range_of(atlas, "FA", "horizontal")
    assert atlas.annotation_reads > reads


# --- resolution ----------------------------------------------------------


def test_find_structure_accepts_acronym_name_and_unique_substring(atlas: Any):
    assert landmarks.find_structure(atlas, "fa")["acronym"] == "FA"
    assert landmarks.find_structure(atlas, "Forebrain area B")["acronym"] == "FB"
    assert landmarks.find_structure(atlas, "hindbrain")["acronym"] == "HB"
    # "forebrain area" matches two structures: ambiguous is not resolved.
    assert landmarks.find_structure(atlas, "forebrain area") is None
    assert landmarks.find_structure(atlas, "") is None


def test_near_misses_names_the_candidates(atlas: Any):
    assert set(landmarks.near_misses(atlas, "forebrain area")) == {"FA", "FB"}
    assert landmarks.near_misses(atlas, "hb") == ["HB"]
    assert landmarks.near_misses(atlas, "") == []


class _AcronymFamilyAtlas:
    """Both matching patterns at once: an acronym family and a name substring.

    "SC" prefixes a family of acronyms AND appears inside the NAME of an
    unrelated one ("visceral"). Ranking names first buried the family the
    query was obviously reaching for.
    """

    atlas_name = "fake_acronym_family"
    structures = {  # noqa: RUF012 - a fixture, not a mutable default
        1: {"id": 1, "acronym": "root", "name": "root", "structure_id_path": [1]},
        2: {"id": 2, "acronym": "VISC", "name": "Visceral area", "structure_id_path": [1, 2]},
        3: {
            "id": 3,
            "acronym": "VISC1",
            "name": "Visceral area, layer 1",
            "structure_id_path": [1, 2, 3],
        },
        4: {
            "id": 4,
            "acronym": "SCm",
            "name": "Superior colliculus, motor related",
            "structure_id_path": [1, 4],
        },
        5: {
            "id": 5,
            "acronym": "SCig",
            "name": "Superior colliculus, intermediate gray",
            "structure_id_path": [1, 4, 5],
        },
    }


def test_near_misses_ranks_acronym_matches_above_name_substrings():
    found = landmarks.near_misses(_AcronymFamilyAtlas(), "SC")  # type: ignore[arg-type]

    # The acronym family leads, shortest first; the "visceral" name hits trail.
    assert found[:2] == ["SCm", "SCig"]
    assert set(found[2:]) == {"VISC", "VISC1"}


def test_descendant_ids_include_the_structure_itself(atlas: Any):
    assert landmarks.descendant_ids(atlas, 2) == {2, 3, 4}
    assert landmarks.descendant_ids(atlas, 5) == {5}
