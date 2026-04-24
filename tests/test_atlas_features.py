"""Checks for BrainGlobe atlas helper integrations."""

from collections.abc import Sequence
from typing import cast

import numpy as np

from langslice_harness.atlas.core import (
    get_additional_reference_slice,
    get_atlas_info,
    get_region_at_position,
    get_structure_hierarchy,
    get_structure_mask_slice,
    list_additional_references,
)


class FakeAtlas:
    atlas_name: str = "fake_mouse_100um"
    orientation: str = "asr"
    reference: np.ndarray
    annotation: np.ndarray
    resolution: Sequence[float]
    metadata: dict[str, object]
    structures: dict[int, dict[str, object]]
    additional_references: dict[str, np.ndarray]
    local_version: tuple[int, int]
    remote_version: tuple[int, int]

    def __init__(self) -> None:
        self.reference = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
        self.annotation = np.zeros((4, 4, 4), dtype=np.uint32)
        self.annotation[:, :, :2] = 2
        self.annotation[:, :, 2:] = 3
        self.resolution = (100.0, 100.0, 100.0)
        self.metadata = {"species": "mouse", "citation": "fake"}
        self.structures = {
            1: {"id": 1, "acronym": "ROOT", "name": "Root", "structure_id_path": [1]},
            2: {
                "id": 2,
                "acronym": "LEFT",
                "name": "Left Region",
                "structure_id_path": [1, 2],
            },
            3: {
                "id": 3,
                "acronym": "RIGHT",
                "name": "Right Region",
                "structure_id_path": [1, 3],
            },
        }
        self.additional_references = {
            "nissl": (self.reference * 2).astype(np.uint16),
        }
        self.local_version = (1, 0)
        self.remote_version = (1, 0)

    def check_latest_version(self, print_warning: bool = True) -> bool:
        _ = print_warning
        return True

    def structure_from_coords(
        self,
        coords: tuple[int, int, int],
        microns: bool = False,
        as_acronym: bool = False,
        hierarchy_lev: int | None = None,
        key_error_string: str = "Outside atlas",
    ) -> int | str:
        ap, dv, ml = coords
        if microns:
            ap = int(ap / self.resolution[0])
            dv = int(dv / self.resolution[1])
            ml = int(ml / self.resolution[2])
        if ap < 0 or dv < 0 or ml < 0:
            raise KeyError(key_error_string)
        if ap >= 4 or dv >= 4 or ml >= 4:
            raise KeyError(key_error_string)

        sid = int(self.annotation[ap, dv, ml])
        if hierarchy_lev is not None:
            path = cast(list[int], self.structures[sid]["structure_id_path"])
            sid = int(path[hierarchy_lev])
        if as_acronym:
            return str(self.structures[sid]["acronym"])
        return sid

    def hemisphere_from_coords(
        self, coords: tuple[int, int, int], microns: bool = False, as_string: bool = False
    ) -> int | str:
        _ = microns
        ml = coords[2]
        hemi_value = 1 if ml < 2 else 2
        if as_string:
            return "left" if hemi_value == 1 else "right"
        return hemi_value

    def get_structure_ancestors(self, structure: int | str) -> list[str]:
        sid = self._to_id(structure)
        path = cast(list[int], self.structures[sid]["structure_id_path"])[:-1]
        return [str(self.structures[node]["acronym"]) for node in path]

    def get_structure_descendants(self, structure: int | str) -> list[str]:
        sid = self._to_id(structure)
        descendants: list[str] = []
        for entry in self.structures.values():
            path = cast(list[int], entry["structure_id_path"])
            if sid in path and int(cast(int, entry["id"])) != sid:
                descendants.append(str(entry["acronym"]))
        return descendants

    def get_structure_mask(self, structure: int | str) -> np.ndarray:
        sid = self._to_id(structure)
        match_ids = [sid]
        for entry in self.structures.values():
            path = cast(list[int], entry["structure_id_path"])
            if sid in path and int(cast(int, entry["id"])) != sid:
                match_ids.append(int(cast(int, entry["id"])))
        return np.where(np.isin(self.annotation, match_ids), sid, 0).astype(np.uint32)

    def _to_id(self, structure: int | str) -> int:
        if isinstance(structure, int):
            return structure
        for entry in self.structures.values():
            if cast(str, entry["acronym"]) == structure:
                return int(cast(int, entry["id"]))
        raise KeyError(structure)


def test_list_additional_references() -> None:
    atlas = FakeAtlas()
    refs = list_additional_references(atlas)
    assert refs == ["nissl"]


def test_additional_reference_slice() -> None:
    atlas = FakeAtlas()

    extra_slice = get_additional_reference_slice(atlas, "nissl", position_mm=0.2)
    assert extra_slice.mode == "L"
    assert extra_slice.size == (4, 4)


def test_region_and_hierarchy_lookup() -> None:
    atlas = FakeAtlas()

    region = get_region_at_position(atlas, 0.2, dv_index=1, ml_index=3, include_hierarchy=True)
    assert region["structure"]["id"] == 3
    assert region["structure"]["acronym"] == "RIGHT"
    assert region["hemisphere"] == "right"
    assert isinstance(region["hierarchy"], dict)

    hier = get_structure_hierarchy(atlas, 2)
    assert hier["ancestors"] == ["ROOT"]
    assert hier["n_descendants"] == 0


def test_structure_mask_slice() -> None:
    atlas = FakeAtlas()

    mask = get_structure_mask_slice(atlas, 2, position_mm=0.1)
    mask_arr = np.asarray(mask)
    assert mask_arr.shape == (4, 4)
    assert int(mask_arr.max()) == 255


def test_atlas_info_projection() -> None:
    atlas = FakeAtlas()

    info = get_atlas_info(atlas)
    assert info["local_version"] == "1.0"
    assert info["is_latest_version"] is True
    assert info["additional_references"] == ["nissl"]
