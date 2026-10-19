"""A deformation record on disk: record.json plus an OME-Zarr 0.6 warp store.

The store's metadata is checked against the OME-NGFF 0.6 JSON schemas
(``tests/fixtures/ngff_0.6``) and its transformations are applied as the
spec defines them, so another program reading the store gets the record's
own mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from langslice.core.deformable import DeformableRecord, Placement, ome_warp
from langslice.core.deformation import RecordStore

SCHEMAS = Path(__file__).parent / "fixtures" / "ngff_0.6"
H, W = 60, 80
MM = 0.02
ATLAS_TO_SECTION = np.array([[1.1, 0.1, 3.0], [0.05, 0.9, -2.0], [0.0, 0.0, 1.0]])
# volume index = M @ [nx, ny, 1]: a coronal plane at slab 200 of an "asr" volume.
NATIVE_TO_VOLUME = [[0.0, 0.0, 200.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def _record(seed: int = 0, parent: DeformableRecord | None = None, *,
            left_out: bool = False, inverse: bool = True) -> DeformableRecord:
    rng = np.random.default_rng(seed)
    placement = Placement("allen_mouse_25um", 5.0, "coronal", 0.0, 0.0, ATLAS_TO_SECTION, MM)
    return DeformableRecord(
        placement=placement, section_size=(W, H), settings=None,
        field_mm=rng.normal(0.0, 0.3, (H, W, 2)).astype(np.float32),
        inverse_field_mm=(rng.normal(0.0, 0.3, (H, W, 2)).astype(np.float32)
                          if inverse else None),
        inverse_source="test", labels=rng.integers(0, 70000, (H, W)).astype(np.uint32),
        footprint=rng.random((H, W)) > 0.5, excluded_ids=[], engine={"name": "test"},
        diagnostics={}, step=0 if parent is None else parent.step + 1, parent=parent,
        atlas={"resolution_um": [25.0, 25.0, 25.0], "orientation": "asr",
               "native_to_volume_index": NATIVE_TO_VOLUME},
        left_out=(rng.random((H, W)) > 0.9) if left_out else None,
    )


def _attributes(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())["attributes"]


def test_a_record_round_trips_at_float16_precision(tmp_path: Path):
    record = _record(left_out=True)
    exact = record.field_mm.copy()
    record.save(tmp_path / "a")
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["record.json", "warp.ome.zarr"]
    loaded = DeformableRecord.load(tmp_path / "a")
    assert loaded.field_mm.dtype == np.float32 and loaded.field_mm.shape == (H, W, 2)
    # float16 rounding: half a unit in the last place, under 0.25 um below 0.5 mm.
    small = np.abs(exact) < 0.5
    assert np.abs(loaded.field_mm - exact)[small].max() <= 0.25e-3
    # The saved record was rounded too, so memory and disk agree exactly.
    assert np.array_equal(loaded.field_mm, record.field_mm)
    assert np.array_equal(loaded.inverse_field_mm, record.inverse_field_mm)
    assert np.array_equal(loaded.labels, record.labels) and loaded.labels.dtype == np.uint32
    assert np.array_equal(loaded.footprint, record.footprint)
    assert loaded.left_out is not None and np.array_equal(loaded.left_out, record.left_out)
    assert loaded.saved_as == "a" and loaded.parent is None


def test_a_record_names_its_parent_and_loads_it_from_the_sibling_folder(tmp_path: Path):
    first = _record(0)
    with pytest.raises(ValueError, match="parent record has no folder"):
        _record(1, parent=first).save(tmp_path / "second")
    first.save(tmp_path / "first")
    second = _record(1, parent=first, inverse=False)
    second.save(tmp_path / "second")
    meta = json.loads((tmp_path / "second" / "record.json").read_text())
    assert meta["parent"] == "first" and meta["version"] == 2
    assert not (tmp_path / "second" / "parent").exists()
    loaded = DeformableRecord.load(tmp_path / "second")
    assert loaded.inverse_field_mm is None and loaded.left_out is None
    assert loaded.parent is not None and loaded.parent.saved_as == "first"
    assert np.array_equal(loaded.parent.field_mm, first.field_mm)
    # A chain that comes back to itself is unreadable, not endless.
    looped = json.loads((tmp_path / "first" / "record.json").read_text())
    (tmp_path / "first" / "record.json").write_text(json.dumps({**looped, "parent": "second"}))
    with pytest.raises(ValueError, match="comes back to it"):
        DeformableRecord.load(tmp_path / "second")
    # A parent that is gone: the error names the parent's record.json.
    (tmp_path / "first" / "record.json").unlink()
    with pytest.raises(ValueError, match=r"first[/\\]record\.json"):
        DeformableRecord.load(tmp_path / "second")


def test_the_store_saves_a_parent_beside_its_child_and_rounds_what_it_holds(tmp_path: Path):
    store = RecordStore(root=tmp_path)
    first, key1 = _record(0), "a" * 64
    exact = first.field_mm.copy()
    store.put(key1, first)  # held in memory, not saved
    assert not np.array_equal(first.field_mm, exact)
    assert np.array_equal(first.field_mm, ome_warp.at_storage_precision(exact))
    second, key2 = _record(1, parent=first), "b" * 64
    store.put(key2, second)
    folder = store.save("s1.tif", key2, second)
    assert folder.name == key2[:24]
    assert (folder.parent / key1[:24] / "record.json").exists()  # the parent, saved first
    store.cache.clear()
    loaded = store.get("s1.tif", key2)
    assert loaded is not None and loaded.parent is not None
    assert np.array_equal(loaded.parent.field_mm, first.field_mm)
    assert np.array_equal(loaded.field_mm, second.field_mm)


def test_an_older_record_format_is_refused(tmp_path: Path):
    _record().save(tmp_path / "a")
    meta = json.loads((tmp_path / "a" / "record.json").read_text())
    meta["version"] = 1
    (tmp_path / "a" / "record.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=r"record\.json: record format 1, not 2"):
        DeformableRecord.load(tmp_path / "a")


def _validator(schema_name: str):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    resources = []
    for path in SCHEMAS.glob("*.schema"):
        schema = json.loads(path.read_text())
        resources.append((schema["$id"], referencing.Resource.from_contents(schema)))
    registry = referencing.Registry().with_resources(resources)
    schema = json.loads((SCHEMAS / schema_name).read_text())
    return jsonschema.Draft202012Validator(schema, registry=registry)


@pytest.mark.parametrize("inverse", [True, False])
def test_the_store_metadata_is_valid_ome_zarr_0_6(tmp_path: Path, inverse: bool):
    _record(left_out=True, inverse=inverse).save(tmp_path / "a")
    store = tmp_path / "a" / ome_warp.WARP_STORE
    root = _attributes(store)
    for name in ("scene.schema", "ome_zarr.schema"):
        _validator(name).validate(root)
    broken = json.loads(json.dumps(root))  # the schemas do refuse: a transform without its output
    del broken["ome"]["scene"]["coordinateTransformations"][1]["output"]
    assert not _validator("scene.schema").is_valid(broken)
    groups = [ome_warp.FIELD_GROUP, *ome_warp.IMAGE_GROUPS.values()]
    if inverse:
        groups.append(ome_warp.INVERSE_GROUP)
    for group in groups:
        attributes = _attributes(store / group)
        for name in ("image.schema", "strict_image.schema", "ome_zarr.schema"):
            _validator(name).validate(attributes)
    # Every coordinate system the transformations name exists where they say.
    scene = root["ome"]["scene"]
    names = {system["name"] for system in scene["coordinateSystems"]}
    for transform in scene["coordinateTransformations"]:
        for end in (transform["input"], transform["output"]):
            if "path" in end:
                multiscale = _attributes(store / end["path"])["ome"]["multiscales"][0]
                assert end["name"] in {s["name"] for s in multiscale["coordinateSystems"]}
            else:
                assert end["name"] in names
    kinds = [t["type"] for t in scene["coordinateTransformations"]]
    assert kinds[:2] == ["bijection" if inverse else "displacements", "sequence"]


def _apply(transform: dict, point: np.ndarray, store: Path) -> np.ndarray:
    """*transform* applied to *point* as OME-NGFF 0.6 defines it (only the
    kinds the store uses; a displacement looked up at a grid point)."""
    kind = transform["type"]
    if kind == "sequence":
        for step in transform["transformations"]:
            point = _apply(step, point, store)
        return point
    if kind == "bijection":
        return _apply(transform["forward"], point, store)
    if kind == "scale":
        return point * np.asarray(transform["scale"])
    if kind == "affine":
        matrix = np.asarray(transform["affine"])
        return matrix[:, :-1] @ point + matrix[:, -1]
    if kind == "displacements":
        group = store / transform["path"]
        scale = _attributes(group)["ome"]["multiscales"][0]["datasets"][0][
            "coordinateTransformations"][0]["scale"]
        import zarr

        array = zarr.open_array(str(group / "s0"), mode="r")[...]
        index = np.rint(point / np.asarray(scale[1:])).astype(int)
        return point + array[:, index[0], index[1]].astype(np.float64)
    raise AssertionError(kind)


def test_the_store_transformations_reproduce_the_records_mapping(tmp_path: Path):
    record = _record()
    record.save(tmp_path / "a")
    store = tmp_path / "a" / ome_warp.WARP_STORE
    warp, placement = _attributes(store)["ome"]["scene"]["coordinateTransformations"][:2]
    inverse = np.linalg.inv(ATLAS_TO_SECTION)
    for row, col in [(0, 0), (17, 33), (H - 1, W - 1)]:
        section = np.array([row, col], dtype=float) * MM  # (y, x) mm
        atlas_um = _apply(placement, _apply(warp, section, store), store)
        # The record's own composition (core.maps.native_points).
        field = record.field_mm[row, col].astype(np.float64)
        native = inverse @ [col + field[0] / MM, row + field[1] / MM, 1.0]
        expected = 25.0 * (np.asarray(NATIVE_TO_VOLUME) @ native)
        assert np.allclose(atlas_um, expected, atol=1e-6)
    # The inverse is the stored inverse field on the placed grid.
    placed = np.array([17, 33], dtype=float) * MM
    back = _apply(warp["inverse"], placed, store)
    assert np.allclose(back - placed, record.inverse_field_mm[17, 33, ::-1], atol=1e-9)
