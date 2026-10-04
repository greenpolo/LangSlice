"""The job folder's public files: ``registration.json`` and each section's maps.

One synthetic stack (the golden recorder's), placed through the library as a
script would: s0 warped by a deformable fit, s1 turned a quarter, flipped
and kept linear, s2 placed without a transform (the identity, as its
pictures show it). Checked: the matrices agree with what every picture
shows (placement pictures and ``fit_deformable`` pictures, through
``coordinate_map``), ``labels.tif`` is the atlas read at ``coords.tif``,
``coords = pixel_to_atlas_um @ (p + residual)``, the QuickNII anchoring and
VisuAlign markers reproduce the maps, the files are what
``docs/file_formats.md`` says, and a job that persists nothing writes none.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
from PIL import Image

from tests.golden.record import (
    ID0,
    ID1,
    ID2,
    apply_patches,
    atlas_loader,
    full_spec,
    write_sections,
)


@pytest.fixture(scope="module")
def placed(tmp_path_factory: pytest.TempPathFactory) -> Any:
    import langslice
    from langslice.doors.jobs import create
    from langslice.linear.spec import NonlinearSpec

    root = tmp_path_factory.mktemp("formats")
    patch = pytest.MonkeyPatch()
    patch.setenv("HOME", str(root / "home"))
    apply_patches()
    folder = root / "stack"
    write_sections(folder)
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"),
                               agent_preprocessing=False)
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    job.set_positions(entries=[{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
                               {"id": ID2, "position_mm": 0.2}])
    job.orient_slices(entries=[{"id": ID1, "flip": True, "rotate_deg": 90}])
    job.adjust_transforms(entries=[
        {"id": ID0, "rotation_deg": 3.0, "scale_x": 1.05, "scale_y": 0.97,
         "translate_x_mm": 0.1, "translate_y_mm": -0.05, "shear": 0.04},
        {"id": ID1, "rotation_deg": -4.0, "scale_x": 1.0, "scale_y": 1.0,
         "translate_x_mm": 0.0, "translate_y_mm": 0.02}])
    assert job.fit_deformable(slices=[ID0], engine="elastix")["status"] == "ok"
    job.fit_deformable(slices=[ID1], keep_linear="kept for the test")
    job.view_placement(entries=[{"id": ID1, "positions_mm": [0.15]}],
                       view={"mode": "overlay", "resolution": 700})
    job.view_placement(entries=[{"id": ID2, "positions_mm": [0.2]}],
                       view={"mode": "overlay", "resolution": 300, "zoom": [0.1, 0.1, 0.9, 0.8]})
    # A picture smaller than the fit grid, so the residual is resampled.
    job.fit_deformable(slices=[ID0], engine="elastix", view={"resolution": 160})
    exported = job.export_maps()
    job.close()
    yield job, Path(job.folder), exported
    patch.undo()


def _views(root: Path, tool: str) -> list[Path]:
    entries = [json.loads(line) for line in (root / "views.jsonl").read_text().splitlines()]
    return [root / entry["path"] for entry in entries if entry["tool"] == tool and entry["layers"]]


def _content_pixels(folder: Path, count: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    frame = json.loads((folder / "view.json").read_text())["frame"]
    x0, y0, x1, y1 = frame["content_box"]
    labels = tifffile.imread(folder / "labels.tif")
    rows, cols = np.nonzero(labels[y0:y1, x0:x1] > 0)
    pick = np.linspace(0, len(rows) - 1, min(count, len(rows))).astype(int)
    return rows[pick] + y0, cols[pick] + x0


def test_export_writes_every_file_with_its_kind(placed):
    _job, root, exported = placed
    assert exported["status"] == "ok" and exported["skipped"] == []
    assert exported["written"] == [ID0, ID1, ID2]
    kinds = {(Path(item["path"]).relative_to(root).as_posix(), item["kind"])
             for item in exported["files"]}
    for stem in ("s0", "s1", "s2"):
        for name, kind in (("coords.tif", "coords"), ("labels.tif", "labels"),
                           ("labels_fiji.tif", "labels_fiji"), ("labels.csv", "labels_csv"),
                           ("maps.json", "maps")):
            assert (f"sections/{stem}/{name}", kind) in kinds
    # Only the warped section has a residual.
    assert ("sections/s0/residual.tif", "residual") in kinds
    assert not (root / "sections/s1/residual.tif").exists()
    assert {("exports/quicknii.json", "quicknii"), ("exports/visualign.json", "visualign"),
            ("registration.json", "registration")} <= kinds


def test_the_map_files_are_what_the_format_says(placed):
    job, root, _exported = placed
    folder = root / "sections" / "s0"
    with tifffile.TiffFile(folder / "coords.tif") as handle:
        assert handle.is_imagej and handle.imagej_metadata["unit"] == "um"
        coords = handle.asarray()
    labels = tifffile.imread(folder / "labels.tif")
    dense = tifffile.imread(folder / "labels_fiji.tif")
    residual = tifffile.imread(folder / "residual.tif")
    working = job.workspace.working_source(ID0)[0].size
    assert coords.dtype == np.float32 and coords.shape == (3, working[1], working[0])
    assert labels.dtype == np.uint32 and labels.shape == coords.shape[1:]
    assert dense.dtype == np.uint16 and residual.dtype == np.float32
    assert residual.shape == (2, *coords.shape[1:])
    outside = ~np.isfinite(coords[0])
    assert outside.any() and (~outside).any()
    assert (labels[outside] == 0).all()
    rows = list(csv.DictReader((folder / "labels.csv").open()))
    ids = {int(row["index"]): int(row["id"]) for row in rows}
    assert set(np.unique(dense)) - {0} == set(ids)
    restored = np.zeros_like(labels)
    for index, uid in ids.items():
        restored[dense == index] = uid
    assert (restored == labels).all()
    assert {row["acronym"] for row in rows} <= {"CTX", "STR", "TH", "HY", "VL"}
    maps = json.loads((folder / "maps.json").read_text())
    assert maps["grid_size"] == list(working) and maps["full_resolution"] is False


def test_labels_are_the_atlas_read_at_the_coordinates(placed):
    job, root, _exported = placed
    atlas = job.workspace.atlas
    resolution = np.asarray(atlas.resolution, dtype=np.float64)
    for stem in ("s0", "s1", "s2"):
        coords = tifffile.imread(root / "sections" / stem / "coords.tif")
        labels = tifffile.imread(root / "sections" / stem / "labels.tif")
        tissue = np.isfinite(coords[0])
        index = np.rint(coords[:, tissue] / resolution[:, None]).astype(int)
        for axis in range(3):
            index[axis] = np.clip(index[axis], 0, atlas.annotation.shape[axis] - 1)
        looked_up = atlas.annotation[index[0], index[1], index[2]]
        assert tissue.sum() > 10_000
        assert np.mean(looked_up == labels[tissue]) > 0.99, stem


def test_coordinates_are_the_matrix_after_the_residual(placed):
    _job, root, _exported = placed
    folder = root / "sections" / "s0"
    coords = tifffile.imread(folder / "coords.tif")
    residual = tifffile.imread(folder / "residual.tif").astype(np.float64)
    matrix = np.asarray(json.loads((folder / "maps.json").read_text())["pixel_to_atlas_um"])
    rows, cols = np.mgrid[0:coords.shape[1], 0:coords.shape[2]].astype(np.float64)
    composed = np.tensordot(matrix, np.stack([rows + residual[0], cols + residual[1],
                                              np.ones_like(rows)]), axes=1)
    tissue = np.isfinite(coords[0])
    assert np.abs(composed[:, tissue] - coords[:, tissue]).max() < 0.01
    # The residual is a real deformation, not zero.
    assert np.abs(residual).max() > 0.5


def test_registration_matrix_agrees_with_the_placement_pictures(placed):
    """A placement picture's coordinate map and registration.json's file
    matrix name the same atlas point for the same section point, including a
    turned and flipped section and a zoomed picture."""
    from langslice.affine import pixel_center_map
    from langslice.core.layers import coordinate_map
    from langslice.core.maps import orientation_matrix, unturned

    job, root, _exported = placed
    registration = json.loads((root / "registration.json").read_text())
    by_id = {entry["id"]: entry for entry in registration["sections"]}
    checked = 0
    for folder in _views(root, "view_placement"):
        view = json.loads((folder / "view.json").read_text())
        section = view["sections"][0]
        record = job.state.by_id(section)
        frame = view["frame"]
        entry = by_id[section]
        rows, cols = _content_pixels(folder)
        section_px = np.linalg.inv(np.asarray(frame["section_to_picture"])) @ np.stack(
            [rows, cols, np.ones_like(rows)]).astype(np.float64)
        render = tuple(frame["section"]["render_size"])
        flat = unturned(render, record.rotation_deg)
        file_to_render = (orientation_matrix(flat, record.rotation_deg, record.flip)
                          @ pixel_center_map(tuple(entry["image"]["size"]), flat))
        fx, fy = (np.linalg.inv(file_to_render)
                  @ np.stack([section_px[1], section_px[0], np.ones_like(rows, float)]))[:2]
        expected = np.asarray(entry["pixel_to_atlas_um"]) @ np.stack([fy, fx, np.ones_like(fx)])
        got = coordinate_map(folder / "view.json")[rows, cols].T
        assert np.abs(got - expected).max() < 0.05, folder.name
        checked += 1
    assert checked >= 2


def test_residual_maps_agree_with_the_fit_deformable_pictures(placed):
    """A fit_deformable picture's coordinate map (its residual layer
    included) and the section's own composed map agree, up to resampling
    the field at the picture's size."""
    from langslice.affine import pixel_center_map
    from langslice.core.layers import coordinate_map
    from langslice.core.maps import native_points, orientation_matrix, section_frame, unturned

    job, root, _exported = placed
    record = job.state.by_id(ID0)
    warp = job.job.deformations.current(job.state, record)
    frame = section_frame(job.state, job.workspace, record)
    folders = _views(root, "fit_deformable")
    assert folders and all((folder / "residual.tif").exists() for folder in folders)
    for folder in folders:
        picture = json.loads((folder / "view.json").read_text())["frame"]
        x0, band, x1, y1 = picture["content_box"]
        rows, cols = _content_pixels(folder)
        grid = pixel_center_map((x1 - x0, y1 - band), tuple(warp.section_size))
        gx, gy = (grid @ np.stack([cols - x0, rows - band, np.ones_like(rows)]).astype(float))[:2]
        flat = unturned(tuple(warp.section_size), record.rotation_deg)
        to_fit = (orientation_matrix(flat, record.rotation_deg, record.flip)
                  @ pixel_center_map(frame.file_size, flat))
        fx, fy = (np.linalg.inv(to_fit) @ np.stack([gx, gy, np.ones_like(gx)]))[:2]
        nx, ny = native_points(frame, warp, fx[None], fy[None])
        expected = frame.native_to_um @ np.stack([ny[0], nx[0], np.ones_like(nx[0])])
        got = coordinate_map(folder / "view.json")[rows, cols].T
        # Under 1 um on a 31 um/px section: two bilinear samplings of one field.
        assert np.abs(got - expected).max() < 1.0, folder.name


def test_a_picture_record_without_its_folder_cannot_be_mapped(placed):
    from langslice.core.layers import coordinate_map

    _job, root, _exported = placed
    folder = _views(root, "fit_deformable")[0]
    record = json.loads((folder / "view.json").read_text())
    with pytest.raises(ValueError, match="residual"):
        coordinate_map(record)
    assert coordinate_map(record, folder=folder).shape[:2] == tuple(
        record["frame"]["picture_size"][::-1])


def test_quicknii_anchoring_and_visualign_markers_reproduce_the_maps(placed):
    from langslice.core.layers import atlas_facts
    from langslice.integrations.quint import atlas_um_to_quicknii_points

    job, root, _exported = placed
    facts = atlas_facts(job.workspace.atlas)
    registration = json.loads((root / "registration.json").read_text())
    by_id = {entry["id"]: entry for entry in registration["sections"]}
    for name in ("quicknii.json", "visualign.json"):
        export = json.loads((root / "exports" / name).read_text())
        assert [item["filename"] for item in export["slices"]] == [ID0, ID1, ID2]
        for item in export["slices"]:
            entry = by_id[item["filename"]]
            assert [item["width"], item["height"]] == entry["image"]["size"]
            anchoring = np.asarray(item["anchoring"])
            matrix = np.asarray(entry["pixel_to_atlas_um"])
            # Pixel centres (row r, col c) sit at QuickNII fractions ((c + .5)/W, (r + .5)/H).
            r, c = np.array([3.0, 100.0, 150.0]), np.array([5.0, 40.0, 200.0])
            width, height = item["width"], item["height"]
            quicknii = (anchoring[:3] + ((c + 0.5) / width)[:, None] * anchoring[3:6]
                        + ((r + 0.5) / height)[:, None] * anchoring[6:9])
            expected = atlas_um_to_quicknii_points(
                (matrix @ np.stack([r, c, np.ones_like(r)])).T, facts)
            assert np.abs(quicknii - expected).max() < 1e-4
            if name == "quicknii.json" or item["filename"] != ID0:
                assert item["markers"] == []
    # The markers of the warped section: each destination's own map is the
    # anchoring at its source.
    export = json.loads((root / "exports" / "visualign.json").read_text())
    item = export["slices"][0]
    markers = np.asarray(item["markers"])
    assert len(markers) > 20
    record = job.state.by_id(ID0)
    from langslice.core.maps import native_points, section_frame

    frame = section_frame(job.state, job.workspace, record)
    warp = job.job.deformations.current(job.state, record)
    nx, ny = native_points(frame, warp, markers[None, :, 2] - 0.5, markers[None, :, 3] - 0.5)
    composed = frame.native_to_um @ np.stack([ny[0], nx[0], np.ones_like(nx[0])])
    anchored = np.asarray(by_id[ID0]["pixel_to_atlas_um"]) @ np.stack(
        [markers[:, 1] - 0.5, markers[:, 0] - 0.5, np.ones(len(markers))])
    assert np.abs(composed - anchored).max() < 1e-3
    assert np.abs(markers[:, :2] - markers[:, 2:]).max() > 0.5  # a real deformation


def test_quicknii_axes_reverse_brainglobe_asr():
    """BrainGlobe asr voxel centre 0 is QuickNII's far edge minus half a voxel."""
    from langslice.integrations.quint import atlas_um_to_quicknii_points

    atlas = {"orientation": "asr", "shape": [528, 320, 456], "resolution_um": [25.0] * 3}
    point = atlas_um_to_quicknii_points([[0.0, 0.0, 0.0]], atlas)[0]
    # QuickNII (x ML, y AP, z DV): every axis reversed against asr.
    assert np.allclose(point, [456 - 0.5, 528 - 0.5, 320 - 0.5])


def test_registration_json_is_rewritten_on_every_write_and_never_by_a_dry_run(placed, tmp_path,
                                                                             monkeypatch):
    import shutil

    import langslice
    from langslice.doors.jobs import open_folder

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _job, root, _exported = placed
    images = tmp_path / "stack"
    shutil.copytree(root.parent, images)
    path = images / "langslice" / "registration.json"
    before = json.loads(path.read_text())
    assert before["format_version"] == 1 and before["atlas"]["orientation"] == "asr"
    entry = before["sections"][2]
    assert entry["mapping"].startswith("linear (identity")
    assert entry["maps"]["current"] is True
    # A dry run writes nothing.
    stamp = path.stat().st_mtime_ns
    opened = open_folder(images, atlas_loader=atlas_loader(), persist=False)
    tools = {tool.__name__: tool for tool in opened.tools().tools}
    dry = tools["export_maps"]()
    tools["set_positions"](entries=[{"id": ID2, "position_mm": 0.25}])
    opened.close()
    assert dry["files_written"] is False and dry["files"]
    assert path.stat().st_mtime_ns == stamp
    # A write rewrites it; the maps written before are no longer current.
    job = langslice.open_job(str(images), atlas_loader=atlas_loader())
    job.set_positions(entries=[{"id": ID2, "position_mm": 0.25}])
    job.close()
    after = json.loads(path.read_text())
    moved = after["sections"][2]
    assert moved["parameters"]["plane"]["position_mm"] == 0.25
    # A flat plane sits on its nearest atlas slab (50 um here), as its pictures do.
    assert moved["pixel_to_atlas_um"][0][2] == 250.0 != entry["pixel_to_atlas_um"][0][2]
    assert moved["maps"]["current"] is False
    assert after["sections"][0]["maps"]["current"] is True


def test_full_resolution_maps_are_on_the_file_pixels(placed, tmp_path, monkeypatch):
    import shutil

    import langslice

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _job, root, _exported = placed
    images = tmp_path / "stack"
    shutil.copytree(root.parent, images)
    job = langslice.open_job(str(images), atlas_loader=atlas_loader())
    reply = job.export_maps(slices=[ID1], full_resolution=True)
    job.close()
    assert reply["written"] == [ID1] and reply["full_resolution"] is True
    with Image.open(images / ID1) as image:
        size = image.size
    coords = tifffile.imread(images / "langslice" / "sections" / "s1" / "coords.tif")
    assert coords.shape == (3, size[1], size[0])


def test_working_size_reads_the_header_as_the_decoder_would(tmp_path):
    from langslice.image_prep import (
        read_working_image,
        read_working_pages,
        working_size,
    )

    rng = np.random.default_rng(0)
    big = (rng.random((3300, 4100, 3)) * 255).astype(np.uint8)
    Image.fromarray(big).save(tmp_path / "big.png")
    Image.fromarray(big).save(tmp_path / "big.jpg", quality=80)
    Image.fromarray(big[:900, :1200]).save(tmp_path / "small.png")
    with tifffile.TiffWriter(tmp_path / "pyramid.tif") as writer:
        writer.write(big, subifds=2, tile=(256, 256))
        writer.write(big[::2, ::2], subfiletype=1, tile=(256, 256))
        writer.write(big[::4, ::4], subfiletype=1, tile=(256, 256))
    pages = (rng.random((3, 700, 900)) * 255).astype(np.uint8)
    tifffile.imwrite(tmp_path / "pages.tif", pages)
    for name in ("big.png", "big.jpg", "small.png", "pyramid.tif"):
        image, factor = read_working_image(tmp_path / name)
        assert working_size(tmp_path / name) == (image.size, factor), name
    decoded, factor = read_working_pages(tmp_path / "pages.tif")
    assert working_size(tmp_path / "pages.tif", pages=True) == (
        (decoded[0].shape[1], decoded[0].shape[0]), factor)


@pytest.mark.parametrize(("rotation", "flip"), [(0, False), (90, False), (180, True),
                                                (270, True), (90, True)])
def test_orientation_matrix_is_the_render_order(rotation, flip):
    from langslice.core.maps import apply, orientation_matrix

    ops = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
           270: Image.Transpose.ROTATE_270}
    width, height = 7, 4
    image = Image.fromarray(np.arange(width * height, dtype=np.uint8).reshape(height, width))
    if rotation:
        image = image.transpose(ops[rotation])
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    turned = np.asarray(image)
    matrix = orientation_matrix((width, height), rotation, flip)
    ys, xs = np.mgrid[0:height, 0:width].astype(float)
    tx, ty = apply(matrix, xs, ys)
    assert (turned[ty.astype(int), tx.astype(int)]
            == np.arange(width * height).reshape(height, width)).all()
