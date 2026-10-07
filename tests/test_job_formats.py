"""The job folder's public files: ``registration.json`` and each section's maps.

One synthetic stack (the golden recorder's), placed through the library as a
script would: s0 warped by a deformable fit (``ops.deformable.fit_deformable``,
Elastix), s1 turned a quarter, flipped and kept linear, s2 placed without a
transform (the identity, as its pictures show it). Checked: the matrices
agree with what every picture shows (``look`` and ``zoom`` placement
pictures and the deformable-fit pictures ``ops.deformable.fit_deformable``
draws, through ``coordinate_map``), ``labels.tif`` is the atlas read at
``coords.tif``, ``coords = pixel_to_atlas_um @ (p + residual)``, the QuickNII anchoring and
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


def _fit_deformable(handle: Any, ids: list[str], *, long_edge: int | None = None) -> Any:
    """``ops.deformable.fit_deformable`` with one Elastix setting, applied, its
    pictures (a deformable-fit picture, with the residual layer) saved as a
    door saves a tool's pictures (``Job.views.shown``)."""
    from langslice.core import deformation
    from langslice.core.display import default_options
    from langslice.ops.deformable import fit_deformable

    choice = deformation.Choice(fit_section=deformation.FIT_LOOK, fit_atlas="template",
                                engine="elastix", stiffness="medium")
    options = (default_options("overlay") if long_edge is None
               else default_options("overlay", long_edge=long_edge))
    job, workspace = handle.job, handle.workspace
    with job.views.shown("fit_deformable", atlas_of=lambda: workspace.atlas) as shown:
        done = fit_deformable(job, workspace, [job.state.by_id(name) for name in ids],
                              [choice], options=options)
        shown.show(list(done.pictures), arguments={"slices": ids})
    job.views.flush()
    return done


@pytest.fixture(scope="module")
def placed(tmp_path_factory: pytest.TempPathFactory) -> Any:
    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.jobs import create
    from langslice.ops.deformable import keep_linear

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
    assert job.position_sections(sections=[
        {"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
        {"id": ID2, "position_mm": 0.2}], view=False)["status"] == "ok"
    assert job.interactive_transform(sections=[
        {"id": ID0, "rotation_deg": 3.0, "scale_x": 1.05, "scale_y": 0.97,
         "translate_x_mm": 0.1, "translate_y_mm": -0.05, "shear": 0.04},
        {"id": ID1, "flip": True, "rotate_quarter": 90, "rotation_deg": -4.0, "scale_x": 1.0,
         "scale_y": 1.0, "translate_x_mm": 0.0, "translate_y_mm": 0.02}],
        view=False)["status"] == "ok"
    fitted = _fit_deformable(job, [ID0])
    assert fitted.written == [ID0], fitted.rows
    keep_linear(job.job, [job.state.by_id(ID1)], "kept for the test")
    job.look(mode="overlay", sections=[ID1], resolution=700)
    job.look(mode="overlay", sections=[ID2], resolution=300)
    job.zoom(box=[30, 25, 270, 200])
    # A picture smaller than the fit grid, so the residual is resampled.
    again = _fit_deformable(job, [ID0], long_edge=160)
    assert again.rows[0]["status"] == "ok" and not again.written  # the same fit
    exported = job.export_maps()
    job.close()
    yield job, Path(job.folder), exported
    patch.undo()


#: The tools whose saved pictures are placement pictures (one atlas plane).
PLACEMENT_TOOLS = ("look", "zoom")


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
                           ("tissue.png", "tissue"), ("maps.json", "maps")):
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
    # The maps cover the section's whole footprint: every pixel the tissue
    # estimate holds (inside the atlas) has coordinates, and tissue.png is
    # that estimate, for a script to mask with.
    with Image.open(folder / "tissue.png") as opened:
        tissue = np.asarray(opened)
    assert tissue.dtype == np.uint8 and tissue.shape == labels.shape
    assert set(np.unique(tissue)) <= {0, 255} and (tissue == 255).any()
    assert np.isfinite(coords[0][tissue == 255]).mean() > 0.999
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
    from langslice.core.affine import pixel_center_map
    from langslice.core.layers import coordinate_map
    from langslice.core.maps import orientation_matrix, unturned

    job, root, _exported = placed
    registration = json.loads((root / "registration.json").read_text())
    by_id = {entry["id"]: entry for entry in registration["sections"]}
    checked = 0
    for folder in [path for tool in PLACEMENT_TOOLS for path in _views(root, tool)]:
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
    """A deformable-fit picture's coordinate map (its residual layer
    included) and the section's own composed map agree, up to resampling
    the field at the picture's size."""
    from langslice.core.affine import pixel_center_map
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
    from langslice.job.quint import atlas_um_to_quicknii_points

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
    from langslice.job.quint import atlas_um_to_quicknii_points

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
    tools["position_sections"](sections=[{"id": ID2, "position_mm": 0.25}], view=False)
    opened.close()
    assert dry["files_written"] is False and dry["files"]
    assert path.stat().st_mtime_ns == stamp
    # A write rewrites it; the maps written before are no longer current.
    job = langslice.open_job(str(images), atlas_loader=atlas_loader())
    job.position_sections(sections=[{"id": ID2, "position_mm": 0.25}], view=False)
    job.close()
    after = json.loads(path.read_text())
    moved = after["sections"][2]
    assert moved["parameters"]["plane"]["position_mm"] == 0.25
    # A flat plane sits on its nearest atlas slab (50 um here), as its pictures do.
    assert moved["pixel_to_atlas_um"][0][2] == 250.0 != entry["pixel_to_atlas_um"][0][2]
    assert moved["maps"]["current"] is False
    assert after["sections"][0]["maps"]["current"] is True


def test_a_starting_position_is_said_in_registration_json_and_the_maps(tmp_path, monkeypatch):
    """A section still at the starting position the job gave it is
    ``parameters.plane.starting_position`` true in registration.json, and maps
    written while it sits there say ``"starting_position": true``. Writing its
    position, even the very same value, makes it the writer's own: both go."""
    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.jobs import create

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    folder = tmp_path / "stack"
    write_sections(folder)
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"),
                               agent_preprocessing=False)
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    root = Path(job.folder)

    def planes() -> dict[str, dict[str, Any]]:
        return {entry["id"]: entry["parameters"]["plane"] for entry in json.loads(
            (root / "registration.json").read_text())["sections"]}

    def maps(stem: str) -> dict[str, Any]:
        return json.loads((root / "sections" / stem / "maps.json").read_text())

    status = {row["id"]: row for row in job.status()["rows"]}
    assert all(row["position_source"] == "default" for row in status.values())
    assert all(plane["starting_position"] is True for plane in planes().values())
    exported = job.export_maps(slices=[ID0, ID1])
    assert exported["written"] == [ID0, ID1]
    assert maps("s0")["starting_position"] is True and maps("s1")["starting_position"] is True

    start = status[ID0]["position_mm"]
    assert job.position_sections(sections=[{"id": ID0, "position_mm": start}],
                                 view=False)["status"] == "ok"
    held = planes()
    assert held[ID0]["starting_position"] is False and held[ID0]["position_mm"] == start
    assert held[ID1]["starting_position"] is True
    job.export_maps(slices=[ID0, ID1])
    job.close()
    assert "starting_position" not in maps("s0")
    assert maps("s1")["starting_position"] is True


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
    from langslice.core.image_prep import (
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


def test_the_footprint_keeps_dark_tissue_and_tears_inside_the_outline():
    """A dark band inside the section, open to the slide through a narrow
    tear, is still the section: the footprint covers it; the tissue
    estimate does not."""
    from types import SimpleNamespace

    import cv2

    from langslice.core.maps import section_footprint

    image = np.full((400, 600, 3), 10, dtype=np.uint8)
    cv2.ellipse(image, (300, 200), (260, 170), 0, 0, 360, (170, 170, 170), -1)
    cv2.ellipse(image, (300, 200), (120, 40), 0, 0, 360, (12, 12, 12), -1)  # dim band
    cv2.line(image, (300, 240), (300, 399), (10, 10, 10), 3)  # a tear to the edge
    workspace = SimpleNamespace(working_source=lambda _id: (Image.fromarray(image), 1.0))
    frame = SimpleNamespace(section_id="x", file_um_per_px=10.0, working_factor=1.0)
    footprint, tissue, found = section_footprint(workspace, frame, (600, 400))  # type: ignore[arg-type]
    assert found
    assert not tissue[200, 300] and footprint[200, 300]      # the dark band
    assert not tissue[300, 300] and footprint[300, 300]      # the tear
    assert not footprint[5, 5] and not footprint[395, 590]   # the slide


# --- sections without a pixel size -------------


def _uncalibrated_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The golden stack with no pixel size anywhere (no host input, the PNGs
    carry none), on an atlas whose anatomy is narrower at AP index 5, so
    the tissue-width estimate differs between 0.10 and 0.25 mm."""
    import langslice
    from langslice.core.spec import NonlinearSpec
    from langslice.doors.jobs import create

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    folder = tmp_path / "stack"
    write_sections(folder)
    loader = atlas_loader()
    loader("x").annotation[5][:, 10:50] = 0
    spec = dataclasses.replace(full_spec(folder), nonlinear=NonlinearSpec(provider="none"),
                               agent_preprocessing=False, inputs={})
    create(spec, atlas_loader=loader).close()
    return langslice.open_job(str(folder), atlas_loader=loader)


def _picture_um_per_section_px(root: Path, section: str) -> float:
    """Micrometres per section-render pixel (along a column step) in the
    last placement picture of *section*."""
    entries = [json.loads(line) for line in (root / "views.jsonl").read_text().splitlines()]
    view = [e for e in entries if e["tool"] == "look" and section in e["sections"]][-1]
    frame = json.loads((root / view["path"] / "view.json").read_text())["frame"]
    to_atlas = np.asarray(frame["pixel_to_atlas_um"]) @ np.asarray(frame["section_to_picture"])
    return float(np.linalg.norm(to_atlas[:, 1]))


def test_an_uncalibrated_section_maps_at_the_scale_its_pictures_draw(tmp_path, monkeypatch):
    """No pixel size: the pictures estimate one from the tissue width at the
    section's CURRENT position; registration.json (and so the maps and the
    exports, the same frame) uses that same scale, not the one stored with
    a transform written at another position."""
    job = _uncalibrated_job(tmp_path, monkeypatch)
    job.position_sections(sections=[{"id": ID0, "position_mm": 0.1}], view=False)
    job.interactive_transform(sections=[{"id": ID0, "rotation_deg": 0.0, "scale_x": 1.0,
                                         "scale_y": 1.0, "translate_x_mm": 0.0,
                                         "translate_y_mm": 0.0}], view=False)
    job.position_sections(sections=[{"id": ID0, "position_mm": 0.25}], view=False)
    job.look(mode="overlay", sections=[ID0], resolution=512)
    job.close()
    root = Path(job.folder)
    entry = {e["id"]: e for e in json.loads(
        (root / "registration.json").read_text())["sections"]}[ID0]
    render = prepared_render_size(root.parent / ID0)
    file_size = entry["image"]["size"]
    reg_um_per_render_px = (np.linalg.norm(np.asarray(entry["pixel_to_atlas_um"])[:, 1])
                            * file_size[0] / render[0])
    picture = _picture_um_per_section_px(root, ID0)
    assert abs(reg_um_per_render_px - picture) < 0.01, (reg_um_per_render_px, picture)
    assert entry["image"]["pixel_size_source"] == "estimated"
    assert "pixel size" in (entry["problem"] or "")


def test_an_uncalibrated_section_is_fitted_at_the_scale_its_pictures_draw(tmp_path,
                                                                           monkeypatch):
    """No pixel size, a transform written at one position, the section then
    moved: the nonlinear start (the trace's canvas, every deformable fit's
    grid, the Elastix affine's start) places the atlas at the scale the
    pictures and registration.json use, not the one stored with the
    transform, so the six numbers mean one placement everywhere."""
    from langslice.core.handoff import prepare_linear_registration
    from langslice.core.sections import PREVIEW_LONG_EDGE

    job = _uncalibrated_job(tmp_path, monkeypatch)
    job.position_sections(sections=[{"id": ID0, "position_mm": 0.1}], view=False)
    job.interactive_transform(sections=[{"id": ID0, "rotation_deg": 0.0, "scale_x": 1.0,
                                         "scale_y": 1.0, "translate_x_mm": 0.0,
                                         "translate_y_mm": 0.0}], view=False)
    stored = job.state.by_id(ID0).transform["calibration"]["section_um_per_px"]
    job.position_sections(sections=[{"id": ID0, "position_mm": 0.25}], view=False)
    job.look(mode="overlay", sections=[ID0], resolution=512)
    handoff = prepare_linear_registration(job.state, job.workspace, ID0,
                                          long_edge=PREVIEW_LONG_EDGE).metadata
    job.close()
    picture = _picture_um_per_section_px(Path(job.folder), ID0)
    assert abs(stored - picture) > 0.1, (stored, picture)  # the scale did change
    assert abs(handoff["section_um_per_px"] - picture) < 0.01, (handoff, picture)
    assert handoff["calibration_source"] == "estimated"


def prepared_render_size(path: Path) -> tuple[int, int]:
    from langslice.core.image_prep import prepared_size
    from langslice.core.sections import PREVIEW_LONG_EDGE

    with Image.open(path) as image:
        return prepared_size(image.size, max_long_edge=PREVIEW_LONG_EDGE)


def test_an_uncalibrated_section_without_a_transform_is_the_identity(tmp_path, monkeypatch,
                                                                     caplog):
    """No pixel size and no transform: the identity at the estimated scale,
    as its pictures draw it, the scale stated as unknown; no traceback per
    section per checkpoint."""
    import logging

    job = _uncalibrated_job(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        job.position_sections(sections=[{"id": ID2, "position_mm": 0.2}], view=False)
        job.look(mode="overlay", sections=[ID2], resolution=512)
    job.close()
    assert not [r for r in caplog.records if r.exc_info], [r.getMessage() for r in caplog.records]
    root = Path(job.folder)
    entry = {e["id"]: e for e in json.loads(
        (root / "registration.json").read_text())["sections"]}[ID2]
    assert entry["pixel_to_atlas_um"] is not None
    assert entry["mapping"].startswith("linear (identity")
    assert "pixel size" in entry["problem"] and "unknown" in entry["problem"]
    render = prepared_render_size(root.parent / ID2)
    reg = (np.linalg.norm(np.asarray(entry["pixel_to_atlas_um"])[:, 1])
           * entry["image"]["size"][0] / render[0])
    assert abs(reg - _picture_um_per_section_px(root, ID2)) < 0.01


def test_the_frame_cache_is_bounded(placed, monkeypatch):
    """A section's frame is memoized under everything it depends on; moving
    it many times keeps one frame per section, and the cache never holds
    more than ``FRAME_CACHE_SECTIONS`` sections (review finding 9)."""
    import dataclasses as dc

    from langslice.core import maps

    job, _root, _exported = placed
    workspace = job.workspace
    record = job.state.by_id(ID2)
    workspace.frame_cache.clear()
    for step in range(40):
        moved = dc.replace(record, position_mm=0.002 * step)
        maps.section_frame(job.state, workspace, moved)
    assert len(workspace.frame_cache) == 1
    # The current key is a hit: the same frame object comes back.
    first = maps.section_frame(job.state, workspace, moved)
    assert maps.section_frame(job.state, workspace, moved) is first
    monkeypatch.setattr(maps, "FRAME_CACHE_SECTIONS", 2)
    for name in (ID0, ID1, ID2):
        maps.section_frame(job.state, workspace, job.state.by_id(name))
    assert len(workspace.frame_cache) == 2


# --- export_maps beside other writers-----------------


def _two_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placed: Any) -> tuple[Any, Any]:
    import shutil

    import langslice

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _job, root, _exported = placed
    images = tmp_path / "stack"
    shutil.copytree(root.parent, images)
    first = langslice.open_job(str(images), atlas_loader=atlas_loader())
    second = langslice.open_job(str(images), atlas_loader=atlas_loader())
    return first, second


def test_export_maps_syncs_before_it_writes(placed, tmp_path, monkeypatch):
    """The operation itself (not only the doors around it) reloads what
    another writer saved before it writes registration.json and the maps."""
    from langslice.ops.exports import export_maps

    first, second = _two_jobs(tmp_path, monkeypatch, placed)
    first.status()  # first holds the state as it was
    second.position_sections(sections=[{"id": ID2, "position_mm": 0.25}], view=False)
    done = export_maps(first.job, first.workspace, [ID2])
    first.close()
    second.close()
    assert done.sections == [ID2]
    root = Path(first.folder)
    entry = {e["id"]: e for e in json.loads(
        (root / "registration.json").read_text())["sections"]}[ID2]
    assert entry["parameters"]["plane"]["position_mm"] == 0.25
    assert entry["maps"]["current"] is True


def test_export_maps_computes_outside_the_lock_and_refuses_a_moved_section(
    placed, tmp_path, monkeypatch,
):
    """The heavy maps are computed outside the job lock (another writer
    gets it meanwhile); a section that writer changed is not written
    (``STALE_INPUT``), the others are."""
    from langslice.core import maps as core_maps
    from langslice.job.lock import FolderLock
    from langslice.ops.exports import export_maps

    first, second = _two_jobs(tmp_path, monkeypatch, placed)
    real = core_maps.section_maps
    seen: list[bool] = []

    def meanwhile(workspace: Any, frame: Any, warp: Any, **kwargs: Any) -> Any:
        # Another process could take the lock now; this thread does not hold it.
        other = FolderLock(Path(first.folder), timeout=0.5)
        with other.held():
            seen.append(True)
        if frame.section_id == ID1 and len(seen) == 2:
            second.position_sections(sections=[{"id": ID1, "position_mm": 0.2}], view=False)
        return real(workspace, frame, warp, **kwargs)

    monkeypatch.setattr(core_maps, "section_maps", meanwhile)
    stamp = (Path(first.folder) / "sections" / "s1" / "maps.json").read_text()
    done = export_maps(first.job, first.workspace, [ID0, ID1, ID2])
    first.close()
    second.close()
    assert seen == [True, True, True]
    assert done.sections == [ID0, ID2]
    assert [row["id"] for row in done.skipped] == [ID1]
    assert "STALE_INPUT" in done.skipped[0]["reason"]
    assert (Path(first.folder) / "sections" / "s1" / "maps.json").read_text() == stamp
