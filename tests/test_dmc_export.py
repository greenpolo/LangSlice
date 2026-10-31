"""The DMC-BrainMap export (``job/dmc_brainmap.py``, ``tools/dmc_export.py``).

DMC-BrainMap itself is not imported: its reader is replicated here from its
source (napari-dmc-brainmap 0.2.0b2: ``results_creator._scale_positions``,
``SliceHandle.getTransform`` / ``get_z_plane`` / ``getVolumeIndex``,
``utils/atlas_utils``), and the conversion is read back through it on
synthetic linear maps cut at every plane, flat, at small angles and at
large ones of either sign. A synthetic job is then exported whole.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from langslice.job.dmc_brainmap import (
    DmcFrame,
    DmcPlacement,
    dmc_decimals,
    dmc_frame,
    dmc_placement,
    natural_key,
)
from tests.synthetic_stack import (
    ID0,
    ID1,
    ID2,
    atlas_loader,
    full_spec,
    write_sections,
)

ALLEN_25 = ("allen_mouse_25um", "asr", (528, 320, 456), (25.0, 25.0, 25.0))
RAT_39 = ("whs_sd_rat_39um", "asr", (1024, 512, 512), (39.0, 39.0, 39.0))
ALLEN_10 = ("allen_mouse_10um", "asr", (1320, 800, 1140), (10.0, 10.0, 10.0))


# --- DMC-BrainMap's reader, replicated ---------------------------------------


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The projective transform from *src* to *dst* (DLT, as skimage's
    ``ProjectiveTransform.from_estimate`` fits it on float32 points)."""
    src = src.astype(np.float32).astype(np.float64)
    dst = dst.astype(np.float32).astype(np.float64)
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _u, _s, vt = np.linalg.svd(np.asarray(rows))
    matrix = vt[-1].reshape(3, 3)
    return matrix / matrix[2, 2]


def _z_plane(
    location: list[float], frame: DmcFrame, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """``SliceHandle.get_z_plane`` at atlas-plane pixels (x, y)."""
    x_angle, y_angle, z_mm = location
    size_x, size_y = frame.x.size, frame.y.size
    z = round(-z_mm / (frame.z.resolution_um / 1000)) + frame.z_bregma
    if x_angle == 0 and y_angle == 0:
        return np.full(x.shape, float(z))
    x_shift = int(np.tan(np.deg2rad(x_angle)) * size_x / 2)
    y_shift = int(np.tan(np.deg2rad(y_angle)) * size_y / 2)
    center = np.array([z, size_y / 2, size_x / 2])
    right = np.array([z + x_shift, size_y / 2, size_x - 1])
    top = np.array([z - y_shift, 0, size_x / 2])
    n = np.cross(right - center, top - center)
    return (-n[1] * (y - center[1]) - n[2] * (x - center[2])) / n[0] + center[
        0
    ]


def _dmc_read(
    placement: DmcPlacement,
    frame: DmcFrame,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """DMC's atlas point for resized-image points *sample* ``(n, 2)``: the
    continuous plane point ``(x, y)`` and the voxel ``(x, y, z)``."""
    matrix = _homography(
        np.asarray(placement.sample_dots), np.asarray(placement.atlas_dots)
    )
    points = matrix @ np.column_stack([sample, np.ones(len(sample))]).T
    post = (points[:2] / points[2]).T
    x = post[:, 0].astype(int)
    y = post[:, 1].astype(int)
    z = _z_plane(
        placement.location(), frame, x.astype(float), y.astype(float)
    ).astype(int)
    return post, np.column_stack([x, y, z])


def _scale_positions(
    positions: np.ndarray, source_max: int, target_max: int
) -> np.ndarray:
    """``results_creator._scale_positions``: ``[0, source_max]`` onto
    ``[0, target_max]``, rounded up."""
    return np.ceil(
        np.asarray(positions, dtype=float) * target_max / source_max
    ).astype(int)


# --- synthetic linear maps ---------------------------------------------------


def _rotation(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [k for k in range(3) if k != axis]
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def _linear_map(
    frame: DmcFrame,
    tilt_a: float,
    tilt_b: float,
    size: tuple[int, int],
) -> np.ndarray:
    """A file pixel ``[row, col, 1]`` -> atlas micrometres map: the section
    cut through the middle of the volume normal to *frame*'s z, tilted about
    its x and y axes, turned 7 degrees in plane, 8 um pixels (a 900 x 700
    image is 7.2 x 5.6 mm), rows not quite square to columns."""
    centre = np.zeros(3)
    rows_dir, cols_dir = np.zeros(3), np.zeros(3)
    cols_dir[frame.x.index], rows_dir[frame.y.index] = 1.0, 1.0
    for axis in (frame.x, frame.y, frame.z):
        centre[axis.index] = axis.size * axis.resolution_um / 2
    turn = (
        _rotation(frame.x.index, tilt_a)
        @ _rotation(frame.y.index, tilt_b)
        @ _rotation(frame.z.index, 7.0)
    )
    cols_um, rows_um = (
        turn @ cols_dir * 8.0,
        turn @ (rows_dir + 0.03 * cols_dir) * 7.5,
    )
    width, height = size
    origin = centre - cols_um * (width - 1) / 2 - rows_um * (height - 1) / 2
    return np.column_stack([rows_um, cols_um, origin])


PLANES = ("coronal", "sagittal", "horizontal")
TILTS = ((0.0, 0.0), (0.3, -0.2), (-0.2, 0.4), (-9.0, 4.0), (6.0, -8.0))


@pytest.mark.parametrize(
    "atlas", [ALLEN_25, RAT_39, ALLEN_10], ids=lambda a: a[0]
)
@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("tilt", TILTS, ids=str)
def test_dmc_reads_the_linear_map_back(
    atlas: tuple[Any, ...], plane: str, tilt: tuple[float, float]
) -> None:
    frame = dmc_frame(*atlas, plane)  # type: ignore[arg-type]
    size = (900, 700)
    matrix = _linear_map(frame, *tilt, size)
    placement = dmc_placement(matrix, size, frame)

    # every pair where DMC's window accepts it, at least five
    atlas_dots = np.asarray(placement.atlas_dots)
    sample_dots = np.asarray(placement.sample_dots)
    assert len(atlas_dots) >= 5
    for dots in (atlas_dots, sample_dots):
        assert np.all(dots >= 0)
        assert np.all(dots[:, 0] < frame.x.size) and np.all(
            dots[:, 1] < frame.y.size
        )
    # the angles give DMC exactly the shifts the conversion chose
    assert (
        int(math.tan(math.radians(placement.x_angle_deg)) * frame.x.size / 2)
        == placement.x_shift
    )
    assert (
        int(math.tan(math.radians(placement.y_angle_deg)) * frame.y.size / 2)
        == placement.y_shift
    )

    # LangSlice's continuous voxel at probe pixels over the section
    cols, rows = np.meshgrid(
        np.linspace(0, size[0] - 1, 41), np.linspace(0, size[1] - 1, 37)
    )
    cols, rows = cols.ravel(), rows.ravel()
    um = (matrix @ np.stack([rows, cols, np.ones(cols.size)])).T
    voxel = np.column_stack(
        [
            um[:, axis.index] / axis.resolution_um
            for axis in (frame.x, frame.y, frame.z)
        ]
    )

    # DMC's transform reproduces the in-plane map (half a voxel on)
    exact = np.column_stack(
        [
            cols * (frame.x.size - 1) / (size[0] - 1) + 0.5,
            rows * (frame.y.size - 1) / (size[1] - 1) + 0.5,
        ]
    )
    post, read = _dmc_read(placement, frame, exact)
    assert np.max(np.abs(post - (voxel[:, :2] + 0.5))) < 2e-3
    off_half = np.all(np.abs((voxel[:, :2] % 1) - 0.5) > 1e-3, axis=1)
    assert np.array_equal(
        read[off_half, :2], np.rint(voxel[off_half, :2]).astype(int)
    )
    # DMC's plane is held to whole voxels: its centre voxel (half a voxel
    # at most), its slopes (half a voxel at the plane's edge) and the
    # truncation of its depth (half a voxel, after the half voxel added)
    depth_error = read[:, 2] - voxel[:, 2]
    assert np.max(np.abs(depth_error)) <= 1.5
    assert float(np.mean(np.abs(depth_error))) < 0.6

    # through the results path (cells rounded up onto the resized image)
    xs = _scale_positions(cols, size[0] - 1, frame.x.size - 1)
    ys = _scale_positions(rows, size[1] - 1, frame.y.size - 1)
    _post, read = _dmc_read(
        placement, frame, np.column_stack([xs, ys]).astype(float)
    )
    per_axis = np.abs(read - voxel)
    assert np.max(per_axis[:, :2]) < 2.0 and np.max(per_axis[:, 2]) < 2.5
    assert np.all(np.abs(np.mean(read - voxel, axis=0)[:2]) < 0.35)


def test_dmc_frame_matches_dmc_get_xyz_and_bregma() -> None:
    coronal = dmc_frame(*ALLEN_25, "coronal")  # type: ignore[arg-type]
    sagittal = dmc_frame(*ALLEN_25, "sagittal")  # type: ignore[arg-type]
    horizontal = dmc_frame(*ALLEN_25, "horizontal")  # type: ignore[arg-type]
    # as DMC 0.2.0b2's get_xyz answers for allen_mouse_25um
    assert [(a.name, a.size) for a in (coronal.x, coronal.y, coronal.z)] == [
        ("rl", 456),
        ("si", 320),
        ("ap", 528),
    ]
    assert [
        (a.name, a.size) for a in (sagittal.x, sagittal.y, sagittal.z)
    ] == [("si", 320), ("ap", 528), ("rl", 456)]
    assert [
        (a.name, a.size) for a in (horizontal.x, horizontal.y, horizontal.z)
    ] == [("rl", 456), ("ap", 528), ("si", 320)]
    # DMC's estimate for an atlas it has no bregma for, and its own table
    assert coronal.bregma == (264, 0, 228)
    assert dmc_frame(*RAT_39, "coronal").bregma == (371, 72, 266)  # type: ignore[arg-type]
    assert dmc_frame(*ALLEN_10, "coronal").bregma == (540, 0, 570)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ap, si and rl"):
        dmc_frame("other", "lsa", (10, 20, 30), (1.0, 1.0, 1.0), "coronal")


def test_z_mm_reads_back_to_the_centre_voxel() -> None:
    assert (dmc_decimals(25.0), dmc_decimals(10.0), dmc_decimals(39.0)) == (
        3,
        2,
        3,
    )
    for atlas in (ALLEN_25, RAT_39, ALLEN_10):
        frame = dmc_frame(*atlas, "coronal")  # type: ignore[arg-type]
        placement = dmc_placement(
            _linear_map(frame, 0.0, 0.0, (500, 400)), (500, 400), frame
        )
        read = (
            round(-placement.z_mm / (frame.z.resolution_um / 1000))
            + frame.z_bregma
        )
        assert read == placement.z_voxel


def test_a_plane_dmc_would_refuse_is_refused() -> None:
    frame = dmc_frame(*ALLEN_25, "coronal")  # type: ignore[arg-type]
    matrix = _linear_map(frame, -12.0, 0.0, (900, 700))
    matrix[frame.z.index, 2] -= 6000.0  # near the front of the volume
    with pytest.raises(ValueError, match="leaves the atlas volume"):
        dmc_placement(matrix, (900, 700), frame)
    away = _linear_map(frame, 0.0, 0.0, (900, 700))
    away[frame.x.index, 2] += 50000.0  # beside the atlas plane
    with pytest.raises(ValueError, match="does not reach"):
        dmc_placement(away, (900, 700), frame)


def test_natural_key_orders_as_natsort() -> None:
    names = [
        "s10_downsampled.tif",
        "s2_downsampled.tif",
        "a_downsampled.tif",
        "10b_downsampled.tif",
        "s1_downsampled.tif",
        "S3_downsampled.tif",
    ]
    assert sorted(names, key=natural_key) == [
        "10b_downsampled.tif",
        "S3_downsampled.tif",
        "a_downsampled.tif",
        "s1_downsampled.tif",
        "s2_downsampled.tif",
        "s10_downsampled.tif",
    ]


# --- a job exported whole ----------------------------------------------------


@pytest.fixture
def exported_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import langslice
    from langslice.doors.jobs import create
    from langslice.job.settings.spec import NonlinearSpec
    from tests.fixtures.example_plugin import ripple_map

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = tmp_path / "M7"
    write_sections(folder)
    spec = dataclasses.replace(
        full_spec(folder), nonlinear=NonlinearSpec(provider="none")
    )
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    assert (
        job.position_sections(
            sections=[
                {"id": ID0, "position_mm": 0.1},
                {"id": ID1, "position_mm": 0.15},
            ],
            view=False,
        )["status"]
        == "ok"
    )
    assert (
        job.interactive_transform(
            sections=[
                {
                    "id": ID0,
                    "rotation_deg": 2.0,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "translate_x_mm": 0.0,
                    "translate_y_mm": 0.0,
                },
                {
                    "id": ID1,
                    "rotation_deg": -4.0,
                    "scale_x": 1.05,
                    "scale_y": 0.97,
                    "translate_x_mm": 0.1,
                    "translate_y_mm": -0.05,
                },
            ],
            view=False,
        )["status"]
        == "ok"
    )
    context = job.context
    field = ripple_map(context.fit_frame(ID0), 1.0)
    reply = context.set_deformations([{"id": ID0, "map": field}]).reply()
    assert reply["written"] == [ID0], reply
    yield job
    job.close()


def test_the_export_writes_a_dmc_animal_folder(exported_job: Any) -> None:
    from langslice.core.refusal import Refused

    done = exported_job.export_dmc_brainmap(channel="green")
    folder = Path(done.folder)
    assert (
        folder == Path(exported_job.folder) / "exports" / "dmc_brainmap" / "M7"
    )
    assert done.written and all(Path(path).is_file() for path in done.files)
    assert done.registered == [ID0, ID1]
    assert [skip.section for skip in done.skipped] == [ID2]
    assert done.warps_left_out == [ID0]

    params = json.loads((folder / "params.json").read_text())
    assert params["general"] == {"animal_id": "M7", "chans_imaged": ["green"]}
    info = params["atlas_info"]
    assert (
        info["atlas"] == "synthetic_ellipses_50um"
        and info["orientation"] == "coronal"
    )
    assert info["xyz_dict"] == {
        "x": ["rl", 160, 50.0],
        "y": ["si", 120, 50.0],
        "z": ["ap", 6, 50.0],
    }
    channel = folder / "sharpy_track" / "green"
    images = sorted(path.name for path in channel.glob("*.tif"))
    assert images == [
        "s0_downsampled.tif",
        "s1_downsampled.tif",
        "s2_downsampled.tif",
    ]
    for name in images:
        pixels = tifffile.imread(channel / name)
        assert pixels.shape == (120, 160) and pixels.dtype == np.uint8

    registration = json.loads((channel / "registration.json").read_text())
    assert registration["imgName"] == {
        str(i): name for i, name in enumerate(images)
    }
    assert sorted(registration["atlasLocation"]) == ["0", "1"]
    for key in ("0", "1"):
        assert len(registration["atlasDots"][key]) >= 5
        assert len(registration["sampleDots"][key]) == len(
            registration["atlasDots"][key]
        )
    x_angle, y_angle, z_mm = registration["atlasLocation"]["0"]
    assert (x_angle, y_angle) == (0.0, 0.0)
    # 0.1 mm from the front is voxel 2 of the 50 um volume; bregma voxel 3
    assert z_mm == pytest.approx((3 - 2) * 0.05)

    record = json.loads((folder / "langslice_export.json").read_text())
    assert record["linear_only"] is True and record["warps_left_out"] == [ID0]
    assert (
        "deformation (a warp) applied in LangSlice is left out"
        in record["note"]
    )
    assert record["not_registered"] == [
        {"section": ID2, "reason": "no position"}
    ]

    # an image an earlier export left is removed: DMC numbers images by place
    (channel / "old_downsampled.tif").write_bytes(b"")
    exported_job.export_dmc_brainmap(channel="green")
    assert not (channel / "old_downsampled.tif").exists()
    with pytest.raises(Refused, match="not a DMC-BrainMap channel"):
        exported_job.export_dmc_brainmap(channel="red")
