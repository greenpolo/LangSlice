"""Independent geometric checks against pixels and analytic affine maps."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import map_coordinates

from langslice.core.landmarks import section_panel, section_picture_point, section_point
from langslice.core.state import SliceState
from langslice.core.triangulation import PiecewiseAffine, visualign_mesh


def test_inverse_keeps_connectivity_when_destination_delaunay_would_flip_diagonal():
    source = np.array([[0., 0.], [4., 0.], [3., 2.], [0., 3.]])
    target = np.array([[0., 0.], [4., 0.], [3., 4.], [0., 3.]])
    mesh = PiecewiseAffine(source, target)
    # Points on both sides of the original diagonal and the diagonal itself.
    probes = np.array([[2., .4], [.5, 1.5], [1.5, 1.]])
    mapped = mesh.map(probes)
    np.testing.assert_allclose(mesh.inverse(mapped), probes, atol=1e-12)
    assert not np.allclose(PiecewiseAffine(target, source).map(mapped), probes)


def test_mesh_reproduces_an_affine_map_between_vertices():
    source = np.array([[0., 0.], [200., 0.], [190., 40.], [0., 30.], [70., 18.]])
    matrix = np.array([[1.1, .3], [-.1, .8]])
    offset = np.array([15., -6.])
    mesh = PiecewiseAffine(source, source @ matrix.T + offset)
    probes = np.array([[20., 10.], [80., 22.], [170., 20.]])
    np.testing.assert_allclose(mesh.map(probes), probes @ matrix.T + offset, atol=1e-12)


def test_fixed_outer_corners_cover_the_entire_rectangular_image_in_both_directions():
    source = np.array([[100., 20.], [250., 25.], [700., 45.]])
    target = source + [[4., 2.], [-5., 3.], [8., -2.]]
    mesh = visualign_mesh(source, target, size=(900, 80))
    yy, xx = np.mgrid[0:80:7, 0:900:23]
    probes = np.stack([xx, yy], axis=-1)
    assert np.isfinite(mesh.map(probes)).all()
    assert np.isfinite(mesh.inverse(probes)).all()
    np.testing.assert_allclose(mesh.inverse(mesh.map(probes)), probes, atol=1e-10)
    np.testing.assert_allclose(mesh.map(source), target, atol=1e-10)


def test_warped_view_endpoint_uses_the_actual_section_point_under_the_pixel():
    width, height = 240, 160
    mm = .025
    yy, xx = np.indices((height, width), dtype=float)
    points = np.stack([xx, yy], axis=-1)
    matrix = np.array([[1.04, .02], [-.03, 1.02]])
    offset = np.array([2., -1.])
    field = (points @ matrix.T + offset - points) * mm
    inverse = ((points - offset) @ np.linalg.inv(matrix).T - points) * mm
    record = SimpleNamespace(field_mm=field, inverse_field_mm=inverse,
                             section_size=(width, height), mm_per_px=mm)
    section = SliceState("section.tif", 0, 0)
    panel = section_panel(section, (width, height), pixel_to_render=np.eye(3),
                          bounds=[0, 0, width, height], warp_key="example")
    original = np.array([83., 59.])
    displayed = matrix @ original + offset
    fractions = (original + .5) / [width, height]
    np.testing.assert_allclose(section_picture_point(panel, fractions, record), displayed,
                               atol=1e-6)
    # OpenCV's image sampler quantizes interpolation coordinates to 1/32 px.
    np.testing.assert_allclose(section_point(panel, displayed, record), fractions, atol=1e-5)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flip", [False, True])
def test_endpoint_pixels_follow_actual_image_rotation_flip_and_crop(rotation, flip):
    # Every acquisition pixel carries its own integer index. Locate it in the
    # actual PIL output rather than generating an expected inverse transform.
    width, height = 13, 9
    pixels = np.arange(width * height, dtype=np.int32).reshape(height, width)
    image = Image.fromarray(pixels)
    turns = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
             270: Image.Transpose.ROTATE_270}
    if rotation:
        image = image.transpose(turns[rotation])
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    record = SliceState("section.tif", 0, 0, rotation_deg=rotation, flip=flip)
    # A displayed crop magnifies by four and offsets its picture by a gutter.
    pixel_to_render = np.array([[.25, 0., 2.], [0., .25, 1.], [0., 0., 1.]])
    panel = section_panel(record, image.size, pixel_to_render=pixel_to_render,
                          bounds=[0, 0, 100, 100])
    rendered = np.asarray(image)
    for acquisition_x, acquisition_y in [(3, 2), (8, 5), (6, 4)]:
        row, col = np.argwhere(rendered == pixels[acquisition_y, acquisition_x])[0]
        picture_xy = np.array([(col - 2.) * 4, (row - 1.) * 4])
        expected = (np.array([acquisition_x, acquisition_y]) + .5) / [width, height]
        np.testing.assert_allclose(section_point(panel, picture_xy), expected, atol=1e-12)
        np.testing.assert_allclose(section_picture_point(panel, expected), picture_xy,
                                   atol=1e-12)


def test_applied_record_interpolates_correspondences_after_sequential_corrections(tmp_path):
    from langslice.ops import landmarks
    from tests.test_landmarks import saved_pairs, setup

    job, workspace, section = setup(tmp_path)
    try:
        for step, delta in enumerate([(5., -3.), (-2., 4.)]):
            edits = saved_pairs(job, workspace, section, delta=delta)
            if step:
                for index, edit in enumerate(edits, 1):
                    edit["id"] = f"L{index}"
            landmarks.set_landmarks(job, workspace, section.id, edits)
            landmarks.landmark_triangulation(job, workspace, section.id)
            applied = job.deformations.current(job.state, section)
            assert applied is not None
            for pair in section.landmarks["pairs"]:
                section_xy = np.array(pair["section_fraction_xy"]) * applied.section_size - .5
                shift = np.array([
                    map_coordinates(applied.field_mm[..., axis],
                                    [[section_xy[1]], [section_xy[0]]], order=1).item()
                    for axis in range(2)
                ])
                placed = section_xy + shift / applied.mm_per_px
                native = np.linalg.inv(applied.placement.atlas_to_section) @ [*placed, 1.]
                # Dense field sampling approximates the analytic mesh at a
                # subpixel vertex; it must still land within 0.1 atlas voxel.
                np.testing.assert_allclose(native[:2], pair["atlas_native_xy"], atol=.1)
    finally:
        job.close()


def test_deleted_landmark_loses_its_influence_when_the_complete_set_is_reapplied(tmp_path):
    from langslice.core import deformation
    from langslice.core import landmarks as geometry
    from langslice.ops import landmarks
    from tests.test_landmarks import saved_pairs, setup

    job, workspace, section = setup(tmp_path)
    try:
        edits = saved_pairs(job, workspace, section)
        atlas_id = edits[0]["atlas"]["picture"]
        section_id = edits[0]["section"]["picture"]
        ap = job.views.lookup(atlas_id).recipe["endpoint_panels"][0]
        sp = job.views.lookup(section_id).recipe["endpoint_panels"][0]
        grid = deformation.fit_grid(job.state, workspace, section)
        fraction = np.array([.5, .45])
        point = fraction * grid.image.size - .5
        native = geometry.transform(
            np.linalg.inv(grid.placement.atlas_to_section), point + [7., 1.]
        )
        atlas_xy = geometry.transform(np.linalg.inv(ap["pixel_to_native"]), native)
        section_xy = geometry.section_picture_point(sp, fraction)
        edits.append({"atlas": {"picture": atlas_id, "x": atlas_xy[0], "y": atlas_xy[1]},
                      "section": {"picture": section_id, "x": section_xy[0], "y": section_xy[1]}})
        landmarks.set_landmarks(job, workspace, section.id, edits)
        landmarks.landmark_triangulation(job, workspace, section.id)
        before = job.deformations.current(job.state, section)
        assert np.max(np.abs(before.field_mm)) > .05
        landmarks.set_landmarks(job, workspace, section.id, delete=["L4"])
        landmarks.landmark_triangulation(job, workspace, section.id)
        after = job.deformations.current(job.state, section)
        # The remaining three correspondences are identity constraints. An
        # application of this complete set must erase L4's previous influence.
        np.testing.assert_allclose(after.field_mm, 0., atol=1e-6)
    finally:
        job.close()
