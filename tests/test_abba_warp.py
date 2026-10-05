"""An applied deformation as ABBA warp-step landmark pairs (``core/abba_warp.py``).

Synthetic sections whose frame and deformation record are built by hand, as
``core.maps.section_frame`` and the deformable fit build them: the pairs'
thin-plate spline reproduces the record's own map (``core.maps.native_points``)
within 5 um off the grid, for every quarter turn and flip; the pairs sit on top
of the affine row (``core.abba_affine.normalized_to_abba_affine``); a
deformation too fine for 33x33 points is sent anyway with its error measured,
the largest grid whose spline does not fold is sent when a finer one folds,
and only a deformation whose spline folds at every grid is refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from langslice.core.abba_affine import normalized_to_abba_affine
from langslice.core.abba_warp import warp_world_landmarks, world_frame
from langslice.core.affine import denormalized_affine, pixel_center_map
from langslice.core.deformable.geometry import Placement
from langslice.core.deformable.record import DeformableRecord
from langslice.core.deformable.settings import FitSettings
from langslice.core.maps import SectionFrame, native_points, orientation_matrix, unturned

FILE = (600, 400)
PIXEL_UM = 20.0
RENDER_UNTURNED = (300, 200)
FIT_UNTURNED = (450, 300)
#: A sheared, scaled, rotated, shifted placement (the six stored numbers).
PARAMS = (1.04, 0.05, 0.02, -0.03, 0.97, -0.015)


def _frame(rotation: int, flip: bool, params=PARAMS) -> SectionFrame:
    render = unturned(RENDER_UNTURNED, rotation) if rotation % 180 else RENDER_UNTURNED
    render_um = PIXEL_UM * FILE[0] / RENDER_UNTURNED[0]
    file_to_render = (orientation_matrix(RENDER_UNTURNED, rotation, flip)
                      @ pixel_center_map(FILE, RENDER_UNTURNED))
    # The atlas plane (25 um voxels) drawn at true scale on the render's frame.
    scale = 25.0 / render_um
    atlas_on_frame = np.array([[scale, 0.0, 12.0], [0.0, scale, -7.0], [0.0, 0.0, 1.0]])
    matrix = np.vstack([denormalized_affine(params, render), [0.0, 0.0, 1.0]])
    return SectionFrame(
        section_id="s.tif", file_size=FILE, working_size=FILE, working_factor=1.0,
        render_size=render, render_um_per_px=render_um, file_um_per_px=PIXEL_UM,
        calibration_source="host", position_mm=5.0, plane="coronal", pitch_deg=0.0,
        yaw_deg=0.0, rotation_deg=rotation, flip=flip, file_to_render=file_to_render,
        atlas_to_render=np.linalg.inv(matrix) @ atlas_on_frame,
        native_to_index=np.eye(3), native_to_um=np.eye(3),
        annotation=np.zeros((4, 4), dtype=np.uint32), atlas={},
    )


def _record(frame: SectionFrame, field: np.ndarray) -> DeformableRecord:
    fit_size = (unturned(FIT_UNTURNED, frame.rotation_deg) if frame.rotation_deg % 180
                else FIT_UNTURNED)
    render_to_fit = pixel_center_map(frame.render_size, fit_size)
    placement = Placement(
        atlas_name="synthetic", position_mm=5.0, plane="coronal", pitch_deg=0.0,
        yaw_deg=0.0, atlas_to_section=render_to_fit @ frame.atlas_to_render,
        section_mm_per_px=frame.render_um_per_px * frame.render_size[0] / fit_size[0] / 1000)
    shape = (fit_size[1], fit_size[0])
    assert field.shape == (*shape, 2)
    empty = np.zeros(shape, dtype=bool)
    return DeformableRecord(
        placement=placement, section_size=fit_size, settings=FitSettings(),
        field_mm=field.astype(np.float32), inverse_field_mm=None, inverse_source="none",
        labels=np.zeros(shape, dtype=np.uint32), tissue=empty, torn_band=empty,
        excluded_ids=[], engine={}, diagnostics={}, atlas={})


def _smooth_field(rotation: int, amplitude_mm: float = 0.08, waves: float = 1.0) -> np.ndarray:
    fit_size = unturned(FIT_UNTURNED, rotation) if rotation % 180 else FIT_UNTURNED
    yy, xx = np.indices((fit_size[1], fit_size[0]), dtype=np.float64)
    u, v = xx / fit_size[0], yy / fit_size[1]
    dx = amplitude_mm * np.sin(2 * math.pi * waves * u) * np.cos(math.pi * waves * v)
    dy = 0.6 * amplitude_mm * np.cos(2 * math.pi * waves * v) * np.sin(math.pi * u + 0.3)
    return np.stack([dx, dy], axis=-1)


def _exact(frame, warp, points, params=PARAMS):
    """(before, after) in world mm of file points, from the record's own map."""
    render_to_world, native_to_world = world_frame(frame, params, size=FILE,
                                                   pixel_size_um=PIXEL_UM)
    nx, ny = native_points(frame, warp, points[None, :, 0], points[None, :, 1])
    native = np.column_stack([nx[0], ny[0], np.ones(len(points))])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    before = (render_to_world @ frame.file_to_render @ homogeneous.T).T[:, :2]
    after = (native_to_world @ native.T).T[:, :2]
    return before, after


@pytest.mark.parametrize("rotation,flip", [(0, False), (0, True), (90, False), (90, True),
                                           (180, False), (270, True)])
def test_pairs_tps_reproduces_the_record_off_the_grid(rotation, flip):
    from langslice.core.thin_plate import ThinPlateKernel

    frame = _frame(rotation, flip)
    warp = _record(frame, _smooth_field(rotation))
    report: dict = {}
    source, target = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                          pixel_size_um=PIXEL_UM, diagnostics=report)
    assert source.shape == target.shape and source.shape[0] == report["points"] <= 33 ** 2
    assert report["max_error_mm"] <= 0.005 and report["within_tolerance"]
    assert 0 < report["p99_error_mm"] <= report["max_error_mm"]
    # ABBA's pull-back (target -> source), applied at random points of the file.
    pullback = ThinPlateKernel(target, source, max_points=33 ** 2)
    rng = np.random.default_rng(rotation + flip)
    points = rng.uniform([-0.5, -0.5], [FILE[0] - 0.5, FILE[1] - 0.5], size=(2000, 2))
    before, after = _exact(frame, warp, points)
    assert np.abs(after - before).max() > 0.04  # the warp moves points
    error = np.linalg.norm(pullback.forward(after) - before, axis=1)
    assert error.max() <= 0.005


@pytest.mark.parametrize("rotation,flip", [(0, False), (90, True), (180, False)])
def test_source_points_are_where_the_affine_row_puts_the_section(rotation, flip):
    """The warp step sits on top of the affine step: its source points are the
    section's pixels as the ABBA affine row (applied to the oriented
    snapshot's centred world frame) places them."""
    frame = _frame(rotation, flip)
    warp = _record(frame, _smooth_field(rotation))
    source, _target = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                           pixel_size_um=PIXEL_UM)
    affine = normalized_to_abba_affine(PARAMS, size=FILE, pixel_size_um=PIXEL_UM,
                                       rotation_deg=rotation)
    n = int(round(math.sqrt(len(source))))
    xs = np.linspace(-0.5, FILE[0] - 0.5, n)
    ys = np.linspace(-0.5, FILE[1] - 0.5, n)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    oriented = (frame.file_to_render @ points.T).T[:, :2]
    extent = np.array(frame.render_size) * frame.render_um_per_px / 1000
    world = extent * (oriented / np.array(frame.render_size) - 0.5)
    placed = world @ affine[:2, :2].T + affine[:2, 3]
    np.testing.assert_allclose(source, placed, atol=1e-9)


def test_without_an_affine_the_world_displacement_is_the_records_field():
    """Identity placement, no turn: a fit-grid pixel centre moves by exactly
    its field (millimetres on the section are millimetres in ABBA)."""
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    frame = _frame(0, False, identity)
    field = _smooth_field(0)
    warp = _record(frame, field)
    fit_to_file = np.linalg.inv(pixel_center_map(FILE, FIT_UNTURNED))
    rows, cols = np.array([10, 150, 290]), np.array([20, 200, 430])
    fit_points = np.column_stack([cols, rows, np.ones(3)]).astype(float)
    points = (fit_to_file @ fit_points.T).T[:, :2]
    before, after = _exact(frame, warp, points, identity)
    np.testing.assert_allclose(after - before, field[rows, cols], atol=1e-5)


def test_a_deformation_whose_spline_folds_at_every_grid_is_refused():
    frame = _frame(0, False)
    fit = FIT_UNTURNED
    yy, xx = np.indices((fit[1], fit[0]), dtype=np.float64)
    mm_per_px = frame.render_um_per_px * frame.render_size[0] / fit[0] / 1000
    # Pull a band backwards faster than the grid advances: the map folds.
    dx = np.where((xx > 150) & (xx < 250), -1.6 * (xx - 150) * mm_per_px, 0.0)
    field = np.stack([dx, np.zeros_like(dx)], axis=-1)
    with pytest.raises(ValueError, match="folds at every grid"):
        warp_world_landmarks(frame, _record(frame, field), PARAMS, size=FILE,
                             pixel_size_um=PIXEL_UM)


def test_a_deformation_too_fine_for_33_points_is_sent_with_its_error():
    """Accuracy never refuses: the 33x33 grid goes out, its error measured
    (the maximum at the probes matches an independent check)."""
    from langslice.core.thin_plate import ThinPlateKernel

    frame = _frame(0, False)
    warp = _record(frame, _smooth_field(0, amplitude_mm=0.02, waves=12.0))
    report: dict = {}
    source, target = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                          pixel_size_um=PIXEL_UM, diagnostics=report)
    assert report["grid_size"] == 33 and len(source) == 33 ** 2 == report["points"]
    assert not report["within_tolerance"] and report["max_error_mm"] > 0.005
    assert 0 < report["p99_error_mm"] <= report["max_error_mm"]
    pullback = ThinPlateKernel(target, source, max_points=33 ** 2)
    rng = np.random.default_rng(7)
    points = rng.uniform([-0.5, -0.5], [FILE[0] - 0.5, FILE[1] - 0.5], size=(4000, 2))
    before, after = _exact(frame, warp, points)
    error = np.linalg.norm(pullback.forward(after) - before, axis=1)
    assert error.max() > 0.005
    assert error.max() == pytest.approx(report["max_error_mm"], rel=0.3)


def test_the_largest_grid_whose_spline_does_not_fold_is_sent(monkeypatch):
    """When the finest grids' splines fold, the largest one that does not is sent."""
    import langslice.core.thin_plate as thin_plate

    class FoldingAbove289(thin_plate.ThinPlateKernel):
        def jacobian(self, points_mm):
            result = super().jacobian(points_mm)
            if len(self.source) > 17 ** 2:
                result[:, 0] *= -1  # a mirrored map: every determinant negative
            return result

    monkeypatch.setattr(thin_plate, "ThinPlateKernel", FoldingAbove289)
    frame = _frame(0, False)
    report: dict = {}
    source, _target = warp_world_landmarks(
        frame, _record(frame, _smooth_field(0, amplitude_mm=0.02, waves=12.0)), PARAMS,
        size=FILE, pixel_size_um=PIXEL_UM, diagnostics=report)
    assert report["grid_size"] == 17 and len(source) == 17 ** 2
    assert not report["within_tolerance"]


def test_another_snapshot_size_is_refused():
    frame = _frame(0, False)
    warp = _record(frame, _smooth_field(0))
    with pytest.raises(ValueError, match="snapshot"):
        warp_world_landmarks(frame, warp, PARAMS, size=(601, 400), pixel_size_um=PIXEL_UM)
