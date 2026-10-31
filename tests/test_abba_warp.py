"""An applied deformation as ABBA warp-step landmark pairs (``core/maps/abba.py``).

Synthetic sections whose frame and deformation record are built by hand, as
``core.maps.section_maps.section_frame`` and the deformable fit build them: the pairs'
thin-plate spline reproduces the record's own map (``core.maps.section_maps.native_points``)
within 5 um off the grid, for every quarter turn and flip; the pairs sit on top
of the affine row (``core.maps.abba.normalized_to_abba_affine``); the placement
is the same on every call; a deformation too fine for 1089 points is sent
anyway with its error measured at held-out points of the atlas overlap only;
pairs that fold the spline are dropped, a start that folds falls back to an
even grid, and only a deformation whose spline folds everywhere is refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from langslice.core.fits.deformable.geometry import Placement
from langslice.core.fits.deformable.record import DeformableRecord
from langslice.core.fits.deformable.settings import FitSettings
from langslice.core.geometry.affine import (
    denormalized_affine,
    pixel_center_map,
)
from langslice.core.maps.abba import (
    MAX_POINTS,
    normalized_to_abba_affine,
    warp_world_landmarks,
    world_frame,
)
from langslice.core.maps.section_maps import (
    SectionFrame,
    native_points,
    orientation_matrix,
    unturned,
)

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


def _record(frame: SectionFrame, field: np.ndarray,
            footprint: np.ndarray | None = None) -> DeformableRecord:
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
        labels=np.zeros(shape, dtype=np.uint32),
        footprint=empty if footprint is None else footprint,
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


def _random_points(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([-0.5, -0.5], [FILE[0] - 0.5, FILE[1] - 0.5], size=(count, 2))


def _section_error(landmarks, before, after):
    from langslice.core.geometry.thin_plate import ThinPlateKernel

    pullback = ThinPlateKernel(landmarks.target_mm, landmarks.source_mm,
                               max_points=MAX_POINTS)
    return np.linalg.norm(pullback.forward(after) - before, axis=1)


@pytest.mark.parametrize("rotation,flip", [(0, False), (0, True), (90, False), (90, True),
                                           (180, False), (270, True)])
def test_pairs_tps_reproduces_the_record_off_the_grid(rotation, flip):
    frame = _frame(rotation, flip)
    warp = _record(frame, _smooth_field(rotation))
    landmarks = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                     pixel_size_um=PIXEL_UM)
    source = landmarks.source_mm
    assert source.shape == landmarks.target_mm.shape and len(source) <= MAX_POINTS
    # No atlas overlap on this synthetic frame: the whole snapshot counts.
    assert landmarks.region == "whole snapshot" and landmarks.placement == "adaptive"
    atlas = landmarks.accuracy.atlas
    assert atlas.max_mm <= 0.005 and landmarks.within_tolerance
    assert 0 < atlas.median_mm <= atlas.p95_mm <= atlas.p99_mm <= atlas.max_mm
    assert landmarks.accuracy.probes > 1000 and landmarks.accuracy.unconverged == 0
    # ABBA's pull-back (target -> source), applied at random points of the file.
    before, after = _exact(frame, warp, _random_points(rotation + flip, 2000))
    assert np.abs(after - before).max() > 0.04  # the warp moves points
    assert _section_error(landmarks, before, after).max() <= 0.005


@pytest.mark.parametrize("rotation,flip", [(0, False), (90, True), (180, False)])
def test_source_points_are_where_the_affine_row_puts_the_section(rotation, flip):
    """The warp step sits on top of the affine step: each pair's source is a
    section pixel as the ABBA affine row (applied to the oriented snapshot's
    centred world frame) places it, and its target is where the record takes
    that pixel."""
    frame = _frame(rotation, flip)
    warp = _record(frame, _smooth_field(rotation))
    landmarks = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                     pixel_size_um=PIXEL_UM)
    affine = normalized_to_abba_affine(PARAMS, size=FILE, pixel_size_um=PIXEL_UM,
                                       rotation_deg=rotation)
    # The file pixels, recovered through the affine row and the centring.
    extent = np.array(frame.render_size) * frame.render_um_per_px / 1000
    world = np.linalg.solve(affine[:2, :2], (landmarks.source_mm - affine[:2, 3]).T).T
    oriented = (world / extent + 0.5) * np.array(frame.render_size)
    homogeneous = np.column_stack([oriented, np.ones(len(oriented))])
    points = (np.linalg.inv(frame.file_to_render) @ homogeneous.T).T[:, :2]
    assert np.all(points >= -0.5 - 1e-6) and np.all(points <= np.array(FILE) - 0.5 + 1e-6)
    before, after = _exact(frame, warp, points)
    np.testing.assert_allclose(landmarks.source_mm, before, atol=1e-9)
    np.testing.assert_allclose(landmarks.target_mm, after, atol=1e-6)


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


def test_the_placement_is_the_same_on_every_call():
    frame = _frame(90, True)
    warp = _record(frame, _smooth_field(90, amplitude_mm=0.02, waves=6.0))
    first = warp_world_landmarks(frame, warp, PARAMS, size=FILE, pixel_size_um=PIXEL_UM)
    second = warp_world_landmarks(frame, warp, PARAMS, size=FILE, pixel_size_um=PIXEL_UM)
    np.testing.assert_array_equal(first.source_mm, second.source_mm)
    np.testing.assert_array_equal(first.target_mm, second.target_mm)
    assert first.accuracy == second.accuracy and first.rounds == second.rounds


def test_a_deformation_whose_spline_folds_everywhere_is_refused():
    frame = _frame(0, False)
    fit = FIT_UNTURNED
    yy, xx = np.indices((fit[1], fit[0]), dtype=np.float64)
    mm_per_px = frame.render_um_per_px * frame.render_size[0] / fit[0] / 1000
    # Pull a band backwards faster than the grid advances: the map folds.
    dx = np.where((xx > 150) & (xx < 250), -1.6 * (xx - 150) * mm_per_px, 0.0)
    field = np.stack([dx, np.zeros_like(dx)], axis=-1)
    with pytest.raises(ValueError, match="folds at every start and every grid"):
        warp_world_landmarks(frame, _record(frame, field), PARAMS, size=FILE,
                             pixel_size_um=PIXEL_UM)


def test_a_deformation_too_fine_for_the_budget_is_sent_with_its_error():
    """Accuracy never refuses: the whole budget goes out, its error measured
    at held-out probes (the section-side maximum matches an independent
    check at random points)."""
    frame = _frame(0, False)
    warp = _record(frame, _smooth_field(0, amplitude_mm=0.02, waves=12.0))
    landmarks = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                     pixel_size_um=PIXEL_UM)
    assert len(landmarks.source_mm) == MAX_POINTS
    assert not landmarks.within_tolerance and landmarks.accuracy.atlas.max_mm > 0.005
    before, after = _exact(frame, warp, _random_points(7, 4000))
    error = _section_error(landmarks, before, after)
    assert error.max() > 0.005
    assert error.max() == pytest.approx(landmarks.accuracy.section.max_mm, rel=0.3)
    # Both sides of the map err by about as much.
    assert landmarks.accuracy.atlas.max_mm == pytest.approx(
        landmarks.accuracy.section.max_mm, rel=0.3)


def test_the_error_is_measured_over_the_atlas_overlap_only():
    """Pairs go where the atlas overlaps the section, and the error is
    recorded there: a fine ripple confined to the background neither draws
    pairs nor counts, while it is still there for a whole-snapshot check."""
    frame = _frame(0, False)
    fit_w, fit_h = FIT_UNTURNED
    footprint = np.zeros((fit_h, fit_w), dtype=bool)
    footprint[:, : fit_w // 2] = True  # the left half holds the atlas
    yy, xx = np.indices((fit_h, fit_w), dtype=np.float64)
    ripple = 0.02 * np.sin(2 * math.pi * 14 * xx / fit_w) * np.sin(2 * math.pi * 9 * yy / fit_h)
    ripple[:, : int(fit_w * 0.7)] = 0.0  # only well inside the background
    field = _smooth_field(0) + np.stack([ripple, ripple], axis=-1)
    warp = _record(frame, field, footprint)
    landmarks = warp_world_landmarks(frame, warp, PARAMS, size=FILE,
                                     pixel_size_um=PIXEL_UM)
    assert landmarks.region == "atlas overlap"
    assert landmarks.accuracy.atlas.max_mm <= 0.005

    # Inside the overlap, an independent check agrees; outside, the ripple
    # the pairs were not spent on remains.
    points = _random_points(11, 4000)
    before, after = _exact(frame, warp, points)
    error = _section_error(landmarks, before, after)
    fit_x = points[:, 0] * fit_w / FILE[0]
    assert error[fit_x < 0.45 * fit_w].max() <= 0.005
    assert error[fit_x > 0.75 * fit_w].max() > 0.005


def _folding_above(limit: int):
    import langslice.core.geometry.thin_plate as thin_plate

    class FoldingAbove(thin_plate.ThinPlateKernel):
        """A spline that reads as folded once it has more than *limit* points."""

        def jacobian(self, points_mm):
            result = super().jacobian(points_mm)
            if len(self.source) > limit:
                result[:, 0] *= -1  # a mirrored map: every determinant negative
            return result

    return FoldingAbove


def test_pairs_that_fold_the_spline_are_dropped(monkeypatch):
    """Every pair added to the start folds the spline: each is dropped, and
    the start's pairs, which do not fold, are sent."""
    import langslice.core.maps.tps_landmarks as tps_landmarks

    monkeypatch.setattr(tps_landmarks, "ThinPlateKernel", _folding_above(300))
    frame = _frame(0, False)
    landmarks = warp_world_landmarks(
        frame, _record(frame, _smooth_field(0, amplitude_mm=0.02, waves=12.0)), PARAMS,
        size=FILE, pixel_size_um=PIXEL_UM)
    assert landmarks.placement == "adaptive" and len(landmarks.source_mm) <= 300
    assert landmarks.dropped > 0 and landmarks.rounds > 0
    assert not landmarks.within_tolerance


def test_a_start_that_folds_falls_back_to_an_even_grid(monkeypatch):
    """Both adaptive starts fold: the largest even snapshot grid whose spline
    does not fold is sent."""
    import langslice.core.geometry.thin_plate as thin_plate
    import langslice.core.maps.tps_landmarks as tps_landmarks

    class FoldingUnless625(thin_plate.ThinPlateKernel):
        def jacobian(self, points_mm):
            result = super().jacobian(points_mm)
            if len(self.source) != 25 ** 2:
                result[:, 0] *= -1
            return result

    monkeypatch.setattr(tps_landmarks, "ThinPlateKernel", FoldingUnless625)
    frame = _frame(0, False)
    landmarks = warp_world_landmarks(frame, _record(frame, _smooth_field(0)), PARAMS,
                                     size=FILE, pixel_size_um=PIXEL_UM)
    assert landmarks.placement == "grid" and len(landmarks.source_mm) == 25 ** 2


def test_another_snapshot_size_is_refused():
    frame = _frame(0, False)
    warp = _record(frame, _smooth_field(0))
    with pytest.raises(ValueError, match="snapshot"):
        warp_world_landmarks(frame, warp, PARAMS, size=(601, 400), pixel_size_um=PIXEL_UM)
