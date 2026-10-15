"""Deformable fit engine on synthetic sections with a known residual field."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from langslice.core.atlas.images import atlas_images, atlas_volume
from langslice.core.atlas.levels import MODEL_DEPTH, labels_at
from langslice.core.atlas.render import region_boundaries
from langslice.core.deformable import (
    CandidateFailure,
    DeformableRecord,
    FitSettings,
    diagnose,
    prepare_fit,
)
from langslice.core.deformable.atlas_images import native_intensity, whole_region_ids
from langslice.core.deformable.fit import fit_prepared
from langslice.core.deformable.geometry import Placement
from langslice.core.deformable.record import jacobian_determinant
from langslice.core.deformable.regions import named_regions
from langslice.core.oblique import plane_index_coordinates, sample_oblique_plane
from tests.deformable_synthetic import (
    CTX,
    SMOOTH_FIELD,
    STR,
    TH,
    VL,
    VS,
    SyntheticAtlas,
    bump_field,
    fit_section,
    placement,
    render_section,
)

#: The deformable engine is FireANTs SyN; a fit test skips without it.
HAVE_ENGINE = importlib.util.find_spec("fireants") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="FireANTs missing")


def _settings(**options) -> FitSettings:
    return FitSettings(**{"engine": "ants", **options})


#: The synthetic sections are flat-intensity regions with no texture, where
#: local correlation has little to work with (synthetic warp error 0.39 of the
#: warp vs 0.12 for mutual information at standard detail). Tests of the fit's
#: mechanics (direction, exclusion, steps, the pool) pin the mutual-information
#: stain metric.
MI = {"stain_metric": "mutual_information", "stain_edges": False}


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


@pytest.fixture(scope="module")
def warped(atlas: SyntheticAtlas):
    field = SMOOTH_FIELD()
    image, truth = render_section(atlas, field)
    return field, image, truth


def _errors(record: DeformableRecord, field: np.ndarray, truth: np.ndarray):
    inner = ndi.binary_erosion(truth > 0, iterations=6)
    error = np.linalg.norm(record.field_mm - field, axis=-1)[inner].mean()
    flipped = np.linalg.norm(record.field_mm + field, axis=-1)[inner].mean()
    return error, flipped, np.linalg.norm(field, axis=-1)[inner].mean()


@needs_engine
def test_engine_recovers_a_known_warp_in_the_documented_direction(atlas, warped):
    field, image, truth = warped
    record = fit_section(image, atlas, placement(),
                         FitSettings(engine="ants", detail="coarse", **MI))
    error, flipped, magnitude = _errors(record, field, truth)
    # atlas point = section point + field: the recovered field matches u, not -u.
    assert error < 0.4 * magnitude
    assert flipped > 1.5 * magnitude
    # The inverse maps placed-atlas points back: v(p + u(p)) = -u(p).
    assert record.inverse_field_mm is not None
    assert record.diagnostics["inverse_consistency_mm"]["p99"] < 0.01
    from langslice.core.deformable.geometry import sample_native

    inverse = np.linalg.inv(record.placement.atlas_to_section)
    yy, xx = np.indices(truth.shape, dtype=np.float64)
    coords = np.stack([inverse[k, 0] * xx + inverse[k, 1] * yy + inverse[k, 2]
                       for k in range(2)], axis=-1)
    placed = sample_native(np.asarray(atlas.annotation[0]), coords)
    linear = np.mean(placed[truth > 0] == truth[truth > 0])
    fitted = np.mean(record.labels[truth > 0] == truth[truth > 0])
    assert fitted > linear
    assert record.inverse_source == "engine"


@needs_engine
def test_excluded_region_does_not_drive_the_fit(atlas):
    engine = "ants"
    # Only the thalamus is displaced; everywhere else the placement is right.
    field = bump_field([(226, 132, 16, 0.12, 0.0)])
    image, _ = render_section(atlas, field)
    th = np.zeros((260, 340), dtype=bool)
    th[110:150, 205:245] = True
    free = fit_section(image, atlas, placement(), FitSettings(engine=engine, detail="coarse", **MI))
    excluded = fit_section(image, atlas, placement(),
                           FitSettings(engine=engine, detail="coarse", exclude=("TH",), **MI))
    moved_free = np.linalg.norm(free.field_mm, axis=-1)[th].mean()
    moved_excluded = np.linalg.norm(excluded.field_mm, axis=-1)[th].mean()
    assert moved_free > 0.03
    assert moved_excluded < 0.3 * moved_free
    assert excluded.excluded_ids == [TH]


def test_exclusion_blanks_intensity_images_and_includes_descendants(atlas, warped):
    _, image, _ = warped
    assert whole_region_ids(atlas, ["VS"]) == frozenset({VS, VL})
    assert whole_region_ids(atlas, [STR]) == frozenset({STR})
    with pytest.raises(ValueError):
        whole_region_ids(atlas, ["NOPE"])
    prepared = prepare_fit(image, atlas, placement(),
                           _settings(detail="coarse", exclude=("STR",)))
    grid_to_native = np.linalg.inv(prepared.grid.section_to_working @ placement().atlas_to_section)
    # Working pixels well inside the placed striatum are blank and outside the mask.
    centre = (prepared.grid.section_to_working @ placement().atlas_to_section) @ [52, 58, 1]
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    assert prepared.inputs.moving[cy - 3:cy + 4, cx - 3:cx + 4].max() == 0
    assert not prepared.inputs.moving_mask[cy - 3:cy + 4, cx - 3:cx + 4].any()
    assert np.isfinite(grid_to_native).all()


def test_synthetic_compression_flags_tissue_but_not_ventricle():
    size = 200
    yy, xx = np.indices((size, size), dtype=np.float64)
    placed = np.zeros((size, size), dtype=np.int32)
    placed[(xx - 60) ** 2 + (yy - 100) ** 2 < 30 ** 2] = CTX
    placed[(xx - 140) ** 2 + (yy - 100) ** 2 < 30 ** 2] = VL
    mm = 0.02
    k = 1.8  # atlas point = c + k (p - c): the section shows each region 1/k^2 as large
    field = np.stack([(k - 1) * (xx - 100), (k - 1) * (yy - 100)], axis=-1) * mm
    native_x = xx + field[..., 0] / mm
    native_y = yy + field[..., 1] / mm
    inside = (native_x >= 0) & (native_x < size) & (native_y >= 0) & (native_y < size)
    warped = np.where(inside, placed[np.clip(np.rint(native_y), 0, size - 1).astype(int),
                                     np.clip(np.rint(native_x), 0, size - 1).astype(int)], 0)
    report = diagnose(field, mm, warped, placed, np.ones((size, size), bool),
                      ventricles=frozenset({VL}), excluded=frozenset(), acronyms={})
    ratios = {r["id"]: r["area_ratio"] for r in report["regions"]}
    assert ratios[CTX] == pytest.approx(1 / k ** 2, rel=0.1)
    assert ratios[VL] == pytest.approx(1 / k ** 2, rel=0.1)
    flagged = {f["id"]: f["code"] for f in report["flags"] if "id" in f}
    assert flagged == {CTX: "REGION_COMPRESSED"}  # ventricles get the looser limit
    assert report["fold_fraction"] == 0


def test_folds_are_reported():
    field = np.zeros((60, 60, 2))
    field[:, 30:, 0] = -0.5  # a step: points right of x=30 map behind their neighbours
    jacobian = jacobian_determinant(field, 0.02)
    assert (jacobian <= 0).any()
    labels = np.ones((60, 60), np.int32)
    report = diagnose(field, 0.02, labels, labels, np.ones((60, 60), bool),
                      ventricles=frozenset(), excluded=frozenset(), acronyms={})
    assert any(f["code"] == "FOLDS" for f in report["flags"])


@needs_engine
def test_lines_against_borders_and_model_regions(atlas):
    field = SMOOTH_FIELD()
    image, truth = render_section(atlas, field)
    merged = labels_at(truth, atlas, MODEL_DEPTH)
    lines = region_boundaries(merged)[0]
    # Lines only constrain the field at boundaries; interiors follow the
    # regularization, so the tolerance is looser than for the stain image.
    borders = fit_section(image, atlas, placement(), _settings(
        section_image="lines", atlas_image="borders_merged"), lines=lines)
    error, flipped, magnitude = _errors(borders, field, truth)
    assert error < 0.6 * magnitude and flipped > 1.5 * magnitude
    labelled = fit_section(image, atlas, placement(), _settings(
        section_image="lines", atlas_image="borders_merged", labels="model"), lines=lines)
    error, flipped, magnitude = _errors(labelled, field, truth)
    assert error < 0.6 * magnitude and flipped > 1.5 * magnitude
    assert set(labelled.engine["step_details"]["label_channels"]) >= {str(STR), str(TH)}


def test_named_regions_drop_a_bubble_loop():
    placed = np.zeros((120, 120), dtype=np.int32)
    placed[10:110, 10:110] = CTX
    placed[30:70, 30:70] = STR
    lines = np.zeros_like(placed, dtype=bool)
    lines[34, 34:76] = lines[76, 34:76] = True  # the model's striatum, shifted
    lines[34:77, 34] = lines[34:77, 76] = True
    yy, xx = np.indices(placed.shape)
    ring = np.abs(np.hypot(xx - 92, yy - 92) - 4) < 0.7  # a loop around a bubble in CTX
    lines |= ring
    tissue = placed > 0
    labels, unnamed, report = named_regions(lines, placed, tissue, mm_per_px=0.02)
    assert labels[55, 55] == STR and labels[20, 20] == CTX
    # The bubble's interior is not a region of its own: it is left unnamed.
    assert labels[92, 92] == 0 and unnamed[92, 92]
    assert any("small_loop_inside_one_region" in r["dropped"] for r in report)
    assert not unnamed[55, 55]


def test_fit_settings_validation():
    with pytest.raises(ValueError, match="SyN-only"):
        FitSettings(engine="elastix", labels="model")
    with pytest.raises(ValueError, match="borders"):
        _settings(section_image="lines", atlas_image="template")
    # A stain against atlas borders: borders are for the image model's lines.
    for kind in ("borders", "borders_merged"):
        with pytest.raises(ValueError, match="traced lines only"):
            FitSettings(section_image="stain", atlas_image=kind)
    with pytest.raises(ValueError, match="stiffness"):
        FitSettings(stiffness="stiff")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="detail"):
        FitSettings(detail="fine")  # type: ignore[arg-type]
    assert FitSettings.from_dict(FitSettings().to_dict()) == FitSettings()
    assert FitSettings().metric == "mutual_information" and FitSettings().stain_edges
    # Elastix has no local correlation: its stand-in.
    assert FitSettings(engine="elastix").metric == "mutual_information"
    assert FitSettings(stain_metric="local_correlation").metric == "local_correlation"
    lines = FitSettings(section_image="lines", atlas_image="borders_merged")
    assert lines.metric == "mean_squares"
    with pytest.raises(ValueError, match="stain_metric"):
        FitSettings(stain_metric="cc")  # type: ignore[arg-type]


@needs_engine
def test_sequential_steps_compose_and_undo(atlas, warped, tmp_path: Path):
    field, image, truth = warped
    first = fit_section(image, atlas, placement(), _settings(detail="coarse",
                                                               stiffness="firm", **MI))
    second = fit_section(image, atlas, placement(),
                         _settings(detail="coarse", structures=("STR",),
                                   neighbourhood_um=250, **MI), previous=first)
    assert second.step == 1 and second.parent is first
    assert second.engine["step_details"]["structures"] == [STR]
    # Far from the striatum the second step leaves the first one's field alone.
    far = np.zeros(truth.shape, dtype=bool)
    far[150:220, 240:300] = True
    assert np.abs(second.field_mm - first.field_mm)[far].max() < 0.01
    striatum = truth == STR
    assert (np.linalg.norm(second.field_mm - field, axis=-1)[striatum].mean()
            <= np.linalg.norm(first.field_mm - field, axis=-1)[striatum].mean() + 0.003)
    second.save(tmp_path / "rec")
    loaded = DeformableRecord.load(tmp_path / "rec")
    assert loaded.step == 1 and loaded.parent is not None and loaded.parent.step == 0
    assert np.array_equal(loaded.labels, second.labels)
    assert np.allclose(loaded.field_mm, second.field_mm)
    assert np.allclose(loaded.placement.atlas_to_section, second.placement.atlas_to_section)
    meta = json.loads((tmp_path / "rec" / "record.json").read_text())
    meta["placement"]["origin"] = 0
    (tmp_path / "rec" / "record.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=r"record\.json: Unknown placement key\(s\) \['origin'\]"):
        DeformableRecord.load(tmp_path / "rec")
    with pytest.raises(ValueError, match="None of the chosen structures"):
        fit_section(image, atlas, placement(), _settings(detail="coarse",
                                                           structures=("VS",), exclude=("VS",)))


def test_taper_weight_holds_the_neighbourhood_and_fades_beyond():
    from langslice.core.deformable.fit import RESTRICTED_TAPER_MM, _taper_weight

    region = np.zeros((100, 200), dtype=bool)
    region[40:60, 20:40] = True
    spacing = 0.01  # mm per working pixel
    weight = _taper_weight(region, spacing, 0.1, (400, 200))  # section grid twice the size
    assert weight.shape == (200, 400)
    distance = ndi.distance_transform_edt(~region) * spacing
    distance = np.kron(distance, np.ones((2, 2)))
    assert weight[distance <= 0.09].min() > 0.999
    assert weight[distance >= 0.1 + RESTRICTED_TAPER_MM + 0.02].max() < 1e-6
    ramp = (distance > 0.12) & (distance < 0.1 + RESTRICTED_TAPER_MM - 0.02)
    assert 0.0 < weight[ramp].min() and weight[ramp].max() < 1.0


def test_inverted_step_undoes_the_tapered_step():
    from langslice.core.deformable.fit import _invert_step

    yy, xx = np.indices((120, 160), dtype=np.float32)
    bump = np.exp(-((xx - 80) ** 2 + (yy - 60) ** 2) / (2 * 20.0 ** 2))
    step = np.stack([6.0 * bump, -4.0 * bump], axis=-1).astype(np.float32)
    inverse = _invert_step(step, np.zeros_like(step))
    # q -> q + v(q) -> + w(q + v(q)) returns to q.
    mx, my = xx + inverse[..., 0], yy + inverse[..., 1]
    import cv2

    back = np.stack([cv2.remap(np.ascontiguousarray(step[..., k]), mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for k in range(2)], -1)
    assert np.abs(inverse + back).max() < 1e-3


@needs_engine
def test_a_restricted_step_leaves_tissue_beyond_its_taper_unmoved(atlas, warped):
    from langslice.core.deformable.fit import RESTRICTED_TAPER_MM

    field, image, truth = warped
    first = fit_section(image, atlas, placement(), _settings(detail="coarse",
                                                               stiffness="firm", **MI))
    second = fit_section(image, atlas, placement(),
                         _settings(detail="coarse", structures=("STR",),
                                   neighbourhood_um=250, **MI), previous=first)
    assert second.engine["step_details"]["taper_um"] == [250.0, 250.0 + 1000 * RESTRICTED_TAPER_MM]
    mm = second.placement.section_mm_per_px
    distance = ndi.distance_transform_edt(first.labels != STR) * mm
    change = np.linalg.norm(second.field_mm - first.field_mm, axis=-1)
    beyond = (distance > 0.25 + RESTRICTED_TAPER_MM + 2 * mm) & (truth > 0)
    assert beyond.any() and change[beyond].max() < 1e-4
    near = (distance <= 0.25) & (truth > 0)
    assert change[near].max() > 1e-3
    assert second.diagnostics["inverse_consistency_mm"]["p99"] < 0.01
    assert second.diagnostics["fold_fraction"] == 0.0 or second.diagnostics["fold_fraction"] < 1e-3


@pytest.mark.slow
@needs_engine
def test_candidates_run_concurrently_and_all_return(atlas, warped):
    field, image, truth = warped
    results = fit_prepared([prepare_fit(image, atlas, placement(), settings) for settings in (
        FitSettings(engine="ants", detail="coarse", **MI),
        FitSettings(engine="ants", detail="coarse", stiffness="firm", **MI),
    )])
    assert len(results) == 2
    for result in results:
        assert not isinstance(result, CandidateFailure), result
        error, flipped, magnitude = _errors(result, field, truth)
        assert error < 0.4 * magnitude


def test_plane_coordinates_match_the_samplers(atlas):
    coords = plane_index_coordinates(atlas, 0.1, "coronal")
    assert np.allclose(coords[0], 2) and coords.shape == (3, 120, 160)
    tilted = plane_index_coordinates(atlas, 0.1, "coronal", 3.0, 2.0)
    from scipy.ndimage import map_coordinates

    expected = sample_oblique_plane(atlas, 0.1, "coronal", 3.0, 2.0, volume="template")
    sampled = map_coordinates(atlas.template.astype(np.float32), tilted, order=1)
    assert np.allclose(sampled, expected, atol=1e-3)


class _TwoImageAtlas:
    """A synthetic atlas that ships one additional reference, as BrainGlobe does."""

    def __init__(self, atlas) -> None:
        self.atlas_name = atlas.atlas_name
        self.orientation = atlas.orientation
        self.resolution = atlas.resolution
        self.metadata = getattr(atlas, "metadata", {})
        self.template = atlas.template
        self.annotation = atlas.annotation
        ap = np.arange(atlas.template.shape[0], dtype=np.float32)[:, None, None]
        self.additional_references = {
            "second": np.broadcast_to(ap, atlas.template.shape).copy()}


def test_atlas_images_are_the_atlas_own(atlas):
    assert atlas_images(atlas) == ("template",)
    two = _TwoImageAtlas(atlas)
    assert atlas_images(two) == ("template", "second")
    assert np.array_equal(atlas_volume(two, "template"), np.asarray(atlas.template))
    with pytest.raises(ValueError, match="its images: template, second"):
        atlas_volume(two, "nissl")


class _Ccfv3At50um:
    """An Allen CCFv3 atlas at 50 um: the extent the borrowed Nissl is offered on."""

    atlas_name = "allen_mouse_50um"
    orientation = "asr"
    resolution = (50.0, 50.0, 50.0)
    template = np.zeros((264, 160, 228), dtype=np.uint8)

    def __init__(self) -> None:
        self.annotation = np.ones((264, 160, 228), dtype=np.uint8)
        self.annotation[:, :10] = 0  # the top rows are outside the brain


def test_a_ccfv3_atlas_borrows_the_augmented_nissl_at_the_offset(atlas, monkeypatch):
    from langslice.core.atlas import images

    ap = np.arange(566, dtype=np.uint16)[:, None, None]
    augmented = type("Augmented", (), {"template": np.broadcast_to(ap, (566, 320, 456))})
    monkeypatch.setattr("langslice.core.atlas.core.load_atlas", lambda name: augmented)
    images._borrowed.clear()
    try:
        ccf = _Ccfv3At50um()
        assert atlas_images(ccf) == ("template", images.BORROWED_NISSL)
        assert atlas_images(atlas) == ("template",)  # not the CCFv3 extent
        volume = atlas_volume(ccf, images.BORROWED_NISSL)
        assert volume.shape == ccf.annotation.shape
        # 50 um voxel i is centred (i + 0.5) * 0.05 mm into the CCFv3, 0.35 mm more
        # into the augmented atlas: index ((i + 0.5) * 0.05 + 0.35) / 0.025 - 0.5.
        expected = (np.arange(264) + 0.5) * 2 + 14 - 0.5
        assert np.allclose(volume[:, 80, 100], np.rint(expected), atol=1)
        assert not volume[:, :10].any()
    finally:
        images._borrowed.clear()


def test_an_additional_reference_is_sampled_like_the_template(atlas):
    two = _TwoImageAtlas(atlas)
    straight = sample_oblique_plane(two, 0.1, "coronal", volume="second")
    ap_index = plane_index_coordinates(two, 0.1, "coronal")[0]
    assert np.allclose(straight, ap_index, atol=1e-3)
    tilted = sample_oblique_plane(two, 0.1, "coronal", 3.0, 0.0, volume="second")
    assert not np.allclose(tilted, straight)
    placement = Placement(atlas_name=two.atlas_name, position_mm=0.1, plane="coronal",
                          pitch_deg=0.0, yaw_deg=0.0, atlas_to_section=np.eye(3),
                          section_mm_per_px=0.01)
    assert np.allclose(native_intensity(two, placement, "second"), ap_index, atol=1e-3)
    with pytest.raises(ValueError, match="Not an image of this atlas"):
        native_intensity(atlas, placement, "second")
