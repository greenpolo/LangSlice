"""Deformable fit engine on synthetic sections with a known residual field."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from langslice.core.atlas.render import family_labels
from langslice.core.deformable import (
    CandidateFailure,
    DeformableRecord,
    FitSettings,
    NisslAtlas,
    diagnose,
    excluded_ids,
    fit_section,
    prepare_fit,
    ventricle_ids,
)
from langslice.core.deformable.engines import FIT_THREADS, RANDOM_SEED, invert_field, run_engine
from langslice.core.deformable.fit import finish_fit, fit_prepared
from langslice.core.deformable.masks import torn_edge_band
from langslice.core.deformable.nissl import AP_OFFSET_MM, NISSL_SHAPE, NISSL_VOXEL_MM
from langslice.core.deformable.record import jacobian_determinant
from langslice.core.deformable.regions import named_regions
from langslice.core.nonlinear.image_gen_helpers import _merge_classified
from langslice.core.oblique import plane_index_coordinates, sample_oblique_plane
from tests.deformable_synthetic import (
    CTX,
    HY,
    SECTION_MM_PER_PX,
    SMOOTH_FIELD,
    STR,
    TH,
    VL,
    VS,
    SyntheticAtlas,
    bump_field,
    placement,
    render_section,
)

#: Elastix ships with LangSlice; ANTs comes with the optional 'registration'
#: extra, so ANTs-only tests skip without it and the rest fall back to Elastix.
HAVE_ANTS = importlib.util.find_spec("ants") is not None
DEFAULT_ENGINE = "ants" if HAVE_ANTS else "elastix"
needs_ants = pytest.mark.skipif(not HAVE_ANTS, reason="antspyx (registration extra) missing")
BOTH_ENGINES = [pytest.param("ants", marks=needs_ants), "elastix"]


def _settings(**options) -> FitSettings:
    return FitSettings(**{"engine": DEFAULT_ENGINE, **options})


#: The synthetic sections are flat-intensity regions with no texture, where
#: local correlation has little to work with (synthetic warp error 0.39 of the
#: warp vs 0.12 for mutual information at standard detail). Tests of the fit's
#: mechanics (direction, exclusion, steps, the pool) pin the mutual-information
#: stain metric; the defaults have their own test.
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


@pytest.mark.parametrize("engine", BOTH_ENGINES)
def test_engine_recovers_a_known_warp_in_the_documented_direction(atlas, warped, engine):
    field, image, truth = warped
    record = fit_section(image, atlas, placement(),
                         FitSettings(engine=engine, detail="coarse", **MI))
    error, flipped, magnitude = _errors(record, field, truth)
    # atlas point = section point + field: the recovered field matches u, not -u.
    assert error < 0.4 * magnitude
    assert flipped > 1.5 * magnitude
    # The inverse maps placed-atlas points back: v(p + u(p)) = -u(p).
    assert record.inverse_field_mm is not None
    assert record.diagnostics["inverse_consistency_mm"]["p99"] < 0.01
    linear = np.mean(record.placed_labels(atlas.annotation[0])[truth > 0] == truth[truth > 0])
    fitted = np.mean(record.labels[truth > 0] == truth[truth > 0])
    assert fitted > linear
    assert record.inverse_source == ("engine" if engine == "ants" else "numerical_fixed_point")


def test_labels_are_clipped_to_tissue(atlas):
    remove = np.zeros((260, 340), dtype=bool)
    remove[:, 250:] = True  # the right side of the section is missing
    image, truth = render_section(atlas, np.zeros((260, 340, 2)), remove=remove)
    record = fit_section(image, atlas, placement(), _settings(detail="coarse"))
    assert not record.labels[~record.tissue].any()
    assert not record.labels[:, 260:].any()
    assert record.labels[truth > 0].astype(bool).mean() > 0.95


def test_torn_edge_band_marks_the_cut_not_the_real_outline(atlas):
    remove = np.zeros((260, 340), dtype=bool)
    remove[:, 230:] = True
    image, _ = render_section(atlas, np.zeros((260, 340, 2)), remove=remove)
    prepared = prepare_fit(image, atlas, placement(), _settings(detail="coarse"))
    band = prepared.torn_band
    assert band[100:160, 226:234].mean() > 0.9  # the straight cut through the brain
    assert not band[:, :120].any()  # the intact left outline stays informative
    assert not (prepared.inputs.fixed_mask & (ndi.zoom(band.astype(float), (
        prepared.grid.size[1] / 260, prepared.grid.size[0] / 340), order=0) > 0)).any()


def test_torn_edge_band_override_is_used(atlas, warped):
    _, image, _ = warped
    band = np.zeros((260, 340), dtype=bool)
    band[50:60, 50:60] = True
    prepared = prepare_fit(image, atlas, placement(), _settings(detail="coarse"),
                           torn_band=band)
    assert np.array_equal(prepared.torn_band, band)


def test_torn_edge_rule_on_a_plain_mask():
    footprint = np.zeros((100, 100), dtype=bool)
    footprint[10:90, 10:90] = True
    tissue = footprint.copy()
    tissue[:, 60:] = False  # torn at x = 60, deep inside the footprint
    band = torn_edge_band(tissue, footprint, mm_per_px=0.02)
    assert band[50, 59] and band[50, 62]
    assert not band[50, 11] and not band[11, 30]


@pytest.mark.parametrize("engine", BOTH_ENGINES)
def test_excluded_region_does_not_drive_the_fit(atlas, engine):
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
    assert excluded_ids(atlas, ["VS"]) == frozenset({VS, VL})
    assert excluded_ids(atlas, [STR]) == frozenset({STR})
    with pytest.raises(ValueError, match="Unknown atlas structures"):
        excluded_ids(atlas, ["NOPE"])
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


def test_numerical_inverse_undoes_a_smooth_field():
    field = bump_field([(40, 40, 15, 0.05, -0.03)], size=(80, 80)).astype(np.float32)
    inverse, residual = invert_field(field, (0.02, 0.02))
    assert residual < 1e-3


@needs_ants
def test_lines_against_borders_and_model_regions(atlas):
    field = SMOOTH_FIELD()
    image, truth = render_section(atlas, field)
    merged = family_labels(truth, atlas)
    lines = np.zeros(merged.shape, dtype=bool)
    lines[:, 1:] |= merged[:, 1:] != merged[:, :-1]
    lines[1:, :] |= merged[1:, :] != merged[:-1, :]
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


def test_label_channels_are_ants_only():
    with pytest.raises(ValueError, match="ANTs-only"):
        FitSettings(engine="elastix", labels="auto")
    with pytest.raises(ValueError, match="borders"):
        _settings(section_image="lines", atlas_image="template")


def test_trimmed_settings_refuse_what_the_ceiling_test_dropped():
    # A stain against atlas borders: borders are for the image model's lines.
    for kind in ("borders", "borders_merged"):
        with pytest.raises(ValueError, match="traced lines only"):
            FitSettings(section_image="stain", atlas_image=kind)
    with pytest.raises(ValueError, match="stiffness"):
        FitSettings(stiffness="stiff")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="detail"):
        FitSettings(detail="fine")  # type: ignore[arg-type]
    # A record saved while line softening was a setting still loads.
    old = {**FitSettings().to_dict(), "line_softening_um": 60.0}
    assert FitSettings.from_dict(old) == FitSettings()
    # Records saved before the stain metric was a setting were fitted with
    # mutual information and no edge channel, and load as such.
    older = {k: v for k, v in FitSettings().to_dict().items()
             if k not in ("stain_metric", "stain_edges")}
    loaded = FitSettings.from_dict(older)
    assert loaded.stain_metric == "mutual_information" and not loaded.stain_edges


def test_stain_metric_per_engine_and_pairing():
    assert FitSettings().metric == "local_correlation" and FitSettings().stain_edges
    # Elastix has no local correlation: its stand-in.
    assert FitSettings(engine="elastix").metric == "mutual_information"
    assert FitSettings(stain_metric="mutual_information").metric == "mutual_information"
    lines = FitSettings(section_image="lines", atlas_image="borders_merged")
    assert lines.metric == "mean_squares"
    with pytest.raises(ValueError, match="stain_metric"):
        FitSettings(stain_metric="cc")  # type: ignore[arg-type]


@pytest.mark.parametrize("engine", BOTH_ENGINES)
def test_default_stain_fit_recovers_the_warp_direction(atlas, warped, engine):
    field, image, truth = warped
    record = fit_section(image, atlas, placement(), FitSettings(engine=engine))
    error, flipped, magnitude = _errors(record, field, truth)
    assert error < 0.6 * magnitude and flipped > 1.4 * magnitude
    parameters = record.engine["native_parameters"]
    if engine == "ants":
        used = parameters["antsRegistration"]
        assert used["syn_metric"] == "CC" and used["edge_channel"]
        assert used["correlation_radius_mm"] == pytest.approx(0.08, abs=0.011)
        assert used["threads"] == FIT_THREADS and used["random_seed"] == RANDOM_SEED
    else:
        assert parameters["edge_channel"] and parameters["threads"] == FIT_THREADS
        assert parameters["requested"]["RandomSeed"] == [str(RANDOM_SEED)]
        assert any("no local correlation" in note for note in record.engine["notes"])


@pytest.mark.parametrize("engine", BOTH_ENGINES)
def test_identical_inputs_give_identical_fields(atlas, warped, engine):
    """Same fit in this process twice and in a pool worker: the same field, bit for bit."""
    _field, image, _truth = warped
    settings = FitSettings(engine=engine, detail="coarse")
    prepared = prepare_fit(image, atlas, placement(), settings)
    first = run_engine(prepared.inputs, settings)
    second = run_engine(prepared.inputs, settings)
    pooled = fit_prepared([prepared, prepare_fit(image, atlas, placement(), settings)])
    assert np.array_equal(first.field_mm, second.field_mm)
    assert np.array_equal(first.inverse_field_mm, second.inverse_field_mm)
    for record in pooled:
        assert not isinstance(record, CandidateFailure), record
        in_process = finish_fit(prepared, first)
        assert np.array_equal(record.field_mm, in_process.field_mm)


@needs_ants
def test_auto_labels_detect_the_ventricle_hole(atlas, warped):
    _, image, _ = warped
    prepared = prepare_fit(image, atlas, placement(),
                           _settings(detail="coarse", labels="auto"))
    assert prepared.details["label_channels"] == ["tissue", "ventricles"]
    assert 0.1 < prepared.details["detected_ventricle_area_mm2"] < 0.5
    assert ventricle_ids(atlas) == frozenset({VS, VL})


def test_sequential_steps_compose_and_undo(atlas, warped, tmp_path: Path):
    field, image, truth = warped
    first = fit_section(image, atlas, placement(), _settings(detail="coarse",
                                                               stiffness="firm", **MI))
    second = fit_section(image, atlas, placement(),
                         _settings(detail="coarse", structures=("STR",),
                                   neighbourhood_um=250, **MI), previous=first)
    assert second.step == 1 and second.undo() is first
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
    assert np.allclose(loaded.native_coordinates(), second.native_coordinates())
    with pytest.raises(ValueError, match="None of the chosen structures"):
        fit_section(image, atlas, placement(), _settings(detail="coarse",
                                                           structures=("VS",), exclude=("VS",)))


def test_record_carries_volume_coordinates(atlas, warped):
    _, image, _ = warped
    record = fit_section(image, atlas, placement(), _settings(detail="coarse"))
    volume = record.volume_coordinates()
    native = record.native_coordinates()
    # Coronal asr: volume axes (AP, DV, ML) = (slice index, native y, native x).
    assert np.allclose(volume[..., 1], native[..., 1])
    assert np.allclose(volume[..., 2], native[..., 0])
    assert np.allclose(volume[..., 0], round(0.1 / 0.05))


@needs_ants
def test_candidates_run_concurrently_and_all_return(atlas, warped):
    field, image, truth = warped
    results = fit_prepared([prepare_fit(image, atlas, placement(), settings) for settings in (
        FitSettings(engine="ants", detail="coarse", **MI),
        FitSettings(engine="elastix", detail="coarse", stiffness="firm", **MI),
    )])
    assert len(results) == 2
    for result in results:
        assert not isinstance(result, CandidateFailure), result
        error, flipped, magnitude = _errors(result, field, truth)
        assert error < 0.4 * magnitude


def test_family_labels_match_the_image_tool_merge(atlas):
    labels = atlas.annotation[0]
    assert np.array_equal(family_labels(labels, atlas), _merge_classified(labels, atlas))


def test_plane_coordinates_match_the_samplers(atlas):
    coords = plane_index_coordinates(atlas, 0.1, "coronal")
    assert np.allclose(coords[0], 2) and coords.shape == (3, 120, 160)
    tilted = plane_index_coordinates(atlas, 0.1, "coronal", 3.0, 2.0)
    from scipy.ndimage import map_coordinates

    expected = sample_oblique_plane(atlas, 0.1, "coronal", 3.0, 2.0, volume="template")
    sampled = map_coordinates(atlas.template.astype(np.float32), tilted, order=1)
    assert np.allclose(sampled, expected, atol=1e-3)


class _Ccfv3At50um:
    """An Allen CCFv3 atlas at 50 um: the extent the Nissl template covers."""

    atlas_name = "allen_mouse_50um"
    orientation = "asr"
    resolution = (50.0, 50.0, 50.0)
    template = np.zeros((264, 160, 228), dtype=np.uint8)
    annotation = np.ones((264, 160, 228), dtype=np.uint8)


class _NisslSource:
    """A stand-in for the augmented atlas: each voxel holds its AP index."""

    def __init__(self) -> None:
        ap = np.arange(NISSL_SHAPE[0], dtype=np.uint16)[:, None, None]
        self.reference = np.broadcast_to(ap, NISSL_SHAPE)


def test_nissl_is_sampled_at_the_ccfv3_offset_inside_the_brain():
    from langslice.core.deformable import nissl as nissl_module

    atlas = _Ccfv3At50um()
    atlas.annotation = atlas.annotation.copy()
    atlas.annotation[:, :10] = 0  # the top rows are outside the brain
    nissl_module._volumes.clear()
    try:
        source = NisslAtlas(loader=lambda name: _NisslSource())
        assert source.compatible(atlas)
        plane = source.sample_plane(atlas, 6.5, "coronal")
        ap_index = plane_index_coordinates(atlas, 6.5, "coronal")[0]
        expected = ((ap_index + 0.5) * 0.05 + AP_OFFSET_MM) / NISSL_VOXEL_MM - 0.5
        assert plane.shape == (160, 228)
        assert np.allclose(plane[20:], expected[20:], atol=1e-3)
        assert np.all(plane[:10] == 0)
    finally:
        nissl_module._volumes.clear()


def test_nissl_is_offered_only_for_the_ccfv3_extent(atlas):
    assert NisslAtlas.compatible(_Ccfv3At50um())
    assert not NisslAtlas.compatible(atlas)


AUGMENTED = Path.home() / ".brainglobe/ccfv3augmented_mouse_25um_v1.0"
ALLEN_25 = Path.home() / ".brainglobe/allen_mouse_25um_v1.2"


@pytest.mark.skipif(not (AUGMENTED.exists() and ALLEN_25.exists()),
                    reason="the cached ccfv3augmented and allen_mouse 25 um atlases")
def test_the_augmented_atlas_holds_the_ccfv3_at_the_offset():
    import tifffile

    allen = tifffile.imread(ALLEN_25 / "annotation.tiff") > 0
    augmented = tifffile.imread(AUGMENTED / "annotation.tiff") > 0
    assert augmented.shape == NISSL_SHAPE
    start = round(AP_OFFSET_MM / NISSL_VOXEL_MM)
    window = augmented[start:start + allen.shape[0]]
    dice = 2 * (allen & window).sum() / (allen.sum() + window.sum())
    assert dice > 0.99


def test_section_mm_per_px_constant_is_the_synthetic_scale():
    assert SECTION_MM_PER_PX == 0.025 and HY == 5


def test_elastix_writes_nothing_into_the_working_directory(atlas, warped, tmp_path, monkeypatch):
    _field, image, _truth = warped
    monkeypatch.chdir(tmp_path)
    record = fit_section(image, atlas, placement(),
                         FitSettings(engine="elastix", detail="coarse", stiffness="firm"))
    assert np.abs(record.field_mm).max() > 0
    assert list(tmp_path.iterdir()) == []
