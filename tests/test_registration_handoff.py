"""Coordinate round trips for the linear-to-nonlinear host boundary."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.affine import normalized_affine
from langslice.core import canvas
from langslice.linear.engine import EngineContext
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState
from langslice.registration_handoff import prepare_linear_registration


def setup_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, known: bool = True):
    pixels = np.zeros((80, 120, 3), dtype=np.uint8)
    pixels[8:30, 9:40] = [240, 70, 20]
    pixels[50:65, 80:105] = [20, 180, 250]
    Image.fromarray(pixels).save(tmp_path / "section.png")
    spec = JobSpec(
        image_folder=str(tmp_path), atlas="test_atlas", preprocess="none",
        inputs={"pixel_size_um": 10.0} if known else {},
    )
    context = EngineContext(
        spec=spec, image_folder=str(tmp_path), job_folder=str(tmp_path / "langslice"),
        results_path=str(tmp_path / "result.json"), model="unused", atlas_loader=lambda _: object(),
    )
    labels = np.zeros((70, 90), dtype=np.int32)
    labels[7:63, 12:78] = 1
    monkeypatch.setattr(canvas, "annotation_slice", lambda *args, **kwargs: labels)
    monkeypatch.setattr(canvas, "atlas_um_per_px", lambda _: 25.0)
    record = SliceState(
        id="section.png", index_original=0, index_corrected=0, position_mm=4.0,
        transform={"params": [1, 0, 0, 0, 1, 0],
                   "calibration": {"section_um_per_px": 10.0, "source": "estimated"}},
    )
    state = StackState(atlas="test_atlas", slices=[record])
    return state, context, record, pixels


def test_shear_nonsquare_round_trip(tmp_path, monkeypatch):
    state, ctx, record, _ = setup_section(tmp_path, monkeypatch)
    matrix = np.array([[1.1, 0.23, 9.0], [-0.14, 0.87, -6.0], [0, 0, 1]])
    assert record.transform is not None
    record.transform["params"] = normalized_affine(matrix[:2], (120, 80))
    before = state.to_dict()
    result = prepare_linear_registration(state, ctx, record.id, long_edge=120)
    geometry = canvas.canvas_geometry((120, 80), 10, ctx.atlas, 4, "coronal")
    points = np.array([[13, 10, 1], [60, 45, 1], [76, 61, 1]]).T
    section_points = result.atlas_to_slice @ points
    placed_section = matrix @ section_points
    placed_section[:2] += np.array(geometry.section_offset)[:, None]
    expected = points.copy().astype(float)
    expected[:2] = points[:2] * geometry.atlas_scale + np.array(geometry.atlas_offset)[:, None]
    np.testing.assert_allclose(placed_section, expected)
    assert state.to_dict() == before


def test_rotation_then_flip_and_requested_grid_calibration(tmp_path, monkeypatch):
    state, ctx, record, _ = setup_section(tmp_path, monkeypatch)
    record.rotation_deg, record.flip = 90, True
    original = Image.open(tmp_path / record.id).convert("RGB")
    expected = original.transpose(Image.Transpose.ROTATE_90).transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )
    full = prepare_linear_registration(state, ctx, record.id, long_edge=120)
    np.testing.assert_array_equal(np.asarray(full.image), np.asarray(expected))
    assert full.image.size == (80, 120)
    small = prepare_linear_registration(state, ctx, record.id, long_edge=60)
    assert small.image.size == (40, 60)
    assert small.metadata["section_um_per_px"] == 20
    assert full.metadata["section_um_per_px"] == 10
    assert full.metadata["atlas_mirror_lr"] is False
    # The padded canvas may round by half a pixel; physical linear scale is exact.
    np.testing.assert_allclose(small.atlas_to_slice[:2, :2] * 2, full.atlas_to_slice[:2, :2])


def test_stored_estimated_calibration_scales_without_reestimating(tmp_path, monkeypatch):
    state, ctx, record, _ = setup_section(tmp_path, monkeypatch, known=False)
    full = prepare_linear_registration(state, ctx, record.id, long_edge=120)
    small = prepare_linear_registration(state, ctx, record.id, long_edge=60)
    assert full.metadata["calibration_source"] == "estimated"
    assert full.metadata["section_um_per_px"] == 10
    assert small.metadata["section_um_per_px"] == 20
    assert record.transform is not None
    del record.transform["calibration"]
    with pytest.raises(ValueError, match="calibration"):
        prepare_linear_registration(state, ctx, record.id)


def test_calibration_uses_requested_resolution_above_preview_cap(tmp_path, monkeypatch):
    state, ctx, record, _ = setup_section(tmp_path, monkeypatch)
    path = tmp_path / record.id
    with Image.open(path) as handle:
        larger = handle.resize((1200, 800))
    larger.save(path)
    prepared = prepare_linear_registration(state, ctx, record.id, long_edge=600)
    assert prepared.image.size == (600, 400)
    assert prepared.metadata["section_um_per_px"] == 20.0
    assert prepared.metadata["source_image_size"] == [1200, 800]


@pytest.mark.parametrize("params", [
    [1, 0, 0, 0, 0, 0], [1, 0, float("nan"), 0, 1, 0],
    [1, 0, 0, 0, float("inf"), 0], [1, 0, 0],
])
def test_invalid_affines_are_rejected(tmp_path, monkeypatch, params):
    state, ctx, record, _ = setup_section(tmp_path, monkeypatch)
    assert record.transform is not None
    record.transform["params"] = params
    with pytest.raises(ValueError, match="affine"):
        prepare_linear_registration(state, ctx, record.id)


def test_missing_position_transform_and_explicit_stale_orientation(tmp_path, monkeypatch):
    state, ctx, record, _ = setup_section(tmp_path, monkeypatch)
    record.position_mm = None
    with pytest.raises(ValueError, match="position"):
        prepare_linear_registration(state, ctx, record.id)
    record.position_mm = 4
    assert record.transform is not None
    record.transform["orientation"] = {"flip": True, "rotation_deg": 0}
    with pytest.raises(ValueError, match="orientation"):
        prepare_linear_registration(state, ctx, record.id)
    record.transform = None
    with pytest.raises(ValueError, match="transform"):
        prepare_linear_registration(state, ctx, record.id)


def test_run_forwards_supplied_geometry_and_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from langslice.nonlinear import image_gen_registration
    from langslice.registration_handoff import run_linear_registration

    state, ctx, record, _ = setup_section(tmp_path, monkeypatch)
    seen = {}

    def fake_run(image, **options):
        seen.update(options)
        seen["image"] = image
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(image_gen_registration, "generate_registration_candidate", fake_run)
    before = state.to_dict()
    candidate = run_linear_registration(state, ctx, record.id, long_edge=120, provider="none")
    assert seen["initial_alignment_source"] == "linear_agent"
    assert seen["initial_atlas_to_slice"].shape == (3, 3)
    assert seen["image"].size == (120, 80)
    assert candidate.metadata["linear_handoff"]["source_image_size"] == [120, 80]
    assert state.to_dict() == before
    with pytest.raises(ValueError, match="cannot be overridden"):
        run_linear_registration(state, ctx, record.id, atlas_mirror_lr=True)
