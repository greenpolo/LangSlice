"""External worker keeps host calibration and initial state intact without Java."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.api.abba_worker import run_linear, run_nonlinear


@pytest.fixture
def params(tmp_path):
    Image.new("L", (80, 40)).save(tmp_path / "section_0001.tif")
    return {"image_folder": str(tmp_path), "pixel_size_um": 25,
            "positions_mm": {"section_0001.tif": 4.0},
            "spec": {"tasks": ["position", "transform"]}}


@pytest.mark.parametrize("initial_position", [4.0, -0.985])
def test_linear_initial_state_is_not_applied_and_affine_uses_snapshot_geometry(
    params, monkeypatch, initial_position,
):
    from langslice.linear import engine

    params["positions_mm"]["section_0001.tif"] = initial_position

    async def run(spec, *, on_write, on_event, emit):
        assert spec.inputs["positions"] == params["positions_mm"]
        assert spec.inputs["pixel_size_um"] == 25
        assert not spec.resume
        row = {"id": "section_0001.tif", "position_mm": initial_position, "rotation_deg": 0,
               "transform": None}
        value = {"slices": [row]}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        row.update(position_mm=4.2, transform={"params": [1, 0, .1, 0, 1, .2]})
        on_write(state)
        on_event({"kind": "image", "images": [{"data": b"raw", "mime_type": "image/png"}],
                  "thought_signature": b"private"})
        return state

    monkeypatch.setattr(engine, "run", run)
    events = []
    result = run_linear(params, events.append)
    assert events[0]["initial"] and events[0]["host_updates"] == []
    update = events[1]["host_updates"][0]
    assert update["position_mm"] == 4.2
    np.testing.assert_allclose(update["affine_mm"],
                               [[1, 0, 0, .2], [0, 1, 0, .2], [0, 0, 1, 0]])
    assert "private" not in json.dumps(events)
    assert "raw" not in json.dumps(events)
    assert result["state"]["slices"][0]["position_mm"] == 4.2


@pytest.mark.parametrize("change, message", [
    ({"pixel_size_um": 0}, "positive"),
    ({"positions_mm": {"missing.tif": 4}}, "exactly"),
    ({"positions_mm": {"section_0001.tif": float("nan")}}, "finite"),
    ({"spec": {"plane": "sagittal"}}, "coronal"),
    ({"host_angles_deg": {"pitch": 4.0}}, "flat atlas"),
    ({"spec": {"transform": {"angles": True}}}, "Cutting-angle"),
    ({"spec": {}, "registered_slices": ["section_0001.tif"]}, "ordering"),
])
def test_bad_host_inputs_refused_before_engine(params, change, message):
    params.update(change)
    with pytest.raises(ValueError, match=message):
        run_linear(params, lambda event: None)


def test_nonlinear_retains_pair_direction_and_grid(tmp_path, monkeypatch):
    import tifffile

    from langslice.integrations import abba

    coords = np.zeros((10, 12, 3), dtype=np.float32)
    np.save(tmp_path / "coords.npy", coords)
    tifffile.imwrite(tmp_path / "image.tif", np.ones((10, 12), dtype=np.uint8))
    source = np.array([[0., 0.], [1., 0.], [0., 1.]])
    target = source + 3

    def compute(actual_coords, histology, config):
        np.testing.assert_array_equal(actual_coords, coords)
        assert histology.shape == (10, 12)
        return source, target

    monkeypatch.setattr(abba, "compute_registration_landmarks", compute)
    result = run_nonlinear({"coords_path": str(tmp_path / "coords.npy"),
                            "histology_path": str(tmp_path / "image.tif")}, lambda event: None)
    assert result["source_points"] == source.tolist()
    assert result["target_points"] == target.tolist()
    assert result["coordinate_frame"] == "fixed_grid_pixels"
