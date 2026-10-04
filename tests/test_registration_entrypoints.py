"""Supplied alignments retain their coordinate frame through public entrypoints."""

import argparse
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from langslice.hosts.api.models import RegisterRequest


class CapturedRequest(Exception):
    pass


@pytest.mark.parametrize("matrix", [
    [[1, 0], [0, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
    [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 1, 0], [0.1, 0, 1]],
])
def test_api_rejects_invalid_alignment(matrix):
    with pytest.raises(ValidationError):
        RegisterRequest(image_path="unused", atlas="test", position_mm=1,
                        initial_atlas_to_slice=matrix)


def test_the_api_door_resolves_the_image_model_it_hands_in(tmp_path, monkeypatch):
    from langslice.hosts.api.runtime import run_register

    image_path = tmp_path / "image.png"
    Image.new("RGB", (40, 30), "gray").save(image_path)
    received = {}

    def capture(**kwargs):
        received.update(kwargs)
        raise CapturedRequest

    monkeypatch.setattr("langslice.core.nonlinear.runtime.estimate_registration", capture)
    request = RegisterRequest(image_path=str(image_path), atlas="test", position_mm=1,
                              provider="chatgpt", preprocess="none")
    with pytest.raises(CapturedRequest):
        run_register(request)
    assert callable(received["image_call"])
    assert received["provider"] == "chatgpt"


@pytest.mark.parametrize("supplied", [False, True])
def test_api_resize_preserves_pixel_centres(tmp_path, monkeypatch, supplied):
    from langslice.hosts.api.runtime import run_register

    image_path = tmp_path / "image.png"
    Image.new("RGB", (301, 199), "gray").save(image_path)
    affine = np.array([[2, 0.2, 17], [-0.1, 3, 12], [0, 0, 1]], dtype=float)
    received = {}

    def capture(**kwargs):
        received.update(kwargs)
        raise CapturedRequest

    monkeypatch.setattr("langslice.core.nonlinear.runtime.estimate_registration", capture)
    request = RegisterRequest(
        image_path=str(image_path), atlas="test", position_mm=1, provider="none",
        preprocess="none", vlm_resolution=100,
        initial_atlas_to_slice=affine.tolist() if supplied else None,
        initial_alignment_source="test-placement", atlas_mirror_lr=True, passes=2,
    )
    with pytest.raises(CapturedRequest):
        run_register(request)
    assert received["passes"] == 2
    assert received["image_call"] is None  # provider none: no model to resolve
    assert received["deformation"] == "deformable"
    assert received["atlas_mirror_lr"] is True
    assert received["initial_alignment_source"] == "test-placement"
    if supplied:
        width, height = received["image"].size
        points = np.array([[0, 0, 1], [30, 15, 1], [70, 55, 1]], dtype=float).T
        original = affine @ points
        expected = (original[:2] + 0.5) * np.array([width / 301, height / 199])[:, None] - 0.5
        actual = np.asarray(received["initial_atlas_to_slice"]) @ points
        np.testing.assert_allclose(actual[:2], expected)
        assert request.initial_atlas_to_slice == affine.tolist()
    else:
        assert received["initial_atlas_to_slice"] is None


def test_nonlinear_runtime_threads_supplied_placement(monkeypatch):
    from langslice.core.nonlinear import runtime

    monkeypatch.setattr(runtime, "load_atlas", lambda _: object())
    monkeypatch.setattr(runtime, "get_composite_slice", lambda *a, **k: Image.new("RGB", (8, 8)))
    received = {}

    def capture(image, **kwargs):
        received.update(kwargs)
        raise CapturedRequest

    monkeypatch.setattr(runtime, "generate_registration_candidate", capture)
    matrix = [[1, 0, 4], [0, 1, 2], [0, 0, 1]]
    with pytest.raises(CapturedRequest):
        runtime.estimate_registration(
            Image.new("RGB", (32, 24)), atlas_name="test", position_mm=2,
            initial_atlas_to_slice=matrix, initial_alignment_source="host",
            atlas_mirror_lr=True, passes=2,
        )
    assert received["passes"] == 2
    assert received["initial_atlas_to_slice"] == matrix
    assert received["initial_alignment_source"] == "host"
    assert received["atlas_mirror_lr"] is True


def test_nonlinear_runtime_threads_omitted_placement_selects_atlas_route(monkeypatch):
    """No ``initial_atlas_to_slice`` forwards ``None`` for it, which is what
    selects route "atlas" inside `border_registration.py` (an explicitly
    supplied placement selects route "supplied" instead; see the test above).
    ``passes``/``atlas_mirror_lr`` still thread through on this route too."""
    from langslice.core.nonlinear import runtime

    monkeypatch.setattr(runtime, "load_atlas", lambda _: object())
    monkeypatch.setattr(runtime, "get_composite_slice", lambda *a, **k: Image.new("RGB", (8, 8)))
    received = {}

    def capture(image, **kwargs):
        received.update(kwargs)
        raise CapturedRequest

    monkeypatch.setattr(runtime, "generate_registration_candidate", capture)
    with pytest.raises(CapturedRequest):
        runtime.estimate_registration(
            Image.new("RGB", (32, 24)), atlas_name="test", position_mm=2,
            atlas_mirror_lr=True, passes=2,
        )
    assert received["initial_atlas_to_slice"] is None
    assert received["passes"] == 2
    assert received["image_call"] is None  # provider none: no model to resolve
    assert received["deformation"] == "deformable"
    assert received["atlas_mirror_lr"] is True


@pytest.mark.parametrize("supplied", [False, True])
def test_api_returns_prepared_frame_provenance_without_remapping_markers(
    tmp_path, monkeypatch, supplied,
):
    from types import SimpleNamespace

    from langslice.core.nonlinear.types import RegistrationAnnotationSession
    from langslice.hosts.api.runtime import run_register

    image_path = tmp_path / "image.png"
    Image.new("RGB", (301, 199), "gray").save(image_path)
    markers = [[7.0, 11.0, 20.0, 23.0]]
    session = RegistrationAnnotationSession(
        workflow="image_gen_registration", target_count=0,
        metadata={"visualign_markers": markers},
    )

    def fake_registration(**kwargs):
        return SimpleNamespace(
            affine_result=SimpleNamespace(
                rotation_deg=0, translation_px=(0, 0), scale=(1, 1), shear=0,
            ),
            annotation_session=session, accepted_correspondences=[], debug_dir=None,
        )

    monkeypatch.setattr("langslice.core.nonlinear.runtime.estimate_registration", fake_registration)
    result = run_register(RegisterRequest(
        image_path=str(image_path), atlas="test", position_mm=1, provider="none",
        preprocess="none", vlm_resolution=100,
        initial_atlas_to_slice=np.eye(3).tolist() if supplied else None,
    ))
    assert result.annotation_session is not None
    metadata = result.annotation_session["metadata"]
    assert isinstance(metadata, dict)
    frame = metadata["api_image_frame"]
    assert frame["input_file_size"] == [301, 199]
    width, height = frame["runtime_image_size"]
    assert width == 100
    scale = np.array([width / 301, height / 199])
    # Top-left input pixel centre moves by the half-pixel resize offset.
    actual = np.asarray(frame["input_to_runtime"]) @ [0, 0, 1]
    np.testing.assert_allclose(actual[:2], (scale - 1) / 2)
    assert metadata["visualign_markers"] == markers
    assert frame["marker_frame"].startswith("runtime prepared image pixels")
    assert "unchanged" in frame["atlas_frame"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_cli_alignment_json_and_mirror(tmp_path, monkeypatch, wrapped):
    from langslice.doors.cli.register import add_register_parser
    from langslice.doors.cli.register import run_register as run_register_cli

    matrix = [[1, 0.1, 4], [0.2, 1, 2], [0, 0, 1]]
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps({"atlas_to_slice": matrix} if wrapped else matrix))
    parser = argparse.ArgumentParser()
    add_register_parser(parser.add_subparsers())
    args = parser.parse_args([
        "register", "unused.png", "--position", "2", "--provider", "none",
        "--initial-alignment", str(path), "--mirror-atlas-lr", "--passes", "2",
        "--out", str(tmp_path / "out"),
    ])
    received = []

    def capture(request, **kwargs):
        received.append(request)
        raise CapturedRequest

    monkeypatch.setattr("langslice.hosts.api.runtime.run_register", capture)
    with pytest.raises(CapturedRequest):
        run_register_cli(args)
    assert received[0].initial_atlas_to_slice == matrix
    assert received[0].atlas_mirror_lr is True
    assert received[0].passes == 2
