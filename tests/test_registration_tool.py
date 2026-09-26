"""The agent controls slice notes, while the tool owns placement and first-result retention."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice import registration_tool as tool
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState
from langslice.nonlinear.border_refinement import border_overlay
from langslice.nonlinear.prompts import border_refinement_prompt
from langslice.registration_handoff import LinearRegistrationInput, prepare_linear_registration


@pytest.fixture
def case(tmp_path, monkeypatch):
    original = Image.new("RGB", (60, 40), (40, 70, 90))
    original.save(tmp_path / "section.png")
    spec = JobSpec(image_folder=str(tmp_path), atlas="test", preprocess="none")
    ctx = SimpleNamespace(
        spec=spec, atlas=object(), image_path=lambda _: str(tmp_path / "section.png"),
    )
    record = SliceState(
        id="section.png", index_original=0, index_corrected=0, position_mm=4,
        transform={"params": [1, 0, 0, 0, 1, 0]},
    )
    state = StackState(atlas="test", slices=[record])
    placement = np.array([[1., 0., 6.], [0., 1., 3.], [0., 0., 1.]])
    labels = np.zeros((40, 60), dtype=np.int64)
    labels[5:30, 5:25] = 100000001
    labels[5:30, 25:45] = 100000003
    prepared = LinearRegistrationInput(original, placement, "test", 4, "coronal", 0, 0, {})
    monkeypatch.setattr(tool, "prepare_linear_registration", lambda *a, **k: prepared)
    monkeypatch.setattr(tool, "annotation_slice", lambda *a, **k: labels)
    monkeypatch.setattr(tool, "_merge_classified", lambda ids, atlas: ids)
    monkeypatch.setattr(tool, "prepare_canvas", lambda image, **k: (image, image.size, 0, 0, 0))
    return state, ctx, record, original, labels, placement


def test_fixed_prompt_placed_input_and_raw_preservation(case, tmp_path, monkeypatch):
    state, ctx, record, original, labels, placement = case
    calls = []
    mask = np.zeros((40, 60), dtype=bool)
    mask[5:30, 23] = True
    raw = border_overlay(Image.new("RGB", original.size, (180, 20, 30)), mask)

    def generate(request):
        calls.append(request)
        return SimpleNamespace(image=raw, route="test")

    monkeypatch.setattr(tool, "generate_warped_segmentation_image", generate)
    before = state.to_dict()
    result = tool.correct_slice(
        state, ctx, record.id, additional_notes="The left piece has shifted.", out=tmp_path / "out",
    )
    assert result["status"] == "ok" and len(calls) == 1
    request = calls[0]
    assert request.prompt.startswith(border_refinement_prompt())
    assert request.prompt.endswith("The left piece has shifted.")
    assert len(request.reference_images) == 1
    np.testing.assert_array_equal(request.reference_images[0], original)
    projected = cv2.warpAffine(
        labels.astype(np.float64), placement[:2], original.size, flags=cv2.INTER_NEAREST,
    ).astype(np.int64)
    expected = border_overlay(original, tool._extract_borders_from_classified(projected) > 0)
    np.testing.assert_array_equal(request.slice_image, expected)
    paths = result["artifact_paths"]
    np.testing.assert_array_equal(Image.open(paths["raw_reply"]), raw)
    overlay = np.asarray(Image.open(paths["lines_on_original"]))
    np.testing.assert_array_equal(overlay[~mask], np.asarray(original)[~mask])
    assert state.to_dict() == before
    assert result["fit_performed"] is False


def test_first_result_is_kept_without_quality_veto_or_notes_retry(case, tmp_path, monkeypatch):
    state, ctx, record, original, *_ = case
    calls = []

    def generate(request):
        calls.append(request)
        return SimpleNamespace(image=original, route="test")

    monkeypatch.setattr(tool, "generate_warped_segmentation_image", generate)
    first = tool.correct_slice(state, ctx, record.id, out=tmp_path / "out")
    repeated = tool.correct_slice(
        state, ctx, record.id, out=tmp_path / "out", additional_notes="Try something else.",
    )
    assert len(calls) == 1
    assert first["status"] == "ok" and first["model_border_pixels"] == 0
    assert calls[0].prompt == border_refinement_prompt()
    assert repeated["cached"] is True and repeated["additional_notes"] == ""
    assert repeated["artifact_dir"] == first["artifact_dir"]
    # A changed linear alignment is a new input, not a veto of the first reply.
    record.position_mm = 4.5
    changed = tool.correct_slice(state, ctx, record.id, out=tmp_path / "out")
    assert len(calls) == 2
    assert changed["geometry_fingerprint"] != first["geometry_fingerprint"]


def test_missing_linear_placement_never_reaches_provider(case, tmp_path, monkeypatch):
    state, ctx, record, *_ = case
    record.transform = None
    monkeypatch.setattr(tool, "prepare_linear_registration", prepare_linear_registration)
    monkeypatch.setattr(
        tool, "generate_warped_segmentation_image", lambda *a: pytest.fail("Unexpected model call"),
    )
    with pytest.raises(ValueError, match="affine transform"):
        tool.correct_slice(state, ctx, record.id, out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_transport_can_retry_without_discarding_any_image_reply(case, tmp_path, monkeypatch):
    state, ctx, record, original, *_ = case
    calls = []

    def fail(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("transport failed")
        return SimpleNamespace(image=original, route="test")

    monkeypatch.setattr(tool, "generate_warped_segmentation_image", fail)
    first = tool.correct_slice(state, ctx, record.id, out=tmp_path / "out")
    repeated = tool.correct_slice(state, ctx, record.id, out=tmp_path / "out")
    assert first["status"] == "error" and first["message"] == "transport failed"
    assert not first["raw_received"]
    assert repeated["status"] == "ok" and repeated["raw_received"]
    assert repeated["attempt"] == 2 and len(calls) == 2
    cached = tool.correct_slice(state, ctx, record.id, out=tmp_path / "out")
    assert cached["cached"] and len(calls) == 2


def test_existing_spline_is_not_silently_reduced_to_affine(case, monkeypatch):
    state, ctx, record, *_ = case
    record.transform["spline"] = {"backend": "elastix"}
    with pytest.raises(ValueError, match="spline"):
        prepare_linear_registration(state, ctx, record.id)
