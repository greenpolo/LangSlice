"""The agent controls minimal prompt edits; the tool owns placement and first-result retention."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from langslice.core.handoff import (
    LinearRegistrationInput,
    prepare_linear_registration,
)
from langslice.core.nonlinear import registration_tool as tool
from langslice.core.nonlinear.border_refinement import border_overlay, smooth_border_overlay
from langslice.core.nonlinear.prompts import border_correction_tool_prompt
from langslice.core.nonlinear.upright import Upright
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.providers.registry import ImageModel, resolve_image_model


def _correct(state, ctx, section_id, **kwargs):
    """One image edit, run in this thread (``start_correction`` and its job)."""
    record, job = tool.start_correction(state, ctx, section_id, **kwargs)
    return job() if job is not None else record


def _model(call, provider="openai-oauth"):
    """The run's image model with its network call replaced (a door resolves it)."""
    return ImageModel(provider, resolve_image_model(provider).model, call)


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
    monkeypatch.setattr(tool, "prepare_canvas", lambda image, **k: (image, np.eye(3)))
    monkeypatch.setattr(tool, "upright", lambda *a, **k: Upright())
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

    model = _model(generate)
    before = state.to_dict()
    result = _correct(
        state, ctx, record.id, out=tmp_path / "out", image_model=model,
        prompt=border_correction_tool_prompt(
            provider="openai-oauth",
        ).replace("Output one image.", "The left piece has shifted. Output one image."),
    )
    assert result["status"] == "ok" and len(calls) == 1
    request = calls[0]
    # OpenAI providers get the GPT twin: Image 1 the clean photograph (edit target),
    # Image 2 the smooth placed borders.
    base = border_correction_tool_prompt(provider="openai-oauth")
    assert request.prompt == base.replace(
        "Output one image.", "The left piece has shifted. Output one image.",
    )
    assert result["prompt_edited"] is True
    paths = result["artifact_paths"]
    assert Path(paths["base_prompt"]).read_text() == base
    assert Path(paths["prompt"]).read_text() == request.prompt
    assert "{+The left piece has shifted.+}" in Path(paths["prompt_diff"]).read_text()
    assert len(request.reference_images) == 1
    np.testing.assert_array_equal(request.slice_image, original)
    expected = smooth_border_overlay(
        original, labels, placement, width_px=tool.BORDER_WIDTH_PX * 60 / 1536,
    )
    np.testing.assert_array_equal(request.reference_images[0], expected)
    assert not np.array_equal(np.asarray(expected), np.asarray(original))
    paths = result["artifact_paths"]
    np.testing.assert_array_equal(Image.open(paths["raw_reply"]), raw)
    overlay = np.asarray(Image.open(paths["lines_on_original"]))
    np.testing.assert_array_equal(overlay[~mask], np.asarray(original)[~mask])
    assert state.to_dict() == before
    assert result["fit_performed"] is False


def test_first_result_is_kept_without_quality_veto_or_edited_retry(case, tmp_path, monkeypatch):
    state, ctx, record, original, *_ = case
    calls = []

    def generate(request):
        calls.append(request)
        return SimpleNamespace(image=original, route="test")

    model = _model(generate)
    first = _correct(state, ctx, record.id, out=tmp_path / "out", image_model=model)
    repeated = _correct(
        state, ctx, record.id, out=tmp_path / "out", image_model=model,
        prompt="Try something else.",
    )
    assert len(calls) == 1
    assert first["status"] == "ok" and first["model_border_pixels"] == 0
    assert calls[0].prompt == border_correction_tool_prompt(provider="openai-oauth")
    assert repeated["cached"] is True and repeated["prompt_edited"] is False
    assert repeated["artifact_dir"] == first["artifact_dir"]
    # A changed linear alignment is a new input, not a veto of the first reply.
    record.position_mm = 4.5
    changed = _correct(state, ctx, record.id, out=tmp_path / "out", image_model=model)
    assert len(calls) == 2
    assert changed["geometry_fingerprint"] != first["geometry_fingerprint"]


def test_missing_linear_placement_never_reaches_provider(case, tmp_path, monkeypatch):
    state, ctx, record, *_ = case
    record.transform = None
    monkeypatch.setattr(tool, "prepare_linear_registration", prepare_linear_registration)
    model = _model(lambda *a: pytest.fail("Unexpected model call"))
    with pytest.raises(ValueError, match="affine transform"):
        _correct(state, ctx, record.id, out=tmp_path / "out", image_model=model)
    assert not (tmp_path / "out").exists()


def test_transport_can_retry_without_discarding_any_image_reply(case, tmp_path, monkeypatch):
    state, ctx, record, original, *_ = case
    calls = []

    def fail(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("transport failed")
        return SimpleNamespace(image=original, route="test")

    model = _model(fail)
    first = _correct(state, ctx, record.id, out=tmp_path / "out", image_model=model)
    repeated = _correct(state, ctx, record.id, out=tmp_path / "out", image_model=model)
    assert first["status"] == "error" and first["message"] == "transport failed"
    assert not first["raw_received"]
    assert repeated["status"] == "ok" and repeated["raw_received"]
    assert repeated["attempt"] == 2 and len(calls) == 2
    cached = _correct(state, ctx, record.id, out=tmp_path / "out", image_model=model)
    assert cached["cached"] and len(calls) == 2


def test_another_image_provider_makes_a_trace_stale(case):
    state, ctx, record, *_ = case
    first = tool.correction_fingerprint(state, ctx, record.id)
    assert tool.correction_fingerprint(state, ctx, record.id) == first
    ctx.spec.nonlinear.provider = "gemini-api"
    assert tool.correction_fingerprint(state, ctx, record.id) != first


def test_a_door_resolves_the_provider_and_none_has_no_image_model():
    resolved = resolve_image_model("openai-oauth")
    assert resolved.provider == "openai-oauth" and resolved.model
    assert resolve_image_model("gemini-api", "some-model").model == "some-model"
    with pytest.raises(ValueError, match="image-model provider"):
        resolve_image_model("none")
