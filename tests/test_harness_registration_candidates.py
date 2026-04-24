from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from PIL import Image

from langslice_harness.harness.registration.types import GeneratedSegmentation


def _candidates():
    from langslice_harness.harness.registration import image_gen_registration

    return image_gen_registration


def _make_slice(size: tuple[int, int] = (12, 8)) -> Image.Image:
    image = Image.new("RGB", size, color=(30, 40, 50))
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), (30 + x, 40 + y, 50))
    return image


def _fake_atlas() -> SimpleNamespace:
    annotation = np.array(
        [
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 2, 2],
            ]
        ],
        dtype=np.int32,
    )
    reference = np.array([[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]])
    structures = {
        1: {"rgb_triplet": [255, 0, 0], "name": "region 1"},
        2: {"rgb_triplet": [0, 255, 0], "name": "region 2"},
        3: {"rgb_triplet": [0, 0, 255], "name": "region 3"},
    }
    return SimpleNamespace(annotation=annotation, reference=reference, structures=structures)


def _install_pipeline_fakes(monkeypatch, tmp_path: Path | None = None) -> dict[str, Any]:
    candidates = _candidates()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(candidates, "load_atlas", lambda atlas_name: _fake_atlas())

    def fake_get_reference_slice(atlas, position_mm):  # noqa: ANN001 - local fake
        return Image.new("L", (4, 3), color=150)

    monkeypatch.setattr(candidates, "get_reference_slice", fake_get_reference_slice)
    monkeypatch.setattr(candidates, "_upscale_to_min_long_edge", lambda image: image)

    def fake_colored_region_slice(atlas, position_mm, target_size=None):  # noqa: ANN001
        image = Image.new("RGB", (4, 3), color=(255, 0, 0))
        image.putpixel((2, 1), (0, 255, 0))
        if target_size is not None:
            return image.resize(target_size, resample=Image.Resampling.NEAREST)
        return image

    monkeypatch.setattr(candidates, "_generate_colored_region_slice", fake_colored_region_slice)
    monkeypatch.setattr(
        candidates,
        "_classify_pixels_to_region_ids",
        lambda model_output_rgb, atlas, position_mm: np.where(
            model_output_rgb[:, :, 0] > model_output_rgb[:, :, 1],
            1,
            2,
        ).astype(np.int32),
    )

    def fake_generate(request):  # noqa: ANN001 - local fake
        calls["request"] = request
        return GeneratedSegmentation(
            image=Image.new("RGB", (5, 4), color=(200, 100, 50)),
            provider=request.provider,
            model=request.model or "fake-model",
            route="openai_images" if request.provider == "openai" else "google_genai",
            revised_prompt="provider revised prompt",
            metadata={"provider_note": "kept", "nested": {"ok": True}},
        )

    monkeypatch.setattr(candidates, "generate_warped_segmentation_image", fake_generate)

    def fake_register(atlas_target_rgb, model_output_rgb):  # noqa: ANN001 - local fake
        calls["atlas_target_shape"] = atlas_target_rgb.shape
        calls["model_output_shape"] = model_output_rgb.shape
        return SimpleNamespace(name="fake-transform"), 1.25

    monkeypatch.setattr(candidates, "_register_colored_images", fake_register)

    def fake_warp(atlas_target_rgb, transform):  # noqa: ANN001 - local fake
        calls["warp_transform"] = transform
        warped = np.zeros_like(atlas_target_rgb)
        warped[:, : warped.shape[1] // 2] = (255, 0, 0)
        warped[:, warped.shape[1] // 2 :] = (0, 255, 0)
        return warped

    monkeypatch.setattr(candidates, "_warp_atlas_rgb", fake_warp)
    monkeypatch.setattr(
        candidates,
        "_extract_visualign_markers",
        lambda transform, scale_to_slice, image_width, image_height: [
            [0.0, 0.0, float(image_width) - 1.0, float(image_height) - 1.0]
        ],
    )

    if tmp_path is not None:
        calls["debug_dir"] = str(tmp_path)
    return calls


def test_generate_registration_candidate_builds_candidate_and_metadata(monkeypatch):
    calls = _install_pipeline_fakes(monkeypatch)
    candidates = _candidates()
    progress: list[str] = []
    traces: list[dict[str, object]] = []

    candidate = candidates.generate_registration_candidate(
        _make_slice(),
        atlas_name="fake_mouse",
        position_mm=1.5,
        provider="openai",
        image_model="gpt-image-2",
        prompt_revision="tighten ventricle boundaries",
        previous_candidate_id="candidate-old",
        candidate_id="candidate-new",
        on_progress=progress.append,
        on_trace=traces.append,
        openai_image_route="images",
        review_model="gpt-4.1",
    )

    assert candidate.candidate_id == "candidate-new"
    assert candidate.generated_segmentation.size == (5, 4)
    assert candidate.warped_atlas.size == (12, 8)
    assert candidate.warped_border_overlay.size == (12, 8)
    assert candidate.markers == [[0.0, 0.0, 11.0, 7.0]]

    request = calls["request"]
    assert request.provider == "openai"
    assert request.model == "gpt-image-2"
    assert request.openai_image_route == "images"
    assert request.review_model == "gpt-4.1"
    assert request.metadata["candidate_id"] == "candidate-new"
    assert request.metadata["previous_candidate_id"] == "candidate-old"
    assert request.metadata["atlas_name"] == "fake_mouse"
    assert request.metadata["position_mm"] == 1.5
    assert "Revision guidance" in request.prompt
    assert "tighten ventricle boundaries" in request.prompt

    assert calls["atlas_target_shape"] == (8, 12, 3)
    assert calls["model_output_shape"] == (8, 12, 3)

    session = candidate.annotation_session
    assert session.workflow == "image_gen_registration"
    assert session.target_count == 0
    assert session.metadata["visualign_markers"] == candidate.markers
    assert session.metadata["n_markers"] == 1
    assert session.metadata["target_size"] == [12, 8]
    assert session.metadata["scale_to_slice"] == 1.0
    assert session.metadata["provider"] == "openai"
    assert session.metadata["model"] == "gpt-image-2"
    assert session.metadata["model_name"] == "gpt-image-2"
    assert session.metadata["route"] == "openai_images"
    assert session.metadata["candidate_id"] == "candidate-new"
    assert session.metadata["previous_candidate_id"] == "candidate-old"
    assert session.metadata["prompt_revision"] == "tighten ventricle boundaries"
    assert session.metadata["atlas_name"] == "fake_mouse"
    assert session.metadata["position_mm"] == 1.5

    assert candidate.metadata["generated"]["provider"] == "openai"
    assert candidate.metadata["generated"]["model"] == "gpt-image-2"
    assert candidate.metadata["generated"]["route"] == "openai_images"
    assert candidate.metadata["generated"]["revised_prompt"] == "provider revised prompt"
    assert candidate.metadata["generated"]["metadata"]["provider_note"] == "kept"
    assert candidate.metadata["previous_candidate_id"] == "candidate-old"
    assert progress
    assert traces


def test_generate_registration_candidate_writes_debug_artifacts(monkeypatch, tmp_path):
    _install_pipeline_fakes(monkeypatch, tmp_path)
    candidates = _candidates()

    candidates.generate_registration_candidate(
        _make_slice(),
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="debug-candidate",
        debug_dir=str(tmp_path),
    )

    artifact_dir = tmp_path / "registration" / "debug-candidate"
    expected = {
        "generated_segmentation.png",
        "warped_atlas.png",
        "warped_border_overlay.png",
        "input_colored_regions.png",
        "input_reference.png",
        "input_slice.png",
    }
    assert {path.name for path in artifact_dir.iterdir()} >= expected


def test_warped_border_overlay_marks_border_pixels(monkeypatch):
    _install_pipeline_fakes(monkeypatch)
    candidates = _candidates()
    base = _make_slice()

    candidate = candidates.generate_registration_candidate(
        base,
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="overlay-candidate",
    )

    base_rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
    overlay_rgb = np.asarray(candidate.warped_border_overlay, dtype=np.uint8)
    changed_pixels = np.any(base_rgb != overlay_rgb, axis=2)

    assert changed_pixels.any()
    changed_colors = overlay_rgb[changed_pixels]
    assert any(
        tuple(color) in {(0, 255, 255), (255, 255, 0)}
        for color in cast(Any, changed_colors.tolist())
    )
