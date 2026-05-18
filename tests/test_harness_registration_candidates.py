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

    def fake_get_reference_slice(atlas, position_mm, *, plane="coronal"):  # noqa: ANN001 - local fake
        del plane
        return Image.new("L", (4, 3), color=150)

    monkeypatch.setattr(candidates, "get_reference_slice", fake_get_reference_slice)
    monkeypatch.setattr(candidates, "_upscale_to_min_long_edge", lambda image: image)

    def fake_colored_region_slice(  # noqa: ANN001
        atlas, position_mm, target_size=None, *, plane="coronal"
    ):
        del plane
        image = Image.new("RGB", (4, 3), color=(255, 0, 0))
        image.putpixel((2, 1), (0, 255, 0))
        if target_size is not None:
            return image.resize(target_size, resample=Image.Resampling.NEAREST)
        return image

    monkeypatch.setattr(candidates, "_generate_colored_region_slice", fake_colored_region_slice)
    monkeypatch.setattr(
        candidates,
        "_classify_pixels_to_region_ids",
        lambda model_output_rgb, atlas, position_mm, *, plane="coronal": np.where(
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

    def fake_inverse_warp(slice_rgb, *, forward_fixed_gray, forward_result_transform):
        calls["inverse_slice_shape"] = slice_rgb.shape
        calls["inverse_forward_fixed_gray_shape"] = forward_fixed_gray.shape
        calls["inverse_forward_result_transform"] = forward_result_transform
        warped = np.zeros_like(slice_rgb)
        warped[:, : warped.shape[1] // 2] = (50, 100, 150)
        warped[:, warped.shape[1] // 2 :] = (200, 150, 100)
        return warped, SimpleNamespace(name="fake-inverse-transform")

    monkeypatch.setattr(candidates, "_run_inverse_warp_for_slice", fake_inverse_warp)

    def fake_build_atlas_root_mask(atlas, position_mm, target_size, *, plane="coronal"):
        # Deterministic alpha: top half opaque, bottom half transparent. Mirrors
        # the shape of a real annotation mask without requiring brainglobe-space
        # to resolve a fake SimpleNamespace atlas.
        del atlas, position_mm, plane
        width, height = target_size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[: height // 2, :] = 255
        calls["root_mask_target_size"] = target_size
        return mask

    monkeypatch.setattr(candidates, "_build_atlas_root_mask", fake_build_atlas_root_mask)

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

    candidate = candidates.generate_registration_candidate(
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
        "slice_warped_to_atlas.png",
        "slice_atlas_border_overlay.png",
    }
    assert {path.name for path in artifact_dir.iterdir()} >= expected

    # Forward + inverse warp absolute paths must be surfaced in metadata so
    # the register CLI can hoist them into its JSON payload.
    metadata_paths = candidate.metadata["artifact_paths"]
    assert metadata_paths["warped_atlas_path"] == str(
        (artifact_dir / "warped_atlas.png").resolve()
    )
    assert metadata_paths["warped_border_overlay_path"] == str(
        (artifact_dir / "warped_border_overlay.png").resolve()
    )
    assert metadata_paths["slice_warped_to_atlas_path"] == str(
        (artifact_dir / "slice_warped_to_atlas.png").resolve()
    )
    assert metadata_paths["slice_atlas_border_overlay_path"] == str(
        (artifact_dir / "slice_atlas_border_overlay.png").resolve()
    )
    # Top-level shortcuts mirror the dict entries for callers that don't
    # want to dig into artifact_paths.
    assert candidate.metadata["slice_warped_to_atlas_path"] == metadata_paths[
        "slice_warped_to_atlas_path"
    ]
    assert candidate.metadata["slice_atlas_border_overlay_path"] == metadata_paths[
        "slice_atlas_border_overlay_path"
    ]
    session_meta = candidate.annotation_session.metadata
    assert session_meta["slice_warped_to_atlas_path"] == metadata_paths[
        "slice_warped_to_atlas_path"
    ]
    assert session_meta["slice_atlas_border_overlay_path"] == metadata_paths[
        "slice_atlas_border_overlay_path"
    ]
    assert session_meta["inverse_warp_status"] == "ok"


def test_build_atlas_root_mask_produces_binary_alpha_at_target_size(monkeypatch):
    """`_build_atlas_root_mask` slices annotation at the AP index for the
    requested plane, marks non-zero structure IDs as opaque (255) and zeros
    as transparent (0), and NEAREST-resizes to *target_size* so alpha stays
    binary -- bilinear interpolation would halo the 3D-viewer silhouette."""
    from langslice_harness.harness.registration import image_gen_helpers

    # Annotation slab: top half has tissue (non-zero IDs), bottom half is bg.
    annotation = np.array(
        [
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.int32,
    )
    atlas = SimpleNamespace(annotation=annotation)

    monkeypatch.setattr(
        image_gen_helpers, "position_mm_to_index", lambda a, p, plane="coronal": 0
    )
    monkeypatch.setattr(
        image_gen_helpers, "slice_axis_index", lambda ctx, plane: 0
    )
    monkeypatch.setattr(
        image_gen_helpers, "atlas_space_context", lambda a: SimpleNamespace()
    )
    monkeypatch.setattr(
        image_gen_helpers, "orient_slice_for_display", lambda a, plane: a
    )

    target_size = (8, 8)  # (W, H) per PIL convention
    mask = image_gen_helpers._build_atlas_root_mask(
        atlas, position_mm=0.0, target_size=target_size, plane="coronal"
    )

    assert mask.shape == (8, 8)  # numpy (H, W)
    assert mask.dtype == np.uint8
    unique_vals = set(np.unique(mask).tolist())
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals
    # Top half opaque (was non-zero), bottom half transparent (was zero).
    assert (mask[0] == 255).all()
    assert (mask[-1] == 0).all()


def test_slice_warped_to_atlas_saved_as_rgba_with_root_mask_alpha(monkeypatch, tmp_path):
    """`slice_warped_to_atlas.png` must be saved as RGBA with the atlas root
    mask as alpha. The Tauri 3D viewer relies on this binary alpha to crop the
    warped slice to a brain-shaped silhouette instead of a rectangular slab."""
    _install_pipeline_fakes(monkeypatch, tmp_path)
    candidates = _candidates()

    candidates.generate_registration_candidate(
        _make_slice(),
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="rgba-mask-candidate",
        debug_dir=str(tmp_path),
    )

    saved = Image.open(
        tmp_path / "registration" / "rgba-mask-candidate" / "slice_warped_to_atlas.png"
    )
    assert saved.mode == "RGBA"
    arr = np.asarray(saved)
    assert arr.shape[-1] == 4  # H, W, RGBA
    alpha = arr[:, :, 3]
    unique_vals = set(np.unique(alpha).tolist())
    assert unique_vals.issubset({0, 255})
    assert (alpha == 0).any()
    assert (alpha == 255).any()
    # Top half opaque (matches the fake _build_atlas_root_mask pattern).
    height = alpha.shape[0]
    assert (alpha[: height // 2] == 255).all()
    assert (alpha[height // 2 :] == 0).all()


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


def _make_fake_itk_module(*, recorder: dict[str, Any]) -> SimpleNamespace:
    """Build a fake itk module that records elastix/transformix invocations."""

    class FakeImage:
        def __init__(self, array: np.ndarray) -> None:
            self.array = np.asarray(array)

    def image_from_array(arr):  # noqa: ANN001
        return FakeImage(arr)

    def array_from_image(img):  # noqa: ANN001
        return img.array

    def elastix_registration_method(  # noqa: ANN001
        fixed_image,
        moving_image,
        *,
        parameter_object=None,
        initial_transform_parameter_file_name=None,
        log_to_console=False,
    ):
        recorder.setdefault("elastix_calls", []).append(
            {
                "fixed_image": fixed_image,
                "moving_image": moving_image,
                "parameter_object": parameter_object,
                "initial_transform_parameter_file_name": (
                    initial_transform_parameter_file_name
                ),
                "log_to_console": log_to_console,
                "fixed_is_moving": fixed_image is moving_image,
            }
        )
        # Return (result_image, inverse_transform) where the transform is just
        # a sentinel object for downstream identification.
        result_image = FakeImage(np.zeros_like(fixed_image.array))
        inverse_transform = SimpleNamespace(name="inverse-transform")
        return result_image, inverse_transform

    def transformix_filter(channel_image, transform):  # noqa: ANN001
        recorder.setdefault("transformix_calls", []).append(
            {"channel_array": channel_image.array, "transform": transform}
        )
        # Pretend the warp is identity for testing purposes.
        return FakeImage(channel_image.array)

    return SimpleNamespace(
        image_from_array=image_from_array,
        array_from_image=array_from_image,
        elastix_registration_method=elastix_registration_method,
        transformix_filter=transformix_filter,
    )


def test_warp_slice_to_atlas_runs_fixed_to_fixed_inverse_pattern(monkeypatch):
    """`_warp_slice_to_atlas` must call elastix with fixed=moving plus the
    forward transform file as initial guess, then run transformix per RGB
    channel of the slice. Matches the canonical itk-elastix Example 11."""
    from langslice_harness.harness.registration import image_gen_helpers

    recorder: dict[str, Any] = {}
    fake_itk = _make_fake_itk_module(recorder=recorder)
    monkeypatch.setitem(__import__("sys").modules, "itk", fake_itk)

    slice_rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    slice_rgb[..., 0] = 10
    slice_rgb[..., 1] = 20
    slice_rgb[..., 2] = 30
    fixed_image_itk = fake_itk.image_from_array(np.zeros((4, 6), dtype=np.float32))
    parameter_object = SimpleNamespace(name="fake-parameter-object")

    warped, inverse_transform = image_gen_helpers._warp_slice_to_atlas(
        slice_rgb=slice_rgb,
        parameter_object=parameter_object,
        forward_transform_params_path="/fake/TransformParameters.1.txt",
        fixed_image_itk=fixed_image_itk,
    )

    # Exactly one Elastix call: fixed-to-fixed with initial forward transform.
    elastix_calls = recorder["elastix_calls"]
    assert len(elastix_calls) == 1
    call = elastix_calls[0]
    assert call["fixed_is_moving"] is True
    assert call["fixed_image"] is fixed_image_itk
    assert call["parameter_object"] is parameter_object
    assert call["initial_transform_parameter_file_name"] == (
        "/fake/TransformParameters.1.txt"
    )

    # Transformix runs once per RGB channel with the inverse transform.
    tf_calls = recorder["transformix_calls"]
    assert len(tf_calls) == 3
    for ch_idx, tf_call in enumerate(tf_calls):
        np.testing.assert_array_equal(tf_call["channel_array"], slice_rgb[:, :, ch_idx])
        assert tf_call["transform"] is inverse_transform

    # Returned warped slice has slice's shape and uint8 dtype.
    assert warped.shape == slice_rgb.shape
    assert warped.dtype == np.uint8


def test_run_inverse_warp_for_slice_writes_forward_transform_to_disk(monkeypatch, tmp_path):
    """`_run_inverse_warp_for_slice` should serialize the forward transform to
    a temp file before delegating to `_warp_slice_to_atlas` -- ITKElastix's
    Python binding only accepts the forward transform as a file path."""
    from langslice_harness.harness.registration import image_gen_helpers

    recorder: dict[str, Any] = {}
    fake_itk = _make_fake_itk_module(recorder=recorder)
    monkeypatch.setitem(__import__("sys").modules, "itk", fake_itk)

    written_paths: list[str] = []

    class FakeResultTransform:
        def __init__(self, n_maps: int = 2) -> None:
            self._n_maps = n_maps

        def GetNumberOfParameterMaps(self) -> int:  # noqa: N802 - itk API name
            return self._n_maps

        def GetParameterMap(self, idx: int):  # noqa: N802 - itk API name
            return SimpleNamespace(idx=idx)

        def WriteParameterFile(self, parameter_map, path):  # noqa: N802 - itk API name
            written_paths.append(str(path))
            # Touch the file so a downstream `os.path.exists` check would pass.
            Path(path).write_text(f"# fake parameter map {parameter_map.idx}\n")

    # Replace ParameterObject construction so we don't need a real itk install.
    monkeypatch.setattr(
        image_gen_helpers,
        "_build_elastix_parameter_object",
        lambda: SimpleNamespace(name="fake-parameter-object"),
    )

    slice_rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    forward_fixed_gray = np.zeros((3, 5), dtype=np.float32)
    forward_transform = FakeResultTransform(n_maps=2)

    warped, _inverse = image_gen_helpers._run_inverse_warp_for_slice(
        slice_rgb,
        forward_fixed_gray=forward_fixed_gray,
        forward_result_transform=forward_transform,
    )

    # Both parameter maps must be written so the chained
    # InitialTransformParametersFileName references resolve correctly.
    assert len(written_paths) == 2
    # The last-stage path is what gets passed to elastix as initial transform.
    last_written = written_paths[-1]
    elastix_calls = recorder["elastix_calls"]
    assert len(elastix_calls) == 1
    assert elastix_calls[0]["initial_transform_parameter_file_name"] == last_written
    assert warped.shape == slice_rgb.shape


def test_register_cli_json_payload_includes_inverse_warp_paths(monkeypatch, tmp_path, capsys):
    """End-to-end: the `langslice register --json` payload must include the
    new inverse-warp file paths so downstream consumers (Tauri GUI) can
    discover the generated PNGs."""
    import argparse
    import json

    import langslice_harness.cli as cli
    import langslice_harness.registration.core as registration_core
    from langslice_harness.registration.types import (
        AffineResult,
        NonlinearResult,
        RegistrationAnnotationSession,
        RegistrationResult,
        identity_affine_matrix,
    )

    inverse_path = tmp_path / "registration" / "candidate-1" / "slice_warped_to_atlas.png"
    overlay_path = (
        tmp_path / "registration" / "candidate-1" / "slice_atlas_border_overlay.png"
    )
    forward_warp_path = tmp_path / "registration" / "candidate-1" / "warped_atlas.png"
    forward_overlay_path = (
        tmp_path / "registration" / "candidate-1" / "warped_border_overlay.png"
    )

    artifact_paths = {
        "warped_atlas_path": str(forward_warp_path),
        "warped_border_overlay_path": str(forward_overlay_path),
        "generated_segmentation_path": None,
        "slice_warped_to_atlas_path": str(inverse_path),
        "slice_atlas_border_overlay_path": str(overlay_path),
    }

    def fake_runtime(image, *, on_progress=None, on_trace=None, **kwargs):  # noqa: ANN001
        del image, on_progress, on_trace, kwargs
        candidate_metadata = {
            **artifact_paths,
            "inverse_warp_status": "ok",
        }
        session_metadata = {
            "visualign_markers": [],
            "n_markers": 0,
            "candidate_metadata": candidate_metadata,
            "artifact_paths": dict(artifact_paths),
            "inverse_warp_status": "ok",
            **artifact_paths,
        }
        session = RegistrationAnnotationSession(
            workflow="image_gen_registration",
            target_count=0,
            metadata=session_metadata,
        )
        affine = AffineResult(
            matrix=identity_affine_matrix(),
            source_size=(16, 16),
            output_size=(16, 16),
            backend="image_gen_registration_dense",
            reasoning="fake",
        )
        nonlinear = NonlinearResult(
            atlas_points=np.zeros((0, 2)),
            slice_points=np.zeros((0, 2)),
            smoothing=0.0,
            backend="elastix_bspline_visualign",
            reasoning="fake",
            output_size=(16, 16),
        )
        return RegistrationResult(
            correspondences=[],
            accepted_correspondences=[],
            affine_result=affine,
            nonlinear_result=nonlinear,
            annotation_session=session,
        )

    monkeypatch.setattr(
        cli, "estimate_registration_runtime", fake_runtime, raising=False
    )
    # The cli imports it as a local symbol; patch the source module too.
    monkeypatch.setattr(
        registration_core, "estimate_registration_runtime", fake_runtime, raising=False
    )

    slice_path = tmp_path / "slice.png"
    Image.new("RGB", (16, 16), color=(120, 130, 140)).save(slice_path)
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        image=str(slice_path),
        atlas="allen_mouse_25um",
        position=1.5,
        plane="coronal",
        registration_mode="direct",
        model=None,
        image_model=None,
        openai_image_route="images",
        review_model=None,
        max_candidates=1,
        vlm_resolution=2048,
        temperature=None,
        thinking=None,
        out=str(out_dir),
        provider="google",
        json=True,
    )

    cli._run_register(args)
    captured = capsys.readouterr().out
    # The JSON payload is the last block separated by a blank line.
    json_blob = captured[captured.find("{") :]
    payload = json.loads(json_blob)

    assert payload["slice_warped_to_atlas_path"] == str(inverse_path)
    assert payload["slice_atlas_border_overlay_path"] == str(overlay_path)
    assert payload["warped_atlas_path"] == str(forward_warp_path)
    assert payload["warped_border_overlay_path"] == str(forward_overlay_path)
    # When the inverse warp succeeds the top-level status is "ok" so the
    # GUI can distinguish success from "no inverse run" from failure.
    assert payload["inverse_warp_status"] == "ok"


def test_generate_registration_candidate_handles_inverse_warp_failure(monkeypatch, tmp_path):
    """If `_run_inverse_warp_for_slice` raises, the candidate must still ship
    with forward artifacts intact, inverse paths absent, and a "failed: ..."
    status surfaced in both session_metadata and candidate_metadata so the
    register CLI can hoist it to its top-level JSON payload."""
    _install_pipeline_fakes(monkeypatch, tmp_path)
    candidates = _candidates()

    def boom(slice_rgb, *, forward_fixed_gray, forward_result_transform):  # noqa: ANN001
        del slice_rgb, forward_fixed_gray, forward_result_transform
        raise RuntimeError("simulated elastix divergence")

    monkeypatch.setattr(candidates, "_run_inverse_warp_for_slice", boom)

    progress: list[str] = []
    candidate = candidates.generate_registration_candidate(
        _make_slice(),
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="inverse-failure-candidate",
        debug_dir=str(tmp_path),
        on_progress=progress.append,
    )

    # Forward artifacts still emitted: the failure must not poison the
    # forward pipeline outputs.
    assert candidate.candidate_id == "inverse-failure-candidate"
    assert candidate.warped_atlas.size == (12, 8)
    assert candidate.warped_border_overlay.size == (12, 8)
    artifact_dir = tmp_path / "registration" / "inverse-failure-candidate"
    forward_artifacts = {path.name for path in artifact_dir.iterdir()}
    assert "warped_atlas.png" in forward_artifacts
    assert "warped_border_overlay.png" in forward_artifacts
    # Inverse PNGs must NOT have been written when the inverse warp failed.
    assert "slice_warped_to_atlas.png" not in forward_artifacts
    assert "slice_atlas_border_overlay.png" not in forward_artifacts

    # Status surfaced in session_metadata and candidate_metadata, starting
    # with "failed:" and containing the original exception message.
    session_meta = candidate.annotation_session.metadata
    assert session_meta["inverse_warp_status"].startswith("failed:")
    assert "RuntimeError" in session_meta["inverse_warp_status"]
    assert "simulated elastix divergence" in session_meta["inverse_warp_status"]
    assert candidate.metadata["inverse_warp_status"] == session_meta["inverse_warp_status"]

    # Inverse-warp artifact paths must be absent (None) on failure.
    assert session_meta["artifact_paths"]["slice_warped_to_atlas_path"] is None
    assert session_meta["artifact_paths"]["slice_atlas_border_overlay_path"] is None
    assert "slice_warped_to_atlas_path" not in session_meta or session_meta.get(
        "slice_warped_to_atlas_path"
    ) in (None,)
    # Forward paths still surfaced as strings (sanity check).
    assert isinstance(session_meta["artifact_paths"]["warped_atlas_path"], str)

    # Progress callback received the user-facing skip message.
    assert any("inverse warp skipped" in msg for msg in progress)


def test_register_cli_json_payload_surfaces_inverse_warp_failure(
    monkeypatch, tmp_path, capsys
):
    """End-to-end: when the inverse warp fails, the `langslice register --json`
    payload must hoist `inverse_warp_status` to the top level with the
    "failed: ..." string and emit `None` for the inverse paths so the Tauri
    GUI can show an error state without digging into nested metadata."""
    import argparse
    import json

    import langslice_harness.cli as cli
    import langslice_harness.registration.core as registration_core
    from langslice_harness.registration.types import (
        AffineResult,
        NonlinearResult,
        RegistrationAnnotationSession,
        RegistrationResult,
        identity_affine_matrix,
    )

    forward_warp_path = tmp_path / "registration" / "candidate-1" / "warped_atlas.png"
    forward_overlay_path = (
        tmp_path / "registration" / "candidate-1" / "warped_border_overlay.png"
    )
    failure_status = "failed: RuntimeError: simulated elastix divergence"

    artifact_paths = {
        "warped_atlas_path": str(forward_warp_path),
        "warped_border_overlay_path": str(forward_overlay_path),
        "generated_segmentation_path": None,
        "slice_warped_to_atlas_path": None,
        "slice_atlas_border_overlay_path": None,
    }

    def fake_runtime(image, *, on_progress=None, on_trace=None, **kwargs):  # noqa: ANN001
        del image, on_progress, on_trace, kwargs
        candidate_metadata = {
            **artifact_paths,
            "inverse_warp_status": failure_status,
        }
        session_metadata = {
            "visualign_markers": [],
            "n_markers": 0,
            "candidate_metadata": candidate_metadata,
            "artifact_paths": dict(artifact_paths),
            "inverse_warp_status": failure_status,
            # Forward paths are still surfaced as flat keys.
            "warped_atlas_path": str(forward_warp_path),
            "warped_border_overlay_path": str(forward_overlay_path),
        }
        session = RegistrationAnnotationSession(
            workflow="image_gen_registration",
            target_count=0,
            metadata=session_metadata,
        )
        affine = AffineResult(
            matrix=identity_affine_matrix(),
            source_size=(16, 16),
            output_size=(16, 16),
            backend="image_gen_registration_dense",
            reasoning="fake",
        )
        nonlinear = NonlinearResult(
            atlas_points=np.zeros((0, 2)),
            slice_points=np.zeros((0, 2)),
            smoothing=0.0,
            backend="elastix_bspline_visualign",
            reasoning="fake",
            output_size=(16, 16),
        )
        return RegistrationResult(
            correspondences=[],
            accepted_correspondences=[],
            affine_result=affine,
            nonlinear_result=nonlinear,
            annotation_session=session,
        )

    monkeypatch.setattr(
        cli, "estimate_registration_runtime", fake_runtime, raising=False
    )
    monkeypatch.setattr(
        registration_core, "estimate_registration_runtime", fake_runtime, raising=False
    )

    slice_path = tmp_path / "slice.png"
    Image.new("RGB", (16, 16), color=(120, 130, 140)).save(slice_path)
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        image=str(slice_path),
        atlas="allen_mouse_25um",
        position=1.5,
        plane="coronal",
        registration_mode="direct",
        model=None,
        image_model=None,
        openai_image_route="images",
        review_model=None,
        max_candidates=1,
        vlm_resolution=2048,
        temperature=None,
        thinking=None,
        out=str(out_dir),
        provider="google",
        json=True,
    )

    cli._run_register(args)
    captured = capsys.readouterr().out
    json_blob = captured[captured.find("{") :]
    payload = json.loads(json_blob)

    # Top-level status carries the failure string verbatim.
    assert payload["inverse_warp_status"] == failure_status
    assert payload["inverse_warp_status"].startswith("failed:")
    assert "simulated elastix divergence" in payload["inverse_warp_status"]
    # Inverse paths are None when the inverse warp failed.
    assert payload["slice_warped_to_atlas_path"] is None
    assert payload["slice_atlas_border_overlay_path"] is None
    # Forward paths still emit, so the GUI can show forward-only artifacts.
    assert payload["warped_atlas_path"] == str(forward_warp_path)
    assert payload["warped_border_overlay_path"] == str(forward_overlay_path)
