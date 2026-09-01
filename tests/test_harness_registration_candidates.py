from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from PIL import Image

from langslice.nonlinear.types import GeneratedSegmentation


def _candidates():
    from langslice.nonlinear import image_gen_registration

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
    template = np.array([[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]])
    structures = {
        1: {"rgb_triplet": [255, 0, 0], "name": "region 1"},
        2: {"rgb_triplet": [0, 255, 0], "name": "region 2"},
        3: {"rgb_triplet": [0, 0, 255], "name": "region 3"},
    }
    return SimpleNamespace(annotation=annotation, template=template, structures=structures)


def _install_pipeline_fakes(monkeypatch, tmp_path: Path | None = None) -> dict[str, Any]:
    candidates = _candidates()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(candidates, "load_atlas", lambda atlas_name: _fake_atlas())
    # The real atlas.resolution/orientation machinery doesn't apply to this
    # stub atlas; the reference band just needs a range wide enough that the
    # +-0.125mm offsets never clamp.
    monkeypatch.setattr(
        candidates, "get_position_range_mm", lambda atlas, plane="coronal": (-10.0, 10.0)
    )

    def fake_colored_region_slice(  # noqa: ANN001
        atlas, position_mm, target_size=None, *, plane="coronal", smooth=True,
        pitch_deg=0.0, yaw_deg=0.0
    ):
        del plane, smooth, pitch_deg, yaw_deg
        image = Image.new("RGB", (4, 3), color=(255, 0, 0))
        image.putpixel((2, 1), (0, 255, 0))
        if target_size is not None:
            return image.resize(target_size, resample=Image.Resampling.NEAREST)
        return image

    monkeypatch.setattr(candidates, "_generate_colored_region_slice", fake_colored_region_slice)
    monkeypatch.setattr(
        candidates,
        "_classify_pixels_to_region_ids",
        lambda model_output_rgb, atlas, position_mm, *, plane="coronal",
        off_palette_background=True, pitch_deg=0.0, yaw_deg=0.0: np.where(
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

    def fake_register(atlas_classified, generated_classified, atlas, fixed_mask=None):  # noqa: ANN001
        calls["atlas_target_shape"] = atlas_classified.shape
        calls["model_output_shape"] = generated_classified.shape
        calls["fixed_mask_used"] = fixed_mask is not None and bool(fixed_mask.any())
        return SimpleNamespace(name="fake-transform"), 1.25

    monkeypatch.setattr(candidates, "_register_region_maps", fake_register)

    def fake_warp_labels(classified, transform):  # noqa: ANN001 - local fake
        calls["warp_transform"] = transform
        warped = np.zeros_like(classified)
        warped[:, : warped.shape[1] // 2] = 1
        warped[:, warped.shape[1] // 2 :] = 2
        return warped

    monkeypatch.setattr(candidates, "_warp_classified_labels", fake_warp_labels)

    def fake_classified_to_rgb(classified, atlas):  # noqa: ANN001 - local fake
        rgb = np.zeros((*classified.shape, 3), dtype=np.uint8)
        rgb[classified == 1] = (255, 0, 0)
        rgb[classified == 2] = (0, 255, 0)
        return rgb

    monkeypatch.setattr(candidates, "_classified_to_rgb", fake_classified_to_rgb)
    monkeypatch.setattr(candidates, "_merge_classified", lambda classified, atlas: classified)
    monkeypatch.setattr(
        candidates, "generation_report", lambda *args, **kwargs: {"flags": []}
    )
    monkeypatch.setattr(
        candidates,
        "_extract_visualign_markers",
        lambda field, scale_to_slice, origin_px=(0.0, 0.0): [[0.0, 0.0, 11.0, 7.0]],
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
        image_prompt="Repaint the section in atlas colors.",
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
    # The agent-authored prompt goes through verbatim; no harness additions.
    assert request.prompt == "Repaint the section in atlas colors."
    assert request.metadata["image_prompt"] == request.prompt

    assert calls["atlas_target_shape"] == (8, 12)
    assert calls["model_output_shape"] == (8, 12)

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
    assert session.metadata["image_prompt"] == "Repaint the section in atlas colors."
    assert isinstance(session.metadata["elastix"]["codes"], list)
    assert candidate.metadata["elastix"] == session.metadata["elastix"]
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
        "input_reference_2.png",
        "input_reference_4.png",
        "input_slice.png",
        "slice_warped_to_atlas.png",
        "slice_atlas_border_overlay.png",
        "region_overlay.png",
        "checkerboard.png",
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
    binary -- bilinear interpolation would halo the 3D-viewer silhouette.

    The implementation now lives in `langslice.atlas.core.get_root_mask` (it is
    an atlas accessor, and the whole-brain transform step needs it too); the
    name here is an alias, so this exercises both."""
    from langslice.atlas import core as atlas_core
    from langslice.nonlinear import image_gen_helpers

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
        atlas_core, "position_mm_to_index", lambda a, p, plane="coronal": 0
    )
    monkeypatch.setattr(atlas_core, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(atlas_core, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(atlas_core, "orient_slice_for_display", lambda a, plane: a)

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
    # Alpha is the atlas silhouette from the letterboxed Elastix-side render:
    # content opaque, letterbox margins transparent.
    assert (alpha == 255).mean() > 0.8
    assert (alpha == 255).any()


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
    from langslice.nonlinear import image_gen_helpers

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
    from langslice.nonlinear import image_gen_helpers

    recorder: dict[str, Any] = {}
    fake_itk = _make_fake_itk_module(recorder=recorder)
    monkeypatch.setitem(__import__("sys").modules, "itk", fake_itk)

    class FakeResultTransform:
        def __init__(self, n_maps: int = 2) -> None:
            self._n_maps = n_maps

        def GetNumberOfParameterMaps(self) -> int:  # noqa: N802 - itk API name
            return self._n_maps

        def GetParameterMap(self, idx: int):  # noqa: N802 - itk API name
            return {
                "Transform": ["AffineTransform"],
                "InitialTransformParameterFileName": [""],
                "Stage": [str(idx)],
            }

    # Replace ParameterObject construction so we don't need a real itk install.
    monkeypatch.setattr(
        image_gen_helpers,
        "_build_elastix_parameter_object",
        lambda grid_spacing=32: SimpleNamespace(name="fake-parameter-object"),
    )

    slice_rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    forward_fixed_gray = np.zeros((3, 5), dtype=np.float32)
    forward_transform = FakeResultTransform(n_maps=2)

    warped, _inverse = image_gen_helpers._run_inverse_warp_for_slice(
        slice_rgb,
        forward_fixed_gray=forward_fixed_gray,
        forward_result_transform=forward_transform,
    )

    # The last-stage path is what gets passed to elastix as initial transform.
    elastix_calls = recorder["elastix_calls"]
    assert len(elastix_calls) == 1
    last_written = Path(elastix_calls[0]["initial_transform_parameter_file_name"])
    assert last_written.name == "TransformParameters.1.txt"
    assert warped.shape == slice_rgb.shape


def test_register_cli_json_payload_includes_inverse_warp_paths(monkeypatch, tmp_path, capsys):
    """End-to-end: the `langslice register --json` payload must include the
    new inverse-warp file paths so downstream consumers (Tauri GUI) can
    discover the generated PNGs."""
    import argparse
    import json

    import langslice.cli as cli
    import langslice.nonlinear.runtime as registration_runtime
    from langslice.nonlinear.types import (
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

    # api.runtime.run_register imports this as a local symbol; patch the
    # source module so the local re-import picks up the fake.
    monkeypatch.setattr(
        registration_runtime, "estimate_registration", fake_runtime, raising=False
    )

    slice_path = tmp_path / "slice.png"
    Image.new("RGB", (16, 16), color=(120, 130, 140)).save(slice_path)
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        image=str(slice_path),
        atlas="allen_mouse_25um",
        position=1.5,
        plane="coronal",
        model=None,
        image_model=None,
        openai_image_route="images",
        review_model=None,
        canvas_pad=0.0,
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

    import langslice.cli as cli
    import langslice.nonlinear.runtime as registration_runtime
    from langslice.nonlinear.types import (
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
        registration_runtime, "estimate_registration", fake_runtime, raising=False
    )

    slice_path = tmp_path / "slice.png"
    Image.new("RGB", (16, 16), color=(120, 130, 140)).save(slice_path)
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        image=str(slice_path),
        atlas="allen_mouse_25um",
        position=1.5,
        plane="coronal",
        model=None,
        image_model=None,
        openai_image_route="images",
        review_model=None,
        canvas_pad=0.0,
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


def test_elastix_report_emits_codes_only_for_implausible_warps():
    from langslice.nonlinear.image_gen_helpers import _elastix_report

    atlas = np.zeros((20, 20), dtype=np.int32)
    atlas[:10] = 1
    atlas[10:15] = 2
    atlas[15:] = 3
    structures = {2: {"acronym": "TH"}, 3: {"acronym": "CB"}}

    # Healthy: identical classification, identity deformation, tissue covered.
    identity = np.zeros((20, 20, 2), dtype=np.float64)
    healthy = _elastix_report(
        atlas_classified=atlas,
        warped_classified=atlas.copy(),
        structures=structures,
        deformation_field=identity,
        tissue_mask=atlas != 0,
    )
    assert healthy["codes"] == []

    # Region 2 vanished, region 3 collapsed to a sliver, and the field folds.
    warped = np.ones((20, 20), dtype=np.int32)
    warped[19, :10] = 3
    folding = np.zeros((20, 20, 2), dtype=np.float64)
    folding[..., 0] = -2.0 * np.arange(20)[np.newaxis, :]
    report = _elastix_report(
        atlas_classified=atlas,
        warped_classified=warped,
        structures=structures,
        deformation_field=folding,
        tissue_mask=None,
    )
    codes = {entry["code"] for entry in report["codes"]}
    assert codes == {"REGION_MISSING", "REGION_COLLAPSED", "WARP_FOLDS"}
    missing = next(e for e in report["codes"] if e["code"] == "REGION_MISSING")
    assert missing["region"] == "TH"

    # All-background warp leaves the real tissue uncovered.
    uncovered = _elastix_report(
        atlas_classified=atlas,
        warped_classified=np.zeros((20, 20), dtype=np.int32),
        structures=structures,
        tissue_mask=np.ones((20, 20), dtype=bool),
    )
    assert {e["code"] for e in uncovered["codes"]} == {"TISSUE_UNCOVERED"}


def test_visualign_markers_sample_the_composed_deformation_field():
    from langslice.nonlinear.image_gen_helpers import _extract_visualign_markers

    field = np.zeros((100, 100, 2), dtype=np.float64)
    field[..., 0] = 3.0  # constant +3 px displacement in x, none in y
    markers = _extract_visualign_markers(field, 2.0)
    assert markers
    for ox, oy, nx, ny in markers:
        assert nx - ox == 6.0  # displacement scaled into slice pixels
        assert ny == oy

    # A padded-canvas origin shifts markers into the original frame, and
    # out-of-bounds coordinates survive: the atlas may exceed the picture.
    shifted = _extract_visualign_markers(field, 2.0, origin_px=(50.0, 50.0))
    assert any(m[0] < 0 for m in shifted)

    assert _extract_visualign_markers(None, 1.0) == []


def test_canvas_pad_grows_working_canvas_and_reports_offsets(monkeypatch):
    _install_pipeline_fakes(monkeypatch)
    candidates = _candidates()

    candidate = candidates.generate_registration_candidate(
        _make_slice(),  # 12x8
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="pad-candidate",
        canvas_pad=0.25,
    )

    # pad = round(0.25 * 12) = 3 px per side -> 18x14 (aspect within range,
    # so no further clamp).
    assert candidate.metadata["pad_px"] == 3
    assert candidate.metadata["canvas_pad"] == 0.25
    assert candidate.metadata["target_size"] == [18, 14]
    assert candidate.metadata["canvas_origin_px"] == [3.0, 3.0]
    assert candidate.warped_atlas.size == (18, 14)


def test_region_ledger_accounts_for_every_region_in_physical_units():
    from langslice.nonlinear.image_gen_helpers import _region_ledger

    atlas = np.zeros((10, 10), dtype=np.int32)
    atlas[:5] = 1
    atlas[5:] = 2
    warped = np.zeros((10, 10), dtype=np.int32)
    warped[:8] = 1  # region 2 shrank; region 3 appears only in the warp
    warped[8:] = 3
    ledger = _region_ledger(
        atlas_classified=atlas,
        warped_classified=warped,
        structures={1: {"acronym": "CTX"}, 2: {"acronym": "TH"}, 3: {"acronym": "CB"}},
        mm2_per_px=0.01,
    )
    rows = {r["region"]: r for r in ledger}
    assert set(rows) == {"CTX", "TH", "CB"}
    assert rows["CTX"]["expected_mm2"] == 0.5 and rows["CTX"]["warped_mm2"] == 0.8
    assert rows["TH"]["warped_px"] == 0
    assert rows["CB"]["expected_px"] == 0 and rows["CB"]["warped_share"] == 0.2


def test_extreme_aspect_ratio_clamps_into_the_supported_range(monkeypatch):
    """gpt-image-2 accepts 1:3..3:1; a 4:1 strip black-pads down to 3:1."""
    _install_pipeline_fakes(monkeypatch)
    candidates = _candidates()

    candidate = candidates.generate_registration_candidate(
        _make_slice((48, 12)),  # 4:1 -> pad height to reach 3:1
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="clamp-candidate",
        image_model="gpt-image-2",
    )

    assert candidate.metadata["target_size"] == [48, 16]  # ceil(48 / 3)
    assert candidate.metadata["canvas_origin_px"] == [0.0, 2.0]  # centered pad
    assert candidate.warped_atlas.size == (48, 16)


def test_in_range_aspect_ratio_is_left_untouched(monkeypatch):
    _install_pipeline_fakes(monkeypatch)
    candidates = _candidates()
    candidate = candidates.generate_registration_candidate(
        _make_slice((24, 9)),  # 2.67:1 — inside 1:3..3:1, no clamp
        atlas_name="fake_mouse",
        position_mm=1.5,
        candidate_id="noop-candidate",
        image_model="gpt-image-2",
    )
    assert candidate.metadata["target_size"] == [24, 9]
    assert candidate.metadata["canvas_origin_px"] == [0.0, 0.0]


def test_classifier_treats_off_palette_pixels_as_background(monkeypatch):
    """A preserved white slide background must classify as background, not as
    the nearest region color."""
    from types import SimpleNamespace

    import langslice.nonlinear.image_gen_helpers as helpers

    atlas = SimpleNamespace(
        annotation=np.array([[[1, 2]]], dtype=np.int32),
        structures={
            1: {"id": 1, "acronym": "A", "name": "A",
                "structure_id_path": [1], "rgb_triplet": [255, 0, 0]},
            2: {"id": 2, "acronym": "B", "name": "B",
                "structure_id_path": [1, 2], "rgb_triplet": [0, 255, 0]},
        },
    )
    monkeypatch.setattr(helpers, "position_mm_to_index", lambda a, p, plane="coronal": 0)
    monkeypatch.setattr(helpers, "slice_axis_index", lambda ctx, plane: 0)
    monkeypatch.setattr(helpers, "atlas_space_context", lambda a: SimpleNamespace())
    monkeypatch.setattr(helpers, "orient_slice_for_display", lambda a, plane: a)

    rgb = np.array(
        [[[255, 255, 255], [250, 4, 6], [0, 0, 0], [0, 250, 10]]], dtype=np.uint8
    )
    classified = helpers._classify_pixels_to_region_ids(rgb, atlas, 0.0)
    assert classified.tolist() == [[0, 1, 0, 2]]  # white and black -> background


def test_prealign_scales_atlas_to_painting_and_clamps_for_fragments():
    candidates = _candidates()
    atlas_map = np.zeros((100, 100), dtype=np.int64)
    atlas_map[40:60, 40:60] = 7  # 20px blob, centered
    paint = np.zeros((100, 100), dtype=np.int64)
    paint[39:61, 38:62] = 7  # slightly bigger, slightly shifted
    aligned, matrix = candidates._prealign_atlas_to_paint(atlas_map, paint)
    assert matrix is not None
    # aligned silhouette covers the painting's extent to within a pixel or two
    ys, xs = np.nonzero(aligned)
    assert abs(ys.min() - 39) <= 2 and abs(ys.max() - 60) <= 2
    assert abs(xs.min() - 38) <= 2 and abs(xs.max() - 61) <= 2
    # a hemibrain-sized painting cannot shrink the atlas onto itself
    frag = np.zeros((100, 100), dtype=np.int64)
    frag[45:55, 45:50] = 7
    _aligned, m2 = candidates._prealign_atlas_to_paint(atlas_map, frag)
    lo, _hi = candidates._PREALIGN_SCALE_BOUNDS
    assert m2 is not None and m2[0, 0] >= lo and m2[1, 1] >= lo
    # degenerate inputs pass through untouched
    same, none_matrix = candidates._prealign_atlas_to_paint(
        atlas_map, np.zeros_like(paint)
    )
    assert none_matrix is None and same is atlas_map
