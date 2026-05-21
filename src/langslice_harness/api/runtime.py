"""Runtime wrappers for the engine API."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from langslice_harness.api.models import (
    EngineLogEvent,
    EngineProgressEvent,
    EstimateRequest,
    EstimateResult,
    ExportRequest,
    ExportResult,
    QuickAffineRequest,
    QuickAffineResult,
    RegisterRequest,
    RegisterResult,
    VersionResult,
)

EngineEmit = Callable[[EngineProgressEvent | EngineLogEvent], None]


def _emit(emit: EngineEmit | None, event: EngineProgressEvent | EngineLogEvent) -> None:
    if emit is not None:
        emit(event)


def _progress(emit: EngineEmit | None, message: str, *, stage: str) -> None:
    _emit(emit, EngineProgressEvent(kind="progress", message=message, stage=stage))


def _log(emit: EngineEmit | None, message: str) -> None:
    _emit(emit, EngineLogEvent(kind="log", message=message))


@contextmanager
def _preserve_runtime_state():
    env_keys = ("LANGSLICE_ENDPOINT", "LANGSLICE_VLM_DEBUG_DIR")
    prior_env = {key: os.environ.get(key) for key in env_keys}
    vlm_module = None
    prior_temperature = None
    prior_thinking = None
    try:
        import langslice_harness.vlm_config as _vlm_config

        vlm_module = _vlm_config
        prior_temperature = getattr(_vlm_config, "TEMPERATURE", None)
        prior_thinking = getattr(_vlm_config, "THINKING_LEVEL", None)
    except Exception:  # pragma: no cover - defensive fallback only
        vlm_module = None

    try:
        yield
    finally:
        for key, value in prior_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if vlm_module is not None:
            vlm_module.TEMPERATURE = prior_temperature
            vlm_module.THINKING_LEVEL = prior_thinking


def get_version() -> VersionResult:
    import langslice_harness

    return VersionResult(version=langslice_harness.__version__)


def run_estimate(request: EstimateRequest, emit: EngineEmit | None = None) -> EstimateResult:
    with _preserve_runtime_state():
        from PIL import Image

        from langslice_harness import vlm_config
        from langslice_harness.estimation import estimate_position
        from langslice_harness.image_prep import (
            adaptive_preprocess,
            normalize_image,
            prepare_image_for_vlm,
        )

        if request.endpoint:
            os.environ["LANGSLICE_ENDPOINT"] = request.endpoint

        debug_dir = request.output_dir
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            os.environ["LANGSLICE_VLM_DEBUG_DIR"] = debug_dir

        _progress(emit, f"Loading image: {request.image_path}", stage="estimate")
        raw_image = Image.open(request.image_path)
        image = prepare_image_for_vlm(normalize_image(raw_image)).image
        if request.preprocess == "auto":
            image = adaptive_preprocess(image)

        if request.provider == "openai":
            import langslice_harness.openai_config as openai_config

            model_name = request.model or openai_config.get_openai_model()
        else:
            model_name = request.model or vlm_config.MODEL_NAME
            if request.temperature is not None:
                vlm_config.set_temperature(request.temperature)
            if request.thinking is not None:
                vlm_config.set_thinking_level(request.thinking)

        workflow = request.workflow
        if workflow is None:
            workflow = (
                "image_gen"
                if request.provider == "google" and vlm_config.is_image_generation_model(model_name)
                else "tool_use"
            )

        _log(emit, f"Running estimate workflow={workflow} provider={request.provider}")

        def on_progress(message: str) -> None:
            _progress(emit, message, stage="estimate")

        if workflow == "image_gen":
            if request.provider != "google":
                raise ValueError("image_gen workflow is only available for google provider")
            if request.plane != "coronal":
                raise ValueError("image_gen workflow currently supports only coronal plane")
            from langslice_harness.harness.estimation.image_gen import estimate_position_image_gen

            result = estimate_position_image_gen(
                image=image,
                atlas_name=request.atlas,
                on_progress=on_progress,
                model_name=model_name,
                debug_dir=debug_dir,
            )
        else:
            result = estimate_position(
                image=image,
                atlas_name=request.atlas,
                plane=request.plane,
                on_progress=on_progress,
                model_name=model_name,
                max_iterations=request.max_iterations,
                media_resolution=request.media_resolution,
                thinking=request.thinking,
                temperature=request.temperature,
                apply_clahe=False,
                debug_dir=debug_dir,
            )

        return EstimateResult(
            position_mm=float(result.position_mm),
            reasoning=str(result.reasoning),
            debug_dir=result.debug_dir,
        )


def run_register(request: RegisterRequest, emit: EngineEmit | None = None) -> RegisterResult:
    with _preserve_runtime_state():
        from PIL import Image

        from langslice_harness.image_prep import (
            adaptive_preprocess,
            normalize_image,
            prepare_image_for_vlm,
        )
        from langslice_harness.registration.core import estimate_registration_runtime
        from langslice_harness.registration.types import annotation_session_to_dict

        if request.endpoint:
            os.environ["LANGSLICE_ENDPOINT"] = request.endpoint

        debug_dir = request.output_dir
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            os.environ["LANGSLICE_VLM_DEBUG_DIR"] = debug_dir

        _progress(emit, f"Loading image: {request.image_path}", stage="register")
        raw_image = Image.open(request.image_path)
        image = prepare_image_for_vlm(normalize_image(raw_image)).image
        if request.preprocess == "auto":
            image = adaptive_preprocess(image)

        if request.provider == "openai":
            import langslice_harness.openai_config as openai_config

            image_model = (
                request.image_model or request.model or openai_config.get_openai_image_model()
            )
            review_model = request.review_model or request.model or openai_config.get_openai_model()
        else:
            import langslice_harness.vlm_config as vlm_config

            image_model = request.image_model or request.model or vlm_config.MODEL_NAME
            review_model = request.review_model or request.model or vlm_config.MODEL_NAME
            if request.temperature is not None:
                vlm_config.set_temperature(request.temperature)
            if request.thinking is not None:
                vlm_config.set_thinking_level(request.thinking)

        _log(emit, f"Running register provider={request.provider}")

        def on_progress(message: str) -> None:
            _progress(emit, message, stage="register")

        result = estimate_registration_runtime(
            image=image,
            atlas_name=request.atlas,
            position_mm=request.position_mm,
            plane=request.plane,
            on_progress=on_progress,
            debug_dir=debug_dir,
            provider=request.provider,
            image_model=image_model,
            review_model=review_model,
        )
        affine = result.affine_result
        session_dict = annotation_session_to_dict(result.annotation_session)
        session_meta = session_dict.get("metadata", {}) if isinstance(session_dict, dict) else {}
        if not isinstance(session_meta, dict):
            session_meta = {}
        candidate_metadata = session_meta.get("candidate_metadata")
        if not isinstance(candidate_metadata, dict):
            candidate_metadata = {}

        artifact_path_keys = (
            "warped_atlas_path",
            "warped_border_overlay_path",
            "generated_segmentation_path",
            "generated_border_overlay_path",
            "slice_warped_to_atlas_path",
            "slice_atlas_border_overlay_path",
        )
        artifact_paths: dict[str, str | None] = {}
        for key in artifact_path_keys:
            value = session_meta.get(key)
            if value is None:
                value = candidate_metadata.get(key)
            artifact_paths[key] = value if isinstance(value, str) else None

        inverse_warp_status_value = session_meta.get("inverse_warp_status")
        if inverse_warp_status_value is None:
            inverse_warp_status_value = candidate_metadata.get("inverse_warp_status")
        inverse_warp_status = (
            inverse_warp_status_value
            if isinstance(inverse_warp_status_value, str)
            else None
        )

        return RegisterResult(
            accepted_correspondence_count=len(result.accepted_correspondences),
            rotation_deg=float(affine.rotation_deg),
            translation_px=(float(affine.translation_px[0]), float(affine.translation_px[1])),
            scale=(float(affine.scale[0]), float(affine.scale[1])),
            shear=float(affine.shear),
            debug_dir=result.debug_dir,
            annotation_session=session_dict,
            warped_atlas_path=artifact_paths["warped_atlas_path"],
            warped_border_overlay_path=artifact_paths["warped_border_overlay_path"],
            generated_segmentation_path=artifact_paths["generated_segmentation_path"],
            generated_border_overlay_path=artifact_paths["generated_border_overlay_path"],
            slice_warped_to_atlas_path=artifact_paths["slice_warped_to_atlas_path"],
            slice_atlas_border_overlay_path=artifact_paths["slice_atlas_border_overlay_path"],
            inverse_warp_status=inverse_warp_status,
        )


def run_quick_affine(
    request: QuickAffineRequest,
    emit: EngineEmit | None = None,
) -> QuickAffineResult:
    from PIL import Image

    from langslice_harness.harness.registration.quick_affine import quick_affine_register

    _progress(emit, f"Loading image: {request.image_path}", stage="quick_affine")
    raw_image = Image.open(request.image_path)
    output_path = request.output_path
    if output_path is None:
        out_dir = Path(request.output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str((out_dir / "quick_affine.png").resolve())
    result = quick_affine_register(
        raw_image,
        atlas_name=request.atlas,
        position_mm=request.position_mm,
        plane=request.plane,
        out_path=Path(output_path),
    )
    return QuickAffineResult.model_validate(result)


def run_export(request: ExportRequest, emit: EngineEmit | None = None) -> ExportResult:
    from PIL import Image

    from langslice_harness.atlas import load_atlas
    from langslice_harness.atlas.space import atlas_space_context
    from langslice_harness.export import build_quint_export, save_quint_json

    _progress(emit, "Loading atlas", stage="export")
    atlas = load_atlas(request.atlas)
    _progress(emit, f"Loading image: {request.image_path}", stage="export")
    image = Image.open(request.image_path)
    width, height = image.size
    atlas_context = atlas_space_context(atlas)

    export = build_quint_export(
        filename=request.image_path,
        position_mm=request.position_mm,
        atlas_name=request.atlas,
        atlas_shape=atlas.reference.shape,
        atlas_resolution=atlas_context.resolution_um,
        image_width=width,
        image_height=height,
        affine_matrix=request.affine_matrix,
        output_width=request.output_width,
        output_height=request.output_height,
        rotation_deg=request.rotation_deg,
        translate_x_pct=request.translate_x_pct,
        translate_y_pct=request.translate_y_pct,
    )
    save_quint_json(export, request.output_path)
    return ExportResult(
        output_path=request.output_path,
        target=export.target,
        aligner=export.aligner,
        slices=len(export.slices),
    )
