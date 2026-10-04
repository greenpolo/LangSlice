"""Runtime wrappers for the engine API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langslice.doors.api.models import (
    EngineLogEvent,
    EngineProgressEvent,
    ExportRequest,
    ExportResult,
    QuickAffineRequest,
    QuickAffineResult,
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


def get_version() -> VersionResult:
    import langslice

    return VersionResult(version=langslice.__version__)


def run_quick_affine(
    request: QuickAffineRequest,
    emit: EngineEmit | None = None,
) -> QuickAffineResult:
    from PIL import Image

    from langslice.core.nonlinear.quick_affine import quick_affine_register

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

    from langslice.core.atlas import load_atlas
    from langslice.core.space import atlas_space_context
    from langslice.job.quint import build_quint_export, save_quint_json

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
        atlas_shape=atlas.template.shape,
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
