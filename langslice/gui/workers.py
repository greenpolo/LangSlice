"""Background worker classes for LangSlice GUI."""

from __future__ import annotations

import importlib
import os
from datetime import datetime

from PIL import Image

from langslice.ai import APResult, estimate_position
from langslice.registration import estimate_registration_runtime

_qtcore = importlib.import_module("PySide6.QtCore")

Signal = _qtcore.Signal
QObject = _qtcore.QObject


def _create_debug_run_dir(atlas_name: str, *, suffix: str | None = None) -> str | None:
    root = os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    if not root:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
    folder_name = f"{timestamp}_{safe_atlas}"
    if suffix:
        folder_name = f"{folder_name}_{suffix}"
    run_dir = os.path.join(root, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class AgentWorker(QObject):
    """Runs AP + registration runtime in a worker thread."""

    step_started = Signal(str)
    step_completed = Signal(str, object)
    step_error = Signal(str, str)
    annotation_session_ready = Signal(object)
    correspondences_ready = Signal(object)
    log_message = Signal(str)
    trace_event = Signal(object)
    finished = Signal()

    def __init__(
        self,
        canonical_image: Image.Image,
        vlm_image: Image.Image,
        atlas_name: str,
        target_landmark_count: int,
        registration_workflow: str,
        show_atlas_borders: bool,
        ap_max_iterations: int,
        enable_code_execution: bool,
        tool_loop_max_steps: int,
    ) -> None:
        super().__init__()
        self.image = canonical_image
        self.vlm_image = vlm_image
        self.atlas_name = atlas_name
        self.target_landmark_count = target_landmark_count
        self.registration_workflow = registration_workflow
        self.show_atlas_borders = bool(show_atlas_borders)
        self.ap_max_iterations = max(1, int(ap_max_iterations))
        self.enable_code_execution = bool(enable_code_execution)
        self.tool_loop_max_steps = max(1, int(tool_loop_max_steps))

    def run(self) -> None:
        try:
            self.step_started.emit("ap")
            self.log_message.emit("Starting position estimation...")
            ap_result = estimate_position(
                image=self.vlm_image,
                atlas_name=self.atlas_name,
                on_progress=self.log_message.emit,
                on_trace=self.trace_event.emit,
                max_iterations=self.ap_max_iterations,
            )
            self.step_completed.emit("ap", ap_result)
        except Exception as exc:
            self.step_error.emit("ap", str(exc))
            self.finished.emit()
            return

        try:
            self.step_started.emit("affine")
            self.log_message.emit(
                "Starting landmark-based registration... "
                f"(workflow={self.registration_workflow}, target_pairs={self.target_landmark_count}, edge_hint_pairs=5)"
            )
            registration_result = estimate_registration_runtime(
                image=self.image,
                on_progress=self.log_message.emit,
                on_trace=self.trace_event.emit,
                atlas_name=self.atlas_name,
                position_mm=ap_result.position_mm,
                target_landmark_count=self.target_landmark_count,
                workflow=self.registration_workflow,
                show_atlas_borders=self.show_atlas_borders,
                on_correspondences=self.correspondences_ready.emit,
                on_annotation_session=self.annotation_session_ready.emit,
                debug_dir=ap_result.debug_dir,
                enable_code_execution=self.enable_code_execution,
                tool_loop_max_steps=self.tool_loop_max_steps,
            )
            self.step_completed.emit("affine", registration_result)
        except Exception as exc:
            self.step_error.emit("affine", str(exc))
        finally:
            self.finished.emit()


class ManualRegistrationWorker(QObject):
    """Runs registration runtime from a manually selected AP position."""

    step_started = Signal(str)
    step_completed = Signal(str, object)
    step_error = Signal(str, str)
    annotation_session_ready = Signal(object)
    correspondences_ready = Signal(object)
    log_message = Signal(str)
    trace_event = Signal(object)
    finished = Signal()

    def __init__(
        self,
        canonical_image: Image.Image,
        atlas_name: str,
        position_mm: float,
        target_landmark_count: int,
        registration_workflow: str,
        show_atlas_borders: bool,
        enable_code_execution: bool,
        tool_loop_max_steps: int,
    ) -> None:
        super().__init__()
        self.image = canonical_image
        self.atlas_name = atlas_name
        self.position_mm = position_mm
        self.target_landmark_count = target_landmark_count
        self.registration_workflow = registration_workflow
        self.show_atlas_borders = bool(show_atlas_borders)
        self.enable_code_execution = bool(enable_code_execution)
        self.tool_loop_max_steps = max(1, int(tool_loop_max_steps))

    def run(self) -> None:
        debug_dir = _create_debug_run_dir(self.atlas_name, suffix="manual")
        try:
            self.step_started.emit("ap")
            self.log_message.emit(f"Using manual AP position: {self.position_mm:.2f} mm")
            self.step_completed.emit(
                "ap",
                APResult(
                    position_mm=self.position_mm,
                    reasoning="Manual AP position",
                    debug_dir=debug_dir,
                ),
            )
        except Exception as exc:
            self.step_error.emit("ap", str(exc))
            self.finished.emit()
            return

        try:
            self.step_started.emit("affine")
            self.log_message.emit(
                "Starting landmark-based registration... "
                f"(workflow={self.registration_workflow}, target_pairs={self.target_landmark_count}, edge_hint_pairs=5)"
            )
            registration_result = estimate_registration_runtime(
                image=self.image,
                on_progress=self.log_message.emit,
                on_trace=self.trace_event.emit,
                atlas_name=self.atlas_name,
                position_mm=self.position_mm,
                target_landmark_count=self.target_landmark_count,
                workflow=self.registration_workflow,
                show_atlas_borders=self.show_atlas_borders,
                on_correspondences=self.correspondences_ready.emit,
                on_annotation_session=self.annotation_session_ready.emit,
                debug_dir=debug_dir,
                enable_code_execution=self.enable_code_execution,
                tool_loop_max_steps=self.tool_loop_max_steps,
            )
            self.step_completed.emit("affine", registration_result)
        except Exception as exc:
            self.step_error.emit("affine", str(exc))
        finally:
            self.finished.emit()
