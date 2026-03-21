"""Main application window for LangSlice."""

from __future__ import annotations

import json
import os
import sys
import importlib
from datetime import datetime
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import stage_event, status_event

_qtcore = importlib.import_module("PySide6.QtCore")
_qtgui = importlib.import_module("PySide6.QtGui")
_qtwidgets = importlib.import_module("PySide6.QtWidgets")

Qt = _qtcore.Qt
Signal = _qtcore.Signal
QObject = _qtcore.QObject
QThread = _qtcore.QThread
QTimer = _qtcore.QTimer

QDragEnterEvent = _qtgui.QDragEnterEvent
QDropEvent = _qtgui.QDropEvent
QImage = _qtgui.QImage
QPainter = _qtgui.QPainter
QPen = _qtgui.QPen
QPixmap = _qtgui.QPixmap
QTransform = _qtgui.QTransform
QColor = _qtgui.QColor

QApplication = _qtwidgets.QApplication
QComboBox = _qtwidgets.QComboBox
QFileDialog = _qtwidgets.QFileDialog
QFrame = _qtwidgets.QFrame
QHBoxLayout = _qtwidgets.QHBoxLayout
QLabel = _qtwidgets.QLabel
QMainWindow = _qtwidgets.QMainWindow
QPushButton = _qtwidgets.QPushButton
QCheckBox = _qtwidgets.QCheckBox
QSizePolicy = _qtwidgets.QSizePolicy
QSlider = _qtwidgets.QSlider
QSplitter = _qtwidgets.QSplitter
QStackedWidget = _qtwidgets.QStackedWidget
QVBoxLayout = _qtwidgets.QVBoxLayout
QWidget = _qtwidgets.QWidget
QDoubleSpinBox = _qtwidgets.QDoubleSpinBox
QSpinBox = _qtwidgets.QSpinBox

from langslice import __version__
from langslice.atlas import (
    DEFAULT_ATLAS_NAME,
    canonicalize_atlas_name,
    list_available_atlases,
    list_downloaded_atlases,
)
from langslice.image_prep import (
    DEFAULT_VLM_MAX_LONG_EDGE,
    LoadedImageState,
    load_image_state,
    prepare_image_for_vlm,
    render_slice_agent_image,
)
from langslice.registration import (
    AffineResult,
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    RegistrationResult,
    estimate_registration_runtime,
)
from langslice.gui.theme import ACCENT, STYLESHEET, TEXT_SECONDARY
from langslice.vlm import APResult, estimate_position
from langslice.vlm.config import (
    AVAILABLE_MODELS,
    AVAILABLE_THINKING_LEVELS,
    CODE_EXECUTION_ENABLED,
    MODEL_NAME,
    TEMPERATURE,
    THINKING_LEVEL,
    ap_use_context_cache,
    ap_use_file_api,
    ap_use_interactions,
    count_tokens_enabled,
    default_registration_workflow,
    get_registration_workflow_options,
    is_image_generation_model,
    set_model_name,
    set_thinking_level,
    set_temperature,
    supports_code_execution,
)
from langslice.export import build_quint_export, save_quint_json
from langslice.gui.run_metadata_dialog import RunMetadataDialog
from langslice.gui.settings_dialog import SettingsDialog
from langslice.gui.overlay_viewer import OverlayGraphicsView

_main_window_components = importlib.import_module("langslice.gui.main_window_components")
CanvasArea = _main_window_components.CanvasArea
ImageLabel = _main_window_components.ImageLabel
StepIndicator = _main_window_components.StepIndicator
affine_matrix_to_qtransform = _main_window_components.affine_matrix_to_qtransform
build_split_view_correspondence_points = (
    _main_window_components.build_split_view_correspondence_points
)
draw_correspondence_markers = _main_window_components.draw_correspondence_markers
pil_to_qpixmap = _main_window_components.pil_to_qpixmap
should_apply_affine_to_slice = _main_window_components.should_apply_affine_to_slice
TraceInspector = importlib.import_module("langslice.gui.trace_inspector").TraceInspector

VLM_RESOLUTION_OPTIONS: list[tuple[str, int]] = [
    ("512", 512),
    ("1K", 1024),
    ("2K", 2048),
    ("4K", 4096),
]

try:
    AtlasViewer = importlib.import_module("langslice.gui.atlas_viewer").AtlasViewer
except Exception:

    class AtlasViewer(QFrame):
        """Fallback atlas widget when atlas_viewer.py is unavailable."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._atlas_name = ""
            self._position_mm: float | None = None
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._label = QLabel("Atlas pending position estimate")
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._label.setStyleSheet(f"color: {TEXT_SECONDARY};")
            layout.addWidget(self._label)

        def set_position(self, position_mm: float) -> None:
            self._position_mm = position_mm
            self._label.setText(f"{self._atlas_name}\nPos: {position_mm:.2f} mm")

        def set_atlas(self, atlas_name: str) -> None:
            self._atlas_name = atlas_name
            if self._position_mm is None:
                self._label.setText(f"{atlas_name}\nAwaiting position estimate")
            else:
                self._label.setText(f"{atlas_name}\nPos: {self._position_mm:.2f} mm")

        def clear(self) -> None:
            self._position_mm = None
            self._label.setText("Atlas pending position estimate")

        def set_correspondence_markers(
            self, markers: list[tuple[float, float, str]] | None
        ) -> None:
            _ = markers

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible


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


class MainWindow(QMainWindow):
    """LangSlice desktop application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LangSlice")
        self.resize(1460, 900)
        self.setMinimumSize(1100, 720)

        self.image_path: str | None = None
        self.loaded_image_state: LoadedImageState | None = None
        self.source_image: Image.Image | None = None
        self.pil_image: Image.Image | None = None
        self.agent_vlm_image: Image.Image | None = None
        self.current_pos: float | None = None
        self._active_pipeline_mode: str = "agent"
        self.ap_result: APResult | None = None
        self.affine_result: AffineResult | None = None
        self.registration_result: RegistrationResult | None = None
        self.preview_annotation_session: RegistrationAnnotationSession | None = None
        self.preview_correspondences: list[RegistrationCorrespondence] | None = None
        self._last_run_context: dict[str, object] | None = None
        self.current_view_mode = "single"

        self.worker_thread: QThread | None = None
        self.worker: AgentWorker | ManualRegistrationWorker | None = None

        self.pixel_size_um: float = 4.0
        self.vlm_max_long_edge: int = DEFAULT_VLM_MAX_LONG_EDGE
        self.registration_landmark_count: int = 8
        self.registration_workflow: str = default_registration_workflow(MODEL_NAME)
        self.ap_max_iterations: int = 20
        self.registration_tool_loop_max_steps: int = 24
        self._pixel_size_source: str = "manual_default"
        self._metadata_pixel_size_um: float | None = None
        self._vlm_scale_factor: float = 1.0
        self._vlm_effective_pixel_size_um: float = self.pixel_size_um
        self._slice_channel_labels: tuple[str, ...] = ("Red", "Green", "Blue")
        self._slice_exposure_ev: float = 0.0
        self._slice_brightness: float = 0.0
        self._slice_contrast: float = 1.0
        self._show_atlas_region_borders: bool = True
        self._code_execution_enabled: bool = CODE_EXECUTION_ENABLED

        self._slice_adjust_timer = QTimer(self)
        self._slice_adjust_timer.setSingleShot(True)
        self._slice_adjust_timer.setInterval(75)
        self._slice_adjust_timer.timeout.connect(self._apply_slice_adjustments)

        self._build_ui()
        self._refresh_model_specific_controls()
        self._set_pipeline_mode("agent")
        self._set_view_mode("single")
        self._set_export_enabled(False)
        self._update_run_buttons()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_agent_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 760, 400])
        root_layout.addWidget(splitter, stretch=1)

    @staticmethod
    def _format_atlas_label(atlas_name: str, downloaded: bool) -> str:
        cleaned = atlas_name.strip()
        parts = [part for part in cleaned.split("_") if part]
        label: str
        if parts and parts[-1].endswith("um") and len(parts) > 1:
            base = " ".join(parts[:-1]).title()
            label = f"{base} ({parts[-1]})"
        else:
            label = cleaned.replace("_", " ").title()
        return f"{label} [local]" if downloaded else label

    @staticmethod
    def _ordered_atlas_names(downloaded: set[str], available: set[str]) -> list[str]:
        ordered: list[str] = []
        local_sorted = sorted(downloaded)
        remote_sorted = sorted(available - downloaded)

        if DEFAULT_ATLAS_NAME in downloaded or DEFAULT_ATLAS_NAME in available:
            ordered.append(DEFAULT_ATLAS_NAME)

        ordered.extend(name for name in local_sorted if name != DEFAULT_ATLAS_NAME)
        ordered.extend(name for name in remote_sorted if name != DEFAULT_ATLAS_NAME)

        if DEFAULT_ATLAS_NAME not in ordered:
            ordered.insert(0, DEFAULT_ATLAS_NAME)

        return ordered

    def _populate_atlas_combo(self) -> None:
        downloaded = {canonicalize_atlas_name(name) for name in list_downloaded_atlases()}
        available = {canonicalize_atlas_name(name) for name in list_available_atlases()}
        atlas_names = self._ordered_atlas_names(downloaded, available)

        self.atlas_combo.blockSignals(True)
        self.atlas_combo.clear()
        for atlas_name in atlas_names:
            label = self._format_atlas_label(atlas_name, atlas_name in downloaded)
            self.atlas_combo.addItem(label, atlas_name)

        default_index = self.atlas_combo.findData(DEFAULT_ATLAS_NAME)
        self.atlas_combo.setCurrentIndex(default_index if default_index >= 0 else 0)
        self.atlas_combo.blockSignals(False)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet("border-bottom: 1px solid rgba(255,255,255,20);")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(10)

        left = QHBoxLayout()
        left.setSpacing(10)
        logo = QLabel("LS")
        logo.setFixedSize(30, 30)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet(
            f"background-color: rgba(99,102,241,40); border: 1px solid rgba(99,102,241,90); border-radius: 8px; color: {ACCENT};"
        )

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel("LangSlice")
        title.setObjectName("heading")
        subtitle = QLabel("VLM Agentic Harness")
        subtitle.setObjectName("subheading")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        left.addWidget(logo)
        left.addLayout(title_col)

        layout.addLayout(left)
        layout.addStretch(1)

        right = QHBoxLayout()
        right.setSpacing(8)

        self.upload_button = QPushButton("Open Image...")
        self.upload_button.setObjectName("secondary")
        self.upload_button.clicked.connect(self._browse_image)

        self.settings_button = QPushButton("Settings")
        self.settings_button.setObjectName("secondary")
        self.settings_button.clicked.connect(self._open_settings)

        self.export_button = QPushButton("Export QUINT/ABBA JSON")
        self.export_button.setObjectName("secondary")
        self.export_button.clicked.connect(self._export_abba)

        right.addWidget(self.upload_button)
        right.addWidget(self.settings_button)
        right.addWidget(self.export_button)
        layout.addLayout(right)
        return header

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.settings_saved.connect(self._on_settings_saved)
        dialog.exec()

    def _build_settings_panel(self) -> QWidget:
        scroll = _qtwidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setStyleSheet(
            "QScrollArea { border: none; border-right: 1px solid rgba(255,255,255,20); background: transparent; } QWidget#settingsPanel { background: transparent; }"
        )

        panel = QWidget()
        panel.setObjectName("settingsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)

        # 1. Project Setup
        project_group = QFrame()
        project_layout = QVBoxLayout(project_group)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.setSpacing(8)
        project_title = QLabel("Project Setup")
        project_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #ffffff;")
        project_layout.addWidget(project_title)

        atlas_label = QLabel("Atlas:")
        atlas_label.setObjectName("subheading")
        self.atlas_combo = QComboBox()
        self._populate_atlas_combo()
        self.atlas_combo.currentIndexChanged.connect(self._on_atlas_changed)
        project_layout.addWidget(atlas_label)
        project_layout.addWidget(self.atlas_combo)

        px_size_label = QLabel("Pixel Size:")
        px_size_label.setObjectName("subheading")
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.1, 100.0)
        self.pixel_size_spin.setValue(self.pixel_size_um)
        self.pixel_size_spin.setSuffix(" um/px")
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setSingleStep(0.5)
        self.pixel_size_spin.setToolTip(
            "Pixel size of the histology image in micrometers per pixel"
        )
        self.pixel_size_spin.valueChanged.connect(self._on_pixel_size_changed)
        project_layout.addWidget(px_size_label)
        project_layout.addWidget(self.pixel_size_spin)

        vlm_resolution_label = QLabel("VLM Resolution:")
        vlm_resolution_label.setObjectName("subheading")
        self.vlm_resolution_combo = QComboBox()
        for label, edge in VLM_RESOLUTION_OPTIONS:
            self.vlm_resolution_combo.addItem(label, edge)
        resolution_index = self.vlm_resolution_combo.findData(self.vlm_max_long_edge)
        self.vlm_resolution_combo.setCurrentIndex(max(0, resolution_index))
        self.vlm_resolution_combo.setToolTip(
            "Maximum long-edge size for the image sent to Gemini. Lower values are faster but lose detail."
        )
        self.vlm_resolution_combo.currentIndexChanged.connect(self._on_vlm_resolution_changed)
        project_layout.addWidget(vlm_resolution_label)
        project_layout.addWidget(self.vlm_resolution_combo)

        layout.addWidget(project_group)

        # 2. Agent Parameters
        agent_group = QFrame()
        agent_layout = QVBoxLayout(agent_group)
        agent_layout.setContentsMargins(0, 0, 0, 0)
        agent_layout.setSpacing(8)
        agent_title = QLabel("Agent Parameters")
        agent_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #ffffff;")
        agent_layout.addWidget(agent_title)

        model_label = QLabel("Model:")
        model_label.setObjectName("subheading")
        self.model_combo = QComboBox()
        for m in AVAILABLE_MODELS:
            self.model_combo.addItem(m)
        self.model_combo.setCurrentText(MODEL_NAME)
        self.model_combo.setToolTip("Gemini model used for AP estimation and registration")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        agent_layout.addWidget(model_label)
        agent_layout.addWidget(self.model_combo)

        workflow_label = QLabel("Workflow:")
        workflow_label.setObjectName("subheading")
        self.workflow_combo = QComboBox()
        self.workflow_combo.setToolTip(
            "Registration workflow. Image-gen mode is only available when an image-generation model is selected."
        )
        self._refresh_registration_workflow_options(self.registration_workflow)
        self.workflow_combo.currentIndexChanged.connect(self._on_registration_workflow_changed)
        agent_layout.addWidget(workflow_label)
        agent_layout.addWidget(self.workflow_combo)

        thinking_label = QLabel("Thinking:")
        thinking_label.setObjectName("subheading")
        self.thinking_combo = QComboBox()
        current_level = THINKING_LEVEL
        selected_index = 0
        for i, (label, level) in enumerate(AVAILABLE_THINKING_LEVELS):
            self.thinking_combo.addItem(label, level)
            if level == current_level:
                selected_index = i
        self.thinking_combo.setCurrentIndex(selected_index)
        self.thinking_combo.setToolTip(
            "Gemini thinking level used for registration. Gemini 3 models use thinking_level rather than token budgets."
        )
        self.thinking_combo.currentIndexChanged.connect(self._on_thinking_level_changed)
        agent_layout.addWidget(thinking_label)
        agent_layout.addWidget(self.thinking_combo)

        row1 = QHBoxLayout()
        landmark_count_label = QLabel("Landmarks:")
        landmark_count_label.setObjectName("subheading")
        self.landmark_count_spin = QSpinBox()
        self.landmark_count_spin.setRange(6, 24)
        self.landmark_count_spin.setValue(self.registration_landmark_count)
        self.landmark_count_spin.setSuffix(" pts")
        self.landmark_count_spin.setToolTip(
            "Total correspondence pairs requested from the registration agent."
        )
        self.landmark_count_spin.valueChanged.connect(self._on_landmark_count_changed)
        row1.addWidget(landmark_count_label)
        row1.addWidget(self.landmark_count_spin)
        agent_layout.addLayout(row1)

        row2 = QHBoxLayout()
        temperature_label = QLabel("Temp:")
        temperature_label.setObjectName("subheading")
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(TEMPERATURE)
        self.temperature_spin.setToolTip("Gemini temperature for AP estimation and registration.")
        self.temperature_spin.valueChanged.connect(self._on_temperature_changed)
        row2.addWidget(temperature_label)
        row2.addWidget(self.temperature_spin)
        agent_layout.addLayout(row2)

        self.code_execution_checkbox = QCheckBox("Enable code execution")
        self.code_execution_checkbox.setChecked(self._code_execution_enabled)
        self.code_execution_checkbox.setToolTip(
            "Lets supported Gemini Flash models use code execution during registration requests."
        )
        self.code_execution_checkbox.toggled.connect(self._on_code_execution_toggled)
        agent_layout.addWidget(self.code_execution_checkbox)

        ap_iterations_row = QHBoxLayout()
        ap_iterations_label = QLabel("AP Max Turns:")
        ap_iterations_label.setObjectName("subheading")
        self.ap_iterations_spin = QSpinBox()
        self.ap_iterations_spin.setRange(1, 50)
        self.ap_iterations_spin.setValue(self.ap_max_iterations)
        self.ap_iterations_spin.setToolTip(
            "Maximum number of AP-estimation tool turns before the agent must stop."
        )
        self.ap_iterations_spin.valueChanged.connect(self._on_ap_max_iterations_changed)
        ap_iterations_row.addWidget(ap_iterations_label)
        ap_iterations_row.addWidget(self.ap_iterations_spin)
        agent_layout.addLayout(ap_iterations_row)

        tool_loop_row = QHBoxLayout()
        tool_loop_label = QLabel("Loop Max Steps:")
        tool_loop_label.setObjectName("subheading")
        self.tool_loop_steps_spin = QSpinBox()
        self.tool_loop_steps_spin.setRange(1, 50)
        self.tool_loop_steps_spin.setValue(self.registration_tool_loop_max_steps)
        self.tool_loop_steps_spin.setToolTip(
            "Maximum number of multimodal tool-loop steps during registration."
        )
        self.tool_loop_steps_spin.valueChanged.connect(self._on_tool_loop_max_steps_changed)
        tool_loop_row.addWidget(tool_loop_label)
        tool_loop_row.addWidget(self.tool_loop_steps_spin)
        agent_layout.addLayout(tool_loop_row)

        layout.addWidget(agent_group)

        # 3. Image Pre-processing
        self.agent_input_wrap = QFrame()
        input_layout = QVBoxLayout(self.agent_input_wrap)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        input_title = QLabel("Image Adjustments")
        input_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #ffffff;")
        input_copy = QLabel("Updates the preview and the image sent to the agents.")
        input_copy.setWordWrap(True)
        input_copy.setObjectName("subheading")
        self.agent_input_status_label = QLabel("Preview matches current agent input.")
        self.agent_input_status_label.setWordWrap(True)
        self.agent_input_status_label.setObjectName("subheading")

        channel_label = QLabel("Channels")
        channel_label.setObjectName("monoLabel")
        channel_row = QHBoxLayout()
        channel_row.setContentsMargins(0, 0, 0, 0)
        channel_row.setSpacing(8)
        self.channel_checkboxes: list[QCheckBox] = []
        for label in ("Red", "Green", "Blue"):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._queue_slice_adjustment_refresh)
            self.channel_checkboxes.append(checkbox)
            channel_row.addWidget(checkbox)
        channel_row.addStretch(1)

        exposure_row = QHBoxLayout()
        exposure_row.setContentsMargins(0, 0, 0, 0)
        exposure_row.setSpacing(8)
        exposure_label = QLabel("Exposure")
        exposure_label.setObjectName("monoLabel")
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-30, 30)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self._queue_slice_adjustment_refresh)
        self.exposure_value_label = QLabel("0.0 EV")
        self.exposure_value_label.setObjectName("monoLabel")
        exposure_row.addWidget(exposure_label)
        exposure_row.addWidget(self.exposure_slider, stretch=1)
        exposure_row.addWidget(self.exposure_value_label)

        brightness_row = QHBoxLayout()
        brightness_row.setContentsMargins(0, 0, 0, 0)
        brightness_row.setSpacing(8)
        brightness_label = QLabel("Brightness")
        brightness_label.setObjectName("monoLabel")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self._queue_slice_adjustment_refresh)
        self.brightness_value_label = QLabel("0%")
        self.brightness_value_label.setObjectName("monoLabel")
        brightness_row.addWidget(brightness_label)
        brightness_row.addWidget(self.brightness_slider, stretch=1)
        brightness_row.addWidget(self.brightness_value_label)

        contrast_row = QHBoxLayout()
        contrast_row.setContentsMargins(0, 0, 0, 0)
        contrast_row.setSpacing(8)
        contrast_label = QLabel("Contrast")
        contrast_label.setObjectName("monoLabel")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._queue_slice_adjustment_refresh)
        self.contrast_value_label = QLabel("100%")
        self.contrast_value_label.setObjectName("monoLabel")
        contrast_row.addWidget(contrast_label)
        contrast_row.addWidget(self.contrast_slider, stretch=1)
        contrast_row.addWidget(self.contrast_value_label)

        self.atlas_borders_checkbox = QCheckBox("Show atlas region borders")
        self.atlas_borders_checkbox.setChecked(True)
        self.atlas_borders_checkbox.toggled.connect(self._on_atlas_borders_toggled)

        reset_adjustments_button = QPushButton("Reset Adjustments")
        reset_adjustments_button.setObjectName("secondary")
        reset_adjustments_button.clicked.connect(self._reset_slice_adjustments)

        input_layout.addWidget(input_title)
        input_layout.addWidget(input_copy)
        input_layout.addWidget(self.agent_input_status_label)
        input_layout.addWidget(channel_label)
        input_layout.addLayout(channel_row)
        input_layout.addLayout(exposure_row)
        input_layout.addLayout(brightness_row)
        input_layout.addLayout(contrast_row)
        input_layout.addWidget(self.atlas_borders_checkbox)
        input_layout.addWidget(reset_adjustments_button)

        self.agent_input_wrap.hide()
        layout.addWidget(self.agent_input_wrap)

        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _build_canvas_panel(self) -> QWidget:
        self.canvas = CanvasArea()
        self.canvas.file_dropped.connect(self._load_image)

        layout = QVBoxLayout(self.canvas)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        toolbar_wrap = QHBoxLayout()
        toolbar_wrap.setContentsMargins(0, 0, 0, 0)
        toolbar_wrap.addStretch(1)
        toolbar_wrap.addWidget(self._build_view_toolbar())
        toolbar_wrap.addStretch(1)
        layout.addLayout(toolbar_wrap)

        self.view_stack = QStackedWidget()
        self.view_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.page_upload = self._build_upload_page()
        self.page_single = self._build_single_page()
        self.page_split = self._build_split_page()
        self.page_overlay = self._build_overlay_page()
        self.view_stack.addWidget(self.page_upload)
        self.view_stack.addWidget(self.page_single)
        self.view_stack.addWidget(self.page_split)
        self.view_stack.addWidget(self.page_overlay)
        layout.addWidget(self.view_stack, stretch=1)
        return self.canvas

    def _build_view_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("glassPanel")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        self.single_btn = QPushButton("Single")
        self.split_btn = QPushButton("Split")
        self.overlay_btn = QPushButton("Overlay")
        for button in (self.single_btn, self.split_btn, self.overlay_btn):
            button.setCheckable(True)
            button.setObjectName("secondary")

        self.single_btn.clicked.connect(lambda: self._set_view_mode("single"))
        self.split_btn.clicked.connect(lambda: self._set_view_mode("split"))
        self.overlay_btn.clicked.connect(lambda: self._set_view_mode("overlay"))

        row.addWidget(self.single_btn)
        row.addWidget(self.split_btn)
        row.addWidget(self.overlay_btn)

        self.opacity_wrap = QFrame()
        opacity_layout = QHBoxLayout(self.opacity_wrap)
        opacity_layout.setContentsMargins(8, 0, 0, 0)
        opacity_layout.setSpacing(6)
        opacity_label = QLabel("Opacity")
        opacity_label.setObjectName("subheading")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        row.addWidget(self.opacity_wrap)
        return bar

    def _build_upload_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        upload_btn = QPushButton("\nUpload Brain Slice\n\nDrop an image here or click to browse\n")
        upload_btn.setFixedSize(330, 330)
        upload_btn.setStyleSheet(
            "QPushButton {"
            "border: 2px dashed rgba(255,255,255,25);"
            "border-radius: 18px;"
            "background-color: rgba(255,255,255,5);"
            "font-size: 14px;"
            "color: #bbbbbb;"
            "padding: 18px;"
            "}"
            "QPushButton:hover {"
            "border: 2px dashed rgba(99,102,241,160);"
            "background-color: rgba(99,102,241,20);"
            "}"
        )
        upload_btn.clicked.connect(self._browse_image)
        layout.addWidget(upload_btn)
        return page

    def _build_single_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.single_image_label = ImageLabel()
        self.single_image_label.setStyleSheet(
            "border: 1px solid rgba(255,255,255,30); border-radius: 10px; background-color: #000000;"
        )
        layout.addWidget(self.single_image_label)
        return page

    def _build_split_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        left = QFrame()
        left.setObjectName("glassPanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        self.split_image_label = ImageLabel()
        left_layout.addWidget(self.split_image_label)

        right = QFrame()
        right.setObjectName("glassPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        self.split_atlas = AtlasViewer()
        right_layout.addWidget(self.split_atlas)

        layout.addWidget(left, stretch=1)
        layout.addWidget(right, stretch=1)
        return page

    def _build_overlay_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.overlay_viewer = OverlayGraphicsView()
        layout.addWidget(self.overlay_viewer)
        return page

    def _build_agent_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(400)
        panel.setStyleSheet("border-left: 1px solid rgba(255,255,255,20);")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QFrame()
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(0, 0, 0, 0)
        hrow.setSpacing(8)
        title = QLabel("Registration Workflows")
        title.setObjectName("heading")
        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.setObjectName("secondary")
        self.clear_all_button.clicked.connect(self._clear_all)
        self.run_button = QPushButton("Run Agent")

        # Give the Run Agent button strong visual pop
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #6366f1; color: white; font-weight: bold; padding: 8px 16px; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #818cf8; }"
            "QPushButton:disabled { background-color: #374151; color: #9ca3af; }"
        )
        self.run_button.clicked.connect(self._run_agent)

        hrow.addWidget(title)
        hrow.addStretch(1)
        hrow.addWidget(self.clear_all_button)
        hrow.addWidget(self.run_button)
        layout.addWidget(header)

        workflow_copy = QLabel(
            "Run Agent for automatic AP estimation, or use the manual position flow below to register from an explicit atlas location."
        )
        workflow_copy.setWordWrap(True)
        workflow_copy.setObjectName("subheading")
        layout.addWidget(workflow_copy)

        self.ap_adjust_wrap = QFrame()
        self.ap_adjust_wrap.setObjectName("glassPanel")
        ap_layout = QVBoxLayout(self.ap_adjust_wrap)
        ap_layout.setContentsMargins(10, 10, 10, 10)
        ap_layout.setSpacing(6)
        manual_title = QLabel("Manual Position Registration")
        manual_title.setStyleSheet("font-size: 13px; font-weight: 600;")
        manual_copy = QLabel(
            "Set AP with the slider, then run landmark registration from that exact manual position (AP agent skipped)."
        )
        manual_copy.setWordWrap(True)
        manual_copy.setObjectName("subheading")
        self.ap_value_label = QLabel("Manual Position: 0.00 mm")
        self.ap_value_label.setObjectName("monoLabel")
        self.ap_slider = QSlider(Qt.Orientation.Horizontal)
        self.ap_slider.setRange(0, 2000)  # Default 0-20mm
        self.ap_slider.setValue(0)
        self.ap_slider.valueChanged.connect(self._on_ap_slider_changed)
        self.run_registration_button = QPushButton("Run Registration at Manual Position")
        self.run_registration_button.clicked.connect(self._run_manual_registration)
        self.manual_registration_status_label = QLabel()
        self.manual_registration_status_label.setWordWrap(True)
        self.manual_registration_status_label.setObjectName("subheading")
        ap_layout.addWidget(manual_title)
        ap_layout.addWidget(manual_copy)
        ap_layout.addWidget(self.ap_value_label)
        ap_layout.addWidget(self.ap_slider)
        ap_layout.addWidget(self.run_registration_button)
        ap_layout.addWidget(self.manual_registration_status_label)
        self.ap_adjust_wrap.hide()
        layout.addWidget(self.ap_adjust_wrap)

        self.step_ap = StepIndicator("Estimate Position (Agent)", has_connector=True)
        self.step_affine = StepIndicator("Landmark Registration", has_connector=False)

        layout.addWidget(self.step_ap)
        layout.addWidget(self.step_affine)

        # Feedback buttons for curating agent traces
        self.feedback_wrap = QFrame()
        fb_layout = QHBoxLayout(self.feedback_wrap)
        fb_layout.setContentsMargins(0, 8, 0, 0)
        fb_layout.setSpacing(8)

        self.success_btn = QPushButton("Mark Success")
        self.success_btn.setObjectName("secondary")
        self.success_btn.clicked.connect(lambda: self._mark_run("success"))

        self.failure_btn = QPushButton("Mark Failure")
        self.failure_btn.setObjectName("secondary")
        self.failure_btn.clicked.connect(lambda: self._mark_run("failure"))

        self.save_trace_btn = QPushButton("Save Agent Trace...")
        self.save_trace_btn.setObjectName("secondary")
        self.save_trace_btn.clicked.connect(self._save_agent_trace)

        fb_layout.addWidget(self.success_btn)
        fb_layout.addWidget(self.failure_btn)
        self.feedback_wrap.hide()
        layout.addWidget(self.feedback_wrap)

        logs_header = QHBoxLayout()
        logs_header.setContentsMargins(0, 0, 0, 0)
        logs_header.setSpacing(8)
        logs_label = QLabel("Agent Trace")
        logs_label.setObjectName("monoLabel")
        logs_header.addWidget(logs_label)
        logs_header.addStretch(1)
        logs_header.addWidget(self.save_trace_btn)
        layout.addLayout(logs_header)

        self.trace_inspector = TraceInspector()
        self.trace_inspector.setMinimumHeight(220)
        layout.addWidget(self.trace_inspector, stretch=1)

        return panel

    def _mark_run(self, outcome: str) -> None:
        if not self.ap_result or not getattr(self.ap_result, "debug_dir", None):
            self._append_log("No debug directory available to save.")
            return

        debug_dir = self.ap_result.debug_dir
        if not isinstance(debug_dir, str) or not debug_dir:
            self._append_log("No debug directory available to save.")
            return
        debug_dir = cast(str, debug_dir)

        if not os.path.exists(debug_dir):
            self._append_log("Debug directory no longer exists on disk.")
            return

        parent_dir = os.path.dirname(debug_dir)
        target_group_dir = os.path.join(parent_dir, outcome)
        os.makedirs(target_group_dir, exist_ok=True)

        folder_name = os.path.basename(debug_dir)
        new_path = self._unique_target_path(os.path.join(target_group_dir, folder_name))

        dialog = RunMetadataDialog(
            outcome=outcome,
            estimated_position_mm=self.ap_result.position_mm,
            parent=self,
        )
        if not dialog.exec():
            self._append_log("Run classification cancelled.")
            return

        metadata = self._build_run_metadata(outcome, dialog.metadata())

        try:
            os.rename(debug_dir, new_path)
            self._write_run_metadata(new_path, metadata)
            self._append_log(f"Agent trace moved to '{outcome}' folder.")
            self.ap_result.debug_dir = new_path  # Update reference
            self.success_btn.setEnabled(False)
            self.failure_btn.setEnabled(False)
        except Exception as exc:
            self._append_log(f"Failed to move trace: {exc}")

    def _save_agent_trace(self) -> None:
        if not self.trace_inspector.has_events():
            self._append_log("No agent trace events available to save.")
            return

        default_name = f"agent_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Agent Trace",
            default_name,
            "JSON (*.json)",
        )
        if not out_path:
            return

        try:
            event_count = self.trace_inspector.export_events(out_path)
            self._append_log(f"Saved {event_count} trace events to {out_path}")
        except Exception as exc:
            self._append_log(f"Failed to save agent trace: {exc}")

    @staticmethod
    def _unique_target_path(base_path: str) -> str:
        if not os.path.exists(base_path):
            return base_path

        root = base_path
        counter = 2
        while True:
            candidate = f"{root}_{counter}"
            if not os.path.exists(candidate):
                return candidate
            counter += 1

    def _build_run_metadata(
        self,
        outcome: str,
        user_metadata: dict[str, object],
    ) -> dict[str, object]:
        actual_position = user_metadata.get("actual_position_mm")
        estimated_position = self.ap_result.position_mm if self.ap_result else None
        signed_error_mm: float | None = None
        absolute_error_mm: float | None = None

        if isinstance(actual_position, (int, float)) and isinstance(
            estimated_position, (int, float)
        ):
            signed_error_mm = float(estimated_position) - float(actual_position)
            absolute_error_mm = abs(signed_error_mm)

        image_name = os.path.basename(self.image_path) if self.image_path else None
        affine_backend = self.affine_result.backend if self.affine_result else None
        run_context = self._last_run_context or {}

        return {
            "classified_at": datetime.now().isoformat(timespec="seconds"),
            "outcome": outcome,
            "image_name": image_name,
            "image_path": self.image_path,
            "atlas_name": run_context.get("atlas_name") or self._current_atlas_name(),
            "model_name": run_context.get("model_name") or self.model_combo.currentText(),
            "temperature": run_context.get("temperature") or self.temperature_spin.value(),
            "pixel_size_um": run_context.get("pixel_size_um") or self.pixel_size_um,
            "pixel_size_source": run_context.get("pixel_size_source") or self._pixel_size_source,
            "metadata_pixel_size_um": run_context.get("metadata_pixel_size_um"),
            "image_width": run_context.get("image_width"),
            "image_height": run_context.get("image_height"),
            "vlm_width": run_context.get("vlm_width"),
            "vlm_height": run_context.get("vlm_height"),
            "vlm_scale_factor": run_context.get("vlm_scale_factor"),
            "vlm_effective_pixel_size_um": run_context.get("vlm_effective_pixel_size_um"),
            "vlm_max_long_edge": run_context.get("vlm_max_long_edge"),
            "vlm_resolution_label": run_context.get("vlm_resolution_label"),
            "registration_landmark_count": run_context.get("registration_landmark_count")
            or self.registration_landmark_count,
            "registration_workflow": run_context.get("registration_workflow")
            or self.registration_workflow,
            "ap_max_iterations": run_context.get("ap_max_iterations") or self.ap_max_iterations,
            "tool_loop_max_steps": run_context.get("tool_loop_max_steps")
            or self.registration_tool_loop_max_steps,
            "code_execution_enabled": run_context.get("code_execution_enabled"),
            "channel_enabled": run_context.get("channel_enabled"),
            "channel_labels": run_context.get("channel_labels"),
            "exposure_ev": run_context.get("exposure_ev"),
            "brightness_pct": run_context.get("brightness_pct"),
            "contrast_pct": run_context.get("contrast_pct"),
            "show_atlas_region_borders": run_context.get("show_atlas_region_borders"),
            "count_tokens_enabled": run_context.get("count_tokens_enabled"),
            "ap_use_file_api": run_context.get("ap_use_file_api"),
            "ap_use_context_cache": run_context.get("ap_use_context_cache"),
            "ap_use_interactions": run_context.get("ap_use_interactions"),
            "run_started_at": run_context.get("run_started_at"),
            "estimated_position_mm": estimated_position,
            "actual_position_mm": actual_position,
            "signed_error_mm": signed_error_mm,
            "absolute_error_mm": absolute_error_mm,
            "affine_backend": affine_backend,
            "notes": user_metadata.get("notes") or "",
        }

    def _write_run_metadata(self, trace_dir: str, metadata: dict[str, object]) -> None:
        metadata_path = os.path.join(trace_dir, "classification.json")
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        self._append_log(f"Saved classification metadata: {metadata_path}")

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)

    def _set_view_mode(self, mode: str) -> None:
        self.current_view_mode = mode
        self.single_btn.setChecked(mode == "single")
        self.split_btn.setChecked(mode == "split")
        self.overlay_btn.setChecked(mode == "overlay")
        self.opacity_wrap.setVisible(mode == "overlay")

        if self.pil_image is None:
            self.view_stack.setCurrentWidget(self.page_upload)
            return

        if mode == "single":
            self.view_stack.setCurrentWidget(self.page_single)
        elif mode == "split":
            self.view_stack.setCurrentWidget(self.page_split)
        else:
            self.view_stack.setCurrentWidget(self.page_overlay)

    def _browse_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Brain Slice",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if file_path:
            self._load_image(file_path)

    def _load_image(self, file_path: str) -> None:
        try:
            loaded_state = load_image_state(
                file_path,
                fallback_pixel_size_um=self.pixel_size_um,
                max_long_edge=self.vlm_max_long_edge,
            )
        except Exception as exc:
            self._append_log(f"Failed to open image: {exc}")
            return

        self.image_path = file_path
        self.loaded_image_state = loaded_state
        self.source_image = loaded_state.canonical_image
        self._slice_channel_labels = loaded_state.channel_labels
        self.pixel_size_um = loaded_state.pixel_size_um
        self._pixel_size_source = loaded_state.pixel_size_source
        self._metadata_pixel_size_um = loaded_state.metadata_pixel_size_um
        self.pixel_size_spin.blockSignals(True)
        self.pixel_size_spin.setValue(loaded_state.pixel_size_um)
        self.pixel_size_spin.blockSignals(False)
        self.current_pos = None
        self._set_pipeline_mode("manual")
        self.ap_result = None
        self.affine_result = None
        self.registration_result = None
        self.preview_annotation_session = None
        self.preview_correspondences = None
        self._last_run_context = None
        self._refresh_agent_input_controls()
        self._reset_slice_adjustments()
        self._reset_steps()
        self._sync_atlas_viewers()
        self._set_ap_slider_value(self.ap_slider.value() / 100.0)
        self._update_display_pixmaps()
        self._set_view_mode(self.current_view_mode)
        self._append_log(f"Loaded image: {os.path.basename(file_path)}")
        self._append_log(f"Manual position initialized at {self.current_pos:.2f} mm")
        if loaded_state.metadata_pixel_size_um is not None:
            self._append_log(
                "Auto-detected pixel size: "
                f"{loaded_state.pixel_size_um:.4f} um/px "
                f"(source: {loaded_state.pixel_size_source})"
            )
        else:
            self._append_log(
                "No pixel-size metadata found; using current manual value: "
                f"{loaded_state.pixel_size_um:.4f} um/px"
            )
        self._append_log(
            "VLM image prepared: "
            f"original={loaded_state.original_size[0]}x{loaded_state.original_size[1]}px, "
            f"vlm={loaded_state.vlm_size[0]}x{loaded_state.vlm_size[1]}px, "
            f"scale={loaded_state.vlm_scale_factor:.4f}, "
            f"effective_pixel_size={loaded_state.vlm_effective_pixel_size_um:.4f} um/px"
        )

    def _clear_all(self) -> None:
        if self._is_worker_running():
            self._append_log("Cannot clear while the agent is running.")
            return

        self.image_path = None
        self.loaded_image_state = None
        self.source_image = None
        self.pil_image = None
        self.agent_vlm_image = None
        self.current_pos = None
        self._set_pipeline_mode("agent")
        self.ap_result = None
        self.affine_result = None
        self.registration_result = None
        self.preview_annotation_session = None
        self.preview_correspondences = None
        self._last_run_context = None
        self._slice_adjust_timer.stop()

        self.trace_inspector.clear_events()
        self.single_image_label.set_source_pixmap(None)
        self.split_image_label.set_source_pixmap(None)
        self.overlay_viewer.clear_all()
        self.split_atlas.clear()

        self.ap_slider.blockSignals(True)
        self.ap_slider.setValue(0)
        self.ap_slider.blockSignals(False)
        self.ap_value_label.setText("Manual Position: 0.00 mm")

        self._reset_steps()
        self._set_view_mode("single")

    def _current_channel_enabled(self) -> tuple[bool, ...]:
        labels = self._slice_channel_labels
        if len(labels) <= 1:
            return (self.channel_checkboxes[0].isChecked(),)
        return tuple(checkbox.isChecked() for checkbox in self.channel_checkboxes)

    def _refresh_agent_input_controls(self) -> None:
        labels = self._slice_channel_labels or ("Red", "Green", "Blue")
        for index, checkbox in enumerate(self.channel_checkboxes):
            visible = index < len(labels) if len(labels) > 1 else index == 0
            checkbox.setVisible(visible)
            if visible:
                if len(labels) == 1:
                    checkbox.setText(labels[0])
                else:
                    checkbox.setText(labels[index])
        self.exposure_value_label.setText(f"{self.exposure_slider.value() / 10.0:+.1f} EV")
        self.brightness_value_label.setText(f"{self.brightness_slider.value():+d}%")
        self.contrast_value_label.setText(f"{self.contrast_slider.value()}%")
        if self.source_image is None:
            self.agent_input_status_label.setText("Load a slice to configure what the agent sees.")
            return
        enabled_labels = []
        current_enabled = self._current_channel_enabled()
        if len(labels) == 1:
            enabled_labels = [labels[0]] if current_enabled[0] else ["none"]
        else:
            enabled_labels = [
                labels[index]
                for index in range(min(len(labels), len(current_enabled)))
                if current_enabled[index]
            ] or ["none"]
        atlas_mode = "on" if self._show_atlas_region_borders else "off"
        self.agent_input_status_label.setText(
            "Preview matches current agent input: "
            + f"channels={', '.join(enabled_labels)}, "
            + f"exposure={self.exposure_slider.value() / 10.0:+.1f} EV, "
            + f"brightness={self.brightness_slider.value():+d}%, "
            + f"contrast={self.contrast_slider.value()}%, "
            + f"vlm={self._current_vlm_resolution_label()}, "
            + f"atlas borders={atlas_mode}."
        )

    def _invalidate_pipeline_outputs(self, reason: str) -> None:
        had_outputs = any(
            value is not None
            for value in (
                self.ap_result,
                self.affine_result,
                self.registration_result,
                self.preview_correspondences,
            )
        )
        self.ap_result = None
        self.affine_result = None
        self.registration_result = None
        self.preview_annotation_session = None
        self.preview_correspondences = None
        self._last_run_context = None
        self._reset_steps()
        self._update_display_pixmaps()
        self._sync_atlas_viewers()
        if had_outputs:
            self._append_log(reason)

    def _apply_slice_adjustments(self) -> None:
        if self.source_image is None:
            return
        self._refresh_agent_input_controls()
        self.pil_image = render_slice_agent_image(
            self.source_image,
            channel_enabled=self._current_channel_enabled(),
            exposure_ev=self.exposure_slider.value() / 10.0,
            brightness=self.brightness_slider.value() / 100.0,
            contrast=self.contrast_slider.value() / 100.0,
        )
        prep = prepare_image_for_vlm(
            self.pil_image,
            pixel_size_um=self.pixel_size_um,
            max_long_edge=self.vlm_max_long_edge,
        )
        self.agent_vlm_image = prep.image
        self._vlm_scale_factor = prep.scale_factor
        self._vlm_effective_pixel_size_um = (
            prep.effective_pixel_size_um
            if prep.effective_pixel_size_um is not None
            else self.pixel_size_um
        )
        self._update_display_pixmaps()
        self._sync_atlas_viewers()

    def _queue_slice_adjustment_refresh(self) -> None:
        self._refresh_agent_input_controls()
        if self.source_image is None:
            return
        if self._is_worker_running():
            return
        self._slice_adjust_timer.start()
        self._invalidate_pipeline_outputs(
            "Agent input adjustments changed; cleared prior AP/registration results."
        )

    def _reset_slice_adjustments(self) -> None:
        for checkbox in self.channel_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        for slider, value in (
            (self.exposure_slider, 0),
            (self.brightness_slider, 0),
            (self.contrast_slider, 100),
        ):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self._refresh_agent_input_controls()
        self._apply_slice_adjustments()
        self._invalidate_pipeline_outputs("Agent input adjustments reset; cleared prior results.")

    def _reset_steps(self) -> None:
        self.step_ap.set_status("idle")
        self.step_affine.set_status("idle")
        self.step_ap.clear_result()
        self.step_affine.clear_result()
        self.agent_input_wrap.setVisible(self.source_image is not None)
        self.ap_adjust_wrap.setVisible(self.pil_image is not None)
        self.feedback_wrap.hide()
        self.success_btn.setEnabled(True)
        self.failure_btn.setEnabled(True)
        self._set_export_enabled(False)
        self._update_run_buttons()
        self.split_atlas.clear()
        self.overlay_viewer.clear()

    def _can_run_agent(self) -> bool:
        return self.pil_image is not None and not self._is_worker_running()

    def _can_run_manual_registration(self) -> bool:
        return (
            self.pil_image is not None
            and self.current_pos is not None
            and not self._is_worker_running()
        )

    def _update_run_buttons(self) -> None:
        agent_enabled = self._can_run_agent()
        manual_enabled = self._can_run_manual_registration()
        input_enabled = self.source_image is not None and not self._is_worker_running()

        self.run_button.setEnabled(agent_enabled)
        self.run_registration_button.setEnabled(manual_enabled)
        self.agent_input_wrap.setEnabled(input_enabled)
        self.ap_adjust_wrap.setEnabled(input_enabled)
        self.run_button.setToolTip(
            f"Run AP agent + landmark registration ({self.registration_workflow})"
        )

        message = self._manual_registration_status_text()
        self.run_registration_button.setToolTip(message)
        self.manual_registration_status_label.setText(message)

    def _manual_registration_status_text(self) -> str:
        if self._is_worker_running():
            return "Manual registration is unavailable while a pipeline run is active."
        if self.pil_image is None:
            return "Load an image to unlock manual position registration."
        if self.current_pos is None:
            return "Pick a manual AP position to enable registration."
        return (
            f"Ready: Run landmark registration from manual position {self.current_pos:.2f} mm "
            "(without AP agent estimation, "
            f"workflow={self.registration_workflow}, target={self.registration_landmark_count} pairs, edge hint=5 pairs)."
        )

    def _current_vlm_resolution_label(self) -> str:
        index = self.vlm_resolution_combo.currentIndex()
        if index < 0:
            return str(self.vlm_max_long_edge)
        label = self.vlm_resolution_combo.itemText(index)
        return label or str(self.vlm_max_long_edge)

    def _current_code_execution_enabled(self) -> bool:
        return bool(self._code_execution_enabled) and supports_code_execution(
            self.model_combo.currentText()
        )

    def _refresh_model_specific_controls(self) -> None:
        supported = supports_code_execution(self.model_combo.currentText())
        self.code_execution_checkbox.setVisible(supported)
        self.code_execution_checkbox.setEnabled(supported)

    def _set_pipeline_mode(self, mode: str) -> None:
        self._active_pipeline_mode = mode
        if mode == "manual":
            self.step_ap.set_title("Manual Position")
            return
        self.step_ap.set_title("Estimate Position (Agent)")

    def _build_run_context(self, atlas_name: str) -> dict[str, object]:
        if self.pil_image is None:
            return {}

        vlm_width = self.pil_image.width
        vlm_height = self.pil_image.height
        vlm_scale_factor = self._vlm_scale_factor
        vlm_effective_pixel_size_um = self._vlm_effective_pixel_size_um
        if self.agent_vlm_image is not None:
            vlm_width = self.agent_vlm_image.width
            vlm_height = self.agent_vlm_image.height

        return {
            "atlas_name": atlas_name,
            "model_name": self.model_combo.currentText(),
            "pixel_size_um": self.pixel_size_um,
            "pixel_size_source": self._pixel_size_source,
            "metadata_pixel_size_um": self._metadata_pixel_size_um,
            "image_width": self.pil_image.width,
            "image_height": self.pil_image.height,
            "vlm_width": vlm_width,
            "vlm_height": vlm_height,
            "vlm_scale_factor": vlm_scale_factor,
            "vlm_effective_pixel_size_um": vlm_effective_pixel_size_um,
            "vlm_max_long_edge": self.vlm_max_long_edge,
            "vlm_resolution_label": self._current_vlm_resolution_label(),
            "registration_landmark_count": self.registration_landmark_count,
            "registration_workflow": self.registration_workflow,
            "temperature": self.temperature_spin.value(),
            "ap_max_iterations": self.ap_max_iterations,
            "tool_loop_max_steps": self.registration_tool_loop_max_steps,
            "code_execution_enabled": self._current_code_execution_enabled(),
            "channel_enabled": list(self._current_channel_enabled()),
            "channel_labels": list(self._slice_channel_labels),
            "exposure_ev": self.exposure_slider.value() / 10.0,
            "brightness_pct": self.brightness_slider.value(),
            "contrast_pct": self.contrast_slider.value(),
            "show_atlas_region_borders": self._show_atlas_region_borders,
            "count_tokens_enabled": count_tokens_enabled(),
            "ap_use_file_api": ap_use_file_api(),
            "ap_use_context_cache": ap_use_context_cache(),
            "ap_use_interactions": ap_use_interactions(),
            "run_started_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _start_worker(self, worker: AgentWorker | ManualRegistrationWorker) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)

        self.worker_thread = thread
        self.worker = worker

        thread.started.connect(worker.run)
        worker.step_started.connect(self._on_step_started)
        worker.step_completed.connect(self._on_step_completed)
        worker.step_error.connect(self._on_step_error)
        worker.annotation_session_ready.connect(self._on_annotation_session_ready)
        worker.correspondences_ready.connect(self._on_correspondences_ready)
        worker.log_message.connect(self._append_log)
        worker.trace_event.connect(self._append_trace_event)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_cleaned)

        self._update_run_buttons()
        thread.start()

    def _run_agent(self) -> None:
        if not self._can_run_agent() or self.pil_image is None:
            return

        self._set_pipeline_mode("agent")
        self.ap_result = None
        self.affine_result = None
        self.registration_result = None
        self.preview_annotation_session = None
        self.preview_correspondences = None
        self._reset_steps()
        self._update_display_pixmaps()
        self.trace_inspector.clear_events()
        self._append_log("Starting agentic registration pipeline...")

        atlas_name = self._current_atlas_name()
        image_copy = self.pil_image.copy()
        vlm_image_copy = (
            self.agent_vlm_image.copy()
            if self.agent_vlm_image is not None
            else self.pil_image.copy()
        )
        self._last_run_context = self._build_run_context(atlas_name)

        self._start_worker(
            AgentWorker(
                image_copy,
                vlm_image_copy,
                atlas_name,
                target_landmark_count=self.registration_landmark_count,
                registration_workflow=self.registration_workflow,
                show_atlas_borders=self._show_atlas_region_borders,
                ap_max_iterations=self.ap_max_iterations,
                enable_code_execution=self._current_code_execution_enabled(),
                tool_loop_max_steps=self.registration_tool_loop_max_steps,
            )
        )

    def _run_manual_registration(self) -> None:
        if (
            not self._can_run_manual_registration()
            or self.pil_image is None
            or self.current_pos is None
        ):
            return

        self._set_pipeline_mode("manual")
        self.ap_result = None
        self.affine_result = None
        self.registration_result = None
        self.preview_annotation_session = None
        self.preview_correspondences = None
        self._reset_steps()
        self._update_display_pixmaps()
        self.trace_inspector.clear_events()
        self._append_log("Starting registration runtime from manual AP position...")

        atlas_name = self._current_atlas_name()
        position_mm = float(self.current_pos)
        self._last_run_context = self._build_run_context(atlas_name)

        self._start_worker(
            ManualRegistrationWorker(
                self.pil_image.copy(),
                atlas_name,
                position_mm,
                target_landmark_count=self.registration_landmark_count,
                registration_workflow=self.registration_workflow,
                show_atlas_borders=self._show_atlas_region_borders,
                enable_code_execution=self._current_code_execution_enabled(),
                tool_loop_max_steps=self.registration_tool_loop_max_steps,
            )
        )

    def _on_step_started(self, step_id: str) -> None:
        if step_id == "ap":
            self.step_ap.set_status("running")
            self._append_trace_event(stage_event("ap", "AP Estimation"))
        elif step_id == "affine":
            self.step_affine.set_status("running")
            self._append_trace_event(stage_event("registration", "Registration"))

    def _on_step_completed(self, step_id: str, result: object) -> None:
        if step_id == "ap" and isinstance(result, APResult):
            manual_mode = self._active_pipeline_mode == "manual"
            self.ap_result = result
            self.current_pos = result.position_mm
            self.step_ap.set_status("completed")
            self.step_ap.show_ap_result(
                result,
                position_label="Manual Position" if manual_mode else "Estimated Position",
            )
            self._set_ap_slider_value(result.position_mm)
            self.ap_adjust_wrap.show()
            if manual_mode:
                self._append_log(f"Manual position locked: {result.position_mm:.2f} mm")
            else:
                self._append_log(f"Final position estimated: {result.position_mm:.2f} mm")
            return

        if step_id == "affine" and isinstance(result, RegistrationResult):
            self.registration_result = result
            self.affine_result = result.affine_result
            self.preview_annotation_session = None
            self.preview_correspondences = None
            affine_result = result.affine_result
            self.step_affine.set_status("completed")
            self.step_affine.show_affine_result(affine_result)
            tx_px, ty_px = affine_result.translation_px
            self._append_log(
                "Affine calculated: "
                f"backend={affine_result.backend}, "
                f"rot={affine_result.rotation_deg:.2f} deg, "
                f"translate=({tx_px:.1f}, {ty_px:.1f}) px, "
                f"scale=({affine_result.scale[0]:.3f}, {affine_result.scale[1]:.3f}), "
                f"shear={affine_result.shear:.3f}, "
                f"accepted_pairs={len(result.accepted_correspondences)}"
            )
            self._update_display_pixmaps()
            self._set_view_mode("split")

        self._set_export_enabled(self._all_steps_completed())

    def _on_correspondences_ready(self, correspondences_obj: object) -> None:
        if not isinstance(correspondences_obj, list):
            return
        correspondences: list[RegistrationCorrespondence] = []
        for item in correspondences_obj:
            if isinstance(item, RegistrationCorrespondence):
                correspondences.append(item)
        if not correspondences:
            return
        self.preview_correspondences = correspondences
        self._append_log(f"Received {len(correspondences)} landmark pairs for visual review")
        self._update_display_pixmaps()
        self._set_view_mode("split")

    def _on_annotation_session_ready(self, session_obj: object) -> None:
        if not isinstance(session_obj, RegistrationAnnotationSession):
            return
        self.preview_annotation_session = session_obj
        if self.current_pos is None:
            position_hint = session_obj.metadata.get("position_mm")
            if isinstance(position_hint, (int, float)):
                self._set_ap_slider_value(float(position_hint))
        atlas_count = len(session_obj.atlas_annotations)
        slice_count = len(session_obj.slice_annotations)
        pass_label = session_obj.metadata.get("pass")
        if pass_label == 1:
            self._append_log(
                f"Atlas annotation pass ready: showing {atlas_count} proposed atlas landmarks"
            )
        elif pass_label == 2:
            self._append_log(
                "Slice transfer pass ready: showing "
                f"{atlas_count} atlas landmarks and {slice_count} slice landmarks"
            )
        else:
            self._append_log(
                f"Registration annotation preview updated: atlas={atlas_count}, slice={slice_count}"
            )
        self._update_display_pixmaps()
        self._set_view_mode("split")

    def _on_step_error(self, step_id: str, message: str) -> None:
        if step_id == "ap":
            self.step_ap.set_status("error")
            self.step_ap.show_error(message)
        elif step_id == "affine":
            self.step_affine.set_status("error")
            self.step_affine.show_error(message)
        self._append_log(f"Error in {step_id}: {message}")
        self._set_export_enabled(False)

    def _on_worker_finished(self) -> None:
        self._append_log("Pipeline completed.")
        self._update_run_buttons()
        self._set_export_enabled(self._all_steps_completed())

        # Enable feedback buttons if debug traces were generated
        if self._all_steps_completed() and getattr(self.ap_result, "debug_dir", None):
            self.feedback_wrap.show()
            self.success_btn.setEnabled(True)
            self.failure_btn.setEnabled(True)

    def _on_thread_cleaned(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._update_run_buttons()

    def _all_steps_completed(self) -> bool:
        return self.ap_result is not None and self.affine_result is not None

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled)

    def _is_worker_running(self) -> bool:
        return self.worker_thread is not None

    def _refresh_registration_workflow_options(self, preferred_workflow: str | None = None) -> None:
        options = get_registration_workflow_options(self.model_combo.currentText())
        previous = preferred_workflow or self.registration_workflow
        self.workflow_combo.blockSignals(True)
        self.workflow_combo.clear()
        selected_index = 0
        for index, (label, value) in enumerate(options):
            self.workflow_combo.addItem(label, value)
            if value == previous:
                selected_index = index
        self.workflow_combo.setCurrentIndex(selected_index)
        self.workflow_combo.blockSignals(False)
        current_value = self.workflow_combo.currentData()
        if isinstance(current_value, str) and current_value:
            self.registration_workflow = current_value
        else:
            self.registration_workflow = default_registration_workflow(
                self.model_combo.currentText()
            )

    def _on_registration_workflow_changed(self, index: int) -> None:
        _ = index
        workflow_value = self.workflow_combo.currentData()
        if not isinstance(workflow_value, str) or not workflow_value:
            return
        self.registration_workflow = workflow_value
        self._append_log(f"Registration workflow set to {workflow_value}")
        self._update_run_buttons()

    def _on_settings_saved(self) -> None:
        self._append_log("Settings saved. Transport toggles will apply to the next run.")

    def _on_model_changed(self, model_name: str) -> None:
        set_model_name(model_name)
        previous_workflow = self.registration_workflow
        self._refresh_registration_workflow_options(self.registration_workflow)
        if previous_workflow != self.registration_workflow:
            self._append_log(
                f"Model changed: {model_name}; registration workflow switched to {self.registration_workflow}"
            )
        else:
            self._append_log(f"Model changed: {model_name}")
        if is_image_generation_model(model_name):
            self.workflow_combo.setToolTip(
                "Image-generation model selected. Registration uses the two-shot image-gen workflow."
            )
        else:
            self.workflow_combo.setToolTip(
                "Registration workflow. Image-gen mode is only available when an image-generation model is selected."
            )
        self._refresh_model_specific_controls()
        self._update_run_buttons()

    def _on_atlas_changed(self) -> None:
        atlas_name = self._current_atlas_name()
        self.split_atlas.set_atlas(atlas_name)
        self.overlay_viewer.set_atlas(atlas_name)
        self.split_atlas.set_show_region_borders(self._show_atlas_region_borders)
        self.overlay_viewer.set_show_region_borders(self._show_atlas_region_borders)
        if self.pil_image is not None:
            self._sync_atlas_viewers()
        self._append_log(f"Atlas selected: {atlas_name}")

    def _current_atlas_name(self) -> str:
        return canonicalize_atlas_name(str(self.atlas_combo.currentData() or DEFAULT_ATLAS_NAME))

    def _sync_atlas_viewers(self) -> None:
        atlas_name = self._current_atlas_name()
        self.split_atlas.set_atlas(atlas_name)
        self.split_atlas.set_show_region_borders(self._show_atlas_region_borders)
        self.overlay_viewer.set_atlas(atlas_name)
        self.overlay_viewer.set_show_region_borders(self._show_atlas_region_borders)

        # Update slider range based on atlas
        try:
            from langslice.atlas import get_position_range_mm, load_atlas

            atlas = load_atlas(atlas_name)
            _, max_pos = get_position_range_mm(atlas)
            self.ap_slider.setRange(0, int(round(max_pos * 100)))
        except Exception:
            pass

        if self.current_pos is None:
            self.split_atlas.clear()
            self.overlay_viewer.clear()
            return

        self.split_atlas.set_position(self.current_pos)
        self.overlay_viewer.set_position(self.current_pos)

    def _on_ap_slider_changed(self, slider_value: int) -> None:
        self._set_ap_slider_value(slider_value / 100.0)

    def _on_pixel_size_changed(self, value: float) -> None:
        self.pixel_size_um = value
        self._pixel_size_source = "manual_override"
        if self.loaded_image_state is not None:
            self.loaded_image_state = self.loaded_image_state.with_pixel_size(
                value,
                source="manual_override",
            )
            self._append_log(
                "Pixel size updated manually: "
                f"{value:.4f} um/px; "
                f"effective VLM pixel size={self.loaded_image_state.vlm_effective_pixel_size_um:.4f} um/px"
            )
        self._apply_slice_adjustments()
        self._sync_atlas_viewers()

    def _on_vlm_resolution_changed(self, index: int) -> None:
        value = self.vlm_resolution_combo.itemData(index)
        if not isinstance(value, int) or value <= 0:
            return
        self.vlm_max_long_edge = int(value)
        self._append_log(
            f"VLM max resolution set to {self._current_vlm_resolution_label()} ({self.vlm_max_long_edge}px long edge)"
        )
        if self.source_image is None or self._is_worker_running():
            return
        self._apply_slice_adjustments()
        self._invalidate_pipeline_outputs(
            "VLM resolution changed; cleared prior AP/registration results."
        )

    def _on_code_execution_toggled(self, checked: bool) -> None:
        self._code_execution_enabled = bool(checked)
        status = "enabled" if self._current_code_execution_enabled() else "disabled"
        self._append_log(f"Registration code execution {status}")

    def _on_ap_max_iterations_changed(self, value: int) -> None:
        self.ap_max_iterations = max(1, int(value))
        self._append_log(f"AP max turns set to {self.ap_max_iterations}")

    def _on_tool_loop_max_steps_changed(self, value: int) -> None:
        self.registration_tool_loop_max_steps = max(1, int(value))
        self._append_log(
            f"Registration tool-loop max steps set to {self.registration_tool_loop_max_steps}"
        )

    def _on_atlas_borders_toggled(self, checked: bool) -> None:
        self._show_atlas_region_borders = bool(checked)
        self._sync_atlas_viewers()
        self._invalidate_pipeline_outputs(
            "Atlas border visibility changed; cleared prior registration results."
        )

    def _on_landmark_count_changed(self, value: int) -> None:
        self.registration_landmark_count = int(value)
        self._append_log(
            f"Registration landmark target set to {self.registration_landmark_count} pairs"
        )
        self._update_run_buttons()

    def _on_thinking_level_changed(self, index: int) -> None:
        level = self.thinking_combo.itemData(index)
        if isinstance(level, str) and level:
            set_thinking_level(level)
            label = self.thinking_combo.itemText(index)
            self._append_log(f"Registration thinking level set to {label} ({level})")

    def _on_temperature_changed(self, value: float) -> None:
        set_temperature(value)
        self._append_log(f"Model temperature set to {value:.2f}")

    def _set_ap_slider_value(self, position_mm: float) -> None:
        min_pos = self.ap_slider.minimum() / 100.0
        max_pos = self.ap_slider.maximum() / 100.0
        clamped_position = max(min_pos, min(max_pos, float(position_mm)))
        value = int(round(clamped_position * 100))
        self.ap_slider.blockSignals(True)
        self.ap_slider.setValue(value)
        self.ap_slider.blockSignals(False)
        self.current_pos = clamped_position
        self.ap_value_label.setText(f"Manual Position: {clamped_position:.2f} mm")
        self._sync_atlas_viewers()
        self._update_run_buttons()

    def _on_opacity_changed(self, value: int) -> None:
        self.overlay_viewer.set_atlas_opacity(value / 100.0)

    def _transformed_pixmap(self) -> QPixmap | None:
        if self.pil_image is None:
            return None

        pixmap = pil_to_qpixmap(self.pil_image)
        if not should_apply_affine_to_slice(
            self.affine_result,
            (self.pil_image.width, self.pil_image.height),
        ):
            return pixmap

        assert self.affine_result is not None
        out_w, out_h = self.affine_result.output_size
        canvas = QPixmap(out_w, out_h)
        canvas.fill(QColor(0, 0, 0, 0))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setTransform(affine_matrix_to_qtransform(self.affine_result.matrix))
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return canvas

    def _update_display_pixmaps(self) -> None:
        base_pixmap = self._transformed_pixmap()
        split_slice_points, split_atlas_points = build_split_view_correspondence_points(
            self.registration_result,
            self.affine_result,
            self.preview_annotation_session,
            self.preview_correspondences,
        )
        single_pixmap = base_pixmap
        if single_pixmap is not None and split_slice_points:
            single_pixmap = draw_correspondence_markers(
                single_pixmap,
                split_slice_points,
                color=QColor(245, 158, 11),
            )
        split_pixmap = base_pixmap
        if split_pixmap is not None and split_slice_points:
            split_pixmap = draw_correspondence_markers(
                split_pixmap,
                split_slice_points,
                color=QColor(245, 158, 11),
            )
        self.single_image_label.set_source_pixmap(single_pixmap)
        self.split_image_label.set_source_pixmap(split_pixmap)
        self.split_atlas.set_correspondence_markers(split_atlas_points)
        self.overlay_viewer.set_slice_pixmap(base_pixmap)
        self.overlay_viewer.set_correspondence_markers(split_slice_points, split_atlas_points)
        self._sync_atlas_viewers()

    def _append_log(self, message: str) -> None:
        stage = "registration" if self.step_affine._status == "running" else "session"
        self._append_trace_event(status_event(stage, message))

    def _append_trace_event(self, event_obj: object) -> None:
        if not isinstance(event_obj, dict):
            return
        self.trace_inspector.append_event(event_obj)

    def _export_abba(self) -> None:
        if not self._all_steps_completed() or self.pil_image is None or self.image_path is None:
            return

        atlas_name = self._current_atlas_name()
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export QUINT/ABBA JSON",
            "registration_abba.json",
            "JSON (*.json)",
        )
        if not out_path:
            return

        # Gather atlas info for anchoring computation
        try:
            from langslice.atlas import load_atlas

            atlas = load_atlas(atlas_name)
            atlas_shape = atlas.reference.shape
            atlas_resolution = atlas.resolution
        except Exception:
            # Fallback: use reasonable defaults for Allen Mouse 25um
            atlas_shape = (528, 320, 456)
            atlas_resolution = (25.0, 25.0, 25.0)

        quint_export = build_quint_export(
            filename=self.image_path,
            position_mm=self.current_pos or 0.0,
            atlas_name=atlas_name,
            atlas_shape=atlas_shape,
            atlas_resolution=atlas_resolution,
            image_width=self.pil_image.width,
            image_height=self.pil_image.height,
            affine_matrix=self.affine_result.matrix if self.affine_result else None,
            output_width=self.affine_result.output_width if self.affine_result else None,
            output_height=self.affine_result.output_height if self.affine_result else None,
        )

        try:
            save_quint_json(quint_export, out_path)
            self._append_log(f"Exported QUINT/ABBA JSON: {out_path}")
        except Exception as exc:
            self._append_log(f"Export failed: {exc}")


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LangSlice")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
