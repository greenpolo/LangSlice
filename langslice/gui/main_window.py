"""Main application window for LangSlice."""

from __future__ import annotations

import os
import sys
import importlib
from typing import Any

from PIL import Image

_qtcore = importlib.import_module("PySide6.QtCore")
_qtgui = importlib.import_module("PySide6.QtGui")
_qtwidgets = importlib.import_module("PySide6.QtWidgets")

Qt = _qtcore.Qt
Signal = _qtcore.Signal
QObject = _qtcore.QObject
QThread = _qtcore.QThread
QTimer = _qtcore.QTimer
QTime = _qtcore.QTime

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
QPlainTextEdit = _qtwidgets.QPlainTextEdit
QPushButton = _qtwidgets.QPushButton
QCheckBox = _qtwidgets.QCheckBox
QSizePolicy = _qtwidgets.QSizePolicy
QSlider = _qtwidgets.QSlider
QSplitter = _qtwidgets.QSplitter
QStackedWidget = _qtwidgets.QStackedWidget
QVBoxLayout = _qtwidgets.QVBoxLayout
QWidget = _qtwidgets.QWidget
QDoubleSpinBox = _qtwidgets.QDoubleSpinBox

from langslice import __version__
from langslice.atlas import (
    DEFAULT_ATLAS_NAME,
    canonicalize_atlas_name,
    list_available_atlases,
    list_downloaded_atlases,
)
from langslice.gui.theme import ACCENT, BG_PRIMARY, ERROR, STYLESHEET, SUCCESS, TEXT_SECONDARY
from langslice.vlm import APResult, AffineResult, PreprocessOptions, estimate_affine, estimate_position
from langslice.export import build_quint_export, save_quint_json
from langslice.gui.settings_dialog import SettingsDialog
from langslice.gui.overlay_viewer import OverlayGraphicsView

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


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    rgba_image = image.convert("RGBA")
    width, height = rgba_image.size
    data = rgba_image.tobytes("raw", "RGBA")
    qt_image = QImage(data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qt_image.copy())


class ImageLabel(QLabel):
    """QLabel that keeps and scales a source pixmap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_source_pixmap(self, pixmap: QPixmap | None) -> None:
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap is None or self.width() <= 1 or self.height() <= 1:
            self.clear()
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class CanvasArea(QFrame):
    """Canvas zone with drag-and-drop support and subtle grid background."""

    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return

        for url in mime.urls():
            if url.isLocalFile() and self._is_image_path(url.toLocalFile()):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()
            if self._is_image_path(file_path):
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        painter = None
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            pen = QPen(QColor(255, 255, 255, 8))
            pen.setWidth(1)
            painter.setPen(pen)
            spacing = 40
            width = self.width()
            height = self.height()
            x = 0
            while x <= width:
                painter.drawLine(x, 0, x, height)
                x += spacing
            y = 0
            while y <= height:
                painter.drawLine(0, y, width, y)
                y += spacing
        finally:
            if painter is not None:
                painter.end()

    @staticmethod
    def _is_image_path(path: str) -> bool:
        return path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))


class AgentWorker(QObject):
    """Runs AP + affine estimation in a worker thread."""

    step_started = Signal(str)
    step_completed = Signal(str, object)
    step_error = Signal(str, str)
    log_message = Signal(str)
    finished = Signal()

    def __init__(
        self,
        image: Image.Image,
        atlas_name: str,
        preprocess_options: PreprocessOptions,
        pixel_size_um: float = 4.0,
    ):
        super().__init__()
        self.image = image
        self.atlas_name = atlas_name
        self.preprocess_options = preprocess_options
        self.pixel_size_um = pixel_size_um

    def run(self) -> None:
        try:
            self.step_started.emit("ap")
            self.log_message.emit("Starting position estimation...")
            ap_result = estimate_position(
                self.image,
                self.atlas_name,
                on_progress=self.log_message.emit,
                preprocess_options=self.preprocess_options,
            )
            self.step_completed.emit("ap", ap_result)
        except Exception as exc:
            self.step_error.emit("ap", str(exc))
            self.finished.emit()
            return

        try:
            self.step_started.emit("affine")
            self.log_message.emit("Starting affine estimation...")
            affine_result = estimate_affine(
                self.image,
                on_progress=self.log_message.emit,
                preprocess_options=self.preprocess_options,
                atlas_name=self.atlas_name,
                position_mm=ap_result.position_mm,
                pixel_size_um=self.pixel_size_um,
            )
            self.step_completed.emit("affine", affine_result)
        except Exception as exc:
            self.step_error.emit("affine", str(exc))
        finally:
            self.finished.emit()


class StepIndicator(QFrame):
    """Step row in the right-side timeline panel."""

    def __init__(self, title: str, has_connector: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = "idle"
        self._pulse = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(0)
        left_col.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.circle = QLabel()
        self.circle.setFixedSize(18, 18)
        self.circle.setStyleSheet("border-radius: 9px;")
        left_col.addWidget(self.circle, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.connector = QFrame()
        self.connector.setFixedWidth(1)
        self.connector.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.connector.setStyleSheet("background-color: rgba(255, 255, 255, 25);")
        self.connector.setVisible(has_connector)
        left_col.addWidget(self.connector, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addLayout(left_col)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"font-weight: 600; color: {TEXT_SECONDARY};")
        right_col.addWidget(self.title_label)

        self.result_panel = QFrame()
        self.result_panel.setObjectName("glassPanel")
        panel_layout = QVBoxLayout(self.result_panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(4)

        self.result_top = QLabel()
        self.result_top.setStyleSheet("font-size: 12px;")
        self.result_reasoning = QLabel()
        self.result_reasoning.setWordWrap(True)
        self.result_reasoning.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY};")
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"font-size: 12px; color: {ERROR};")

        panel_layout.addWidget(self.result_top)
        panel_layout.addWidget(self.result_reasoning)
        panel_layout.addWidget(self.error_label)

        self.result_panel.hide()
        right_col.addWidget(self.result_panel)
        root.addLayout(right_col, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._tick)
        self.set_status("idle")

    def set_status(self, status: str) -> None:
        self._status = status
        if status == "running":
            self._timer.start()
            self.title_label.setStyleSheet(f"font-weight: 600; color: {ACCENT};")
        elif status == "completed":
            self._timer.stop()
            self.title_label.setStyleSheet(f"font-weight: 600; color: {SUCCESS};")
            self.circle.setStyleSheet(f"background-color: {SUCCESS}; border-radius: 9px;")
        elif status == "error":
            self._timer.stop()
            self.title_label.setStyleSheet(f"font-weight: 600; color: {ERROR};")
            self.circle.setStyleSheet(f"background-color: {ERROR}; border-radius: 9px;")
        else:
            self._timer.stop()
            self.title_label.setStyleSheet(f"font-weight: 600; color: {TEXT_SECONDARY};")
            self.circle.setStyleSheet("background-color: #555555; border-radius: 9px;")

    def show_ap_result(self, result: APResult) -> None:
        self.result_top.setText(f"Estimated Position: <span style='color:{SUCCESS}'>{result.position_mm:.2f} mm</span>")
        self.result_reasoning.setText(result.reasoning)
        self.error_label.clear()
        self.result_panel.show()

    def show_affine_result(self, result: AffineResult) -> None:
        self.result_top.setText(
            " | ".join(
                [
                    f"Rotation: {result.rotation:.2f}°",
                    f"Translate X: {result.translateX:.2f}%",
                    f"Translate Y: {result.translateY:.2f}%",
                ]
            )
        )
        self.result_reasoning.setText(result.reasoning)
        self.error_label.clear()
        self.result_panel.show()

    def show_error(self, message: str) -> None:
        self.result_top.clear()
        self.result_reasoning.clear()
        self.error_label.setText(message)
        self.result_panel.show()

    def clear_result(self) -> None:
        self.result_top.clear()
        self.result_reasoning.clear()
        self.error_label.clear()
        self.result_panel.hide()

    def _tick(self) -> None:
        if self._status != "running":
            return
        self._pulse = not self._pulse
        color = ACCENT if self._pulse else "#4f46e5"
        self.circle.setStyleSheet(f"background-color: {color}; border-radius: 9px;")


class MainWindow(QMainWindow):
    """LangSlice desktop application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LangSlice")
        self.resize(1460, 900)
        self.setMinimumSize(1100, 720)

        self.image_path: str | None = None
        self.pil_image: Image.Image | None = None
        self.current_pos: float | None = None
        self.ap_result: APResult | None = None
        self.affine_result: AffineResult | None = None
        self.current_view_mode = "single"

        self.worker_thread: QThread | None = None
        self.worker: AgentWorker | None = None

        self.pixel_size_um: float = 4.0

        self._build_ui()
        self._set_view_mode("single")
        self._set_export_enabled(False)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_agent_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1020, 400])
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

        self.atlas_combo = QComboBox()
        self._populate_atlas_combo()
        self.atlas_combo.currentIndexChanged.connect(self._on_atlas_changed)

        px_size_label = QLabel("Pixel Size:")
        px_size_label.setObjectName("subheading")
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.1, 100.0)
        self.pixel_size_spin.setValue(self.pixel_size_um)
        self.pixel_size_spin.setSuffix(" µm/px")
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setSingleStep(0.5)
        self.pixel_size_spin.setFixedWidth(130)
        self.pixel_size_spin.setToolTip("Pixel size of the histology image in micrometers per pixel")
        self.pixel_size_spin.valueChanged.connect(self._on_pixel_size_changed)

        self.upload_button = QPushButton("Open Image...")
        self.upload_button.setObjectName("secondary")
        self.upload_button.clicked.connect(self._browse_image)

        self.settings_button = QPushButton("⚙ Settings")
        self.settings_button.setObjectName("secondary")
        self.settings_button.clicked.connect(self._open_settings)

        self.export_button = QPushButton("Export ABBA")
        self.export_button.setObjectName("secondary")
        self.export_button.clicked.connect(self._export_abba)

        right.addWidget(self.atlas_combo)
        right.addWidget(px_size_label)
        right.addWidget(self.pixel_size_spin)
        right.addWidget(self.upload_button)
        right.addWidget(self.settings_button)
        right.addWidget(self.export_button)
        layout.addLayout(right)
        return header

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec()

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
        self.single_image_label.setStyleSheet("border: 1px solid rgba(255,255,255,20); border-radius: 10px;")
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
        self.overlay_viewer.set_pixel_size(self.pixel_size_um)
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
        title = QLabel("Agent Workflow")
        title.setObjectName("heading")
        self.run_button = QPushButton("Run Agent")
        self.run_button.clicked.connect(self._run_agent)
        hrow.addWidget(title)
        hrow.addStretch(1)
        hrow.addWidget(self.run_button)
        layout.addWidget(header)

        self.ap_adjust_wrap = QFrame()
        self.ap_adjust_wrap.setObjectName("glassPanel")
        ap_layout = QVBoxLayout(self.ap_adjust_wrap)
        ap_layout.setContentsMargins(10, 10, 10, 10)
        self.ap_value_label = QLabel("Manual Position: 0.00 mm")
        self.ap_value_label.setObjectName("monoLabel")
        self.ap_slider = QSlider(Qt.Orientation.Horizontal)
        self.ap_slider.setRange(0, 2000)  # Default 0-20mm
        self.ap_slider.setValue(0)
        self.ap_slider.valueChanged.connect(self._on_ap_slider_changed)
        ap_layout.addWidget(self.ap_value_label)
        ap_layout.addWidget(self.ap_slider)
        self.ap_adjust_wrap.hide()
        layout.addWidget(self.ap_adjust_wrap)

        self.step_ap = StepIndicator("Estimate Position", has_connector=True)
        self.step_affine = StepIndicator("Affine Transformation", has_connector=False)

        preprocess_panel = QFrame()
        preprocess_panel.setObjectName("glassPanel")
        preprocess_layout = QVBoxLayout(preprocess_panel)
        preprocess_layout.setContentsMargins(10, 10, 10, 10)
        preprocess_layout.setSpacing(6)

        self.preprocess_checkbox = QCheckBox("Enable image preprocessing")
        self.preprocess_checkbox.setChecked(False)
        self.preprocess_hint = QLabel("Crops tissue, normalizes contrast, and resizes before VLM calls.")
        self.preprocess_hint.setWordWrap(True)
        self.preprocess_hint.setStyleSheet(f"font-size: 12px; color: {TEXT_SECONDARY};")
        preprocess_layout.addWidget(self.preprocess_checkbox)
        preprocess_layout.addWidget(self.preprocess_hint)

        layout.addWidget(preprocess_panel)
        layout.addWidget(self.step_ap)
        layout.addWidget(self.step_affine)

        logs_label = QLabel("Agent Logs")
        logs_label.setObjectName("monoLabel")
        layout.addWidget(logs_label)

        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setPlaceholderText("Waiting for input...")
        self.logs.setMinimumHeight(180)
        layout.addWidget(self.logs, stretch=1)

        return panel

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
            with Image.open(file_path) as loaded:
                image = loaded.copy()
        except Exception as exc:
            self._append_log(f"Failed to open image: {exc}")
            return

        self.image_path = file_path
        self.pil_image = image
        self.current_pos = None
        self.ap_result = None
        self.affine_result = None
        self._reset_steps()
        self._update_display_pixmaps()
        self._set_view_mode(self.current_view_mode)
        self._append_log(f"Loaded image: {os.path.basename(file_path)}")

    def _reset_steps(self) -> None:
        self.step_ap.set_status("idle")
        self.step_affine.set_status("idle")
        self.step_ap.clear_result()
        self.step_affine.clear_result()
        self.ap_adjust_wrap.hide()
        self._set_export_enabled(False)
        self.run_button.setEnabled(self.pil_image is not None and not self._is_worker_running())
        self.split_atlas.clear()
        self.overlay_viewer.clear()

    def _run_agent(self) -> None:
        if self.pil_image is None or self._is_worker_running():
            return

        self._reset_steps()
        self._append_log("Starting agentic registration pipeline...")

        atlas_name = self._current_atlas_name()
        image_copy = self.pil_image.copy()
        preprocess_options = PreprocessOptions(enabled=self.preprocess_checkbox.isChecked())

        if preprocess_options.enabled:
            self._append_log("Preprocessing enabled: crop tissue + normalize contrast + resize.")
        else:
            self._append_log("Preprocessing disabled: sending original image content.")

        thread = QThread(self)
        worker = AgentWorker(image_copy, atlas_name, preprocess_options=preprocess_options, pixel_size_um=self.pixel_size_um)
        worker.moveToThread(thread)

        self.worker_thread = thread
        self.worker = worker

        thread.started.connect(worker.run)
        worker.step_started.connect(self._on_step_started)
        worker.step_completed.connect(self._on_step_completed)
        worker.step_error.connect(self._on_step_error)
        worker.log_message.connect(self._append_log)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_cleaned)

        self.run_button.setEnabled(False)
        thread.start()

    def _on_step_started(self, step_id: str) -> None:
        if step_id == "ap":
            self.step_ap.set_status("running")
        elif step_id == "affine":
            self.step_affine.set_status("running")

    def _on_step_completed(self, step_id: str, result: object) -> None:
        if step_id == "ap" and isinstance(result, APResult):
            self.ap_result = result
            self.current_pos = result.position_mm
            self.step_ap.set_status("completed")
            self.step_ap.show_ap_result(result)
            self._set_ap_slider_value(result.position_mm)
            self.ap_adjust_wrap.show()
            self._sync_atlas_viewers()
            self._append_log(f"Final position estimated: {result.position_mm:.2f} mm")
            return

        if step_id == "affine" and isinstance(result, AffineResult):
            self.affine_result = result
            self.step_affine.set_status("completed")
            self.step_affine.show_affine_result(result)
            self._append_log(
                f"Affine calculated: rot {result.rotation:.2f}°, X {result.translateX:.2f}%, Y {result.translateY:.2f}%"
            )
            self._update_display_pixmaps()
            self._set_view_mode("split")

        self._set_export_enabled(self._all_steps_completed())

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
        self.run_button.setEnabled(self.pil_image is not None)
        self._set_export_enabled(self._all_steps_completed())

    def _on_thread_cleaned(self) -> None:
        self.worker = None
        self.worker_thread = None

    def _all_steps_completed(self) -> bool:
        return self.ap_result is not None and self.affine_result is not None

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled)

    def _is_worker_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def _on_atlas_changed(self) -> None:
        atlas_name = self._current_atlas_name()
        self.split_atlas.set_atlas(atlas_name)
        self.overlay_viewer.set_atlas(atlas_name)
        if self.current_pos is not None:
            self._sync_atlas_viewers()
        self._append_log(f"Atlas selected: {atlas_name}")

    def _current_atlas_name(self) -> str:
        return canonicalize_atlas_name(str(self.atlas_combo.currentData() or DEFAULT_ATLAS_NAME))

    def _sync_atlas_viewers(self) -> None:
        atlas_name = self._current_atlas_name()
        self.split_atlas.set_atlas(atlas_name)
        self.overlay_viewer.set_atlas(atlas_name)
        self.overlay_viewer.set_pixel_size(self.pixel_size_um)

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
        self.current_pos = slider_value / 100.0
        self.ap_value_label.setText(f"Manual Position: {self.current_pos:.2f} mm")
        self._sync_atlas_viewers()

    def _on_pixel_size_changed(self, value: float) -> None:
        self.pixel_size_um = value
        self._sync_atlas_viewers()

    def _set_ap_slider_value(self, position_mm: float) -> None:
        value = int(round(position_mm * 100))
        self.ap_slider.blockSignals(True)
        self.ap_slider.setValue(value)
        self.ap_slider.blockSignals(False)
        self.ap_value_label.setText(f"Manual Position: {position_mm:.2f} mm")

    def _on_opacity_changed(self, value: int) -> None:
        self.overlay_viewer.set_atlas_opacity(value / 100.0)

    def _transformed_pixmap(self) -> QPixmap | None:
        if self.pil_image is None:
            return None

        pixmap = pil_to_qpixmap(self.pil_image)
        if self.affine_result is None:
            return pixmap

        tx_px = pixmap.width() * (self.affine_result.translateX / 100.0)
        ty_px = pixmap.height() * (self.affine_result.translateY / 100.0)
        transform = QTransform()
        transform.translate((pixmap.width() / 2.0) + tx_px, (pixmap.height() / 2.0) + ty_px)
        transform.rotate(self.affine_result.rotation)
        transform.translate(-pixmap.width() / 2.0, -pixmap.height() / 2.0)
        return pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)

    def _update_display_pixmaps(self) -> None:
        pixmap = self._transformed_pixmap()
        self.single_image_label.set_source_pixmap(pixmap)
        self.split_image_label.set_source_pixmap(pixmap)
        self.overlay_viewer.set_slice_pixmap(pixmap)
        self._sync_atlas_viewers()

    def _append_log(self, message: str) -> None:
        now = QTime.currentTime().toString("HH:mm:ss")
        self.logs.appendPlainText(f"[{now}] {message}")
        scrollbar = self.logs.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _export_abba(self) -> None:
        if not self._all_steps_completed() or self.pil_image is None or self.image_path is None:
            return

        atlas_name = self._current_atlas_name()
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ABBA JSON",
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

        rotation = self.affine_result.rotation if self.affine_result else 0.0
        tx = self.affine_result.translateX if self.affine_result else 0.0
        ty = self.affine_result.translateY if self.affine_result else 0.0

        quint_export = build_quint_export(
            filename=self.image_path,
            position_mm=self.current_pos or 0.0,
            atlas_name=atlas_name,
            atlas_shape=atlas_shape,
            atlas_resolution=atlas_resolution,
            image_width=self.pil_image.width,
            image_height=self.pil_image.height,
            rotation_deg=rotation,
            translate_x_pct=tx,
            translate_y_pct=ty,
        )

        try:
            save_quint_json(quint_export, out_path)
            self._append_log(f"Exported ABBA JSON: {out_path}")
        except Exception as exc:
            self._append_log(f"Export failed: {exc}")


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LangSlice")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
