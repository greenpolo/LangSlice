from __future__ import annotations

import importlib
from typing import Optional

from PIL import Image
QtCore = importlib.import_module("PySide6.QtCore")
QtGui = importlib.import_module("PySide6.QtGui")
QtWidgets = importlib.import_module("PySide6.QtWidgets")

QObject = QtCore.QObject
QPoint = QtCore.QPoint
Qt = QtCore.Qt
QThread = QtCore.QThread
QTimer = QtCore.QTimer
Signal = QtCore.Signal
Slot = QtCore.Slot

QImage = QtGui.QImage
QPixmap = QtGui.QPixmap
QResizeEvent = QtGui.QResizeEvent

QFrame = QtWidgets.QFrame
QLabel = QtWidgets.QLabel
QProgressBar = QtWidgets.QProgressBar
QStackedLayout = QtWidgets.QStackedLayout
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

from langslice.atlas.core import get_reference_slice, load_atlas
from langslice.gui.theme import ACCENT, BG_PANEL_SOLID, BORDER_SUBTLE, FONT_MONO, TEXT_PRIMARY, TEXT_SECONDARY


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode == "RGB":
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
    elif img.mode == "RGBA":
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, 4 * img.width, QImage.Format.Format_RGBA8888)
    elif img.mode == "L":
        data = img.tobytes("raw", "L")
        qimg = QImage(data, img.width, img.height, img.width, QImage.Format.Format_Grayscale8)
    else:
        img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class AtlasLoaderWorker(QObject):
    slice_ready = Signal(QPixmap)
    error = Signal(str)
    finished = Signal()

    def __init__(self, atlas_name: str, position_mm: float) -> None:
        super().__init__()
        self._atlas_name = atlas_name
        self._position_mm = position_mm

    @Slot()
    def load(self) -> None:
        try:
            atlas = load_atlas(self._atlas_name)
            image = get_reference_slice(atlas, self._position_mm)
            self.slice_ready.emit(pil_to_qpixmap(image))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class AtlasViewer(QFrame):
    _STATE_PLACEHOLDER = 0
    _STATE_LOADING = 1
    _STATE_ERROR = 2
    _STATE_IMAGE = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("glassPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._atlas_name: Optional[str] = None
        self._position_mm: Optional[float] = None
        self._generation = 0
        self._queued_generation = 0
        self._active_generation = 0
        self._source_pixmap: Optional[QPixmap] = None

        self._active_thread: Optional[QThread] = None
        self._active_worker: Optional[AtlasLoaderWorker] = None

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._start_loading_request)

        self._stack_host = QWidget(self)
        self._stack = QStackedLayout(self._stack_host)
        self._stack.setContentsMargins(10, 10, 10, 10)

        self._placeholder_widget = self._build_placeholder_widget()
        self._loading_widget = self._build_loading_widget()
        self._error_widget = self._build_error_widget()
        self._image_widget = self._build_image_widget()

        self._stack.addWidget(self._placeholder_widget)
        self._stack.addWidget(self._loading_widget)
        self._stack.addWidget(self._error_widget)
        self._stack.addWidget(self._image_widget)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._stack_host)

        self._ap_badge = QLabel(self)
        self._ap_badge.setObjectName("atlasApBadge")
        self._ap_badge.setText("Pos: 0.00 mm")
        self._ap_badge.setVisible(False)
        self._ap_badge.setStyleSheet(
            f"""
            QLabel#atlasApBadge {{
                background-color: rgba(0, 0, 0, 204);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 51);
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 11px;
                font-family: {FONT_MONO};
            }}
            """
        )

        self.setStyleSheet(
            f"""
            AtlasViewer {{
                background-color: {BG_PANEL_SOLID};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 8px;
            }}
            QLabel#atlasIcon {{
                color: {ACCENT};
                font-size: 48px;
            }}
            QLabel#atlasHeading {{
                color: {TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
            }}
            QLabel#atlasSubtext {{
                color: {TEXT_SECONDARY};
                font-size: 12px;
            }}
            QLabel#atlasPlaceholderText {{
                color: rgba(99, 102, 241, 204);
                font-size: 12px;
            }}
            QLabel#atlasErrorText {{
                color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
            """
        )

        self._show_placeholder_state()

    def set_position(self, position_mm: float) -> None:
        self._position_mm = float(position_mm)
        self._queue_reload()

    def set_atlas(self, atlas_name: str) -> None:
        cleaned = atlas_name.strip()
        self._atlas_name = cleaned if cleaned else None
        self._queue_reload()

    def clear(self) -> None:
        self._atlas_name = None
        self._position_mm = None
        self._source_pixmap = None
        self._generation += 1
        self._queued_generation = self._generation
        self._debounce_timer.stop()
        self._cancel_active_request()
        self._ap_badge.setVisible(False)
        self._image_label.clear()
        self._show_placeholder_state()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()
        self._position_ap_badge()

    def _queue_reload(self) -> None:
        self._generation += 1
        self._queued_generation = self._generation

        if self._position_mm is None or self._atlas_name is None:
            self._debounce_timer.stop()
            self._source_pixmap = None
            self._image_label.clear()
            self._ap_badge.setVisible(False)
            self._show_placeholder_state()
            return

        self._set_ap_badge_text(self._position_mm)
        self._ap_badge.setVisible(True)
        self._position_ap_badge()
        self._debounce_timer.start()

    @Slot()
    def _start_loading_request(self) -> None:
        if self._position_mm is None or self._atlas_name is None:
            self._show_placeholder_state()
            return

        request_generation = self._queued_generation
        atlas_name = self._atlas_name
        position_mm = self._position_mm

        self._active_generation = request_generation
        self._cancel_active_request()
        self._show_loading_state()

        thread = QThread(self)
        worker = AtlasLoaderWorker(atlas_name, position_mm)
        worker.moveToThread(thread)

        thread.started.connect(worker.load)
        worker.slice_ready.connect(lambda pixmap, g=request_generation: self._on_slice_ready(g, pixmap))
        worker.error.connect(lambda msg, g=request_generation: self._on_error(g, msg))
        worker.finished.connect(lambda g=request_generation: self._on_worker_finished(g))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    def _cancel_active_request(self) -> None:
        if self._active_thread is not None:
            if self._active_thread.isRunning():
                self._active_thread.requestInterruption()
                self._active_thread.quit()
            self._active_thread = None
            self._active_worker = None

    def _on_slice_ready(self, generation: int, pixmap: QPixmap) -> None:
        if generation != self._generation:
            return
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()
        self._stack.setCurrentIndex(self._STATE_IMAGE)

    def _on_error(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._error_message_label.setText(message or "Failed to load atlas slice")
        self._stack.setCurrentIndex(self._STATE_ERROR)

    def _on_worker_finished(self, generation: int) -> None:
        if generation == self._active_generation:
            self._active_thread = None
            self._active_worker = None

    def _show_placeholder_state(self) -> None:
        self._stack.setCurrentIndex(self._STATE_PLACEHOLDER)

    def _show_loading_state(self) -> None:
        self._stack.setCurrentIndex(self._STATE_LOADING)

    def _set_ap_badge_text(self, position_mm: float) -> None:
        self._ap_badge.setText(f"Pos: {position_mm:.2f} mm")
        self._ap_badge.adjustSize()

    def _position_ap_badge(self) -> None:
        if not self._ap_badge.isVisible():
            return
        x = self.width() - self._ap_badge.width() - 14
        y = self.height() - self._ap_badge.height() - 14
        self._ap_badge.move(QPoint(max(x, 0), max(y, 0)))
        self._ap_badge.raise_()

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return

        target_size = self._image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def _build_placeholder_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

        icon = QLabel("🧠", widget)
        icon.setObjectName("atlasIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        heading = QLabel("Atlas Reference", widget)
        heading.setObjectName("atlasHeading")
        heading.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        text = QLabel("Awaiting Position Estimate...", widget)
        text.setObjectName("atlasPlaceholderText")
        text.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(icon)
        layout.addWidget(heading)
        layout.addWidget(text)
        layout.addStretch(1)
        return widget

    def _build_loading_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addStretch(1)

        spinner = QProgressBar(widget)
        spinner.setRange(0, 0)
        spinner.setTextVisible(False)
        spinner.setFixedWidth(120)

        spinner_wrap = QWidget(widget)
        spinner_layout = QVBoxLayout(spinner_wrap)
        spinner_layout.setContentsMargins(0, 0, 0, 0)
        spinner_layout.addWidget(spinner, 0, Qt.AlignmentFlag.AlignHCenter)

        text = QLabel("Loading atlas slice...", widget)
        text.setObjectName("atlasSubtext")
        text.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(spinner_wrap)
        layout.addWidget(text)
        layout.addStretch(1)
        return widget

    def _build_error_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addStretch(1)

        icon = QLabel("!", widget)
        icon.setObjectName("atlasIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        title = QLabel("Unable to load atlas slice", widget)
        title.setObjectName("atlasHeading")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._error_message_label = QLabel("", widget)
        self._error_message_label.setObjectName("atlasErrorText")
        self._error_message_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._error_message_label.setWordWrap(True)

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(self._error_message_label)
        layout.addStretch(1)
        return widget

    def _build_image_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel(widget)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setScaledContents(False)

        layout.addWidget(self._image_label, 1)
        return widget
