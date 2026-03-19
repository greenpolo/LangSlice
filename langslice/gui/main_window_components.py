"""Reusable helpers and widgets for the main GUI window."""

from __future__ import annotations

import importlib
from typing import Any

from PIL import Image

_qtcore = importlib.import_module("PySide6.QtCore")
_qtgui = importlib.import_module("PySide6.QtGui")
_qtwidgets = importlib.import_module("PySide6.QtWidgets")

Qt = _qtcore.Qt
Signal = _qtcore.Signal
QTimer = _qtcore.QTimer

QDragEnterEvent = _qtgui.QDragEnterEvent
QDropEvent = _qtgui.QDropEvent
QImage = _qtgui.QImage
QPainter = _qtgui.QPainter
QPen = _qtgui.QPen
QPixmap = _qtgui.QPixmap
QTransform = _qtgui.QTransform
QColor = _qtgui.QColor

QFrame = _qtwidgets.QFrame
QHBoxLayout = _qtwidgets.QHBoxLayout
QLabel = _qtwidgets.QLabel
QSizePolicy = _qtwidgets.QSizePolicy
QVBoxLayout = _qtwidgets.QVBoxLayout
QWidget = _qtwidgets.QWidget

from langslice.registration import (
    AffineResult,
    RegistrationCorrespondence,
    RegistrationResult,
    build_annotation_session_from_correspondences,
    get_session_marker_points,
)
from langslice.gui.theme import ACCENT, ERROR, SUCCESS, TEXT_SECONDARY
from langslice.vlm import APResult


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert PIL Image to QPixmap."""
    rgba_image = image.convert("RGBA")
    width, height = rgba_image.size
    data = rgba_image.tobytes("raw", "RGBA")
    qt_image = QImage(data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qt_image.copy())


def affine_matrix_to_qtransform(matrix: object) -> QTransform:
    """Convert a row-major 3x3 affine matrix into a QTransform."""
    arr = getattr(matrix, "tolist", None)
    values = arr() if callable(arr) else matrix
    m: Any = values
    return QTransform(
        float(m[0][0]),
        float(m[1][0]),
        0.0,
        float(m[0][1]),
        float(m[1][1]),
        0.0,
        float(m[0][2]),
        float(m[1][2]),
        1.0,
    )


def build_split_view_correspondence_points(
    registration_result: RegistrationResult | None,
    affine_result: AffineResult | None,
    preview_correspondences: list[RegistrationCorrespondence] | None = None,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """Return paired marker points for split-view slice and atlas panes."""
    _ = affine_result
    session = None
    if registration_result is not None:
        session = registration_result.annotation_session
        if session is None:
            session = build_annotation_session_from_correspondences(
                registration_result.accepted_correspondences
            )
    elif preview_correspondences is not None:
        session = build_annotation_session_from_correspondences(preview_correspondences)
    if session is None:
        return [], []

    slice_points = get_session_marker_points(session, image_role="slice")
    atlas_points = get_session_marker_points(session, image_role="atlas")
    return slice_points, atlas_points


def should_apply_affine_to_slice(
    affine_result: AffineResult | None,
    slice_size: tuple[int, int] | None,
) -> bool:
    """Return True when an affine result is defined in the slice-image source frame."""
    if affine_result is None or slice_size is None:
        return False
    direction = str(affine_result.provenance.get("transform_direction", "")).strip().lower()
    if direction == "atlas_to_slice":
        return False
    return affine_result.source_size == (int(slice_size[0]), int(slice_size[1]))


def draw_correspondence_markers(
    pixmap: QPixmap,
    points: list[tuple[float, float, str]],
    *,
    color: QColor,
) -> QPixmap:
    """Draw labeled correspondence markers onto a pixmap copy."""
    if pixmap.isNull() or not points:
        return pixmap

    annotated = QPixmap(pixmap)
    painter = QPainter(annotated)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    text_color = QColor(240, 240, 240)

    width = annotated.width()
    height = annotated.height()
    radius = max(5, int(round(max(width, height) / 125.0)))
    pen = QPen(color)
    pen.setWidth(max(2, radius // 4))
    painter.setPen(pen)
    font = painter.font()
    font.setPixelSize(max(10, int(round(radius * 1.4))))
    painter.setFont(font)
    for x, y, label in points:
        if x < 0.0 or y < 0.0 or x >= width or y >= height:
            continue
        painter.drawEllipse(int(round(x - radius)), int(round(y - radius)), radius * 2, radius * 2)
        painter.setPen(text_color)
        painter.drawText(int(round(x + radius + 4)), int(round(y - radius)), label)
        painter.setPen(pen)

    painter.end()
    return annotated


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

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def show_ap_result(
        self,
        result: APResult,
        *,
        position_label: str = "Estimated Position",
    ) -> None:
        self.result_top.setText(
            f"{position_label}: <span style='color:{SUCCESS}'>{result.position_mm:.2f} mm</span>"
        )
        self.result_reasoning.setText(result.reasoning)
        self.error_label.clear()
        self.result_panel.show()

    def show_affine_result(self, result: AffineResult) -> None:
        tx_px, ty_px = result.translation_px
        scale_x, scale_y = result.scale
        self.result_top.setText(
            " | ".join(
                [
                    f"Backend: {result.backend}",
                    f"Rotation: {result.rotation_deg:.2f} deg",
                    f"Translate: ({tx_px:.1f}, {ty_px:.1f}) px",
                    f"Scale: ({scale_x:.3f}, {scale_y:.3f})",
                    f"Shear: {result.shear:.3f}",
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
