"""Structured trace viewer widgets for the GUI."""

from __future__ import annotations

import importlib
import os
from typing import Any, cast

_qtcore = importlib.import_module("PySide6.QtCore")
_qtgui = importlib.import_module("PySide6.QtGui")
_qtwidgets = importlib.import_module("PySide6.QtWidgets")

Qt = _qtcore.Qt
QBuffer = _qtcore.QBuffer
QByteArray = _qtcore.QByteArray
QIODevice = _qtcore.QIODevice
Signal = _qtcore.Signal

QCursor = _qtgui.QCursor
QPixmap = _qtgui.QPixmap

QDialog = _qtwidgets.QDialog
QDialogButtonBox = _qtwidgets.QDialogButtonBox
QFrame = _qtwidgets.QFrame
QHBoxLayout = _qtwidgets.QHBoxLayout
QLabel = _qtwidgets.QLabel
QPushButton = _qtwidgets.QPushButton
QScrollArea = _qtwidgets.QScrollArea
QVBoxLayout = _qtwidgets.QVBoxLayout
QWidget = _qtwidgets.QWidget


def _stage_accent(stage: str) -> str:
    if stage == "ap":
        return "#dbeafe"
    if stage in {"affine", "registration"}:
        return "#dcfce7"
    return "#f3f4f6"


def _role_label(event_type: str, role: str) -> str:
    if event_type == "tool_call":
        return "Tool"
    if event_type == "tool_result":
        return "Tool Result"
    if role == "model":
        return "Model"
    if role == "system":
        return "Stage"
    return "Runtime"


class _TraceThumbnail(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event: Any) -> None:
        if getattr(event, "button", lambda: None)() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImagePreviewDialog(QDialog):
    def __init__(
        self,
        *,
        title: str,
        pixmap: QPixmap,
        path: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(920, 720)

        layout = QVBoxLayout(self)
        info = QLabel(path or "In-memory trace image")
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        image_wrap = QWidget()
        image_layout = QVBoxLayout(image_wrap)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(pixmap)
        image_layout.addWidget(image_label)
        scroll.setWidget(image_wrap)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class TraceEntryWidget(QFrame):
    def __init__(self, event: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._event = event
        self.setFrameShape(QFrame.Shape.StyledPanel)
        stage = str(event.get("stage", ""))
        self.setStyleSheet(
            "QFrame {"
            f"border: 1px solid #d1d5db; border-left: 4px solid {_stage_accent(stage)};"
            "border-radius: 8px; background: white; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel(str(event.get("title", "")))
        title.setStyleSheet("font-size: 13px; font-weight: 600; color: #111827;")
        title.setWordWrap(True)
        layout.addWidget(title)

        meta = QLabel(self._build_meta_line(event))
        meta.setStyleSheet("font-size: 11px; color: #6b7280;")
        meta.setWordWrap(True)
        layout.addWidget(meta)

        summary = event.get("summary")
        if isinstance(summary, str) and summary:
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            summary_label.setStyleSheet("font-size: 12px; color: #1f2937;")
            layout.addWidget(summary_label)

        parts = event.get("parts", [])
        for part in cast(list[object], parts if isinstance(parts, list) else []):
            if not isinstance(part, dict):
                continue
            widget = self._build_part_widget(part)
            if widget is not None:
                layout.addWidget(widget)

        metadata = event.get("metadata")
        if isinstance(metadata, dict) and metadata:
            metadata_label = QLabel(self._format_metadata(metadata))
            metadata_label.setWordWrap(True)
            metadata_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            metadata_label.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace; font-size: 11px; color: #4b5563;"
            )
            layout.addWidget(metadata_label)

    def _build_meta_line(self, event: dict[str, object]) -> str:
        parts = []
        created_at = event.get("created_at")
        if isinstance(created_at, str) and created_at:
            parts.append(created_at.replace("T", " "))
        parts.append(_role_label(str(event.get("event_type", "")), str(event.get("role", ""))))
        stage = str(event.get("stage", "")).strip()
        if stage:
            parts.append(stage.upper())
        return "  •  ".join(parts)

    def _format_metadata(self, metadata: dict[str, object]) -> str:
        entries: list[str] = []
        for key, value in metadata.items():
            entries.append(f"{key}: {value}")
        return "\n".join(entries)

    def _build_part_widget(self, part: dict[str, object]) -> QWidget | None:
        kind = part.get("kind")
        if kind == "text":
            return self._build_text_part(part)
        if kind == "image":
            return self._build_image_part(part)
        return None

    def _build_text_part(self, part: dict[str, object]) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label_text = part.get("label")
        if isinstance(label_text, str) and label_text:
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 11px; font-weight: 600; color: #374151;")
            layout.addWidget(label)

        body = QLabel(str(part.get("text", "")))
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if part.get("monospace"):
            body.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace; font-size: 11px; color: #111827; background: #f9fafb; padding: 6px; border-radius: 6px;"
            )
        else:
            body.setStyleSheet("font-size: 12px; color: #111827;")

        if part.get("collapsible"):
            body.setVisible(False)
            toggle = QPushButton(f"Show {label_text or 'details'}")
            toggle.setObjectName("secondary")

            def _toggle() -> None:
                visible = not body.isVisible()
                body.setVisible(visible)
                toggle.setText(
                    f"Hide {label_text or 'details'}"
                    if visible
                    else f"Show {label_text or 'details'}"
                )

            toggle.clicked.connect(_toggle)
            layout.addWidget(toggle)
        layout.addWidget(body)
        return wrap

    def _build_image_part(self, part: dict[str, object]) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label_text = str(part.get("label", "Image"))
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 11px; font-weight: 600; color: #374151;")
        layout.addWidget(label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        thumbnail = _TraceThumbnail()
        thumbnail.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        thumbnail.setFixedSize(160, 120)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail.setStyleSheet(
            "border: 1px solid #d1d5db; background: #f9fafb; border-radius: 6px;"
        )

        pixmap = self._pixmap_from_part(part)
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                thumbnail.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumbnail.setPixmap(scaled)
            thumbnail.clicked.connect(
                lambda: ImagePreviewDialog(
                    title=label_text,
                    pixmap=pixmap,
                    path=cast(str | None, part.get("path")),
                    parent=self,
                ).exec()
            )
        else:
            thumbnail.setText("Image unavailable")
        row.addWidget(thumbnail)

        info = QLabel(self._image_info(part))
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setStyleSheet("font-size: 11px; color: #4b5563;")
        row.addWidget(info, stretch=1)

        layout.addLayout(row)
        return wrap

    def _pixmap_from_part(self, part: dict[str, object]) -> QPixmap | None:
        data = part.get("data")
        if isinstance(data, (bytes, bytearray)):
            pixmap = QPixmap()
            pixmap.loadFromData(bytes(data))
            return pixmap
        path = part.get("path")
        if isinstance(path, str) and path and os.path.exists(path):
            return QPixmap(path)
        return None

    def _image_info(self, part: dict[str, object]) -> str:
        width = part.get("width")
        height = part.get("height")
        mime_type = part.get("mime_type")
        entries = [f"{width}x{height}px" if width and height else None, mime_type]
        path = part.get("path")
        if isinstance(path, str) and path:
            entries.append(path)
        metadata = part.get("metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                entries.append(f"{key}: {value}")
        return "\n".join(str(item) for item in entries if item)


class TraceStageWidget(QFrame):
    def __init__(self, event: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 4)
        layout.setSpacing(8)

        line_left = QFrame()
        line_left.setFrameShape(QFrame.Shape.HLine)
        line_left.setStyleSheet("color: #d1d5db;")
        line_right = QFrame()
        line_right.setFrameShape(QFrame.Shape.HLine)
        line_right.setStyleSheet("color: #d1d5db;")
        title = QLabel(str(event.get("title", "")))
        title.setStyleSheet("font-size: 12px; font-weight: 700; color: #374151;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(line_left, stretch=1)
        layout.addWidget(title)
        layout.addWidget(line_right, stretch=1)


class TraceInspector(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)

        self._placeholder = QLabel("Waiting for input...")
        self._placeholder.setStyleSheet("color: #6b7280; font-size: 12px;")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._layout.insertWidget(0, self._placeholder)

        self.setWidget(self._content)

    def clear_events(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._placeholder.show()

    def append_event(self, event: dict[str, object]) -> None:
        scrollbar = self.verticalScrollBar()
        was_near_bottom = scrollbar.value() >= max(0, scrollbar.maximum() - 24)
        self._placeholder.hide()
        widget: QWidget
        if str(event.get("event_type", "")) == "stage":
            widget = TraceStageWidget(event)
        else:
            widget = TraceEntryWidget(event)
        self._layout.insertWidget(self._layout.count() - 1, widget)
        if was_near_bottom:
            scrollbar.setValue(scrollbar.maximum())
