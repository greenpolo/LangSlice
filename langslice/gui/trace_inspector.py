"""Structured trace viewer widgets for the GUI."""

from __future__ import annotations

import importlib
import json
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
QResizeEvent = _qtgui.QResizeEvent

QDialog = _qtwidgets.QDialog
QDialogButtonBox = _qtwidgets.QDialogButtonBox
QFrame = _qtwidgets.QFrame
QHBoxLayout = _qtwidgets.QHBoxLayout
QLabel = _qtwidgets.QLabel
QPushButton = _qtwidgets.QPushButton
QScrollArea = _qtwidgets.QScrollArea
QVBoxLayout = _qtwidgets.QVBoxLayout
QWidget = _qtwidgets.QWidget


def _event_colors(event_type: str, role: str) -> tuple[str, str, str]:
    # Returns (Border Color, Background Color, Accent Color)
    if event_type == "tool_call":
        return ("rgba(59, 130, 246, 0.4)", "rgba(59, 130, 246, 0.05)", "#3b82f6")  # Blue
    if event_type == "tool_result":
        return ("rgba(34, 197, 94, 0.4)", "rgba(34, 197, 94, 0.05)", "#22c55e")  # Green
    if role == "model":
        return ("rgba(99, 102, 241, 0.4)", "rgba(99, 102, 241, 0.05)", "#6366f1")  # Indigo
    if event_type == "error":
        return ("rgba(239, 68, 68, 0.4)", "rgba(239, 68, 68, 0.05)", "#ef4444")  # Red

    # Default (Runtime / System)
    return ("rgba(255, 255, 255, 0.15)", "rgba(255, 255, 255, 0.03)", "#888888")  # Dark Gray


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


class _TraceImageLabel(_TraceThumbnail):
    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(180)
        self.setStyleSheet(
            "border: 1px solid rgba(255,255,255,0.12);"
            "background: rgba(255,255,255,0.04); border-radius: 8px;"
        )
        self._update_scaled_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap.isNull() or self.width() <= 8:
            self.clear()
            return
        target_width = max(64, self.width() - 12)
        target_height = max(160, min(420, int(round(target_width * 0.75))))
        scaled = self._pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


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
        self.resize(1100, 820)

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
        event_type = str(event.get("event_type", ""))
        role = str(event.get("role", ""))
        border_color, bg_color, accent_color = _event_colors(event_type, role)

        self.setStyleSheet(
            "QFrame {"
            f"border: 1px solid {border_color}; border-left: 4px solid {accent_color};"
            f"border-radius: 8px; background: {bg_color}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel(str(event.get("title", "")))
        title.setStyleSheet("font-size: 13px; font-weight: 600; color: #ffffff;")
        title.setWordWrap(True)
        layout.addWidget(title)

        meta = QLabel(self._build_meta_line(event))
        meta.setStyleSheet("font-size: 11px; color: #888888;")
        meta.setWordWrap(True)
        layout.addWidget(meta)

        summary = event.get("summary")
        if isinstance(summary, str) and summary:
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            summary_label.setStyleSheet("font-size: 12px; color: #ededed;")
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
                "font-family: Consolas, 'Courier New', monospace; font-size: 11px; color: #888888;"
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
            label.setStyleSheet("font-size: 11px; font-weight: 600; color: #bbbbbb;")
            layout.addWidget(label)

        body = QLabel(str(part.get("text", "")))
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if part.get("monospace"):
            body.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace; font-size: 11px; color: #ffffff; background: rgba(0,0,0,0.3); padding: 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,10);"
            )
        else:
            body.setStyleSheet("font-size: 12px; color: #ededed;")

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
        label.setStyleSheet("font-size: 11px; font-weight: 600; color: #bbbbbb;")
        layout.addWidget(label)

        pixmap = self._pixmap_from_part(part)
        if pixmap is not None and not pixmap.isNull():
            image_label = _TraceImageLabel(pixmap)
            image_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            image_label.clicked.connect(
                lambda: ImagePreviewDialog(
                    title=label_text,
                    pixmap=pixmap,
                    path=cast(str | None, part.get("path")),
                    parent=self,
                ).exec()
            )
            layout.addWidget(image_label)
        else:
            image_label = QLabel("Image unavailable")
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setMinimumHeight(120)
            image_label.setStyleSheet(
                "border: 1px solid rgba(255,255,255,0.12); color: #bbbbbb;"
                "background: rgba(255,255,255,0.04); border-radius: 8px;"
            )
            layout.addWidget(image_label)

        info = QLabel(self._image_info(part))
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setStyleSheet("font-size: 11px; color: #9ca3af;")
        layout.addWidget(info)
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
        self._events: list[dict[str, object]] = []

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
        self._events = []
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._placeholder.show()

    def append_event(self, event: dict[str, object]) -> None:
        self._events.append(dict(event))
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

    def has_events(self) -> bool:
        return bool(self._events)

    def export_events(self, json_path: str) -> int:
        export_path = os.path.abspath(json_path)
        export_dir = os.path.dirname(export_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(export_path))[0] or "agent_trace"
        assets_dir = os.path.join(export_dir or os.getcwd(), f"{base_name}_assets")
        os.makedirs(assets_dir, exist_ok=True)

        exported_events: list[dict[str, object]] = []
        image_count = 0
        for event_index, event in enumerate(self._events, start=1):
            exported_event = dict(event)
            exported_parts: list[dict[str, object]] = []
            for part_index, part in enumerate(cast(list[object], event.get("parts", [])), start=1):
                if not isinstance(part, dict):
                    continue
                exported_part = dict(part)
                if part.get("kind") == "image":
                    image_count += 1
                    mime_type = str(part.get("mime_type", "image/jpeg"))
                    ext = ".png" if mime_type == "image/png" else ".jpg"
                    label = (
                        str(part.get("label", f"image_{image_count}")).strip()
                        or f"image_{image_count}"
                    )
                    safe_label = "".join(
                        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label
                    )
                    image_name = f"event_{event_index:03d}_part_{part_index:02d}_{safe_label}{ext}"
                    image_path = os.path.join(assets_dir, image_name)
                    data = part.get("data")
                    if isinstance(data, (bytes, bytearray)):
                        with open(image_path, "wb") as fh:
                            fh.write(bytes(data))
                    elif isinstance(part.get("path"), str) and os.path.exists(
                        str(part.get("path"))
                    ):
                        with (
                            open(str(part.get("path")), "rb") as src,
                            open(image_path, "wb") as dst,
                        ):
                            dst.write(src.read())
                    exported_part["saved_path"] = os.path.relpath(
                        image_path, os.path.dirname(export_path) or os.getcwd()
                    )
                    exported_part.pop("data", None)
                exported_parts.append(exported_part)
            exported_event["parts"] = exported_parts
            exported_events.append(exported_event)

        with open(export_path, "w", encoding="utf-8") as fh:
            json.dump(exported_events, fh, indent=2)
        return len(exported_events)
