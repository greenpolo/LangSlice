"""Physical-scale overlay viewer for brain slice + atlas registration.

Renders both the histology slice and atlas reference in a shared coordinate
space (slice-pixel units), scaling the atlas by the ratio of physical pixel
sizes so that structures match in real-world dimensions — analogous to how
ABBA uses BigDataViewer's physical coordinate system.
"""

from __future__ import annotations

import importlib
from typing import Optional

from PIL import Image

_qtcore = importlib.import_module("PySide6.QtCore")
_qtgui = importlib.import_module("PySide6.QtGui")
_qtwidgets = importlib.import_module("PySide6.QtWidgets")

Qt = _qtcore.Qt
QObject = _qtcore.QObject
QPointF = _qtcore.QPointF
QRectF = _qtcore.QRectF
QThread = _qtcore.QThread
QTimer = _qtcore.QTimer
Signal = _qtcore.Signal
Slot = _qtcore.Slot

QImage = _qtgui.QImage
QPixmap = _qtgui.QPixmap

QFrame = _qtwidgets.QFrame
QGraphicsPixmapItem = _qtwidgets.QGraphicsPixmapItem
QGraphicsScene = _qtwidgets.QGraphicsScene
QGraphicsView = _qtwidgets.QGraphicsView
QLabel = _qtwidgets.QLabel
QSizePolicy = _qtwidgets.QSizePolicy
QVBoxLayout = _qtwidgets.QVBoxLayout
QWidget = _qtwidgets.QWidget

from langslice.atlas.core import get_composite_slice, load_atlas
from langslice.gui.theme import (
    ACCENT,
    BG_PANEL_SOLID,
    BORDER_SUBTLE,
    FONT_MONO,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL Image to QPixmap."""
    if img.mode == "RGBA":
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, 4 * img.width, QImage.Format.Format_RGBA8888)
    elif img.mode == "RGB":
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format.Format_RGB888)
    elif img.mode == "L":
        data = img.tobytes("raw", "L")
        qimg = QImage(data, img.width, img.height, img.width, QImage.Format.Format_Grayscale8)
    else:
        img = img.convert("RGBA")
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, 4 * img.width, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Background atlas loader (same pattern as atlas_viewer.py)
# ---------------------------------------------------------------------------

class _AtlasLoaderWorker(QObject):
    """Load an atlas composite slice in a background thread."""
    slice_ready = Signal(QPixmap, float)  # pixmap, atlas_resolution_um
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
            # Composite = reference + green boundary lines
            composite = get_composite_slice(atlas, self._position_mm, opacity=0.4)
            # Atlas pixel size along coronal axes (DV and ML are axes 1, 2)
            atlas_res_um = float(atlas.resolution[1])
            self.slice_ready.emit(_pil_to_qpixmap(composite), atlas_res_um)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class OverlayGraphicsView(QFrame):
    """QGraphicsView-based viewer that renders slice + atlas in shared
    physical coordinate space.

    Coordinate convention (all in *slice-pixel* units):
    - The slice pixmap is placed at the origin with scale 1.0
    - The atlas pixmap is scaled by ``atlas_resolution_um / pixel_size_um``
      so that one atlas pixel covers the correct number of slice pixels.
    - Both items are centered on the scene origin.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("glassPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # State
        self._pixel_size_um: float = 4.0
        self._atlas_resolution_um: float = 25.0
        self._atlas_name: Optional[str] = None
        self._position_mm: Optional[float] = None
        self._atlas_opacity: float = 0.5
        self._generation: int = 0

        # Async loader
        self._active_thread: Optional[QThread] = None
        self._active_worker: Optional[_AtlasLoaderWorker] = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._start_atlas_load)

        # Scene / view
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene, self)
        self._view.setRenderHints(
            _qtgui.QPainter.RenderHint.Antialiasing
            | _qtgui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setFrameShape(QFrame.Shape.NoFrame)
        self._view.setStyleSheet("background: transparent;")

        # Pixmap items
        self._slice_item: Optional[QGraphicsPixmapItem] = None
        self._atlas_item: Optional[QGraphicsPixmapItem] = None

        # Placeholder label (shown when nothing loaded)
        self._placeholder = QLabel("Load an image and run the agent\nto see the overlay", self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")

        # AP badge
        self._ap_badge = QLabel(self)
        self._ap_badge.setVisible(False)
        self._ap_badge.setStyleSheet(
            f"""
            QLabel {{
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

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._view)

        self._placeholder.raise_()
        self._view.setVisible(False)

    # --- Public API --------------------------------------------------------

    def set_slice_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        """Set the histology slice image."""
        if self._slice_item is not None:
            self._scene.removeItem(self._slice_item)
            self._slice_item = None

        if pixmap is None or pixmap.isNull():
            self._fit_scene()
            return

        self._slice_item = self._scene.addPixmap(pixmap)
        self._slice_item.setZValue(0)
        self._layout_items()

    def set_atlas(self, atlas_name: str) -> None:
        self._atlas_name = atlas_name.strip() or None
        self._queue_atlas_reload()

    def set_position(self, position_mm: float) -> None:
        self._position_mm = position_mm
        self._queue_atlas_reload()

    def set_pixel_size(self, um_per_px: float) -> None:
        self._pixel_size_um = max(0.01, um_per_px)
        self._layout_items()

    def set_atlas_opacity(self, opacity: float) -> None:
        self._atlas_opacity = max(0.0, min(1.0, opacity))
        if self._atlas_item is not None:
            self._atlas_item.setOpacity(self._atlas_opacity)

    def clear(self) -> None:
        self._atlas_name = None
        self._position_mm = None
        self._generation += 1
        self._debounce.stop()
        self._cancel_active()
        if self._atlas_item is not None:
            self._scene.removeItem(self._atlas_item)
            self._atlas_item = None
        self._ap_badge.setVisible(False)
        self._fit_scene()

    def clear_all(self) -> None:
        """Clear both slice and atlas."""
        self.clear()
        if self._slice_item is not None:
            self._scene.removeItem(self._slice_item)
            self._slice_item = None
        self._fit_scene()

    # --- Resize ------------------------------------------------------------

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._fit_scene()
        self._position_badges()
        self._position_placeholder()

    # --- Internal: layout + fit --------------------------------------------

    def _layout_items(self) -> None:
        """Position and scale both items centered on the scene origin."""
        if self._slice_item is not None:
            pm = self._slice_item.pixmap()
            self._slice_item.setPos(-pm.width() / 2.0, -pm.height() / 2.0)

        if self._atlas_item is not None:
            scale = self._atlas_resolution_um / self._pixel_size_um
            self._atlas_item.setScale(scale)
            pm = self._atlas_item.pixmap()
            # After scaling, the effective size is pm.size() * scale.
            # Center the scaled atlas on the origin.
            self._atlas_item.setPos(
                -(pm.width() * scale) / 2.0,
                -(pm.height() * scale) / 2.0,
            )
            self._atlas_item.setOpacity(self._atlas_opacity)

        self._fit_scene()

    def _fit_scene(self) -> None:
        """Fit the visible content into the view."""
        has_content = self._slice_item is not None or self._atlas_item is not None
        self._view.setVisible(has_content)
        self._placeholder.setVisible(not has_content)

        if not has_content:
            return

        rect = self._scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        # Add a small margin
        margin = max(rect.width(), rect.height()) * 0.04
        rect.adjust(-margin, -margin, margin, margin)
        self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _position_badges(self) -> None:
        if self._ap_badge.isVisible():
            x = self.width() - self._ap_badge.width() - 14
            y = self.height() - self._ap_badge.height() - 14
            self._ap_badge.move(max(x, 0), max(y, 0))
            self._ap_badge.raise_()

    def _position_placeholder(self) -> None:
        self._placeholder.setGeometry(0, 0, self.width(), self.height())

    # --- Internal: async atlas loading ------------------------------------

    def _queue_atlas_reload(self) -> None:
        self._generation += 1
        if self._atlas_name is None or self._position_mm is None:
            self._debounce.stop()
            if self._atlas_item is not None:
                self._scene.removeItem(self._atlas_item)
                self._atlas_item = None
            self._ap_badge.setVisible(False)
            self._fit_scene()
            return

        self._update_ap_badge(self._position_mm)
        self._ap_badge.setVisible(True)
        self._position_badges()
        self._debounce.start()

    def _start_atlas_load(self) -> None:
        if self._atlas_name is None or self._position_mm is None:
            return

        gen = self._generation
        self._cancel_active()

        thread = QThread(self)
        worker = _AtlasLoaderWorker(self._atlas_name, self._position_mm)
        worker.moveToThread(thread)

        thread.started.connect(worker.load)
        worker.slice_ready.connect(lambda pm, res, g=gen: self._on_atlas_ready(g, pm, res))
        worker.error.connect(lambda msg, g=gen: self._on_atlas_error(g, msg))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda g=gen: self._on_loader_done(g))

        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    def _cancel_active(self) -> None:
        if self._active_thread is not None:
            if self._active_thread.isRunning():
                self._active_thread.requestInterruption()
                self._active_thread.quit()
            self._active_thread = None
            self._active_worker = None

    def _on_atlas_ready(self, gen: int, pixmap: QPixmap, atlas_res_um: float) -> None:
        if gen != self._generation:
            return
        self._atlas_resolution_um = atlas_res_um

        if self._atlas_item is not None:
            self._scene.removeItem(self._atlas_item)
        self._atlas_item = self._scene.addPixmap(pixmap)
        self._atlas_item.setZValue(1)
        self._layout_items()

    def _on_atlas_error(self, gen: int, message: str) -> None:
        if gen != self._generation:
            return
        # Silently ignore atlas load errors in the overlay view;
        # the split-view AtlasViewer already shows errors to the user.

    def _on_loader_done(self, gen: int) -> None:
        if gen == self._generation:
            self._active_thread = None
            self._active_worker = None

    def _update_ap_badge(self, position_mm: float) -> None:
        self._ap_badge.setText(f"Pos: {position_mm:.2f} mm")
        self._ap_badge.adjustSize()
