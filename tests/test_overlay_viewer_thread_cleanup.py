"""Checks overlay-viewer loader thread cleanup safety."""

from __future__ import annotations

from PIL import Image

from langslice.gui import overlay_viewer


class _DeletedThreadStub(overlay_viewer.QThread):
    def __init__(self, parent: overlay_viewer.QWidget | None = None) -> None:
        super().__init__(parent)

    def isRunning(self) -> bool:  # noqa: N802
        raise RuntimeError("Internal C++ object (PySide6.QtCore.QThread) already deleted")


class _RunningThreadStub(overlay_viewer.QThread):
    def __init__(self, parent: overlay_viewer.QWidget | None = None) -> None:
        super().__init__(parent)
        self.interrupted = False
        self.quit_called = False

    def isRunning(self) -> bool:  # noqa: N802
        return True

    def requestInterruption(self) -> None:  # noqa: N802
        self.interrupted = True

    def quit(self) -> None:
        self.quit_called = True


def test_cancel_active_tolerates_deleted_thread(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        overlay_viewer._qtwidgets.QApplication.instance()
        or overlay_viewer._qtwidgets.QApplication([])
    )
    viewer = overlay_viewer.OverlayGraphicsView()

    viewer._active_thread = _DeletedThreadStub(viewer)
    viewer._active_worker = overlay_viewer._AtlasLoaderWorker("allen_mouse_25um", 0.0)

    viewer._cancel_active()

    assert viewer._active_thread is None
    assert viewer._active_worker is None
    viewer.close()
    app.processEvents()


def test_cancel_active_interrupts_running_thread(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        overlay_viewer._qtwidgets.QApplication.instance()
        or overlay_viewer._qtwidgets.QApplication([])
    )
    viewer = overlay_viewer.OverlayGraphicsView()

    thread = _RunningThreadStub(viewer)
    viewer._active_thread = thread
    viewer._active_worker = overlay_viewer._AtlasLoaderWorker("allen_mouse_25um", 0.0)

    viewer._cancel_active()

    assert thread.interrupted
    assert thread.quit_called
    assert viewer._active_thread is None
    assert viewer._active_worker is None
    viewer.close()
    app.processEvents()


def test_queue_reload_invalid_state_cancels_active_thread(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        overlay_viewer._qtwidgets.QApplication.instance()
        or overlay_viewer._qtwidgets.QApplication([])
    )
    viewer = overlay_viewer.OverlayGraphicsView()

    thread = _RunningThreadStub(viewer)
    viewer._active_thread = thread
    viewer._active_worker = overlay_viewer._AtlasLoaderWorker("allen_mouse_25um", 0.0)
    viewer._atlas_name = None
    viewer._position_mm = 0.0

    viewer._queue_atlas_reload()

    assert thread.interrupted
    assert thread.quit_called
    assert viewer._active_thread is None
    assert viewer._active_worker is None
    viewer.close()
    app.processEvents()


def test_loader_done_clears_matching_thread_even_with_generation_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        overlay_viewer._qtwidgets.QApplication.instance()
        or overlay_viewer._qtwidgets.QApplication([])
    )
    viewer = overlay_viewer.OverlayGraphicsView()

    thread = _RunningThreadStub(viewer)
    viewer._active_thread = thread
    viewer._active_worker = overlay_viewer._AtlasLoaderWorker("allen_mouse_25um", 0.0)
    viewer._generation = 10

    viewer._on_loader_done(9, thread)

    assert viewer._active_thread is None
    assert viewer._active_worker is None
    viewer.close()
    app.processEvents()


def test_overlay_viewer_persistent_marker_layer(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        overlay_viewer._qtwidgets.QApplication.instance()
        or overlay_viewer._qtwidgets.QApplication([])
    )
    viewer = overlay_viewer.OverlayGraphicsView()

    slice_pixmap = overlay_viewer._pil_to_qpixmap(Image.new("RGB", (120, 80), (255, 255, 255)))
    atlas_pixmap = overlay_viewer._pil_to_qpixmap(Image.new("RGB", (120, 80), (0, 0, 0)))
    viewer.set_slice_pixmap(slice_pixmap)
    viewer._on_atlas_ready(0, atlas_pixmap, (528, 320, 456))
    viewer.set_correspondence_markers([(10.0, 12.0, "1")], [(20.0, 24.0, "1")])

    assert len(viewer._marker_items) == 4

    viewer.set_correspondence_markers([], [])

    assert viewer._marker_items == []
    viewer.close()
    app.processEvents()
