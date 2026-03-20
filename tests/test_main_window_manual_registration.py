"""Targeted coverage for manual registration in the main window."""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

import langslice.atlas as atlas
from langslice.gui import main_window
from langslice.image_prep import LoadedImageState
from langslice.registration import (
    AffineResult,
    LandmarkAnnotation,
    NonlinearResult,
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    RegistrationResult,
)


def _build_registration_result(translate_x_px: float) -> RegistrationResult:
    corr = RegistrationCorrespondence(
        slice_xy=(15.0, 18.0),
        atlas_xy=(12.0, 16.0),
        label="A",
        confidence="high",
    )
    affine = AffineResult(
        matrix=np.array(
            [[1.0, 0.0, translate_x_px], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        source_size=(120, 80),
        output_size=(120, 80),
        backend="test_backend",
        reasoning="synthetic",
    )
    nonlinear = NonlinearResult(
        atlas_points=np.array([[12.0, 16.0]], dtype=np.float64),
        slice_points=np.array([[15.0, 18.0]], dtype=np.float64),
        smoothing=1.0,
        backend="tps",
        reasoning="synthetic",
        output_size=(120, 80),
    )
    return RegistrationResult(
        correspondences=[corr],
        accepted_correspondences=[corr],
        rejected_correspondences=[],
        affine_result=affine,
        nonlinear_result=nonlinear,
        qc_state="accepted",
    )


def test_load_image_initializes_manual_position_and_manual_ui(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("LANGSLICE_VLM_DEBUG_DIR", raising=False)
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.positions: list[float] = []

        def set_position(self, position_mm: float) -> None:
            self.positions.append(position_mm)

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            _ = markers

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.positions: list[float] = []

        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            self.positions.append(position_mm)

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            _ = slice_markers, atlas_markers

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    dummy_state = LoadedImageState(
        canonical_image=Image.new("RGB", (120, 80), (255, 255, 255)),
        vlm_image=Image.new("RGB", (120, 80), (255, 255, 255)),
        pixel_size_um=4.0,
        pixel_size_source="manual_default",
        metadata_pixel_size_um=None,
        original_size=(120, 80),
        vlm_size=(120, 80),
        vlm_scale_factor=1.0,
        vlm_effective_pixel_size_um=4.0,
        channel_labels=("Red", "Green", "Blue"),
    )
    monkeypatch.setattr(main_window, "load_image_state", lambda *_args, **_kwargs: dummy_state)
    monkeypatch.setattr(atlas, "load_atlas", lambda _atlas_name: object())
    monkeypatch.setattr(atlas, "get_position_range_mm", lambda _atlas_obj: (0.0, 8.0))

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()
    window._load_image("synthetic.png")
    app.processEvents()

    assert window.current_pos == 0.0
    assert window.ap_value_label.text() == "Manual Position: 0.00 mm"
    assert window.step_ap.title_label.text() == "Manual Position"
    assert window.run_button.isEnabled()
    assert window.run_registration_button.isEnabled()
    assert "manual position 0.00 mm" in window.run_registration_button.toolTip()
    assert "target=8 pairs" in window.run_registration_button.toolTip()
    assert "edge hint=5 pairs" in window.run_registration_button.toolTip()
    assert "without AP agent estimation" in window.manual_registration_status_label.text()
    assert not window.ap_adjust_wrap.isHidden()

    assert window.split_atlas.positions
    assert window.overlay_viewer.positions
    assert window.split_atlas.positions[-1] == 0.0
    assert window.overlay_viewer.positions[-1] == 0.0

    window.close()


def test_manual_registration_path_updates_state(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("LANGSLICE_VLM_DEBUG_DIR", raising=False)
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            _ = markers

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            _ = slice_markers, atlas_markers

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()

    stale_result = _build_registration_result(translate_x_px=4.0)
    fresh_result = _build_registration_result(translate_x_px=14.0)
    captured: dict[str, object] = {}

    def fake_estimate_registration_runtime(
        *,
        image,
        on_progress,
        on_trace,
        on_correspondences,
        on_annotation_session,
        atlas_name,
        position_mm,
        pixel_size_um,
        target_landmark_count,
        workflow,
        show_atlas_borders,
        debug_dir,
        enable_code_execution,
        tool_loop_max_steps,
    ):
        captured["image_size"] = image.size
        captured["atlas_name"] = atlas_name
        captured["position_mm"] = position_mm
        captured["pixel_size_um"] = pixel_size_um
        captured["target_landmark_count"] = target_landmark_count
        captured["workflow"] = workflow
        captured["show_atlas_borders"] = show_atlas_borders
        captured["debug_dir"] = debug_dir
        captured["enable_code_execution"] = enable_code_execution
        captured["tool_loop_max_steps"] = tool_loop_max_steps
        _ = on_trace, on_annotation_session
        if on_correspondences is not None:
            on_correspondences(fresh_result.accepted_correspondences)
        time.sleep(0.05)
        if on_progress is not None:
            on_progress("manual registration test runtime")
        return fresh_result

    monkeypatch.setattr(
        main_window, "estimate_registration_runtime", fake_estimate_registration_runtime
    )

    window.source_image = Image.new("RGB", (120, 80), (255, 255, 255))
    window.pil_image = window.source_image.copy()
    window.agent_vlm_image = window.pil_image.copy()
    window.current_pos = 1.23
    window.landmark_count_spin.setValue(18)
    window.tool_loop_steps_spin.setValue(9)
    window.code_execution_checkbox.setChecked(False)
    window.ap_result = main_window.APResult(position_mm=0.45, reasoning="stale", debug_dir="old")
    window.affine_result = stale_result.affine_result
    window.registration_result = stale_result
    window.feedback_wrap.show()
    window._update_run_buttons()

    assert window.run_registration_button.isEnabled()

    window._run_manual_registration()

    assert not window.run_button.isEnabled()
    assert not window.run_registration_button.isEnabled()

    deadline = time.monotonic() + 5.0
    while (
        window._is_worker_running() or window.worker_thread is not None
    ) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert not window._is_worker_running()
    assert window.worker_thread is None
    assert window.worker is None

    assert captured["image_size"] == (120, 80)
    assert captured["atlas_name"] == "allen_mouse_25um"
    assert captured["position_mm"] == 1.23
    assert captured["target_landmark_count"] == 18
    assert captured["workflow"] == "single_pass"
    assert captured["show_atlas_borders"] is True
    assert captured["enable_code_execution"] is False
    assert captured["tool_loop_max_steps"] == 9

    assert window.ap_result is not None
    assert window.ap_result.position_mm == 1.23
    assert window.ap_result.reasoning == "Manual AP position"
    assert captured["debug_dir"] == window.ap_result.debug_dir
    assert window.registration_result is fresh_result
    assert window.registration_result is not stale_result
    assert window.affine_result is fresh_result.affine_result
    if window.ap_result.debug_dir:
        assert not window.feedback_wrap.isHidden()
    else:
        assert window.feedback_wrap.isHidden()
    assert window.step_ap._status == "completed"
    assert window.step_affine._status == "completed"
    assert window.step_ap.title_label.text() == "Manual Position"
    assert "Manual Position" in window.step_ap.result_top.text()
    assert "Estimated Position" not in window.step_ap.result_top.text()
    assert window.run_registration_button.isEnabled()

    window.close()


def test_successful_registration_displays_annotation_markers(monkeypatch) -> None:
    """Annotations from a successful image_gen_two_shot registration must appear on viewers."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("LANGSLICE_VLM_DEBUG_DIR", raising=False)
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.marker_history: list[list[tuple[float, float, str]]] = []

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            self.marker_history.append(list(markers or []))

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.marker_history: list[tuple[list, list]] = []

        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(self, slice_markers, atlas_markers) -> None:
            self.marker_history.append((list(slice_markers), list(atlas_markers)))

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()
    window.source_image = Image.new("RGB", (120, 80), (255, 255, 255))
    window.pil_image = window.source_image.copy()
    window.agent_vlm_image = window.pil_image.copy()
    window.current_pos = 1.23
    window.registration_workflow = "image_gen_two_shot"
    window._update_run_buttons()

    # Build a result with annotation_session (as the real runtime does).
    corr1 = RegistrationCorrespondence(
        slice_xy=(15.0, 18.0), atlas_xy=(12.0, 16.0), label="1", confidence="high",
    )
    corr2 = RegistrationCorrespondence(
        slice_xy=(50.0, 40.0), atlas_xy=(30.0, 25.0), label="2", confidence="high",
    )
    result_with_session = RegistrationResult(
        correspondences=[corr1, corr2],
        accepted_correspondences=[corr1, corr2],
        rejected_correspondences=[],
        affine_result=AffineResult(
            matrix=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64,
            ),
            source_size=(120, 80),
            output_size=(120, 80),
            backend="test",
            reasoning="synthetic",
        ),
        nonlinear_result=NonlinearResult(
            atlas_points=np.array([[12.0, 16.0], [30.0, 25.0]], dtype=np.float64),
            slice_points=np.array([[15.0, 18.0], [50.0, 40.0]], dtype=np.float64),
            smoothing=1.0,
            backend="tps",
            reasoning="synthetic",
            output_size=(120, 80),
        ),
        qc_state="accepted",
        annotation_session=RegistrationAnnotationSession(
            workflow="image_gen_two_shot",
            target_count=2,
            atlas_annotations=[
                LandmarkAnnotation(image_role="atlas", pixel_xy=(12.0, 16.0), label="1"),
                LandmarkAnnotation(image_role="atlas", pixel_xy=(30.0, 25.0), label="2"),
            ],
            slice_annotations=[
                LandmarkAnnotation(image_role="slice", pixel_xy=(15.0, 18.0), label="1"),
                LandmarkAnnotation(image_role="slice", pixel_xy=(50.0, 40.0), label="2"),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 2},
        ),
    )
    pass2_session = RegistrationAnnotationSession(
        workflow="image_gen_two_shot",
        target_count=2,
        atlas_annotations=[
            LandmarkAnnotation(image_role="atlas", pixel_xy=(12.0, 16.0), label="1"),
            LandmarkAnnotation(image_role="atlas", pixel_xy=(30.0, 25.0), label="2"),
        ],
        slice_annotations=[
            LandmarkAnnotation(image_role="slice", pixel_xy=(15.0, 18.0), label="1"),
            LandmarkAnnotation(image_role="slice", pixel_xy=(50.0, 40.0), label="2"),
        ],
        metadata={"pass": 2},
    )

    def fake_estimate_registration_runtime(
        *,
        image,
        on_progress,
        on_trace,
        on_correspondences,
        on_annotation_session,
        atlas_name,
        position_mm,
        pixel_size_um,
        target_landmark_count,
        workflow,
        show_atlas_borders,
        debug_dir,
        enable_code_execution,
        tool_loop_max_steps,
    ):
        _ = (
            image, on_progress, on_trace, atlas_name, position_mm,
            pixel_size_um, target_landmark_count, workflow,
            show_atlas_borders, debug_dir, enable_code_execution,
            tool_loop_max_steps,
        )
        # Emit intermediate annotation session (pass 2 with both atlas and slice).
        if on_annotation_session is not None:
            on_annotation_session(pass2_session)
        if on_correspondences is not None:
            on_correspondences(result_with_session.accepted_correspondences)
        time.sleep(0.05)
        return result_with_session

    monkeypatch.setattr(
        main_window, "estimate_registration_runtime", fake_estimate_registration_runtime
    )

    window._run_manual_registration()
    deadline = time.monotonic() + 5.0
    while (
        window._is_worker_running() or window.worker_thread is not None
    ) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert not window._is_worker_running()
    assert window.registration_result is result_with_session
    assert window.preview_annotation_session is None  # Cleared on success.

    # After successful registration, split_atlas must have received atlas markers.
    assert window.split_atlas.marker_history, "Atlas viewer should have received marker updates"
    last_atlas_markers = window.split_atlas.marker_history[-1]
    assert len(last_atlas_markers) == 2
    assert last_atlas_markers == [(12.0, 16.0, "1"), (30.0, 25.0, "2")]

    # Overlay viewer must also have received both slice and atlas markers.
    assert window.overlay_viewer.marker_history, "Overlay viewer should have received markers"
    last_slice_markers, last_atlas_markers_ov = window.overlay_viewer.marker_history[-1]
    assert len(last_slice_markers) == 2
    assert last_slice_markers == [(15.0, 18.0, "1"), (50.0, 40.0, "2")]
    assert len(last_atlas_markers_ov) == 2
    assert last_atlas_markers_ov == [(12.0, 16.0, "1"), (30.0, 25.0, "2")]

    window.close()


def test_manual_registration_shows_preview_pairs_on_solver_failure(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("LANGSLICE_VLM_DEBUG_DIR", raising=False)
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.marker_history: list[list[tuple[float, float, str]]] = []

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            self.marker_history.append(list(markers or []))

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            _ = slice_markers, atlas_markers

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()
    window.source_image = Image.new("RGB", (120, 80), (255, 255, 255))
    window.pil_image = window.source_image.copy()
    window.agent_vlm_image = window.pil_image.copy()
    window.current_pos = 1.23
    window._update_run_buttons()

    preview_corr = RegistrationCorrespondence(
        slice_xy=(15.0, 18.0),
        atlas_xy=(12.0, 16.0),
        label="A",
        confidence="high",
    )

    def fake_estimate_registration_runtime(
        *,
        image,
        on_progress,
        on_trace,
        on_correspondences,
        on_annotation_session,
        atlas_name,
        position_mm,
        pixel_size_um,
        target_landmark_count,
        workflow,
        show_atlas_borders,
        debug_dir,
        enable_code_execution,
        tool_loop_max_steps,
    ):
        _ = (
            image,
            on_progress,
            on_trace,
            atlas_name,
            position_mm,
            pixel_size_um,
            target_landmark_count,
            workflow,
            show_atlas_borders,
            debug_dir,
            enable_code_execution,
            tool_loop_max_steps,
            on_annotation_session,
        )
        if on_correspondences is not None:
            on_correspondences([preview_corr])
        raise ValueError("Landmark spread too small (coverage=0.0514)")

    monkeypatch.setattr(
        main_window, "estimate_registration_runtime", fake_estimate_registration_runtime
    )

    window._run_manual_registration()
    deadline = time.monotonic() + 5.0
    while (
        window._is_worker_running() or window.worker_thread is not None
    ) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert window.step_affine._status == "error"
    assert window.preview_correspondences is not None
    assert len(window.preview_correspondences) == 1
    assert window.preview_correspondences[0].label == "A"
    assert window.split_atlas.marker_history
    assert window.split_atlas.marker_history[-1] == [(12.0, 16.0, "A")]
    assert window.registration_result is None

    window.close()


def test_manual_registration_shows_image_gen_first_pass_annotations(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("LANGSLICE_VLM_DEBUG_DIR", raising=False)
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.marker_history: list[list[tuple[float, float, str]]] = []

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            self.marker_history.append(list(markers or []))

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.marker_history: list[
                tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]
            ] = []

        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            self.marker_history.append((list(slice_markers or []), list(atlas_markers or [])))

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()
    window.source_image = Image.new("RGB", (120, 80), (255, 255, 255))
    window.pil_image = window.source_image.copy()
    window.agent_vlm_image = window.pil_image.copy()
    window.current_pos = 1.23
    window.registration_workflow = "image_gen_two_shot"
    window._update_run_buttons()

    first_pass_session = RegistrationAnnotationSession(
        workflow="image_gen_two_shot",
        target_count=2,
        atlas_annotations=[
            LandmarkAnnotation(image_role="atlas", pixel_xy=(12.0, 16.0), label="1"),
            LandmarkAnnotation(image_role="atlas", pixel_xy=(32.0, 36.0), label="2"),
        ],
        slice_annotations=[],
        metadata={"pass": 1},
    )

    def fake_estimate_registration_runtime(
        *,
        image,
        on_progress,
        on_trace,
        on_correspondences,
        on_annotation_session,
        atlas_name,
        position_mm,
        pixel_size_um,
        target_landmark_count,
        workflow,
        show_atlas_borders,
        debug_dir,
        enable_code_execution,
        tool_loop_max_steps,
    ):
        _ = (
            image,
            on_progress,
            on_trace,
            on_correspondences,
            atlas_name,
            position_mm,
            pixel_size_um,
            target_landmark_count,
            workflow,
            show_atlas_borders,
            debug_dir,
            enable_code_execution,
            tool_loop_max_steps,
        )
        if on_annotation_session is not None:
            on_annotation_session(first_pass_session)
        raise ValueError("Need at least 3 correspondence pairs to fit registration, got 2")

    monkeypatch.setattr(
        main_window, "estimate_registration_runtime", fake_estimate_registration_runtime
    )

    window._run_manual_registration()
    deadline = time.monotonic() + 5.0
    while (
        window._is_worker_running() or window.worker_thread is not None
    ) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert window.step_affine._status == "error"
    assert window.preview_annotation_session is first_pass_session
    assert window.split_atlas.marker_history
    assert window.split_atlas.marker_history[-1] == [(12.0, 16.0, "1"), (32.0, 36.0, "2")]
    assert window.overlay_viewer.marker_history
    assert window.overlay_viewer.marker_history[-1] == (
        [],
        [(12.0, 16.0, "1"), (32.0, 36.0, "2")],
    )
    assert window.registration_result is None

    window.close()


def test_annotation_session_uses_position_hint_when_ap_not_set(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.positions: list[float] = []
            self.marker_history: list[list[tuple[float, float, str]]] = []

        def set_position(self, position_mm: float) -> None:
            self.positions.append(position_mm)

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            self.marker_history.append(list(markers or []))

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.positions: list[float] = []

        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            self.positions.append(position_mm)

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            _ = slice_markers, atlas_markers

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()
    window.source_image = Image.new("RGB", (120, 80), (255, 255, 255))
    window.pil_image = window.source_image.copy()
    window.agent_vlm_image = window.pil_image.copy()
    window.current_pos = None

    session = RegistrationAnnotationSession(
        workflow="image_gen_two_shot",
        target_count=2,
        atlas_annotations=[
            LandmarkAnnotation(image_role="atlas", pixel_xy=(12.0, 16.0), label="1"),
            LandmarkAnnotation(image_role="atlas", pixel_xy=(32.0, 36.0), label="2"),
        ],
        slice_annotations=[],
        metadata={"pass": 1, "position_mm": 2.5},
    )

    window._on_annotation_session_ready(session)
    app.processEvents()

    assert window.current_pos == 2.5
    assert window.split_atlas.positions
    assert window.split_atlas.positions[-1] == 2.5
    assert window.overlay_viewer.positions
    assert window.overlay_viewer.positions[-1] == 2.5
    assert window.split_atlas.marker_history
    assert window.split_atlas.marker_history[-1] == [(12.0, 16.0, "1"), (32.0, 36.0, "2")]

    window.close()


def test_image_model_selection_gates_registration_workflow(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            _ = markers

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            _ = slice_markers, atlas_markers

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()

    models = [window.model_combo.itemText(i) for i in range(window.model_combo.count())]
    assert "gemini-3-pro-image-preview" in models
    assert "gemini-3.1-flash-image-preview" in models

    window.model_combo.setCurrentText("gemini-3-pro-image-preview")
    app.processEvents()

    image_workflows = [
        window.workflow_combo.itemData(i) for i in range(window.workflow_combo.count())
    ]
    assert image_workflows == ["image_gen_two_shot"]
    assert window.registration_workflow == "image_gen_two_shot"
    assert window.code_execution_checkbox.isHidden()

    window.model_combo.setCurrentText("gemini-3-flash-preview")
    app.processEvents()

    text_workflows = [
        window.workflow_combo.itemData(i) for i in range(window.workflow_combo.count())
    ]
    assert "image_gen_two_shot" not in text_workflows
    assert "single_pass" in text_workflows
    assert "multimodal_tool_loop" in text_workflows
    assert not window.code_execution_checkbox.isHidden()

    main_window.set_model_name("gemini-3-flash-preview")
    window.close()


def test_save_agent_trace_exports_visible_events(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(main_window, "list_downloaded_atlases", lambda: ["allen_mouse_25um"])
    monkeypatch.setattr(main_window, "list_available_atlases", lambda: ["allen_mouse_25um"])

    class DummyAtlasViewer(main_window.QFrame):
        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def set_correspondence_markers(
            self,
            markers: list[tuple[float, float, str]] | None,
        ) -> None:
            _ = markers

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    class DummyOverlayGraphicsView(main_window.QFrame):
        def set_pixel_size(self, pixel_size_um: float) -> None:
            _ = pixel_size_um

        def set_atlas(self, atlas_name: str) -> None:
            _ = atlas_name

        def clear(self) -> None:
            return

        def clear_all(self) -> None:
            return

        def set_position(self, position_mm: float) -> None:
            _ = position_mm

        def set_slice_pixmap(self, pixmap) -> None:
            _ = pixmap

        def set_correspondence_markers(
            self,
            slice_markers: list[tuple[float, float, str]] | None = None,
            atlas_markers: list[tuple[float, float, str]] | None = None,
        ) -> None:
            _ = slice_markers, atlas_markers

        def set_atlas_opacity(self, opacity: float) -> None:
            _ = opacity

        def set_show_region_borders(self, visible: bool) -> None:
            _ = visible

    monkeypatch.setattr(main_window, "AtlasViewer", DummyAtlasViewer)
    monkeypatch.setattr(main_window, "OverlayGraphicsView", DummyOverlayGraphicsView)

    output_path = tmp_path / "saved_trace.json"
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(output_path), "JSON (*.json)")),
    )

    app = main_window.QApplication.instance() or main_window.QApplication([])
    window = main_window.MainWindow()
    window._append_trace_event(
        main_window.status_event(
            "registration", "Registration image-gen atlas pass: request started"
        )
    )

    window._save_agent_trace()
    app.processEvents()

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["event_type"] == "status"

    window.close()
