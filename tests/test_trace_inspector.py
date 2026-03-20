from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from langslice.agent_trace import image_part_from_pil, runtime_event
from langslice.gui import trace_inspector


def test_trace_entry_renders_inline_image_part(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        trace_inspector._qtwidgets.QApplication.instance()
        or trace_inspector._qtwidgets.QApplication([])
    )

    image = Image.new("RGB", (120, 80), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    event = runtime_event(
        stage="registration",
        title="Prepared request",
        parts=[
            image_part_from_pil(
                image, label="Input image", image_format="PNG", image_bytes=buf.getvalue()
            )
        ],
    )

    widget = trace_inspector.TraceEntryWidget(event)
    image_widgets = widget.findChildren(trace_inspector._TraceImageLabel)

    assert len(image_widgets) == 1
    assert image_widgets[0].pixmap() is not None
    widget.deleteLater()
    app.processEvents()


def test_trace_inspector_exports_events_with_image_assets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = (
        trace_inspector._qtwidgets.QApplication.instance()
        or trace_inspector._qtwidgets.QApplication([])
    )

    image = Image.new("RGB", (80, 60), (0, 255, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    inspector = trace_inspector.TraceInspector()
    inspector.append_event(
        runtime_event(
            stage="registration",
            title="Prepared request",
            parts=[
                image_part_from_pil(
                    image, label="Input image", image_format="PNG", image_bytes=buf.getvalue()
                )
            ],
        )
    )

    output_path = tmp_path / "agent_trace.json"
    exported_count = inspector.export_events(str(output_path))

    assert exported_count == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    saved_path = payload[0]["parts"][0]["saved_path"]
    assert isinstance(saved_path, str)
    assert (tmp_path / saved_path).exists()

    inspector.deleteLater()
    app.processEvents()
