"""Script-style checks for standalone ANTs affine worker mode."""

from __future__ import annotations

from PIL import Image

import langslice.gui.main_window as main_window
from langslice.registration.types import AffineResult


captured: dict[str, object] = {}


def fake_estimate_affine_registration(image, *, on_progress, atlas_name, position_mm, pixel_size_um, fallback):
    captured["atlas_name"] = atlas_name
    captured["position_mm"] = position_mm
    captured["pixel_size_um"] = pixel_size_um
    captured["fallback"] = fallback
    captured["image_size"] = image.size
    if on_progress is not None:
        on_progress("fake ants run")
    return AffineResult.from_legacy_params(
        image_width=image.width,
        image_height=image.height,
        rotation_deg=1.5,
        translate_x_pct=0.0,
        translate_y_pct=0.0,
        backend="ants",
        reasoning="fake ants result",
    )


def fake_estimate_position(*args, **kwargs):
    raise AssertionError("AP estimation should not run in affine_only mode")


original_affine = main_window.estimate_affine_registration
original_position = main_window.estimate_position

try:
    main_window.estimate_affine_registration = fake_estimate_affine_registration
    main_window.estimate_position = fake_estimate_position

    worker = main_window.AgentWorker(
        Image.new("RGB", (640, 480), (255, 255, 255)),
        Image.new("RGB", (320, 240), (255, 255, 255)),
        "allen_mouse_25um",
        pixel_size_um=4.0,
        position_mm=3.25,
        mode="affine_only",
    )

    events: list[tuple[str, object]] = []
    logs: list[str] = []
    worker.step_started.connect(lambda step: events.append(("started", step)))
    worker.step_completed.connect(lambda step, result: events.append((step, result)))
    worker.step_error.connect(lambda step, msg: events.append((f"error:{step}", msg)))
    worker.log_message.connect(logs.append)
    worker.run()

    assert captured["atlas_name"] == "allen_mouse_25um"
    assert captured["position_mm"] == 3.25
    assert captured["pixel_size_um"] == 4.0
    assert captured["fallback"] is None
    assert captured["image_size"] == (640, 480)
    assert ("started", "affine") in events
    assert any(kind == "affine" for kind, _ in events)
    assert all(not kind.startswith("error:") for kind, _ in events)
    assert any("standalone ANTs affine" in entry for entry in logs)
finally:
    main_window.estimate_affine_registration = original_affine
    main_window.estimate_position = original_position

print("Affine-only worker OK")
