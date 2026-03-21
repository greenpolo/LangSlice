"""Checks for registration-agent correspondence constraints."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

import langslice.registration.agents as agents


class _DummyModels:
    def generate_content(self, *, model: str, contents: Any, config: Any) -> object:
        _ = model, contents, config
        return object()

    def generate_content_stream(self, *, model: str, contents: Any, config: Any) -> object:
        _ = model, contents, config
        return iter([])


class _DummyFiles:
    """Stub for client.files used by the tool loop File API path."""

    def upload(self, *, file: Any, config: Any) -> SimpleNamespace:
        return SimpleNamespace(name="fake-file-id")

    def delete(self, *, name: str) -> None:
        pass


class _DummyClient:
    def __init__(self) -> None:
        self.models = _DummyModels()
        self.files = _DummyFiles()


def _correspondence(
    atlas_norm_y: int,
    atlas_norm_x: int,
    slice_norm_y: int,
    slice_norm_x: int,
    label: str,
    *,
    status: str = "found",
) -> dict[str, object]:
    """Build a fake paired correspondence in normalised [y, x] 0-1000 format."""
    return {
        "atlas_point_2d": [atlas_norm_y, atlas_norm_x],
        "slice_point_2d": [slice_norm_y, slice_norm_x],
        "label": label,
        "status": status,
    }


def _patch_common(monkeypatch: Any) -> None:
    """Apply atlas/vlm mocks shared by all tests."""
    atlas = SimpleNamespace(
        atlas_name="allen_mouse_25um",
        orientation="asr",
        reference=np.ones((1, 320, 456), dtype=np.uint8),
        annotation=np.ones((1, 320, 456), dtype=np.int32),
        resolution=(25.0, 25.0, 25.0),
        metadata={},
        structures={1: {"acronym": "CTX", "name": "Cortex"}},
    )
    monkeypatch.setattr(agents, "load_atlas", lambda _name: atlas)
    monkeypatch.setattr(
        agents,
        "get_atlas_info",
        lambda _atlas: {"shape": (528, 320, 456), "resolution_um": (25, 25, 25)},
    )
    monkeypatch.setattr(
        agents,
        "get_slice_region_metadata",
        lambda _atlas, _pos: [
            {
                "acronym": "CTX",
                "name": "Cortex",
                "centroid_normalized": (500, 500),
                "area_fraction": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        agents,
        "get_composite_slice",
        lambda _atlas, _pos: Image.new("RGB", (456, 320), (0, 0, 0)),
    )
    monkeypatch.setattr(agents, "normalize_image", lambda img: img)
    monkeypatch.setattr(
        agents,
        "prepare_image_for_vlm",
        lambda img, **kw: SimpleNamespace(image=img, output_size=img.size, original_size=img.size),
    )

    vlm_config = SimpleNamespace(
        MODEL_NAME="test-model",
        CODE_EXECUTION_ENABLED=False,
        THINKING_LEVEL="HIGH",
        TEMPERATURE=0.5,
        count_tokens_enabled=lambda: False,
        get_client=lambda: _DummyClient(),
        supports_file_api=lambda: False,
    )
    real_import_module = agents.importlib.import_module
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            vlm_config if name == "langslice.ai.config" else real_import_module(name, **kwargs)
        ),
    )


# ---------------------------------------------------------------------------
# Helper: build mock function-call responses for the tool loop
# ---------------------------------------------------------------------------


def _fc_response(*calls: tuple[str, dict[str, Any]]) -> SimpleNamespace:
    """Build a mock Gemini response containing one or more FunctionCall parts.

    Each *call* is ``(tool_name, tool_args)``.  The returned object has the
    same shape the tool-loop code accesses:

    - ``response.candidates[0].content.parts[i].function_call.name``
    - ``response.candidates[0].content.parts[i].function_call.args``
    - ``response.usage_metadata`` (set to None for simplicity)
    """
    parts = []
    for name, args in calls:
        parts.append(
            SimpleNamespace(
                function_call=SimpleNamespace(name=name, args=args, id=None),
                text=None,
                thought=False,
            )
        )
    content = SimpleNamespace(parts=parts, role="model")
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        usage_metadata=None,
    )


# ---------------------------------------------------------------------------
# Shared helper tests
# ---------------------------------------------------------------------------


def test_image_to_inline_data_uses_rgb_png() -> None:
    part = agents._image_to_inline_data(Image.new("RGBA", (8, 6), (255, 0, 0, 128)))
    assert isinstance(part, dict)

    inline_data = part["inline_data"]
    assert isinstance(inline_data, dict)
    assert inline_data["mime_type"] == "image/png"

    import io

    encoded_data = inline_data["data"]
    assert isinstance(encoded_data, (bytes, bytearray))

    decoded = Image.open(io.BytesIO(encoded_data))
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 6)


# ---------------------------------------------------------------------------
# Temperature / thinking level config tests
# ---------------------------------------------------------------------------


def test_registration_request_uses_configured_temperature(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    captured: dict[str, object] = {}

    # The new code uses place -> zoom -> confirm -> finish.
    # For a 1-landmark run: border_count=1, interior_count=0 (50/50 split of 1)
    tool_loop_responses = [
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d": [100, 120], "slice_point_2d": [140, 160],
            "feature_description": "anchor",
        })),
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [100, 120], "slice_center_2d": [140, 160],
        })),
        _fc_response(("confirm_point", {"label": "1"})),
        _fc_response(("finish", {})),
    ]

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args
        if "config" not in captured:
            captured["config"] = kwargs["config"]
        return tool_loop_responses.pop(0)

    current_import_module = agents.importlib.import_module
    monkeypatch.setattr(agents, "_retry_generate", mock_generate)
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(
                MODEL_NAME="test-model",
                CODE_EXECUTION_ENABLED=False,
                THINKING_LEVEL="LOW",
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
                supports_file_api=lambda: False,
            )
            if name == "langslice.ai.config"
            else current_import_module(name, **kwargs)
        ),
    )

    agents.estimate_registration_correspondences(
        Image.new("RGB", (120, 100), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        target_landmark_count=1,
        min_edge_landmarks=0,
    )

    # Config is now a types.GenerateContentConfig, not a dict.
    from google.genai import types

    config = captured["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.temperature == 0.2


def test_registration_request_uses_configured_thinking_level(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    captured: dict[str, object] = {}

    tool_loop_responses = [
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d": [100, 120], "slice_point_2d": [140, 160],
            "feature_description": "anchor",
        })),
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [100, 120], "slice_center_2d": [140, 160],
        })),
        _fc_response(("confirm_point", {"label": "1"})),
        _fc_response(("finish", {})),
    ]

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args
        if "config" not in captured:
            captured["config"] = kwargs["config"]
        return tool_loop_responses.pop(0)

    current_import_module = agents.importlib.import_module
    monkeypatch.setattr(agents, "_retry_generate", mock_generate)
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(
                MODEL_NAME="test-model",
                CODE_EXECUTION_ENABLED=False,
                THINKING_LEVEL="LOW",
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
                supports_file_api=lambda: False,
            )
            if name == "langslice.ai.config"
            else current_import_module(name, **kwargs)
        ),
    )

    agents.estimate_registration_correspondences(
        Image.new("RGB", (120, 100), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        target_landmark_count=1,
        min_edge_landmarks=0,
    )

    from google.genai import types

    config = captured["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == "LOW"


# ---------------------------------------------------------------------------
# Image-gen workflow tests (unchanged — they don't use the tool loop)
# ---------------------------------------------------------------------------


def _make_image_with_markers(
    base: Image.Image,
    markers: list[tuple[int, int]],
    colour: tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
) -> Image.Image:
    """Draw bright circles on *base* at *markers* positions (x, y)."""
    img = base.copy()
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    for x, y in markers:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)
    return img


def _fake_image_response(
    image: Image.Image,
    text: str | None = None,
    *,
    mime_type: str = "image/png",
) -> SimpleNamespace:
    """Build a mock Gemini response containing an inline image and optional text."""
    import io

    buf = io.BytesIO()
    fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
    image.save(buf, format=fmt)
    parts = [
        SimpleNamespace(
            inline_data=SimpleNamespace(
                data=buf.getvalue(),
                mime_type=mime_type,
            ),
            text=None,
            thought=False,
        )
    ]
    if text is not None:
        parts.append(SimpleNamespace(inline_data=None, text=text, thought=False))
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        # Also expose text at top level for _extract_json_payload fallback.
        text=text,
        parsed=None,
    )


def _typed_part_bytes(part: Any) -> bytes | None:
    inline_data = getattr(part, "inline_data", None)
    if inline_data is None:
        return None
    data = getattr(inline_data, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return None


def test_image_gen_two_shot_runs_both_passes(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    atlas_base = Image.new("RGB", (456, 320), (128, 128, 128))
    slice_base = Image.new("RGB", (200, 120), (128, 128, 128))

    atlas_annotated = _make_image_with_markers(atlas_base, [(100, 50)], colour=(255, 0, 0))
    slice_annotated = _make_image_with_markers(slice_base, [(30, 20)], colour=(255, 0, 0))

    responses = [
        _fake_image_response(atlas_annotated),
        _fake_image_response(slice_annotated),
    ]

    current_import_module = agents.importlib.import_module
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(
                MODEL_NAME="gemini-3-pro-image-preview",
                CODE_EXECUTION_ENABLED=False,
                THINKING_LEVEL="MEDIUM",
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
                supports_file_api=lambda: False,
            )
            if name == "langslice.ai.config"
            else current_import_module(name, **kwargs)
        ),
    )
    monkeypatch.setattr(agents, "_retry_generate_stream", lambda *args, **kwargs: responses.pop(0))

    # Marker extraction is not yet implemented — the image-gen workflow
    # returns an empty correspondence list, which the caller raises on.
    with pytest.raises(RuntimeError, match="no usable correspondences"):
        agents.estimate_registration_correspondences(
            slice_base,
            atlas_name="allen_mouse_25um",
            position_mm=2.0,
            target_landmark_count=4,
            min_edge_landmarks=2,
            workflow="image_gen_two_shot",
        )


def test_image_gen_two_shot_raises_on_no_atlas_image(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    slice_base = Image.new("RGB", (200, 120), (128, 128, 128))

    # Return an empty response (no image parts) for the atlas pass.
    empty_response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
        text=None,
        parsed=None,
    )

    current_import_module = agents.importlib.import_module
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(
                MODEL_NAME="gemini-3-pro-image-preview",
                CODE_EXECUTION_ENABLED=False,
                THINKING_LEVEL="MEDIUM",
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
                supports_file_api=lambda: False,
            )
            if name == "langslice.ai.config"
            else current_import_module(name, **kwargs)
        ),
    )
    monkeypatch.setattr(
        agents, "_retry_generate_stream", lambda *args, **kwargs: empty_response
    )

    with pytest.raises(RuntimeError, match="did not return an image"):
        agents.estimate_registration_correspondences(
            slice_base,
            atlas_name="allen_mouse_25um",
            position_mm=2.0,
            target_landmark_count=2,
            min_edge_landmarks=1,
            workflow="image_gen_two_shot",
        )


def test_image_gen_config_requests_image_text_1k_and_high_thinking(monkeypatch: Any) -> None:
    import langslice.registration.agents_image_gen as image_gen_agents

    real_import_module = image_gen_agents.importlib.import_module
    monkeypatch.setattr(
        image_gen_agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(
                supports_image_model_thinking=lambda model_name: (
                    model_name == "gemini-3.1-flash-image-preview"
                )
            )
            if name == "langslice.ai.config"
            else real_import_module(name, **kwargs)
        ),
    )

    config = image_gen_agents._build_image_gen_config(
        model_name="gemini-3.1-flash-image-preview",
        thinking_level="HIGH",
    )

    assert config.response_modalities == ["IMAGE", "TEXT"]
    assert config.image_config.image_size == "1K"
    assert config.thinking_config.thinking_level == "HIGH"


def test_image_gen_config_skips_thinking_when_model_does_not_support_it(monkeypatch: Any) -> None:
    import langslice.registration.agents_image_gen as image_gen_agents

    real_import_module = image_gen_agents.importlib.import_module
    monkeypatch.setattr(
        image_gen_agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(supports_image_model_thinking=lambda _model_name: False)
            if name == "langslice.ai.config"
            else real_import_module(name, **kwargs)
        ),
    )

    config = image_gen_agents._build_image_gen_config(
        model_name="gemini-3-pro-image-preview",
        thinking_level="HIGH",
    )

    assert config.response_modalities == ["IMAGE", "TEXT"]
    assert config.image_config.image_size == "1K"
    assert config.thinking_config is None


def test_extract_generated_image_prefers_last_image_part() -> None:
    import langslice.registration.agents_image_gen as image_gen_agents

    first = Image.new("RGB", (8, 8), (255, 0, 0))
    last = Image.new("RGB", (8, 8), (0, 255, 0))

    response = _fake_image_response(first)
    response.candidates[0].content.parts.append(
        _fake_image_response(last).candidates[0].content.parts[0]
    )

    extracted = image_gen_agents._extract_generated_image(response)

    assert extracted is not None
    assert extracted.getpixel((0, 0)) == (0, 255, 0)


def test_image_gen_slice_request_matches_ai_studio_part_order() -> None:
    import langslice.registration.agents_image_gen as image_gen_agents

    annotated_atlas = Image.new("RGB", (11, 7), (10, 20, 30))
    slice_image = Image.new("RGB", (13, 9), (40, 50, 60))
    request = image_gen_agents._build_image_gen_slice_request(
        annotated_atlas=annotated_atlas,
        slice_prep=SimpleNamespace(image=slice_image),
        species="mouse",
    )

    assert len(request) == 1
    parts = request[0].parts
    assert len(parts) == 3

    import io

    first_bytes = _typed_part_bytes(parts[0])
    second_bytes = _typed_part_bytes(parts[1])
    assert first_bytes is not None
    assert second_bytes is not None
    assert Image.open(io.BytesIO(first_bytes)).size == slice_image.size
    assert Image.open(io.BytesIO(second_bytes)).size == annotated_atlas.size
    assert isinstance(parts[2].text, str)


def test_image_gen_slice_request_reuses_generated_atlas_payload_bytes() -> None:
    import io

    import langslice.registration.agents_image_gen as image_gen_agents

    atlas_image = Image.new("RGB", (11, 7), (10, 20, 30))
    slice_image = Image.new("RGB", (13, 9), (40, 50, 60))
    atlas_response = _fake_image_response(atlas_image, mime_type="image/jpeg")
    atlas_payloads = image_gen_agents._extract_generated_images(atlas_response)

    assert len(atlas_payloads) == 1
    request = image_gen_agents._build_image_gen_slice_request(
        annotated_atlas=atlas_payloads[0],
        slice_prep=SimpleNamespace(image=slice_image),
        species="mouse",
    )

    parts = request[0].parts
    atlas_bytes = _typed_part_bytes(parts[1])
    assert atlas_bytes is not None
    assert atlas_bytes == atlas_payloads[0].data
    assert parts[1].inline_data.mime_type == "image/jpeg"
    assert Image.open(io.BytesIO(atlas_bytes)).size == atlas_image.size


# ---------------------------------------------------------------------------
# Tool-loop workflow tests (updated for native function calling + confirm gate)
# ---------------------------------------------------------------------------


def test_multimodal_tool_loop_places_and_confirms_before_finish(monkeypatch: Any) -> None:
    """Full place -> zoom -> confirm -> finish workflow for 2 border points."""
    _patch_common(monkeypatch)

    responses = [
        # Place point 1
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d": [100, 120], "slice_point_2d": [140, 160],
            "feature_description": "outer contour notch",
        })),
        # Zoom to inspect point 1
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [100, 120], "slice_center_2d": [140, 160],
        })),
        # Confirm point 1
        _fc_response(("confirm_point", {"label": "1"})),
        # Place point 2
        _fc_response(("place_point_pair", {
            "label": "2", "category": "border",
            "atlas_point_2d": [500, 520], "slice_point_2d": [540, 560],
            "feature_description": "ventricle corner",
        })),
        # Zoom to inspect point 2
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [500, 520], "slice_center_2d": [540, 560],
        })),
        # Confirm point 2
        _fc_response(("confirm_point", {"label": "2"})),
        # Finish
        _fc_response(("finish", {})),
    ]

    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (200, 120), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=2,
        min_edge_landmarks=1,
        workflow="multimodal_tool_loop",
        border_count=2,
        interior_count=0,
    )

    assert [corr.label for corr in result] == ["1", "2"]
    assert result[0].rationale.startswith("status=found")


def test_multimodal_tool_loop_accepts_zoom_local_coordinates(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    responses = [
        # Zoom first (overview context)
        _fc_response(("view_zoom_pair", {
            "zoom": 2.0, "atlas_center_2d": [500, 500], "slice_center_2d": [500, 500],
        })),
        # Place with local coordinates
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d_local": [0, 0], "slice_point_2d_local": [0, 0],
            "feature_description": "upper-left corner of zoom window",
        })),
        # Zoom to inspect
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [250, 250], "slice_center_2d": [250, 250],
        })),
        # Confirm
        _fc_response(("confirm_point", {"label": "1"})),
        # Finish
        _fc_response(("finish", {})),
    ]

    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (456, 320), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=1,
        min_edge_landmarks=1,
        workflow="multimodal_tool_loop",
        border_count=1,
        interior_count=0,
    )

    assert [corr.label for corr in result] == ["1"]
    # The local [0,0] maps to the top-left corner of the zoom window.
    # With zoom=2.0 centered at [500,500], the window is roughly the middle half.
    # Exact values depend on atlas upscale dimensions, so use relative tolerance.
    assert result[0].atlas_normalized_yx == pytest.approx(
        (250.0, 250.0), rel=0.01
    )
    assert result[0].slice_normalized_yx == pytest.approx(
        (250.0, 250.0), rel=0.01
    )


def test_multimodal_tool_loop_uses_explicit_step_limit(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    # This response will be returned once (the only allowed step), and since
    # it doesn't finish, the loop should exceed max steps.
    responses = [
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d": [100, 120], "slice_point_2d": [140, 160],
            "feature_description": "outer contour notch",
        })),
    ]

    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    with pytest.raises(RuntimeError, match="exceeded the maximum number of steps"):
        agents.estimate_registration_correspondences(
            Image.new("RGB", (200, 120), (0, 0, 0)),
            atlas_name="allen_mouse_25um",
            position_mm=2.0,
            target_landmark_count=2,
            min_edge_landmarks=1,
            workflow="multimodal_tool_loop",
            tool_loop_max_steps=1,
        )


def test_multimodal_tool_loop_confirm_rejected_without_zoom(monkeypatch: Any) -> None:
    """Confirm gate: trying to confirm without zooming returns an error."""
    _patch_common(monkeypatch)

    responses = [
        # Place point 1
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d": [100, 120], "slice_point_2d": [140, 160],
            "feature_description": "notch",
        })),
        # Try to confirm without zoom — should get rejection
        _fc_response(("confirm_point", {"label": "1"})),
        # Now zoom
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [100, 120], "slice_center_2d": [140, 160],
        })),
        # Now confirm succeeds
        _fc_response(("confirm_point", {"label": "1"})),
        # Finish
        _fc_response(("finish", {})),
    ]

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args
        return responses.pop(0)

    monkeypatch.setattr(agents, "_retry_generate", mock_generate)

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (200, 120), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=1,
        min_edge_landmarks=0,
        workflow="multimodal_tool_loop",
        border_count=1,
        interior_count=0,
    )

    assert [corr.label for corr in result] == ["1"]


def test_multimodal_tool_loop_border_interior_default_split(monkeypatch: Any) -> None:
    """With target_count=3, border=2 and interior=1 by default (extra goes to border)."""
    _patch_common(monkeypatch)

    # We need 2 border + 1 interior confirmed
    responses = [
        # Place border 1
        _fc_response(("place_point_pair", {
            "label": "1", "category": "border",
            "atlas_point_2d": [100, 120], "slice_point_2d": [140, 160],
            "feature_description": "notch",
        })),
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [100, 120], "slice_center_2d": [140, 160],
        })),
        _fc_response(("confirm_point", {"label": "1"})),
        # Place border 2
        _fc_response(("place_point_pair", {
            "label": "2", "category": "border",
            "atlas_point_2d": [200, 220], "slice_point_2d": [240, 260],
            "feature_description": "edge",
        })),
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [200, 220], "slice_center_2d": [240, 260],
        })),
        _fc_response(("confirm_point", {"label": "2"})),
        # Place interior 1
        _fc_response(("place_point_pair", {
            "label": "3", "category": "interior",
            "atlas_point_2d": [500, 500], "slice_point_2d": [500, 500],
            "feature_description": "deep feature",
        })),
        _fc_response(("view_zoom_pair", {
            "zoom": 3.0, "atlas_center_2d": [500, 500], "slice_center_2d": [500, 500],
        })),
        _fc_response(("confirm_point", {"label": "3"})),
        # Finish
        _fc_response(("finish", {})),
    ]

    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (200, 120), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=3,
        min_edge_landmarks=0,
        workflow="multimodal_tool_loop",
        # border_count and interior_count not set — should default to 2/1
    )

    assert len(result) == 3
    assert [corr.label for corr in result] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Heartbeat / progress tests (unchanged)
# ---------------------------------------------------------------------------


def test_registration_progress_heartbeat_reports_wait_and_completion() -> None:
    messages: list[str] = []

    result = agents._run_with_progress_heartbeat(
        lambda: (time.sleep(0.03), "ok")[1],
        request_label="Registration test request",
        on_progress=messages.append,
        heartbeat_interval_s=0.01,
    )

    assert result == "ok"
    assert messages[0] == "Registration test request: request started"
    assert any("still waiting for Gemini" in message for message in messages)
    assert messages[-1].startswith("Registration test request: response received in ")
