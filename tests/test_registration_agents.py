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


class _DummyClient:
    def __init__(self) -> None:
        self.models = _DummyModels()


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
        lambda img: SimpleNamespace(image=img, output_size=img.size, original_size=img.size),
    )

    vlm_config = SimpleNamespace(
        MODEL_NAME="test-model",
        CODE_EXECUTION_ENABLED=False,
        THINKING_LEVEL="HIGH",
        TEMPERATURE=0.5,
        count_tokens_enabled=lambda: False,
        get_client=lambda: _DummyClient(),
    )
    real_import_module = agents.importlib.import_module
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            vlm_config if name == "langslice.vlm.config" else real_import_module(name, **kwargs)
        ),
    )


def test_single_pass_pairs_atlas_and_slice_landmarks(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    correspondences = [
        _correspondence(10, 10, 10, 10, "e1"),
        _correspondence(450, 10, 950, 10, "e2"),
        _correspondence(200, 140, 300, 400, "i1"),
        _correspondence(210, 150, 350, 450, "i2"),
        _correspondence(220, 160, 400, 500, "i3"),
        _correspondence(230, 170, 450, 550, "i4"),
        _correspondence(240, 180, 500, 600, "i5"),
        _correspondence(250, 190, 550, 650, "i6"),
    ]

    call_count = 0

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args, kwargs
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(parsed={"correspondences": correspondences})

    monkeypatch.setattr(agents, "_retry_generate", mock_generate)

    image = Image.new("RGB", (3790, 2844), (0, 0, 0))
    result = agents.estimate_registration_correspondences(
        image,
        atlas_name="allen_mouse_25um",
        position_mm=4.28,
        target_landmark_count=8,
        min_edge_landmarks=5,
    )
    assert len(result) == 8
    assert call_count == 1
    assert result[0].label == "e1"
    assert result[1].label == "e2"


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


def test_single_pass_keeps_raw_slice_coordinate_frame(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    correspondences = [
        _correspondence(20, 20, 2, 2, "p1"),
        _correspondence(430, 22, 43, 3, "p2"),
        _correspondence(24, 300, 2, 36, "p3"),
        _correspondence(430, 300, 43, 36, "p4"),
        _correspondence(230, 20, 23, 2, "p5"),
        _correspondence(230, 300, 23, 36, "p6"),
        _correspondence(20, 160, 2, 19, "p7"),
        _correspondence(430, 160, 43, 19, "p8"),
        _correspondence(120, 90, 12, 11, "p9"),
        _correspondence(180, 110, 18, 13, "p10"),
        _correspondence(260, 140, 26, 17, "p11"),
        _correspondence(300, 180, 30, 22, "p12"),
    ]

    call_count = 0

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args, kwargs
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(parsed={"correspondences": correspondences})

    monkeypatch.setattr(agents, "_retry_generate", mock_generate)

    image = Image.new("RGB", (3780, 3174), (0, 0, 0))
    result = agents.estimate_registration_correspondences(
        image,
        atlas_name="allen_mouse_25um",
        position_mm=4.28,
        target_landmark_count=12,
        min_edge_landmarks=8,
    )
    assert len(result) == 12
    assert call_count == 1
    # Without atlas-frame rescue, tiny normalized slice coordinates remain small.
    assert result[3].slice_xy[0] < 200.0
    assert result[3].slice_xy[1] < 200.0
    assert result[0].slice_xy[0] < 20.0


def test_single_pass_normalized_coordinates_map_to_exact_pixel_xy(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        agents,
        "get_composite_slice",
        lambda _atlas, _pos: Image.new("RGB", (123, 77), (0, 0, 0)),
    )

    correspondences = [
        _correspondence(1000, 500, 250, 1000, "endpoint"),
        _correspondence(0, 0, 0, 0, "origin"),
        _correspondence(500, 0, 500, 0, "mid_left"),
        _correspondence(0, 1000, 0, 500, "top_right"),
        _correspondence(1000, 1000, 1000, 1000, "bottom_right"),
        _correspondence(500, 500, 500, 500, "center"),
    ]

    monkeypatch.setattr(
        agents,
        "_retry_generate",
        lambda *args, **kwargs: SimpleNamespace(parsed={"correspondences": correspondences}),
    )

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (301, 201), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=4.28,
        target_landmark_count=6,
        min_edge_landmarks=5,
    )

    assert len(result) == 6
    assert result[0].atlas_xy == (61.0, 76.0)
    assert result[0].slice_xy == (300.0, 50.0)
    assert result[4].atlas_xy == (122.0, 76.0)
    assert result[4].slice_xy == (300.0, 200.0)


def test_single_pass_keeps_out_of_range_points_without_filtering(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    correspondences = [
        _correspondence(10, 10, 10, 10, "keep_1"),
        _correspondence(100, 100, 100, 100, "keep_2"),
        _correspondence(200, 200, 200, 200, "keep_3"),
        _correspondence(300, 300, 300, 300, "keep_4"),
        _correspondence(400, 400, 400, 400, "keep_5"),
        _correspondence(500, 500, 500, 500, "keep_6"),
        _correspondence(1001, 600, 600, 600, "reject_atlas"),
        _correspondence(600, 600, -1, 600, "keep_negative_slice"),
    ]

    monkeypatch.setattr(
        agents,
        "_retry_generate",
        lambda *args, **kwargs: SimpleNamespace(parsed={"correspondences": correspondences}),
    )

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (3790, 2844), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=4.28,
        target_landmark_count=8,
        min_edge_landmarks=5,
    )

    labels = [corr.label for corr in result]
    assert len(result) == 8
    assert "reject_atlas" in labels
    assert "keep_negative_slice" in labels


def test_single_pass_only_filters_not_visible_pairs(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    correspondences = [
        _correspondence(10, 10, 10, 10, "keep_1"),
        _correspondence(100, 100, 100, 100, "keep_2"),
        _correspondence(200, 200, 200, 200, "keep_3"),
        _correspondence(300, 300, 300, 300, "keep_4"),
        _correspondence(400, 400, 400, 400, "keep_5"),
        _correspondence(500, 500, 500, 500, "drop_not_visible", status="not_visible"),
        _correspondence(1001, 600, 600, 600, "keep_out_of_range"),
        _correspondence(600, 600, 1001, 600, "keep_slice_out_of_range"),
    ]

    monkeypatch.setattr(
        agents,
        "_retry_generate",
        lambda *args, **kwargs: SimpleNamespace(parsed={"correspondences": correspondences}),
    )

    image = Image.new("RGB", (3790, 2844), (0, 0, 0))
    result = agents.estimate_registration_correspondences(
        image,
        atlas_name="allen_mouse_25um",
        position_mm=4.28,
        target_landmark_count=8,
        min_edge_landmarks=5,
    )
    assert len(result) == 7


def test_registration_request_uses_configured_temperature(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    captured: dict[str, object] = {}

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args
        captured["config"] = kwargs["config"]
        return SimpleNamespace(parsed={"correspondences": [_correspondence(10, 10, 10, 10, "a")]})

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
            )
            if name == "langslice.vlm.config"
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

    assert isinstance(captured["config"], dict)
    assert captured["config"]["temperature"] == 0.2


def test_registration_request_can_force_code_execution(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    captured: dict[str, object] = {}

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args
        captured["config"] = kwargs["config"]
        return SimpleNamespace(parsed={"correspondences": [_correspondence(10, 10, 10, 10, "a")]})

    current_import_module = agents.importlib.import_module
    monkeypatch.setattr(agents, "_retry_generate", mock_generate)
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name, **kwargs: (
            SimpleNamespace(
                MODEL_NAME="gemini-3-flash-preview",
                CODE_EXECUTION_ENABLED=False,
                THINKING_LEVEL="LOW",
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
            )
            if name == "langslice.vlm.config"
            else current_import_module(name, **kwargs)
        ),
    )

    agents.estimate_registration_correspondences(
        Image.new("RGB", (120, 100), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        target_landmark_count=1,
        min_edge_landmarks=0,
        enable_code_execution=True,
    )

    assert isinstance(captured["config"], dict)
    assert captured["config"]["tools"] == [{"code_execution": {}}]


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
            )
            if name == "langslice.vlm.config"
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
            )
            if name == "langslice.vlm.config"
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
            if name == "langslice.vlm.config"
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
            if name == "langslice.vlm.config"
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


def test_multimodal_tool_loop_places_pairs_before_finish(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    responses = [
        SimpleNamespace(
            parsed={
                "tool_name": "place_point_pair",
                "tool_args": {
                    "label": "1",
                    "atlas_point_2d": [100, 120],
                    "slice_point_2d": [140, 160],
                    "feature_description": "outer contour notch",
                    "status": "found",
                },
            }
        ),
        SimpleNamespace(
            parsed={
                "tool_name": "place_point_pair",
                "tool_args": {
                    "label": "2",
                    "atlas_point_2d": [500, 520],
                    "slice_point_2d": [540, 560],
                    "feature_description": "ventricle corner",
                    "status": "found",
                },
            }
        ),
        SimpleNamespace(parsed={"tool_name": "finish", "tool_args": {}}),
    ]

    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (200, 120), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=2,
        min_edge_landmarks=1,
        workflow="multimodal_tool_loop",
    )

    assert [corr.label for corr in result] == ["1", "2"]
    assert result[0].rationale.startswith("status=found")


def test_multimodal_tool_loop_accepts_zoom_local_coordinates(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    responses = [
        SimpleNamespace(
            parsed={
                "tool_name": "view_zoom_pair",
                "tool_args": {
                    "zoom": 2.0,
                    "atlas_center_2d": [500, 500],
                    "slice_center_2d": [500, 500],
                },
            }
        ),
        SimpleNamespace(
            parsed={
                "tool_name": "place_point_pair",
                "tool_args": {
                    "label": "1",
                    "atlas_point_2d_local": [0, 0],
                    "slice_point_2d_local": [0, 0],
                    "feature_description": "upper-left corner of zoom window",
                    "status": "found",
                },
            }
        ),
        SimpleNamespace(parsed={"tool_name": "finish", "tool_args": {}}),
    ]

    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (456, 320), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=1,
        min_edge_landmarks=1,
        workflow="multimodal_tool_loop",
    )

    assert [corr.label for corr in result] == ["1"]
    assert result[0].atlas_normalized_yx == pytest.approx((250.78369905956112, 250.54945054945054))
    assert result[0].slice_normalized_yx == pytest.approx((250.78369905956112, 250.54945054945054))


def test_multimodal_tool_loop_uses_explicit_step_limit(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    responses = [
        SimpleNamespace(
            parsed={
                "tool_name": "place_point_pair",
                "tool_args": {
                    "label": "1",
                    "atlas_point_2d": [100, 120],
                    "slice_point_2d": [140, 160],
                    "feature_description": "outer contour notch",
                    "status": "found",
                },
            }
        )
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


def test_registration_request_uses_configured_thinking_level(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    captured: dict[str, object] = {}

    def mock_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        _ = args
        captured["config"] = kwargs["config"]
        return SimpleNamespace(parsed={"correspondences": [_correspondence(10, 10, 10, 10, "a")]})

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
            )
            if name == "langslice.vlm.config"
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

    assert isinstance(captured["config"], dict)
    assert captured["config"]["thinking_config"] == {"thinking_level": "LOW"}


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
