"""Checks for registration-agent correspondence constraints."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import langslice.registration.agents as agents


class _DummyModels:
    def generate_content(self, *, model: str, contents: Any, config: Any) -> object:
        _ = model, contents, config
        return object()


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
        lambda img: SimpleNamespace(image=img, output_size=img.size),
    )

    vlm_config = SimpleNamespace(
        MODEL_NAME="test-model",
        CODE_EXECUTION_ENABLED=False,
        REGISTRATION_THINKING_BUDGET=8192,
        TEMPERATURE=0.5,
        count_tokens_enabled=lambda: False,
        get_client=lambda: _DummyClient(),
    )
    real_import_module = agents.importlib.import_module
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name: vlm_config if name == "langslice.vlm.config" else real_import_module(name),
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
        lambda name: (
            SimpleNamespace(
                MODEL_NAME="test-model",
                CODE_EXECUTION_ENABLED=False,
                REGISTRATION_THINKING_BUDGET=8192,
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
            )
            if name == "langslice.vlm.config"
            else current_import_module(name)
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


def test_image_gen_two_shot_pairs_numbered_landmarks(monkeypatch: Any) -> None:
    _patch_common(monkeypatch)

    responses = [
        SimpleNamespace(
            parsed={
                "landmarks": [
                    {
                        "label": str(index),
                        "atlas_point_2d": [50 * index, 60 * index],
                        "status": "found",
                    }
                    for index in range(1, 5)
                ]
            }
        ),
        SimpleNamespace(
            text=json.dumps(
                {
                    "landmarks": [
                        {
                            "label": str(index),
                            "slice_point_2d": [70 * index, 80 * index],
                            "status": "found",
                        }
                        for index in range(1, 5)
                    ]
                }
            )
        ),
    ]

    current_import_module = agents.importlib.import_module
    monkeypatch.setattr(
        agents.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(
                MODEL_NAME="gemini-3-pro-image-preview",
                CODE_EXECUTION_ENABLED=False,
                REGISTRATION_THINKING_BUDGET=8192,
                TEMPERATURE=0.2,
                count_tokens_enabled=lambda: False,
                get_client=lambda: _DummyClient(),
                supports_structured_image_output=lambda model: (
                    model == "gemini-3-pro-image-preview"
                ),
            )
            if name == "langslice.vlm.config"
            else current_import_module(name)
        ),
    )
    monkeypatch.setattr(agents, "_retry_generate", lambda *args, **kwargs: responses.pop(0))

    result = agents.estimate_registration_correspondences(
        Image.new("RGB", (200, 120), (0, 0, 0)),
        atlas_name="allen_mouse_25um",
        position_mm=2.0,
        target_landmark_count=4,
        min_edge_landmarks=2,
        workflow="image_gen_two_shot",
    )

    assert [corr.label for corr in result] == ["1", "2", "3", "4"]
    assert result[0].atlas_normalized_yx == (50.0, 60.0)
    assert result[0].slice_normalized_yx == (70.0, 80.0)


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
