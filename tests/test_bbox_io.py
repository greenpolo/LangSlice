"""Tests for bbox_io: manifest schema, batch packing, mm-strip filter."""
from __future__ import annotations

import bbox_io


def test_mm_strip_passes_clean_caption():
    text = "The hippocampus appears as a thin crescent and broadens caudally."
    assert bbox_io.mm_strip_filter(text) == text


def test_mm_strip_rejects_mm_value():
    text = "At 5.4 mm the hippocampus appears..."
    assert bbox_io.mm_strip_filter(text) is None


def test_mm_strip_rejects_micron_value():
    text = "The dentate gyrus expands by 200 μm caudally."
    assert bbox_io.mm_strip_filter(text) is None


def test_mm_strip_rejects_section_index():
    text = "In Section 3, the region begins to broaden."
    assert bbox_io.mm_strip_filter(text) is None


def test_mm_strip_rejects_bregma():
    text = "Posterior to bregma, the region narrows."
    assert bbox_io.mm_strip_filter(text) is None


def test_length_gate_too_short():
    text = "It broadens."
    assert bbox_io.length_gate(text) is False


def test_length_gate_ok():
    text = (
        "The hippocampus appears as a thin crescent. It broadens caudally "
        "and reveals dentate gyrus subdivisions."
    )
    assert bbox_io.length_gate(text) is True


def test_length_gate_too_long():
    sentences = ["This is a sentence."] * 7
    text = " ".join(sentences)
    assert bbox_io.length_gate(text) is False


def test_pack_batch_request_whole_brain():
    example = {
        "id": "bbox_000001",
        "atlas": "allen_mouse_25um",
        "orientation": "coronal",
        "region": "Hippocampal Formation",
        "is_hemisphere": False,
        "section_image_paths": [],   # the test stubs PNG bytes via inject
        "bboxes": [
            {"left": [10, 20, 30, 40], "right": [60, 20, 80, 40]},
            {"left": [12, 22, 32, 42], "right": [62, 22, 82, 42]},
            {"left": [14, 24, 34, 44], "right": [64, 24, 84, 44]},
            {"left": [16, 26, 36, 46], "right": [66, 26, 86, 46]},
        ],
    }
    images_b64 = ["AAAA", "BBBB", "CCCC", "DDDD"]
    line = bbox_io.pack_batch_request(example, images_b64=images_b64)
    assert line["key"] == "bbox_000001"
    contents = line["request"]["contents"]
    assert len(contents) == 1
    parts = contents[0]["parts"]
    # 4 images + 1 text part.
    assert sum(1 for p in parts if "inline_data" in p) == 4
    text_parts = [p for p in parts if "text" in p]
    assert len(text_parts) == 1
    text = text_parts[0]["text"]
    assert "Hippocampal Formation" in text
    assert "left=[10, 20, 30, 40]" in text
    assert "right=[60, 20, 80, 40]" in text
    assert "anterior to posterior" in text
    assert "millimeter" in text.lower() or "Do NOT mention" in text
    config = line["request"]["generation_config"]
    assert config["thinking_config"]["thinking_level"] == "LOW"
    assert config["temperature"] == 1.0
    assert "max_output_tokens" not in config


def test_pack_batch_request_sagittal_single_bbox():
    example = {
        "id": "bbox_000002",
        "atlas": "allen_mouse_25um",
        "orientation": "sagittal",
        "region": "Dentate Gyrus",
        "is_hemisphere": True,
        "section_image_paths": [],
        "bboxes": [
            [220, 140, 410, 320],
            [218, 138, 408, 318],
            [216, 136, 406, 316],
            [214, 134, 404, 314],
        ],
    }
    images_b64 = ["A", "B", "C", "D"]
    line = bbox_io.pack_batch_request(example, images_b64=images_b64)
    text = next(p["text"] for p in line["request"]["contents"][0]["parts"] if "text" in p)
    assert "medial to lateral" in text
    assert "[220, 140, 410, 320]" in text
    assert "left=" not in text
    assert "right=" not in text
