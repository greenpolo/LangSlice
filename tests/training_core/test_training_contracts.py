from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from langslice_training.contracts import (
    CONTRACT_SCHEMA_VERSION,
    SFTExample,
    export_contract_json_schema,
    validate_sft_example,
)


def _minimal_valid_row() -> dict[str, object]:
    return {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "v1.2",
        "plane": "coronal",
        "subject_id": "mouse-001",
        "system_prompt_kind": "single_slice",
        "query_image_paths": ["images/query.png"],
        "user_prompt_text": "Estimate the position of this slice.",
        "trace": [
            {
                "tool_call": {
                    "name": "fetch_atlas",
                    "args": {"position_mm": 2.25},
                },
                "tool_result": {
                    "image_paths": ["images/atlas.png"],
                    "text": "Fetched atlas image.",
                },
            },
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {"position_mm": 2.30, "reasoning": "Best match."},
                }
            },
        ],
        "quality": {"label": "good"},
        "gemini_reasoning": "Internal chain-of-thought summary placeholder.",
    }


def test_validate_sft_example_accepts_minimal_row() -> None:
    row = _minimal_valid_row()
    parsed = validate_sft_example(row)
    assert isinstance(parsed, SFTExample)
    assert parsed.plane == "coronal"
    assert parsed.trace[-1].submit is not None
    assert CONTRACT_SCHEMA_VERSION


def test_validate_sft_example_accepts_json_string() -> None:
    row = _minimal_valid_row()
    parsed = validate_sft_example(json.dumps(row))
    assert parsed.subject_id == "mouse-001"


def test_validate_sft_example_rejects_wrong_plane() -> None:
    row = _minimal_valid_row()
    row["plane"] = "axial"
    with pytest.raises(ValidationError):
        validate_sft_example(row)


def test_validate_sft_example_rejects_missing_query_image() -> None:
    row = _minimal_valid_row()
    row["query_image_paths"] = []
    with pytest.raises(ValidationError):
        validate_sft_example(row)


def test_validate_sft_example_rejects_missing_final_submit() -> None:
    row = _minimal_valid_row()
    row["trace"] = row["trace"][:-1]
    with pytest.raises(ValidationError):
        validate_sft_example(row)


def test_validate_sft_example_rejects_nonnumeric_position() -> None:
    row = _minimal_valid_row()
    trace = row["trace"]
    assert isinstance(trace, list)
    submit = trace[-1]["submit"]
    assert isinstance(submit, dict)
    args = submit["args"]
    assert isinstance(args, dict)
    args["position_mm"] = "2.3mm"
    with pytest.raises(ValidationError):
        validate_sft_example(row)


def test_export_contract_json_schema_includes_sft_example() -> None:
    schema = export_contract_json_schema()
    assert "$defs" in schema
    assert "SFTExample" in schema["$defs"]

