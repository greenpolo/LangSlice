from __future__ import annotations

import json
from pathlib import Path

from langslice_harness.harness.estimation.trace_collection import (
    AgentTraceRecorder,
    TracePricing,
    categorize_trace,
    collect_manifest_traces,
    estimate_cost_usd,
    export_sft_messages,
    load_manifest,
)


def test_load_manifest_accepts_single_and_group_records(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join([
            json.dumps({
                "id": "single-1",
                "kind": "single",
                "image": "slice.png",
                "atlas": "allen_mouse_25um",
                "plane": "coronal",
                "position_mm": 6.2,
            }),
            json.dumps({
                "id": "group-1",
                "kind": "group",
                "images": ["a.png", "b.png"],
                "atlas": "allen_mouse_25um",
                "plane": "coronal",
                "positions_mm": [6.0, 6.2],
                "interval_um": 200,
                "thickness_um": 50,
            }),
        ]),
        encoding="utf-8",
    )

    rows = load_manifest(manifest)

    assert [row.id for row in rows] == ["single-1", "group-1"]
    assert rows[0].truth_positions_mm == [6.2]
    assert rows[1].truth_positions_mm == [6.0, 6.2]
    assert rows[1].interval_um == 200


def test_estimate_cost_counts_thinking_as_output():
    pricing = TracePricing(input_per_million=2.0, output_per_million=12.0)
    usage = {
        "prompt_token_count": 1_000_000,
        "candidates_token_count": 100_000,
        "thoughts_token_count": 50_000,
    }

    cost = estimate_cost_usd(usage, pricing)

    assert cost == 3.8


def test_categorize_trace_marks_clean_and_hard_negative():
    clean = categorize_trace(
        truth_positions=[6.0],
        submitted_positions=[6.05],
        fetched_positions=[2.0, 4.0, 6.0, 5.8, 6.0, 6.2],
        tolerance_mm=0.2,
        turn_count=4,
        median_turn_count=6,
        had_broad_restart=False,
        submit_rejections=0,
        fallback=False,
    )
    hard = categorize_trace(
        truth_positions=[6.0],
        submitted_positions=[6.05],
        fetched_positions=[1.0, 3.0, 11.0, 5.8, 6.0, 6.2],
        tolerance_mm=0.2,
        turn_count=8,
        median_turn_count=6,
        had_broad_restart=True,
        submit_rejections=1,
        fallback=False,
    )

    assert clean["accepted"] is True
    assert clean["label"] == "clean_success"
    assert hard["accepted"] is True
    assert hard["label"] == "hard_negative_recovered"


def test_export_sft_messages_excludes_teacher_thought_summaries():
    raw_events = [
        {"role": "user", "parts": [{"kind": "text", "text": "Target"}]},
        {
            "role": "assistant",
            "visible_text": "Landmarks: corpus callosum thick.",
            "teacher_thought_summaries": ["Hidden teacher summary"],
            "tool_calls": [{"name": "fetch_atlas", "args": {"positions_mm": [4, 6, 8]}}],
        },
        {"role": "tool", "name": "fetch_atlas", "response": {"status": "ok"}},
    ]

    messages = export_sft_messages(raw_events)

    rendered = json.dumps(messages)
    assert "corpus callosum thick" in rendered
    assert "fetch_atlas" in rendered
    assert "Hidden teacher summary" not in rendered


def test_export_sft_messages_can_include_rationale_summaries_and_user_images():
    raw_events = [
        {
            "role": "assistant",
            "visible_text": "Calling a broad sweep.",
            "teacher_thought_summaries": ["The target looks mid-coronal."],
            "tool_calls": [{"name": "fetch_atlas", "args": {"positions_mm": [4, 6, 8]}}],
        },
        {
            "role": "tool",
            "name": "fetch_atlas",
            "response": {"status": "ok", "description": "Fetched atlas"},
            "artifacts": [
                {
                    "type": "image_path",
                    "path": "tool_images/fetch_atlas_001_img01.jpg",
                    "mime_type": "image/jpeg",
                }
            ],
        },
    ]
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Determine this slice position."},
            {"type": "image_path", "path": "target_images/target_001.jpg"},
        ],
    }

    messages = export_sft_messages(
        raw_events,
        user_message=user_message,
        include_rationale_summaries=True,
    )

    rendered = json.dumps(messages)
    assert messages[0] == user_message
    assert "Rationale summary: The target looks mid-coronal." in rendered
    assert "Calling a broad sweep." in rendered
    assert "tool_images/fetch_atlas_001_img01.jpg" in rendered


def test_agent_trace_recorder_captures_visible_text_thoughts_and_tool_calls():
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    recorder = AgentTraceRecorder()
    thought = types.Part.from_text(text="I compared the target to the broad sweep.")
    thought.thought = True
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                thought,
                types.Part.from_text(text="Landmarks: corpus callosum thick."),
                types.Part.from_function_call(
                    name="fetch_atlas", args={"positions_mm": [4.0, 6.0, 8.0]}
                ),
            ],
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=3,
            total_token_count=18,
        ),
    )

    recorder.record_model_response(response)

    event = recorder.raw_events[0]
    assert event["role"] == "assistant"
    assert event["visible_text"] == "Landmarks: corpus callosum thick."
    assert event["teacher_thought_summaries"] == [
        "I compared the target to the broad sweep."
    ]
    assert event["tool_calls"] == [
        {"name": "fetch_atlas", "args": {"positions_mm": [4.0, 6.0, 8.0]}}
    ]
    assert recorder.usage_totals["thoughts_token_count"] == 3


def test_agent_trace_recorder_persists_multimodal_tool_images(tmp_path: Path):
    from google.genai import types

    recorder = AgentTraceRecorder(tool_artifact_dir=tmp_path / "tool_images")
    image_part = types.Part.from_bytes(
        mime_type="image/jpeg",
        data=b"fake-jpeg-bytes",
    )

    recorder.record_tool_result(
        tool_name="fetch_atlas",
        tool_args={"positions_mm": [4.0]},
        result={
            "status": "ok",
            "positions_mm": [4.0],
            "_langslice_multimodal_parts": [
                types.Part.from_text(text="Atlas at 4.00 mm:"),
                image_part,
            ],
        },
    )

    event = recorder.raw_events[0]
    artifacts = event["artifacts"]
    assert artifacts == [
        {
            "type": "image_path",
            "path": "tool_images/fetch_atlas_001_img01.jpg",
            "mime_type": "image/jpeg",
        }
    ]
    saved = tmp_path / "tool_images" / "fetch_atlas_001_img01.jpg"
    assert saved.read_bytes() == b"fake-jpeg-bytes"


def test_agent_trace_recorder_captures_tool_results_without_private_parts():
    recorder = AgentTraceRecorder()

    recorder.record_tool_result(
        tool_name="fetch_atlas",
        tool_args={"positions_mm": [4.0, 6.0, 8.0]},
        result={
            "status": "ok",
            "positions_mm": [4.0, 6.0, 8.0],
            "_langslice_multimodal_parts": ["private"],
        },
    )

    event = recorder.raw_events[0]
    assert event["role"] == "tool"
    assert event["name"] == "fetch_atlas"
    assert "_langslice_multimodal_parts" not in event["response"]


def test_collect_manifest_traces_writes_raw_sft_and_summary(tmp_path: Path):
    from PIL import Image

    from langslice_harness.harness.estimation._types import PositionResult

    image_path = tmp_path / "slice.png"
    Image.new("RGB", (24, 24), color=128).save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "id": "single-1",
            "kind": "single",
            "image": str(image_path),
            "atlas": "allen_mouse_25um",
            "plane": "coronal",
            "position_mm": 6.0,
            "modality": "nissl",
        }),
        encoding="utf-8",
    )
    out_dir = tmp_path / "traces"

    async def fake_single(**kwargs):
        recorder = kwargs["trace_recorder"]
        recorder.raw_events.append({
            "role": "assistant",
            "visible_text": "Landmarks: corpus callosum thick.",
            "teacher_thought_summaries": ["teacher-only"],
            "tool_calls": [
                {"name": "submit_estimate", "args": {"position_mm": 6.05}}
            ],
        })
        recorder.usage_totals.update({
            "prompt_token_count": 100,
            "candidates_token_count": 10,
            "thoughts_token_count": 5,
        })
        return PositionResult(position_mm=6.05, reasoning="fake")

    collect_manifest_traces(
        manifest_path=manifest,
        out_dir=out_dir,
        model="gemini-3.1-pro-preview",
        single_runner=fake_single,
    )

    raw = json.loads((out_dir / "runs" / "single-1" / "raw_trace.json").read_text())
    sft = json.loads((out_dir / "runs" / "single-1" / "sft_trace.json").read_text())
    sft_deployment = json.loads(
        (out_dir / "runs" / "single-1" / "sft_trace_deployment.json").read_text()
    )
    sft_rationale = json.loads(
        (out_dir / "runs" / "single-1" / "sft_trace_rationale.json").read_text()
    )
    results = [
        json.loads(line)
        for line in (out_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((out_dir / "trace_summary.json").read_text())

    assert raw["manifest"]["id"] == "single-1"
    assert "teacher-only" in json.dumps(raw)
    assert "teacher-only" not in json.dumps(sft)
    assert "target_images/target_001.jpg" in json.dumps(sft_deployment)
    assert "teacher-only" not in json.dumps(sft_deployment)
    assert "Rationale summary: teacher-only" in json.dumps(sft_rationale)
    assert (out_dir / "runs" / "single-1" / "target_images" / "target_001.jpg").exists()
    assert raw["sft_quality"]["accepted_for_sft"] is True
    assert results[0]["category"]["accepted"] is True
    assert results[0]["sft_quality"]["accepted_for_sft"] is True
    assert summary["total_runs"] == 1
    assert summary["accepted_runs"] == 1


def test_runner_sessions_accept_trace_collection_kwargs():
    import inspect

    from langslice_harness.harness.estimation.runner import (
        run_group_session,
        run_single_slice_session,
    )

    single_params = inspect.signature(run_single_slice_session).parameters
    group_params = inspect.signature(run_group_session).parameters

    assert "trace_recorder" in single_params
    assert "include_thought_summaries" in single_params
    assert "trace_recorder" in group_params
    assert "include_thought_summaries" in group_params
