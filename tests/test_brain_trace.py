"""Full-content JSONL traces of whole-brain agent sessions.

No live model calls: a scripted fake ``BaseLlm`` is swapped in through
``LLMRegistry.new_llm`` (same seam as tests/fakes.py), so the trace under test
is written from a session whose every turn is known in advance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice import cli
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import ingest
from langslice.linear.whole_brain.state import BrainConfig, StackState
from langslice.linear.whole_brain.survey import run_survey_session
from langslice.linear.whole_brain.trace import TRACE_DIR_ENV, open_trace

_SUBMIT_ARGS: dict[str, Any] = {
    "axis_directions": {"ap": "anterior_to_posterior"},
    "interval_breaks": [2],
    "notes": ["gap after section 1"],
    "clean": False,
    "summary": "Looked at two sections; one interval break.",
}
_VIEWED = ["slice_1.png", "slice_2.png"]


class _ScriptedLlm(BaseLlm):
    """Turn 1: reason in prose and call ``view_slices``. Turn 2: submit."""

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        responses = sum(
            1
            for content in (llm_request.contents or [])
            for part in (content.parts or [])
            if getattr(part, "function_response", None) is not None
        )
        if responses == 0:
            parts = [
                types.Part.from_text(text="Zooming in on the first two sections."),
                types.Part.from_function_call(
                    name="view_slices", args={"slice_ids": _VIEWED}
                ),
            ]
        else:
            parts = [
                types.Part.from_function_call(name="submit_survey", args=_SUBMIT_ARGS)
            ]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            turn_complete=True,
        )


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


def _ingested(folder: Path) -> tuple[StackState, Any]:
    for index in range(1, 4):
        Image.new("RGB", (64, 48), (30, 30, 30)).save(folder / f"slice_{index}.png")
    ctx = build_context(
        BrainConfig(image_folder=str(folder)),
        emit=lambda _msg: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    return state, ctx


def _run_survey(folder: Path, monkeypatch) -> None:
    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry, "new_llm", staticmethod(lambda model: _ScriptedLlm(model=model))
    )
    state, ctx = _ingested(folder)
    asyncio.run(
        run_survey_session(
            state=state, ctx=ctx, species="mouse", pos_lo=0.0, pos_hi=13.2
        )
    )


def _records(trace_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(trace_dir.glob("*.jsonl"))
    assert len(paths) == 1, paths
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def _of_kind(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [record for record in records if record["kind"] == kind]


def test_trace_captures_a_whole_survey_session(tmp_path: Path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv(TRACE_DIR_ENV, str(trace_dir))
    _run_survey(tmp_path, monkeypatch)

    records = _records(trace_dir)
    assert [record["kind"] for record in records][:2] == ["session", "seed"]
    assert all("at" in record for record in records)

    # --- the session header carries the system instruction, once -------------
    session = records[0]
    assert session["run_label"] == "whole_brain_survey"
    assert session["agent"] == "whole_brain_survey"
    assert session["model"] == "gemini-3-flash-preview"
    assert "allen_mouse_25um" in session["system_instruction"]
    assert "submit_survey" in session["system_instruction"]
    assert len(_of_kind(records, "session")) == 1

    # --- the seed message: text verbatim, images as labelled descriptors -----
    seed_parts = records[1]["parts"]
    texts = [part["text"] for part in seed_parts if part["kind"] == "text"]
    assert any("3 sections of the stack follow" in text for text in texts)
    assert "0: slice_1.png" in texts
    images = [part for part in seed_parts if part["kind"] == "inline_data"]
    assert len(images) == 3
    assert images[0]["label"] == "0: slice_1.png"
    assert images[0]["mime_type"] == "image/jpeg"
    assert images[0]["byte_count"] > 0
    assert images[0]["dimensions"] == [64, 48]
    assert "data" not in images[0]

    # --- model turns: full text and full function-call arguments ------------
    model_turns = _of_kind(records, "model")
    assert model_turns[0]["text"] == "Zooming in on the first two sections."
    assert model_turns[0]["function_calls"] == [
        {"name": "view_slices", "args": {"slice_ids": _VIEWED}}
    ]
    submit = model_turns[-1]["function_calls"][0]
    assert submit["name"] == "submit_survey"
    assert submit["args"] == _SUBMIT_ARGS  # verbatim, values and all

    # --- tool results: the whole payload, plus media descriptors ------------
    results = _of_kind(records, "tool_result")
    view = next(r for r in results if r["name"] == "view_slices")
    assert view["response"]["status"] == "ok"
    assert view["response"]["slice_ids"] == _VIEWED
    assert "rendered with any flips already applied" in view["response"]["description"]
    assert "images" not in view["response"]  # media key never lands in the JSON
    assert [media["mime_type"] for media in view["media"]] == ["image/jpeg"] * 2
    assert all(media["byte_count"] > 0 for media in view["media"])
    assert all("data" not in media for media in view["media"])
    submit_result = next(r for r in results if r["name"] == "submit_survey")
    assert submit_result["response"]["status"] == "ok"

    # --- and the closing summary --------------------------------------------
    assert _of_kind(records, "summary") == [
        {
            "kind": "summary",
            "tool_calls": 2,
            "turns": 1,  # both model turns arrived on one run_async pass
            "submitted": True,
            "at": records[-1]["at"],
        }
    ]


def test_trace_carries_no_image_bytes(tmp_path: Path, monkeypatch):
    """Descriptors only — neither stringified bytes nor base64 JPEG payload."""
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv(TRACE_DIR_ENV, str(trace_dir))
    _run_survey(tmp_path, monkeypatch)

    text = next(iter(trace_dir.glob("*.jsonl"))).read_text()
    assert "\\xff\\xd8" not in text  # repr of a JPEG's magic bytes
    assert "/9j/" not in text  # base64 of the same
    assert '"byte_count": 674' in text  # ... but the size is on record


def test_nudges_that_were_sent_are_recorded(tmp_path: Path, monkeypatch):
    class _SilentLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream, llm_request
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="Looks fine.")]
                ),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    trace_dir = tmp_path / "traces"
    monkeypatch.setenv(TRACE_DIR_ENV, str(trace_dir))
    monkeypatch.setattr(
        LLMRegistry, "new_llm", staticmethod(lambda model: _SilentLlm(model=model))
    )
    state, ctx = _ingested(tmp_path)
    asyncio.run(
        run_survey_session(
            state=state,
            ctx=ctx,
            species="mouse",
            pos_lo=0.0,
            pos_hi=13.2,
            max_iterations=2,
        )
    )

    records = _records(trace_dir)
    # Two turns, so exactly one nudge was sent — the one built at the end of
    # the last turn never left the driver and is not on record.
    nudges = _of_kind(records, "nudge")
    assert len(nudges) == 1
    assert nudges[0]["turn"] == 2
    assert "did not call a tool" in nudges[0]["text"]
    assert _of_kind(records, "model")[0]["text"] == "Looks fine."
    assert _of_kind(records, "summary")[0]["submitted"] is False


def test_no_trace_dir_means_no_recorder_and_no_files(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(TRACE_DIR_ENV, raising=False)
    assert open_trace("whole_brain_survey", agent=object()) is None

    _run_survey(tmp_path, monkeypatch)
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_sessions_sharing_a_run_label_get_distinct_files(tmp_path: Path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv(TRACE_DIR_ENV, str(trace_dir))
    _run_survey(tmp_path, monkeypatch)
    _run_survey(tmp_path, monkeypatch)

    paths = sorted(trace_dir.glob("*.jsonl"))
    assert len(paths) == 2
    assert all(path.name.startswith("whole_brain_survey_") for path in paths)


def test_estimate_brain_trace_dir_flag(tmp_path: Path):
    parsed: argparse.Namespace = cli._build_parser().parse_args(
        ["linear", "estimate-brain", str(tmp_path), "--trace-dir", str(tmp_path / "t")]
    )
    assert parsed.trace_dir == str(tmp_path / "t")
    assert cli._build_parser().parse_args(
        ["linear", "estimate-brain", str(tmp_path)]
    ).trace_dir is None
