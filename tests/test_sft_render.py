"""Tests for models/langslice-gemma-4/training/sft/render.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from sft.dataset import load_examples
from sft.render import (
    AtlasMetaCache,
    RenderedExample,
    build_system_prompt,
    build_tools_schema,
    render_example,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sft_traces"


def _allen_mouse_25um_cached() -> bool:
    """Check whether allen_mouse_25um is downloaded under ~/.brainglobe/.

    BrainGlobe directory names include version suffix (e.g. allen_mouse_25um_v1.2),
    so we glob rather than check an exact path.
    """
    brainglobe = Path("~/.brainglobe").expanduser()
    if not brainglobe.is_dir():
        return False
    return any(brainglobe.glob("allen_mouse_25um*"))


_ATLAS_AVAILABLE = _allen_mouse_25um_cached()


def test_tools_schema_single_slice_has_fetch_atlas_and_submit_estimate() -> None:
    tools = build_tools_schema("single_slice")
    names = [t["function"]["name"] for t in tools]
    assert "fetch_atlas" in names
    assert "submit_estimate" in names
    assert "submit_group_estimate" not in names


def test_tools_schema_function_shape_matches_hf_format() -> None:
    tools = build_tools_schema("single_slice")
    fetch = next(t for t in tools if t["function"]["name"] == "fetch_atlas")
    assert fetch["type"] == "function"
    assert "description" in fetch["function"]
    assert "parameters" in fetch["function"]
    params = fetch["function"]["parameters"]
    assert params["type"] == "object"
    assert "positions_mm" in params["properties"]


def test_submit_estimate_schema_requires_position_and_reasoning() -> None:
    tools = build_tools_schema("single_slice")
    submit = next(t for t in tools if t["function"]["name"] == "submit_estimate")
    params = submit["function"]["parameters"]
    assert set(params["required"]) == {"position_mm", "reasoning"}
    assert "reasoning" in params["properties"]


@pytest.mark.skipif(
    not _ATLAS_AVAILABLE,
    reason="atlas not downloaded locally",
)
def test_atlas_meta_cache_returns_same_instance_for_same_atlas() -> None:
    cache = AtlasMetaCache()
    a = cache.get("allen_mouse_25um", "coronal")
    b = cache.get("allen_mouse_25um", "coronal")
    assert a is b  # identity, not just equality


@pytest.mark.skipif(
    not _ATLAS_AVAILABLE,
    reason="atlas not downloaded locally",
)
def test_build_system_prompt_single_slice_uses_production_prompt() -> None:
    cache = AtlasMetaCache()
    prompt = build_system_prompt(
        kind="single_slice",
        atlas_name="allen_mouse_25um",
        plane="coronal",
        atlas_meta_cache=cache,
    )
    # Production prompt mentions "AP" axis label for coronal
    assert "AP" in prompt
    assert "allen_mouse_25um" in prompt


def test_build_tools_schema_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown system_prompt_kind"):
        build_tools_schema("group")


def test_build_system_prompt_rejects_unknown_kind_without_atlas_load() -> None:
    # Must NOT load the atlas — fail fast on bad kind.
    cache = AtlasMetaCache()
    with pytest.raises(ValueError, match="unknown system_prompt_kind"):
        build_system_prompt(
            kind="group",
            atlas_name="allen_mouse_25um",
            plane="coronal",
            atlas_meta_cache=cache,
        )
    # Cache should be empty — no atlas loaded.
    assert cache._cache == {}  # type: ignore[attr-defined]  # private inspection in test


@pytest.fixture
def staged_single_slice(tmp_path: Path) -> Path:
    """Stage the single_slice fixture jsonl + dummy PNGs into tmp_path; return jsonl path."""
    src = FIXTURES / "single_slice_minimal.jsonl"
    dest = tmp_path / "single_slice_minimal.jsonl"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("query.png", "a3.png", "a5.png", "a7.png"):
        Image.new("RGB", (32, 32), color=(128, 128, 128)).save(tmp_path / name)
    return dest


@pytest.mark.skipif(
    not _ATLAS_AVAILABLE,
    reason="atlas not downloaded locally",
)
def test_render_single_slice_minimal(staged_single_slice: Path) -> None:
    examples = load_examples(staged_single_slice)
    cache = AtlasMetaCache()
    rendered = render_example(examples[0], atlas_meta_cache=cache)

    assert isinstance(rendered, RenderedExample)
    msgs = rendered.messages
    # system + user + (assistant + tool) + (assistant final submit)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # User turn has 1 image + 1 text content block
    user_content = msgs[1]["content"]
    assert sum(1 for c in user_content if c.get("type") == "image") == 1
    assert sum(1 for c in user_content if c.get("type") == "text") == 1
    # Find the first assistant turn — must carry tool_calls with fetch_atlas
    first_assistant = next(m for m in msgs[2:] if m["role"] == "assistant")
    assert first_assistant["tool_calls"][0]["function"]["name"] == "fetch_atlas"
    args = json.loads(first_assistant["tool_calls"][0]["function"]["arguments"])
    assert args["positions_mm"] == [3.0, 5.0, 7.0]
    # Tool message must have matching tool_call_id and 3 images + 1 text
    first_tool = msgs[msgs.index(first_assistant) + 1]
    assert first_tool["role"] == "tool"
    assert first_tool["tool_call_id"] == first_assistant["tool_calls"][0]["id"]
    assert sum(1 for c in first_tool["content"] if c.get("type") == "image") == 3
    # Final assistant turn has the submit
    final_assistant = msgs[-1]
    assert final_assistant["role"] == "assistant"
    fn = final_assistant["tool_calls"][0]["function"]
    assert fn["name"] == "submit_estimate"
    submit_args = json.loads(fn["arguments"])
    assert submit_args["position_mm"] == pytest.approx(5.2)
    assert submit_args["reasoning"]


@pytest.mark.skipif(
    not _ATLAS_AVAILABLE,
    reason="atlas not downloaded locally",
)
def test_render_unique_tool_call_ids(staged_single_slice: Path) -> None:
    """Every assistant tool_call.id must be unique within the trace."""
    rendered = render_example(
        load_examples(staged_single_slice)[0], atlas_meta_cache=AtlasMetaCache()
    )
    ids = []
    for m in rendered.messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                ids.append(tc["id"])
    assert len(ids) == len(set(ids))
