"""Tests for models/langslice-gemma-4/training/sft/render.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from sft.render import (
    AtlasMetaCache,
    build_system_prompt,
    build_tools_schema,
)


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
