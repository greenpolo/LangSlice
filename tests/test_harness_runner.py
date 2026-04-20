import asyncio

import pytest
from PIL import Image

from langslice.harness.estimation._types import MultiSliceResult, PositionResult
from langslice.harness.estimation.group import build_group_agent
from langslice.harness.estimation.runner import (
    run_group_session,
    run_single_slice_session,
)
from langslice.harness.estimation.session import (
    ARTIFACT_TARGET,
    build_initial_state,
)
from langslice.harness.estimation.single_slice import build_single_slice_agent
from langslice.harness.estimation.validators import gate_submit_tool


def test_initial_state_single_slice():
    state = build_initial_state(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        pos_lo=0.0, pos_hi=13.2,
        n_slices=1, interval_mm=0.0, thickness_um=50,
        max_iterations=20,
    )
    assert state["atlas"] == "allen_mouse_25um"
    assert state["plane"] == "coronal"
    assert state["axis_label"] == "AP"
    assert state["n_slices"] == 1
    assert state["fetched_positions"] == []
    assert state["saw_broad_sweep"] is False
    assert state["saw_narrow_sweep"] is False
    assert state["submit_attempts"] == 0
    assert state["result"] is None


def test_artifact_target_constant():
    assert ARTIFACT_TARGET == "target"


def test_build_single_slice_agent_registers_four_tools():
    agent = build_single_slice_agent(
        atlas_name="allen_mouse_25um", plane="coronal",
        species="mouse", pos_lo=0.0, pos_hi=13.2,
        model="gemini-3-flash-preview",
    )
    tool_names = {getattr(t, "__name__", None) or getattr(t, "name", None) for t in agent.tools}
    assert "fetch_atlas" in tool_names
    assert "zoom" in tool_names
    assert "side_by_side" in tool_names
    assert "submit_estimate" in tool_names
    assert agent.instruction  # non-empty prompt


def test_build_group_agent_registers_four_tools_and_callback():
    agent = build_group_agent(
        atlas_name="allen_mouse_25um", plane="coronal",
        species="mouse", pos_lo=0.0, pos_hi=13.2,
        n_slices=4, interval_mm=0.200, thickness_um=50,
        model="gemini-3-flash-preview",
    )
    assert agent.name == "group_position_estimator"
    assert len(agent.tools) == 4
    tool_names = {getattr(t, "__name__", None) or getattr(t, "name", None) for t in agent.tools}
    assert tool_names == {"fetch_atlas", "zoom", "side_by_side", "submit_group_estimate"}
    assert agent.before_tool_callback is gate_submit_tool
    assert agent.instruction  # non-empty prompt


def test_run_single_slice_session_happy_path(monkeypatch):
    """Runner completes when the model follows broad -> narrow -> submit."""
    try:
        from tests.fakes import install_fake_adk_model_scripted_submit
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")
    try:
        install_fake_adk_model_scripted_submit(monkeypatch)
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")

    img = Image.new("RGB", (456, 320), 128)
    result = asyncio.run(
        run_single_slice_session(
            image=img,
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
        )
    )
    assert result.position_mm is not None
    assert 0.0 <= result.position_mm <= 13.2


def test_run_group_session_drives_fake_to_completion(monkeypatch):
    """Runner completes when the group model follows broad -> narrow -> submit_group."""
    try:
        from tests.fakes import install_fake_adk_group_model_scripted_submit
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")
    try:
        install_fake_adk_group_model_scripted_submit(
            monkeypatch, n_slices=4, interval_mm=0.2,
        )
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")

    # Blank images keep CLAHE a no-op and avoid any OpenCV dependency in tests.
    images = [Image.new("L", (256, 256), color=128) for _ in range(4)]
    result = asyncio.run(
        run_group_session(
            images=images,
            atlas_name="allen_mouse_25um",
            interval_mm=0.2,
            thickness_um=50,
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=12,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert isinstance(result, MultiSliceResult)
    assert len(result.positions) == 4
    for pos in result.positions:
        assert isinstance(pos, PositionResult)
        assert 0.0 <= pos.position_mm <= 15.0  # loose sanity on atlas range
    # Monotonic, matching the scripted centre of 6.0 mm with 0.2 mm spacing.
    position_values = [p.position_mm for p in result.positions]
    assert position_values == sorted(position_values)
    assert all(abs(p - 6.0) <= 1.0 for p in position_values)
    # Reasoning text is the one propagated from the scripted fake — confirms
    # the runner reached submit rather than falling back to midpoint.
    lowered = result.group_reasoning.lower()
    assert "broad" in lowered and "narrow" in lowered
    assert "fell back" not in lowered
