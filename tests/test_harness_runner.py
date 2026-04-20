import asyncio

import pytest
from PIL import Image

from langslice.harness.estimation.group import build_group_agent
from langslice.harness.estimation.runner import run_single_slice_session
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
