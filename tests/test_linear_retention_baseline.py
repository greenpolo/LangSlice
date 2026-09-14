"""Retiring the acceptance experiment must leave the established workflow intact."""

import pytest

from langslice.adk.plugins import ToolMediaDeliveryPlugin, WorkingSetImages
from langslice.linear.session import build_plugins
from tests.test_linear_toolbox import _box, _tool


def test_completion_configuration_fails_instead_of_silently_changing_policy(tmp_path):
    with pytest.raises(ValueError, match="completion retirement was removed"):
        _box(tmp_path, image_retention="completion")


@pytest.mark.parametrize("policy", [None, "legacy"])
def test_baseline_has_no_acceptance_tool_or_inspection_gate(tmp_path, policy):
    kwargs = {} if policy is None else {"image_retention": policy}
    state, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"], **kwargs)
    assert "accept_views" not in box.names
    _tool(box, "adjust_transform")("s0.png", 0.0, 1.0, 1.0, 0.0, 0.0)
    # No model-delivery boundary has been announced: baseline submit needs a
    # transform, not a separate acceptance or final-image inspection action.
    assert _tool(box, "validate")([])["would_submit"]
    assert _tool(box, "submit")("done", [], [])["status"] == "ok"
    assert _tool(box, "undo")()["status"] == "ok"
    assert not state.submitted


def test_session_uses_working_set_then_delivery_tracking(monkeypatch):
    monkeypatch.delenv("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR", raising=False)
    monkeypatch.delenv("LANGSLICE_ADK_MODEL_CALL_DELAY_S", raising=False)
    plugins = build_plugins("test", tool_media_delivered=lambda _ids: None)
    assert isinstance(plugins[0]._custom_filter, WorkingSetImages)
    assert isinstance(plugins[1], ToolMediaDeliveryPlugin)
