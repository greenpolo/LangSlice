"""The default workflow has no acceptance tool or inspection gate."""

from langslice.agent.plugins import WorkingSetImages
from langslice.agent.session import build_plugins
from tests.linear_tool_helpers import box as _box
from tests.linear_tool_helpers import single_transform
from tests.linear_tool_helpers import tool_named as _tool


def test_baseline_has_no_acceptance_tool_or_inspection_gate(tmp_path):
    state, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"])
    assert "accept_views" not in box.names
    single_transform(_tool(box, "interactive_transform"))("s0.png", 0.5, 1.0, 1.0, 0.0, 0.0)
    # No model-delivery boundary has been announced: baseline submit needs a
    # transform, not a separate acceptance or final-image inspection action.
    assert _tool(box, "submit")("done", [], [])["status"] == "ok"
    assert _tool(box, "undo")()["status"] == "ok"
    assert not state.submitted


def test_session_uses_the_working_set_filter_first(monkeypatch):
    monkeypatch.delenv("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR", raising=False)
    monkeypatch.delenv("LANGSLICE_ADK_MODEL_CALL_DELAY_S", raising=False)
    plugins = build_plugins("test")
    assert isinstance(plugins[0]._custom_filter, WorkingSetImages)
