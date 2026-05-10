"""Unit tests for tools.expert_iteration.trace_format.

Uses a small mocked ADK event log fixture and on-disk dummy atlas image
artifacts; does NOT require the model or live atlas data. We do touch
``langslice_harness.atlas.core.load_atlas`` indirectly (via plane_tolerance_mm
+ get_in_plane_long_edge), so the BrainGlobe atlas registry must resolve
``allen_mouse_25um`` — that's a project invariant; the same atlas is used
across the SFT/RLVR test suites.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(127, 64, 32)).save(path, format="JPEG")


def _build_minimal_events(run_dir: Path) -> list[dict]:
    """Mimic the AgentTraceRecorder raw_events shape for one rollout.

    Two assistant turns: one fetch_atlas call, then a submit_estimate call.
    The fetch turn is followed by a 'tool' event carrying the atlas image
    artifacts (paths relative to run_dir, mirroring the recorder's layout).
    """
    tool_artifact_dir = run_dir / "tool_artifacts"
    img1 = tool_artifact_dir / "fetch_atlas_001_img01.jpg"
    img2 = tool_artifact_dir / "fetch_atlas_001_img02.jpg"
    _write_dummy_image(img1)
    _write_dummy_image(img2)

    return [
        {
            "role": "assistant",
            "visible_text": "Let me fetch some atlas sections.",
            "teacher_thought_summaries": [],
            "tool_calls": [{
                "name": "fetch_atlas",
                "args": {"positions_mm": [4.0, 6.0]},
            }],
            "usage_metadata": {},
        },
        {
            "role": "tool",
            "name": "fetch_atlas",
            "args": {"positions_mm": [4.0, 6.0]},
            "response": {
                "positions_mm": [4.0, 6.0],
                "description": "Fetched 2 atlas sections at: 4.00, 6.00 mm",
                "status": "ok",
            },
            "artifacts": [
                {
                    "type": "image_path",
                    "path": "tool_artifacts/fetch_atlas_001_img01.jpg",
                    "mime_type": "image/jpeg",
                },
                {
                    "type": "image_path",
                    "path": "tool_artifacts/fetch_atlas_001_img02.jpg",
                    "mime_type": "image/jpeg",
                },
            ],
        },
        {
            "role": "assistant",
            "visible_text": "Submitting estimate.",
            "teacher_thought_summaries": [],
            "tool_calls": [{
                "name": "submit_estimate",
                "args": {
                    "position_mm": 5.0,
                    "reasoning": "Position lies between sections 4 and 6 mm.",
                },
            }],
            "usage_metadata": {},
        },
    ]


# ──────────────────────────────────────────────────────────────────────────
# convert_trace
# ──────────────────────────────────────────────────────────────────────────

def test_convert_trace_returns_alternating_steps_with_submit(tmp_path: Path) -> None:
    from tools.expert_iteration.trace_format import convert_trace

    run_dir = tmp_path / "rollout_xyz"
    run_dir.mkdir()
    events = _build_minimal_events(run_dir)
    out_dir = tmp_path / "iterative"
    out_dir.mkdir()

    trace = convert_trace(
        events,
        run_dir=run_dir,
        atlas_name="allen_mouse_25um",
        plane="coronal",
        out_dir=out_dir,
    )
    assert trace is not None
    # Expect exactly 2 steps: one (tool_call, tool_result) + one submit.
    assert len(trace) == 2
    assert "tool_call" in trace[0] and "tool_result" in trace[0]
    assert trace[0]["tool_call"]["name"] == "fetch_atlas"
    assert trace[0]["tool_call"]["args"]["positions_mm"] == [4.0, 6.0]
    assert len(trace[0]["tool_result"]["image_paths"]) == 2
    assert "submit" in trace[1]
    assert trace[1]["submit"]["name"] == "submit_estimate"
    assert trace[1]["submit"]["args"]["position_mm"] == 5.0
    assert "Position lies" in trace[1]["submit"]["args"]["reasoning"]


def test_convert_trace_returns_none_when_no_submit(tmp_path: Path) -> None:
    from tools.expert_iteration.trace_format import convert_trace

    run_dir = tmp_path / "rollout_xyz"
    run_dir.mkdir()
    events = _build_minimal_events(run_dir)
    # Strip the final submit turn.
    events = events[:-1]

    out_dir = tmp_path / "iterative"
    out_dir.mkdir()
    assert convert_trace(
        events,
        run_dir=run_dir,
        atlas_name="allen_mouse_25um",
        plane="coronal",
        out_dir=out_dir,
    ) is None


def test_convert_trace_stages_atlas_images_under_out_dir(tmp_path: Path) -> None:
    from tools.expert_iteration.trace_format import convert_trace

    run_dir = tmp_path / "rollout_xyz"
    run_dir.mkdir()
    events = _build_minimal_events(run_dir)
    out_dir = tmp_path / "iterative"
    out_dir.mkdir()

    trace = convert_trace(
        events,
        run_dir=run_dir,
        atlas_name="allen_mouse_25um",
        plane="coronal",
        out_dir=out_dir,
    )
    assert trace is not None
    # Atlas image paths in the trace are relative to out_dir; resolve them.
    for path_rel in trace[0]["tool_result"]["image_paths"]:
        full = (out_dir / path_rel).resolve()
        assert full.is_file(), f"missing staged atlas image: {full}"
        # Layout should match build_sft_corpus convention.
        assert "atlas" in path_rel and "coronal" in path_rel


# ──────────────────────────────────────────────────────────────────────────
# trim_trace + classify_quality
# ──────────────────────────────────────────────────────────────────────────

def test_trim_trace_caps_fetch_calls(tmp_path: Path) -> None:
    from tools.expert_iteration.trace_format import trim_trace

    # Five fetch turns + one submit; positions_mm progressively closer to 5.0.
    trace = []
    positions = [[1.0, 2.0], [3.0, 4.0], [5.5], [4.9, 5.1], [4.95, 5.05]]
    for pos in positions:
        trace.append({
            "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": pos}},
            "tool_result": {
                "image_paths": [f"atlas/p{i}.jpg" for i in range(len(pos))],
                "text": "ok",
            },
        })
    trace.append({
        "submit": {
            "name": "submit_estimate",
            "args": {"position_mm": 5.0, "reasoning": "..."},
        },
    })

    trimmed, meta = trim_trace(trace, max_fetch_calls=2, max_total_images=12)
    assert sum(1 for s in trimmed if "tool_call" in s) == 2
    assert "submit" in trimmed[-1]
    assert meta["original_num_fetch_calls"] == 5
    assert meta["kept_num_fetch_calls"] == 2


def test_classify_quality_marks_in_tolerance() -> None:
    from tools.expert_iteration.trace_format import classify_quality

    quality = classify_quality(
        model="litellm-proxy:langslice-ft",
        err=0.1,
        tol=0.2,
        rescue=0.9,
        sub_canonical_mm=5.0,
        events=[],
    )
    # err=0.1 == 0.5 * tol → "tight"
    assert quality["accuracy"] == "tight"
    assert quality["max_error_mm"] == 0.1
    assert quality["fetch_calls"] == 0


# ──────────────────────────────────────────────────────────────────────────
# build_row (end-to-end)
# ──────────────────────────────────────────────────────────────────────────

def test_build_row_produces_validated_row(tmp_path: Path) -> None:
    from tools.expert_iteration.trace_format import build_row
    sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))
    from sft.dataset import _validate_row  # type: ignore[import-not-found]

    run_dir = tmp_path / "rollout_xyz"
    run_dir.mkdir()
    events = _build_minimal_events(run_dir)
    out_dir = tmp_path / "iterative"
    out_dir.mkdir()

    # Stage a query image into out_dir/queries/<sid>.jpg manually.
    queries_dir = out_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    query_image = queries_dir / "test_section.jpg"
    _write_dummy_image(query_image)
    query_rel = "queries/test_section.jpg"

    row = build_row(
        section_id="test_section",
        subject_id="test_subject",
        atlas_name="allen_mouse_25um",
        plane="coronal",
        query_image_rel=query_rel,
        events=events,
        submitted_position_mm=5.0,
        sub_canonical_mm=5.0,
        abs_err_mm=0.1,
        model="litellm-proxy:langslice-ft",
        iteration_round=0,
        out_dir=out_dir,
        run_dir=run_dir,
        apply_clahe=False,
        max_fetch_calls=2,
        max_total_images=12,
    )
    assert row is not None
    # Expected audit-trail fields.
    assert row["quality"]["source"] == "expert_iteration"
    assert row["quality"]["iteration_round"] == 0
    assert row["bucket"] == 1
    assert row["plane"] == "coronal"
    assert row["query_image_paths"] == [query_rel]
    assert row["trace"][-1]["submit"]["args"]["position_mm"] == 5.0

    # Should pass the SFT trainer's row validator.
    _validate_row(row, lineno=1, root=out_dir)


def test_build_row_returns_none_for_traceless_events(tmp_path: Path) -> None:
    from tools.expert_iteration.trace_format import build_row

    run_dir = tmp_path / "rollout_xyz"
    run_dir.mkdir()
    out_dir = tmp_path / "iterative"
    out_dir.mkdir()
    queries_dir = out_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    _write_dummy_image(queries_dir / "test_section.jpg")

    # Events without a terminal submit_estimate.
    events = [
        {
            "role": "assistant",
            "visible_text": "thinking...",
            "teacher_thought_summaries": [],
            "tool_calls": [],
            "usage_metadata": {},
        },
    ]
    row = build_row(
        section_id="test_section",
        subject_id="test_subject",
        atlas_name="allen_mouse_25um",
        plane="coronal",
        query_image_rel="queries/test_section.jpg",
        events=events,
        submitted_position_mm=5.0,
        sub_canonical_mm=5.0,
        abs_err_mm=0.1,
        model="litellm-proxy:langslice-ft",
        iteration_round=0,
        out_dir=out_dir,
        run_dir=run_dir,
        apply_clahe=False,
    )
    assert row is None
