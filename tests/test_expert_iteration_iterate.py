"""Mock-heavy tests for the multi-round iterate.py orchestration.

These tests don't actually call vLLM, the SFT trainer, or slicebench.
Instead we monkey-patch the phase-runner functions to produce a known
sequence of artifacts and observe:
  * phases run in the correct order
  * state.json is written after every phase boundary
  * resume from each phase picks up at the right next step
  * --rounds 1 reproduces wave-1 behavior (single round, no extra phases)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))


def _make_test_args(
    *,
    out_dir: Path,
    iter_dir: Path,
    tmp_path: Path,
    rounds: int = 1,
    run_id: str = "t",
    skip_retrain: bool = True,
    skip_eval: bool = True,
) -> types.SimpleNamespace:
    """Build the SimpleNamespace iterate._run_round expects.

    Centralizes the field list so adding new CLI flags doesn't fan out
    across every test in this file.
    """
    return types.SimpleNamespace(
        base_checkpoint=Path("/fake/base"),
        base_corpus=tmp_path / "base.jsonl",
        iterative_corpus_dir=iter_dir,
        allocation_root=tmp_path / "manifest",
        output_dir=out_dir,
        rounds=rounds, start_round=0,
        rollouts_per_prompt=1, prompts_per_round=1,
        filter_mode="best-of-n", threshold_pct=0.015,
        temperature=0.9, concurrency=1,
        max_iterations=20, max_retries=2,
        media_resolution="medium", seed=1,
        apply_clahe=False, max_fetch_calls=2, max_total_images=12,
        planes=["coronal"],
        vllm_url="http://127.0.0.1:8000/v1",
        vllm_base_compose=Path("docker-compose.training.yml"),
        training_container_name="langslice-training-dev",
        skip_container_check=True,
        vllm_served_name="langslice-ft", vllm_max_model_len=8192,
        vllm_gpu_mem_util=0.85, vllm_startup_timeout=10.0,
        manage_vllm=False, skip_vllm_check=True,
        model_alias="langslice-ft",
        # LoRA hot-swap mode (off by default — preserves merge+restart flow).
        vllm_lora_mode=False,
        vllm_lora_max_rank=16,
        vllm_lora_max_loras=4,
        vllm_base_model_path=None,
        sft_config=Path("configs/sft_default.toml"),
        sft_initial_adapter=None,
        skip_retrain=skip_retrain, skip_eval=skip_eval,
        eval_bench="tiny",
        eval_num_generations=1,
        distilled_sample_n=None,
        distilled_sample_seed=0,
        atlas_embedding_cache=None,
        query_embedding_cache=None,
        bucketed_shape_sampler=False,
        clahe_augment_fraction=0.0,
        synthetic_reasoning_mode="region_dump",
        repo_root=tmp_path,
        no_validate=True,
        run_id=run_id,
        # Curriculum (Phase C): None disables curriculum bias.
        curriculum_weights_dir=None,
        curriculum_alpha=1.0,
        curriculum_max_weight_change=3.0,
        curriculum_floor_fraction=0.1,
        curriculum_smoothing=0.5,
    )


# ──────────────────────────────────────────────────────────────────────────
# Test fixtures: stand-in objects + canned events
# ──────────────────────────────────────────────────────────────────────────

def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(127, 64, 32)).save(path, format="JPEG")


def _make_events(run_dir: Path) -> list[dict]:
    """Synthesize a minimal ADK event log with a single fetch + submit."""
    artifacts = run_dir / "tool_artifacts"
    img = artifacts / "fetch_atlas_001_img01.jpg"
    _write_dummy_image(img)
    return [
        {
            "role": "assistant", "visible_text": "fetching",
            "teacher_thought_summaries": [],
            "tool_calls": [{"name": "fetch_atlas",
                            "args": {"positions_mm": [4.0]}}],
            "usage_metadata": {},
        },
        {
            "role": "tool", "name": "fetch_atlas",
            "args": {"positions_mm": [4.0]},
            "response": {"positions_mm": [4.0], "description": "ok",
                         "status": "ok"},
            "artifacts": [{"type": "image_path",
                          "path": "tool_artifacts/fetch_atlas_001_img01.jpg",
                          "mime_type": "image/jpeg"}],
        },
        {
            "role": "assistant", "visible_text": "submitting",
            "teacher_thought_summaries": [],
            "tool_calls": [{"name": "submit_estimate",
                            "args": {"position_mm": 5.0,
                                     "reasoning": "between 4 and 6 mm"}}],
            "usage_metadata": {},
        },
    ]


@pytest.fixture
def fake_alloc_example():
    """A duck-typed RLVR allocation row matching SingleSliceExample shape."""
    return types.SimpleNamespace(
        section_id="sub-001/0001",
        subject_id="sub-001",
        image_path=Path("references/TestImages/M01.jpg"),
        atlas_name="allen_mouse_25um",
        plane="coronal",
        ap_mm=5.0,
    )


@pytest.fixture
def fake_rollout_factory(tmp_path):
    """Factory producing a fake RolloutResult for a given (sid, gen_idx)."""
    from iSFT.rollout import RolloutResult, RolloutSpec

    def make(spec_dict: dict, gen_idx: int, *, error: str | None = None) -> Any:
        run_dir = tmp_path / f"runs/{spec_dict['section_id']}_g{gen_idx}".replace("/", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        events = _make_events(run_dir) if error is None else []
        spec = RolloutSpec(
            section_id=spec_dict["section_id"],
            subject_id=spec_dict["subject_id"],
            image_path=spec_dict["image_path"],
            atlas_name=spec_dict["atlas_name"],
            plane=spec_dict["plane"],
            truth_mm=spec_dict["truth_mm"],
            plane_extent_mm=spec_dict["plane_extent_mm"],
            generation_idx=gen_idx,
        )
        return RolloutResult(
            spec=spec,
            submitted_position_mm=5.0 if error is None else None,
            reasoning="ok" if error is None else "",
            events=events,
            run_dir=run_dir,
            error=error,
        )
    return make


# ──────────────────────────────────────────────────────────────────────────
# Resume detection (the cheap, fully-isolated test)
# ──────────────────────────────────────────────────────────────────────────

def test_state_module_imports_via_iterate(tmp_path: Path) -> None:
    """Sanity: iterate.py imports state_mod; check the round-orchestration helper."""
    from iSFT import iterate
    from iSFT import state as state_mod
    assert iterate is not None
    assert state_mod.PHASES[0] == "sampled"


def test_round_skips_completed_phases(tmp_path: Path, monkeypatch) -> None:
    """When state.json marks 'unioned' as last-completed, only train+eval should run."""
    from iSFT import iterate
    from iSFT import state as state_mod

    out_dir = tmp_path / "run_001"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()

    # Pre-stage state.json claiming we got through "unioned" already.
    s = state_mod.RunState(run_id="t", rounds_total=1)
    s.mark_phase(0, "unioned")
    state_mod.save_state(s, out_dir)

    # Pre-stage every phase artifact so resume-from-unioned doesn't crash
    # trying to load earlier phases' on-disk outputs.
    (out_dir / "round_0_rollouts.json").write_text("[]", encoding="utf-8")
    (out_dir / "round_0_scored.json").write_text("[]", encoding="utf-8")
    (out_dir / "round_0_filtered.json").write_text("[]", encoding="utf-8")
    (out_dir / "round_0_corpus.jsonl").write_text("", encoding="utf-8")

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=1, run_id="t", skip_retrain=True, skip_eval=True,
    )
    # Counters: any phase that re-runs on resume bumps its counter; expectation
    # is that all phases <= "unioned" stay at 0.
    counters = {"rollouts": 0, "scored": 0, "filtered": 0, "appended": 0,
                "unioned": 0, "trained": 0, "evaluated": 0}

    def _fake_phase_rollouts(**kw):
        counters["rollouts"] += 1
        return [], []
    def _fake_score(rs):
        counters["scored"] += 1
        return []
    def _fake_filter(scored, *, filter_mode, threshold_pct):
        counters["filtered"] += 1
        return []
    def _fake_append(**kw):
        counters["appended"] += 1
        return iter_dir / "round_0.jsonl", {"n_kept": 0,
                                            "n_dropped_trace": 0,
                                            "n_dropped_dup": 0}
    def _fake_union(**kw):
        counters["unioned"] += 1
        return out_dir / "round_0_corpus.jsonl", {"rows_kept": 0}
    def _fake_train(**kw):
        counters["trained"] += 1
        return kw["output_adapter_dir"]
    def _fake_eval(**kw):
        counters["evaluated"] += 1
        return kw["eval_out_dir"] / "summary.json"

    monkeypatch.setattr(iterate, "_phase_rollouts", _fake_phase_rollouts)
    monkeypatch.setattr(iterate, "_score_rollouts", _fake_score)
    monkeypatch.setattr(iterate, "_kept_rollouts_for_filter", _fake_filter)
    monkeypatch.setattr(iterate, "_phase_append_iterative", _fake_append)
    monkeypatch.setattr(iterate, "_phase_union", _fake_union)
    monkeypatch.setattr(iterate, "_phase_train", _fake_train)
    monkeypatch.setattr(iterate, "_phase_eval", _fake_eval)
    # Disable vllm management.
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round", lambda **kw: None)
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round_for_eval", lambda **kw: None)
    monkeypatch.setattr(iterate, "_tear_down_vllm_for_round", lambda **kw: None)

    # --skip-retrain and --skip-eval are set, so phases past "unioned" should
    # also not run. The point of this test is that the EARLIER phases stay at 0.
    iterate._run_round(args=args, round_idx=0, run_state=s)

    # No phase up to "unioned" should have re-run.
    assert counters["rollouts"] == 0
    assert counters["scored"] == 0
    assert counters["filtered"] == 0
    assert counters["appended"] == 0
    assert counters["unioned"] == 0


def test_round_runs_phases_in_order_when_no_resume(tmp_path: Path, monkeypatch) -> None:
    """Fresh state — every phase runs exactly once, in the expected order."""
    from iSFT import iterate
    from iSFT import state as state_mod

    out_dir = tmp_path / "run_002"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    s = state_mod.init_or_resume(out_dir, run_id="t2", rounds_total=1)

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=1, run_id="t2", skip_retrain=False, skip_eval=False,
    )

    order: list[str] = []
    monkeypatch.setattr(iterate, "_phase_rollouts",
                        lambda **kw: (order.append("rollouts") or ([], [])))
    monkeypatch.setattr(iterate, "_score_rollouts",
                        lambda rs: (order.append("scored") or []))
    monkeypatch.setattr(iterate, "_kept_rollouts_for_filter",
                        lambda *a, **kw: (order.append("filtered") or []))
    monkeypatch.setattr(iterate, "_phase_append_iterative",
                        lambda **kw: (order.append("appended"),
                                      (iter_dir / "round_0.jsonl",
                                       {"n_kept": 0, "n_dropped_trace": 0,
                                        "n_dropped_dup": 0}))[1])
    monkeypatch.setattr(iterate, "_phase_union",
                        lambda **kw: (order.append("unioned"),
                                      (out_dir / "round_0_corpus.jsonl",
                                       {"rows_kept": 0}))[1])
    monkeypatch.setattr(iterate, "_phase_train",
                        lambda **kw: (order.append("trained")
                                      or kw["output_adapter_dir"]))
    monkeypatch.setattr(iterate, "_phase_eval",
                        lambda **kw: (order.append("evaluated")
                                      or kw["eval_out_dir"] / "summary.json"))
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round", lambda **kw: None)
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round_for_eval", lambda **kw: None)
    monkeypatch.setattr(iterate, "_tear_down_vllm_for_round", lambda **kw: None)

    iterate._run_round(args=args, round_idx=0, run_state=s)

    assert order == [
        "rollouts", "scored", "filtered", "appended",
        "unioned", "trained", "evaluated",
    ]
    # state.json should now reflect "done" for round 0.
    loaded = state_mod.load_state(out_dir)
    assert loaded is not None
    assert loaded.phase == "done"


def test_state_json_written_after_each_phase(tmp_path: Path, monkeypatch) -> None:
    """Verify state.json's phase field updates between phases (not just at end)."""
    from iSFT import iterate
    from iSFT import state as state_mod

    out_dir = tmp_path / "run_003"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    s = state_mod.init_or_resume(out_dir, run_id="t3", rounds_total=1)

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=1, run_id="t3", skip_retrain=True, skip_eval=True,
    )

    observed_phases_during_run: list[str | None] = []

    def _capturing_score(rs):
        # When this phase runs, state.json should already say "rollouts".
        loaded = state_mod.load_state(out_dir)
        observed_phases_during_run.append(loaded.phase if loaded else None)
        return []

    def _capturing_filter(
        scored, *, filter_mode, threshold_pct,
        adaptive_buffer=None, adaptive_quantile=0.95, adaptive_warmup_n=50,
    ):
        loaded = state_mod.load_state(out_dir)
        observed_phases_during_run.append(loaded.phase if loaded else None)
        return []

    monkeypatch.setattr(iterate, "_phase_rollouts", lambda **kw: ([], []))
    monkeypatch.setattr(iterate, "_score_rollouts", _capturing_score)
    monkeypatch.setattr(iterate, "_kept_rollouts_for_filter", _capturing_filter)
    monkeypatch.setattr(iterate, "_phase_append_iterative",
                        lambda **kw: (iter_dir / "round_0.jsonl",
                                      {"n_kept": 0, "n_dropped_trace": 0,
                                       "n_dropped_dup": 0}))
    monkeypatch.setattr(iterate, "_phase_union",
                        lambda **kw: (out_dir / "round_0_corpus.jsonl",
                                      {"rows_kept": 0}))
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round", lambda **kw: None)
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round_for_eval", lambda **kw: None)
    monkeypatch.setattr(iterate, "_tear_down_vllm_for_round", lambda **kw: None)

    iterate._run_round(args=args, round_idx=0, run_state=s)
    # Score saw "rollouts" already saved; filter saw "scored" already saved.
    assert observed_phases_during_run[0] == "rollouts"
    assert observed_phases_during_run[1] == "scored"


def test_multi_round_advances_round_counter(tmp_path: Path, monkeypatch) -> None:
    """Round 0 → done → advance_round → Round 1 starts at phase=None."""
    from iSFT import iterate
    from iSFT import state as state_mod

    out_dir = tmp_path / "run_004"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()

    # Stub out everything so each round just walks through phases.
    monkeypatch.setattr(iterate, "_phase_rollouts", lambda **kw: ([], []))
    monkeypatch.setattr(iterate, "_score_rollouts", lambda rs: [])
    monkeypatch.setattr(iterate, "_kept_rollouts_for_filter", lambda *a, **kw: [])
    monkeypatch.setattr(iterate, "_phase_append_iterative",
                        lambda **kw: (iter_dir / f"round_{kw['round_idx']}.jsonl",
                                      {"n_kept": 0, "n_dropped_trace": 0,
                                       "n_dropped_dup": 0}))
    monkeypatch.setattr(iterate, "_phase_union",
                        lambda **kw: (out_dir / f"round_{kw['round_idx']}_corpus.jsonl",
                                      {"rows_kept": 0}))
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round", lambda **kw: None)
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round_for_eval", lambda **kw: None)
    monkeypatch.setattr(iterate, "_tear_down_vllm_for_round", lambda **kw: None)

    argv = [
        "--base-checkpoint", "/fake/base",
        "--base-corpus", str(tmp_path / "base.jsonl"),
        "--iterative-corpus-dir", str(iter_dir),
        "--allocation-root", str(tmp_path / "manifest"),
        "--output-dir", str(out_dir),
        "--rounds", "3",
        "--skip-retrain",
        "--skip-eval",
        "--no-validate",
        "--skip-vllm-check",
    ]
    rc = iterate.main(argv)
    assert rc == 0
    loaded = state_mod.load_state(out_dir)
    assert loaded is not None
    # After 3 rounds with --skip-retrain we still mark unioned as last,
    # but the round counter should have advanced.
    # On the final round we don't call advance_round (it's the last one);
    # round=2 with phase=("unioned" if skip_retrain else "done") is fine.
    assert loaded.round == 2


def test_rounds_arg_must_be_positive(tmp_path: Path) -> None:
    from iSFT import iterate
    rc = iterate.main([
        "--base-checkpoint", "/fake/base",
        "--base-corpus", "/fake/base.jsonl",
        "--iterative-corpus-dir", str(tmp_path),
        "--allocation-root", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--rounds", "0",
        "--skip-vllm-check",
    ])
    assert rc == 2


def test_help_runs_and_includes_rounds_flag(capsys) -> None:
    from iSFT import iterate
    with pytest.raises(SystemExit) as excinfo:
        iterate.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--rounds" in out
    assert "--manage-vllm" in out
    assert "--skip-retrain" in out
    assert "--skip-eval" in out


# ──────────────────────────────────────────────────────────────────────────
# Resume from each individual phase
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("completed_phase,phases_that_should_run", [
    (None, ["rollouts", "scored", "filtered", "appended", "unioned"]),
    ("rollouts", ["scored", "filtered", "appended", "unioned"]),
    ("scored", ["filtered", "appended", "unioned"]),
    ("filtered", ["appended", "unioned"]),
    ("appended", ["unioned"]),
    ("unioned", []),
])
def test_resume_from_arbitrary_phase_runs_only_remaining(
    tmp_path: Path, monkeypatch, completed_phase, phases_that_should_run,
) -> None:
    from iSFT import iterate
    from iSFT import state as state_mod

    out_dir = tmp_path / f"run_resume_{completed_phase or 'fresh'}"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()

    s = state_mod.RunState(run_id="t", rounds_total=1)
    if completed_phase is not None:
        s.mark_phase(0, completed_phase)
    state_mod.save_state(s, out_dir)
    # Pre-stage artifacts the resumed phases would have written if they ran.
    # Each phase writes a deterministic file the next phase consumes; resume
    # must find these on disk to skip ahead correctly.
    if completed_phase is not None:
        (out_dir / "round_0_rollouts.json").write_text("[]", encoding="utf-8")
        (out_dir / "round_0_scored.json").write_text("[]", encoding="utf-8")
        (out_dir / "round_0_filtered.json").write_text("[]", encoding="utf-8")
        (out_dir / "round_0_corpus.jsonl").write_text("", encoding="utf-8")

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=1, run_id="t", skip_retrain=True, skip_eval=True,
    )

    ran: list[str] = []
    monkeypatch.setattr(iterate, "_phase_rollouts",
                        lambda **kw: (ran.append("rollouts") or ([], [])))
    monkeypatch.setattr(iterate, "_score_rollouts",
                        lambda rs: (ran.append("scored") or []))
    monkeypatch.setattr(iterate, "_kept_rollouts_for_filter",
                        lambda *a, **kw: (ran.append("filtered") or []))
    monkeypatch.setattr(iterate, "_phase_append_iterative",
                        lambda **kw: (ran.append("appended"),
                                      (iter_dir / "round_0.jsonl",
                                       {"n_kept": 0, "n_dropped_trace": 0,
                                        "n_dropped_dup": 0}))[1])
    monkeypatch.setattr(iterate, "_phase_union",
                        lambda **kw: (ran.append("unioned"),
                                      (out_dir / "round_0_corpus.jsonl",
                                       {"rows_kept": 0}))[1])
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round", lambda **kw: None)
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round_for_eval", lambda **kw: None)
    monkeypatch.setattr(iterate, "_tear_down_vllm_for_round", lambda **kw: None)

    iterate._run_round(args=args, round_idx=0, run_state=s)
    assert ran == phases_that_should_run


# ──────────────────────────────────────────────────────────────────────────
# Atlas-embedding splice plumbing — verify --atlas-embedding-cache is
# threaded through to the train_sft.py subprocess command.
# ──────────────────────────────────────────────────────────────────────────


def _capture_phase_train_cmd(
    *, monkeypatch, args, tmp_path: Path
) -> list[str]:
    """Helper: invoke _phase_train, intercept subprocess.run, return its cmd."""
    from iSFT import iterate

    captured: list[list[str]] = []

    def _fake_run(cmd, *_args, **_kwargs):
        captured.append(cmd)

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(iterate.subprocess, "run", _fake_run)

    unioned = tmp_path / "round_0_corpus.jsonl"
    unioned.write_text("", encoding="utf-8")
    out_adapter = tmp_path / "round_0_adapter"
    iterate._phase_train(
        args=args, round_idx=0,
        unioned_jsonl=unioned, output_adapter_dir=out_adapter,
    )
    assert len(captured) == 1
    return captured[0]


def test_phase_train_passes_atlas_cache_when_set(
    tmp_path: Path, monkeypatch
) -> None:
    """When args.atlas_embedding_cache is set, the train_sft.py cmd contains
    --atlas-embedding-cache pointing at the in-container path translation.
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    cache_dir = tmp_path / "out" / "atlas_embeddings"
    cache_dir.mkdir(parents=True)

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=1, skip_retrain=False, skip_eval=True,
    )
    args.atlas_embedding_cache = cache_dir

    cmd = _capture_phase_train_cmd(monkeypatch=monkeypatch, args=args, tmp_path=tmp_path)

    # Final cmd is ["docker", "compose", "-f", ..., "run", ..., "bash", "-lc", "<full bash>"]
    bash_payload = cmd[-1]
    assert "--atlas-embedding-cache" in bash_payload
    # In-container path uses /workspace/LangSlice prefix.
    assert "/workspace/LangSlice/out/atlas_embeddings" in bash_payload


def test_phase_train_omits_atlas_cache_when_none(
    tmp_path: Path, monkeypatch
) -> None:
    """When args.atlas_embedding_cache is None (default), the cmd must NOT
    include --atlas-embedding-cache — splice stays off.
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=1, skip_retrain=False, skip_eval=True,
    )
    assert args.atlas_embedding_cache is None  # default from helper

    cmd = _capture_phase_train_cmd(monkeypatch=monkeypatch, args=args, tmp_path=tmp_path)
    bash_payload = cmd[-1]
    assert "--atlas-embedding-cache" not in bash_payload
