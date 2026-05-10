"""Mock-heavy tests for the --vllm-lora-mode hot-swap branch in iterate.py
and the underlying REST helpers in vllm_lifecycle.py.

These tests do NOT require a live vLLM. They verify:

* ``write_compose_override(enable_lora=True)`` produces a YAML that
  carries the ``--enable-lora`` family of flags + the
  ``VLLM_ALLOW_RUNTIME_LORA_UPDATING`` env.
* ``load_lora_adapter`` / ``unload_lora_adapter`` POST the correct
  URLs and JSON payloads, surface non-2xx responses as ``VLLMLoRAError``,
  and the unload helper swallows ``404 not found`` idempotently.
* ``list_loaded_models`` round-trips through the ``/v1/models`` shape.
* Round 0 in lora-mode routes rollouts to the bare base alias and
  hot-loads no adapter (round 0 has no prior best model).
* Round k>0 in lora-mode hot-loads ``round_<k-1>_adapter`` under name
  ``round_<k>_rollout_adapter`` and routes rollouts to that name.
* Eval phase always hot-loads the *just-trained* adapter under
  ``round_<k>_eval_adapter`` and routes slicebench to that name.
* The eviction logic respects ``--max-loras`` by unloading the oldest
  convention-named adapter before loading a new one.
* Default ``--vllm-lora-mode`` is off, so the merge+restart flow is
  unchanged for back-compat.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tests"))

# isort: off
# Reuse the sibling test module's _make_test_args so the CLI-arg shape
# stays in sync as flags are added/renamed. The sys.path mutation above
# must run before this import, so it cannot live in the top import block.
from test_expert_iteration_iterate import _make_test_args  # noqa: E402
# isort: on


# ──────────────────────────────────────────────────────────────────────────
# write_compose_override(enable_lora=True)
# ──────────────────────────────────────────────────────────────────────────

def test_compose_override_enable_lora_emits_flags(tmp_path: Path) -> None:
    from tools.expert_iteration import vllm_lifecycle

    out = tmp_path / "lora.override.yml"
    vllm_lifecycle.write_compose_override(
        output_path=out,
        model_path="/models/sft-base",
        served_model_name="langslice-ft",
        max_model_len=12288,
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=4,
    )
    text = out.read_text(encoding="utf-8")
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in text
    assert "--enable-lora" in text
    assert "--max-lora-rank" in text and '"16"' in text
    assert "--max-loras" in text and '"4"' in text
    # Tool-call parsing flags must still be present so served behavior is
    # consistent with the merge-mode override.
    assert "gemma4" in text
    assert "--tool-call-parser" in text


def test_compose_override_disable_lora_omits_flags(tmp_path: Path) -> None:
    from tools.expert_iteration import vllm_lifecycle

    out = tmp_path / "merge.override.yml"
    vllm_lifecycle.write_compose_override(
        output_path=out,
        model_path="/workspace/foo/merged",
        served_model_name="langslice-ft",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        # enable_lora defaults to False
    )
    text = out.read_text(encoding="utf-8")
    assert "--enable-lora" not in text
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in text
    assert "--max-lora-rank" not in text


# ──────────────────────────────────────────────────────────────────────────
# REST helpers (load / unload / list)
# ──────────────────────────────────────────────────────────────────────────

class _FakeResp:
    """Minimal stand-in for the urllib response context manager."""
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_load_lora_adapter_posts_correct_payload() -> None:
    from tools.expert_iteration import vllm_lifecycle

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["content_type"] = req.headers.get("Content-type")
        return _FakeResp(b"Success: LoRA adapter 'a' added successfully.", 200)

    with patch.object(vllm_lifecycle.urllib.request, "urlopen", side_effect=fake_urlopen):
        vllm_lifecycle.load_lora_adapter(
            base_url="http://127.0.0.1:8000/v1",
            lora_name="round_3_rollout_adapter",
            lora_path="/workspace/run/round_2_adapter",
        )
    assert captured["url"] == "http://127.0.0.1:8000/v1/load_lora_adapter"
    assert captured["method"] == "POST"
    assert captured["body"] == {
        "lora_name": "round_3_rollout_adapter",
        "lora_path": "/workspace/run/round_2_adapter",
    }
    assert captured["content_type"] == "application/json"


def test_load_lora_adapter_raises_on_http_error() -> None:
    from tools.expert_iteration import vllm_lifecycle

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b"adapter directory not found"),
        )

    with patch.object(vllm_lifecycle.urllib.request, "urlopen", side_effect=fake_urlopen):
        with pytest.raises(vllm_lifecycle.VLLMLoRAError) as excinfo:
            vllm_lifecycle.load_lora_adapter(
                base_url="http://127.0.0.1:8000/v1",
                lora_name="bad", lora_path="/no/such/dir",
            )
    assert "HTTP 400" in str(excinfo.value)
    assert "not found" in str(excinfo.value)


def test_unload_lora_adapter_swallows_not_found() -> None:
    from tools.expert_iteration import vllm_lifecycle

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b"The lora adapter 'xx' was not found."),
        )

    with patch.object(vllm_lifecycle.urllib.request, "urlopen", side_effect=fake_urlopen):
        # Must NOT raise — idempotent on missing adapter.
        vllm_lifecycle.unload_lora_adapter(
            base_url="http://127.0.0.1:8000/v1", lora_name="xx",
        )


def test_list_loaded_models_parses_data_array() -> None:
    from tools.expert_iteration import vllm_lifecycle

    body = json.dumps({"object": "list", "data": [
        {"id": "langslice-ft"},
        {"id": "round_2_eval_adapter", "parent": "langslice-ft"},
    ]}).encode("utf-8")

    def fake_urlopen(url, timeout):
        return _FakeResp(body, 200)

    with patch.object(vllm_lifecycle.urllib.request, "urlopen", side_effect=fake_urlopen):
        ids = vllm_lifecycle.list_loaded_models(base_url="http://127.0.0.1:8000/v1")
    assert ids == ["langslice-ft", "round_2_eval_adapter"]


# ──────────────────────────────────────────────────────────────────────────
# iterate.py LoRA-mode branches
# ──────────────────────────────────────────────────────────────────────────

def _patch_compose_calls(monkeypatch, vllm_lifecycle):
    """No-op the docker compose subprocess wrappers + readiness probe."""
    monkeypatch.setattr(vllm_lifecycle, "compose_up_vllm", lambda **kw: None)
    monkeypatch.setattr(vllm_lifecycle, "compose_down_vllm", lambda **kw: None)
    monkeypatch.setattr(vllm_lifecycle, "wait_for_vllm", lambda *a, **kw: True)


def test_round0_lora_mode_does_not_load_adapter(tmp_path: Path, monkeypatch) -> None:
    """Round 0 in lora-mode should bring up vLLM with the base + --enable-lora,
    but NOT hot-load any adapter (no prior trained adapter exists)."""
    from tools.expert_iteration import iterate, vllm_lifecycle

    args = _make_test_args(
        out_dir=tmp_path / "out",
        iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=1,
    )
    args.manage_vllm = True
    args.vllm_lora_mode = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _patch_compose_calls(monkeypatch, vllm_lifecycle)
    load_calls: list[dict] = []
    monkeypatch.setattr(vllm_lifecycle, "load_lora_adapter",
                        lambda **kw: load_calls.append(kw))
    monkeypatch.setattr(vllm_lifecycle, "unload_lora_adapter",
                        lambda **kw: load_calls.append({"unload": kw}))
    monkeypatch.setattr(vllm_lifecycle, "list_loaded_models",
                        lambda **kw: ["langslice-ft"])

    override = iterate._bring_up_vllm_for_round(
        args=args, round_idx=0, run_dir=args.output_dir,
    )
    assert override.is_file()
    assert override.name == "lora_base_compose.override.yml"
    text = override.read_text(encoding="utf-8")
    assert "--enable-lora" in text
    # Round 0 must not register any adapter.
    assert load_calls == []


def test_roundN_lora_mode_hotloads_prev_adapter(tmp_path: Path, monkeypatch) -> None:
    """Round 2 in lora-mode hot-loads round_1_adapter under model name
    round_2_rollout_adapter (so subsequent rollouts route to round 1's
    trained best-of)."""
    from tools.expert_iteration import iterate, vllm_lifecycle

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Pre-stage round_1_adapter so the resolver finds it.
    (out_dir / "round_1_adapter").mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=3,
    )
    args.manage_vllm = True
    args.vllm_lora_mode = True

    _patch_compose_calls(monkeypatch, vllm_lifecycle)
    monkeypatch.setattr(vllm_lifecycle, "list_loaded_models",
                        lambda **kw: ["langslice-ft"])
    load_calls: list[dict] = []
    monkeypatch.setattr(vllm_lifecycle, "load_lora_adapter",
                        lambda **kw: load_calls.append(kw))
    monkeypatch.setattr(vllm_lifecycle, "unload_lora_adapter",
                        lambda **kw: load_calls.append({"unload": kw}))

    iterate._bring_up_vllm_for_round(
        args=args, round_idx=2, run_dir=out_dir,
    )
    assert len(load_calls) == 1
    call = load_calls[0]
    assert call["lora_name"] == "round_2_rollout_adapter"
    assert call["lora_path"].endswith("round_1_adapter")


def test_roundN_lora_mode_eval_hotloads_just_trained(tmp_path: Path, monkeypatch) -> None:
    from tools.expert_iteration import iterate, vllm_lifecycle

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    adapter_dir = out_dir / "round_2_adapter"
    adapter_dir.mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=3,
    )
    args.manage_vllm = True
    args.vllm_lora_mode = True

    _patch_compose_calls(monkeypatch, vllm_lifecycle)
    monkeypatch.setattr(vllm_lifecycle, "list_loaded_models",
                        lambda **kw: ["langslice-ft"])
    load_calls: list[dict] = []
    monkeypatch.setattr(vllm_lifecycle, "load_lora_adapter",
                        lambda **kw: load_calls.append(kw))
    monkeypatch.setattr(vllm_lifecycle, "unload_lora_adapter",
                        lambda **kw: load_calls.append({"unload": kw}))

    iterate._bring_up_vllm_for_round_for_eval(
        args=args, round_idx=2, run_dir=out_dir, adapter_dir=adapter_dir,
    )
    assert len(load_calls) == 1
    call = load_calls[0]
    assert call["lora_name"] == "round_2_eval_adapter"
    assert call["lora_path"].endswith("round_2_adapter")


def test_lora_mode_eviction_when_at_max_loras(tmp_path: Path, monkeypatch) -> None:
    """When --max-loras is hit, the oldest convention-named adapter is
    unloaded before the new one is loaded."""
    from tools.expert_iteration import iterate, vllm_lifecycle

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "round_3_adapter").mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=5,
    )
    args.manage_vllm = True
    args.vllm_lora_mode = True
    args.vllm_lora_max_loras = 2  # Tighten cap for the test.

    _patch_compose_calls(monkeypatch, vllm_lifecycle)
    # Pretend two convention adapters are already loaded — round 1 + 2.
    # New load attempt for round 4 needs to evict round 1 (oldest).
    monkeypatch.setattr(vllm_lifecycle, "list_loaded_models",
                        lambda **kw: [
                            "langslice-ft",
                            "round_1_eval_adapter",
                            "round_2_eval_adapter",
                        ])
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        vllm_lifecycle, "load_lora_adapter",
        lambda **kw: events.append(("load", kw)),
    )
    monkeypatch.setattr(
        vllm_lifecycle, "unload_lora_adapter",
        lambda **kw: events.append(("unload", kw)),
    )

    iterate._bring_up_vllm_for_round(
        args=args, round_idx=4, run_dir=out_dir,
    )
    # Expect: at least one unload of the OLDEST convention adapter, then load.
    kinds = [e[0] for e in events]
    assert kinds[0] == "unload"
    assert events[0][1]["lora_name"] == "round_1_eval_adapter"
    assert events[-1][0] == "load"
    assert events[-1][1]["lora_name"] == "round_4_rollout_adapter"


def test_lora_mode_skips_load_if_adapter_already_loaded(tmp_path: Path, monkeypatch) -> None:
    """If the target adapter name is already registered, skip the POST."""
    from tools.expert_iteration import iterate, vllm_lifecycle

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "round_1_adapter").mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=3,
    )
    args.manage_vllm = True
    args.vllm_lora_mode = True

    _patch_compose_calls(monkeypatch, vllm_lifecycle)
    monkeypatch.setattr(
        vllm_lifecycle, "list_loaded_models",
        lambda **kw: ["langslice-ft", "round_2_rollout_adapter"],
    )
    events: list[str] = []
    monkeypatch.setattr(
        vllm_lifecycle, "load_lora_adapter",
        lambda **kw: events.append("load"),
    )
    monkeypatch.setattr(
        vllm_lifecycle, "unload_lora_adapter",
        lambda **kw: events.append("unload"),
    )

    iterate._bring_up_vllm_for_round(
        args=args, round_idx=2, run_dir=out_dir,
    )
    # No load/unload — the adapter was already registered under the right name.
    assert events == []


def test_lora_mode_default_off_preserves_merge_path(tmp_path: Path, monkeypatch) -> None:
    """With --vllm-lora-mode unset, --enable-lora must not appear in the
    written override and the merge_lora_to_bf16 path must still be used
    for round k>0."""
    from tools.expert_iteration import iterate, vllm_lifecycle

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "round_0_adapter").mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=2,
    )
    args.manage_vllm = True
    args.vllm_lora_mode = False  # Explicit: legacy flow

    _patch_compose_calls(monkeypatch, vllm_lifecycle)
    merge_calls: list[dict] = []
    monkeypatch.setattr(
        vllm_lifecycle, "merge_lora_to_bf16",
        lambda **kw: merge_calls.append(kw)
        or Path(kw["output_path"]).mkdir(parents=True, exist_ok=True)
        or (Path(kw["output_path"]) / "config.json").write_text("{}", encoding="utf-8"),
    )
    # In merge mode the LoRA REST helpers must NOT be touched.
    load_calls: list[dict] = []
    monkeypatch.setattr(
        vllm_lifecycle, "load_lora_adapter",
        lambda **kw: load_calls.append(kw),
    )

    override = iterate._bring_up_vllm_for_round(
        args=args, round_idx=1, run_dir=out_dir,
    )
    assert override.name == "round_1_compose.override.yml"
    text = override.read_text(encoding="utf-8")
    assert "--enable-lora" not in text
    assert len(merge_calls) == 1
    assert load_calls == []


def test_run_round_lora_mode_routes_rollouts_to_adapter_alias(
    tmp_path: Path, monkeypatch,
) -> None:
    """End-to-end ``_run_round`` smoke: in lora-mode for round k>0, the
    model_str routed to rollouts must be the round_k_rollout_adapter alias,
    not the bare model_alias."""
    from tools.expert_iteration import iterate
    from tools.expert_iteration import state as state_mod

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    (out_dir / "round_0_adapter").mkdir()

    args = _make_test_args(
        out_dir=out_dir, iter_dir=iter_dir, tmp_path=tmp_path,
        rounds=2, run_id="t-lora",
        skip_retrain=True, skip_eval=True,
    )
    args.vllm_lora_mode = True
    s = state_mod.init_or_resume(out_dir, run_id="t-lora", rounds_total=2)

    captured_models: list[str] = []

    def fake_rollouts(*, args, round_idx, artifacts_root, model_str):
        captured_models.append(model_str)
        return [], []

    monkeypatch.setattr(iterate, "_phase_rollouts", fake_rollouts)
    monkeypatch.setattr(iterate, "_score_rollouts", lambda rs: [])
    monkeypatch.setattr(iterate, "_kept_rollouts_for_filter",
                        lambda *a, **kw: [])
    monkeypatch.setattr(
        iterate, "_phase_append_iterative",
        lambda **kw: (iter_dir / "round_1.jsonl",
                      {"n_kept": 0, "n_dropped_trace": 0, "n_dropped_dup": 0}),
    )
    monkeypatch.setattr(
        iterate, "_phase_union",
        lambda **kw: (out_dir / "round_1_corpus.jsonl", {"rows_kept": 0}),
    )
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round", lambda **kw: None)
    monkeypatch.setattr(iterate, "_bring_up_vllm_for_round_for_eval",
                        lambda **kw: None)
    monkeypatch.setattr(iterate, "_tear_down_vllm_for_round", lambda **kw: None)

    iterate._run_round(args=args, round_idx=1, run_state=s)
    assert captured_models == ["litellm-proxy:round_1_rollout_adapter"]


def test_phase_eval_uses_served_name_override(tmp_path: Path, monkeypatch) -> None:
    """``_phase_eval(served_name=...)`` must thread the override into the
    slicebench --model arg, not the static args.vllm_served_name."""
    from tools.expert_iteration import iterate

    args = _make_test_args(
        out_dir=tmp_path / "out", iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_out = args.output_dir / "round_3_slicebench"
    eval_out.mkdir()

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr(iterate.subprocess, "run", fake_run)
    iterate._phase_eval(
        args=args, round_idx=3, eval_out_dir=eval_out,
        served_name="round_3_eval_adapter",
    )
    cmd_str = " ".join(captured["cmd"])
    assert "--model litellm-proxy:round_3_eval_adapter" in cmd_str
    assert "litellm-proxy:langslice-ft" not in cmd_str


def test_phase_eval_default_served_name_falls_back(tmp_path: Path, monkeypatch) -> None:
    """When ``served_name`` is None, fall back to args.vllm_served_name (merge mode)."""
    from tools.expert_iteration import iterate

    args = _make_test_args(
        out_dir=tmp_path / "out", iter_dir=tmp_path / "iter",
        tmp_path=tmp_path, rounds=1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_out = args.output_dir / "round_0_slicebench"
    eval_out.mkdir()

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr(iterate.subprocess, "run", fake_run)
    iterate._phase_eval(args=args, round_idx=0, eval_out_dir=eval_out)
    cmd_str = " ".join(captured["cmd"])
    assert "--model litellm-proxy:langslice-ft" in cmd_str


# ──────────────────────────────────────────────────────────────────────────
# CLI flag default
# ──────────────────────────────────────────────────────────────────────────

def test_cli_flag_default_off() -> None:
    from tools.expert_iteration import iterate

    ns = iterate._parse_args([
        "--base-checkpoint", "/x",
        "--base-corpus", "/x.jsonl",
        "--iterative-corpus-dir", "/x",
        "--allocation-root", "/x",
        "--output-dir", "/x",
    ])
    assert ns.vllm_lora_mode is False
    assert ns.vllm_lora_max_rank == 16
    assert ns.vllm_lora_max_loras == 4


def test_cli_flag_enables_lora_mode() -> None:
    from tools.expert_iteration import iterate

    ns = iterate._parse_args([
        "--base-checkpoint", "/x",
        "--base-corpus", "/x.jsonl",
        "--iterative-corpus-dir", "/x",
        "--allocation-root", "/x",
        "--output-dir", "/x",
        "--vllm-lora-mode",
        "--vllm-lora-max-loras", "8",
    ])
    assert ns.vllm_lora_mode is True
    assert ns.vllm_lora_max_loras == 8
