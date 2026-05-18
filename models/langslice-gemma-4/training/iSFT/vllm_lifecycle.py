"""vLLM serve-profile lifecycle helpers for multi-round expert iteration.

Each round needs vLLM to serve a different fine-tune adapter. Wave 2 picks
**Option A** (simpler): merge the LoRA into a bf16 checkpoint between rounds,
write a per-round docker-compose override that points the ``vllm`` service
at the new merged path, then bring the service up + tear it down per round.

The override yml is regenerated each round into the run directory. The base
``docker-compose.training.yml`` continues to define the vllm service shape
(image, ports, GPU, env); the override only swaps the ``--model`` argument
and (optionally) tweaks ``--max-model-len`` / ``--gpu-memory-utilization``.

Round 0 is special: the base SFT checkpoint is already merged-bf16, so we
skip the merge step and feed its path straight through.

External dependency: ``docker compose`` CLI must be on PATH. We invoke
``docker compose -f base.yml -f override.yml --profile serve up -d vllm``
which honors both files (override takes priority for matching keys).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# Default poll interval/timeout for /v1/models readiness probe.
_PROBE_INTERVAL_S = 2.0
_PROBE_TIMEOUT_S = 300.0  # 5 min


# ──────────────────────────────────────────────────────────────────────────
# Per-round docker-compose override
# ──────────────────────────────────────────────────────────────────────────

def _format_yaml_list(items: Sequence[str], indent: int = 8) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {_yaml_quote(it)}" for it in items)


def _yaml_quote(value: str) -> str:
    """Quote a value for embedding into the compose override YAML.

    Compose accepts plain scalars for paths/options that don't contain
    spaces, colons, or other YAML metacharacters. We always wrap in
    double quotes to avoid edge cases (paths with colons on Windows
    bind mounts, --max-model-len numeric values that YAML would
    otherwise coerce, etc.).
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_compose_override(
    *,
    output_path: Path,
    model_path: str,
    served_model_name: str = "langslice-ft",
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.85,
    extra_args: Sequence[str] = (),
    enable_lora: bool = False,
    max_lora_rank: int = 16,
    max_loras: int = 4,
) -> Path:
    """Write a per-round docker-compose override that swaps the vllm --model.

    The override pins the ``vllm`` service's command to a complete arg list,
    overriding (not merging with) the base compose's ``command``. We mirror
    every flag the base specifies so the served behavior stays consistent
    across rounds.

    When ``enable_lora`` is True, three additional things happen:

    1. ``--enable-lora --max-lora-rank <r> --max-loras <n>`` are appended.
    2. The override sets ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`` in the
       service environment so the ``/v1/(load|unload)_lora_adapter`` REST
       endpoints are reachable.
    3. The base ``--model`` should point at the *base* checkpoint (e.g.
       the SFT-merged-bf16) — adapters are then hot-swapped via REST.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_args = [
        "--model", model_path,
        "--served-model-name", served_model_name,
        "--max-model-len", str(int(max_model_len)),
        "--gpu-memory-utilization", f"{float(gpu_memory_utilization):.3f}",
        "--dtype", "bfloat16",
    ]
    if enable_lora:
        cmd_args.extend([
            "--enable-lora",
            "--max-lora-rank", str(int(max_lora_rank)),
            "--max-loras", str(int(max_loras)),
        ])
    cmd_args.extend([
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
        "--reasoning-parser", "gemma4",
    ])
    cmd_args.extend(extra_args)
    env_block = ""
    if enable_lora:
        env_block = (
            "    environment:\n"
            "      VLLM_ALLOW_RUNTIME_LORA_UPDATING: \"True\"\n"
        )
    yaml = (
        "services:\n"
        "  vllm:\n"
        f"{env_block}"
        "    command:\n"
        f"{_format_yaml_list(cmd_args)}\n"
    )
    output_path.write_text(yaml, encoding="utf-8")
    logger.info(
        "Wrote vllm compose override: %s (model=%s, enable_lora=%s)",
        output_path, model_path, enable_lora,
    )
    return output_path


# ──────────────────────────────────────────────────────────────────────────
# Runtime LoRA management via vLLM REST API
# ──────────────────────────────────────────────────────────────────────────

class VLLMLoRAError(RuntimeError):
    """Raised when a /v1/(load|unload)_lora_adapter call fails."""


def _post_json(
    url: str, payload: dict, *, timeout_s: float = 60.0,
) -> tuple[int, str]:
    """POST a JSON body and return (status, decoded text). Never raises on HTTP errors."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 — fixed-form POST
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # Body is on the exception object for HTTPError.
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
        return exc.code, text


def load_lora_adapter(
    *, base_url: str, lora_name: str, lora_path: str, timeout_s: float = 60.0,
) -> None:
    """``POST /v1/load_lora_adapter`` — register a LoRA adapter at runtime.

    ``base_url`` should include the OpenAI ``/v1`` suffix. ``lora_path`` is
    the *in-container* path to the adapter directory (must contain
    ``adapter_config.json`` + ``adapter_model.safetensors``).

    Raises :class:`VLLMLoRAError` on non-2xx response. Does not raise if the
    adapter is already loaded under the same name (vLLM returns a 200 with
    a "Success" message in either case).
    """
    url = base_url.rstrip("/") + "/load_lora_adapter"
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    status, text = _post_json(url, payload, timeout_s=timeout_s)
    if status != 200:
        raise VLLMLoRAError(
            f"load_lora_adapter({lora_name}) failed: HTTP {status}: {text}"
        )
    logger.info("Loaded LoRA adapter %r from %s (resp=%s)", lora_name, lora_path, text.strip())


def unload_lora_adapter(
    *, base_url: str, lora_name: str, timeout_s: float = 60.0,
) -> None:
    """``POST /v1/unload_lora_adapter`` — drop a LoRA adapter at runtime.

    Idempotent-ish: vLLM returns 400 if the name isn't registered. We swallow
    that case (logged at INFO) so callers can free a name without first
    checking whether it was ever loaded.
    """
    url = base_url.rstrip("/") + "/unload_lora_adapter"
    payload = {"lora_name": lora_name}
    status, text = _post_json(url, payload, timeout_s=timeout_s)
    if status == 200:
        logger.info("Unloaded LoRA adapter %r (resp=%s)", lora_name, text.strip())
        return
    if status == 400 and "not found" in text.lower():
        logger.info("Unload skipped: adapter %r not registered (resp=%s)",
                    lora_name, text.strip())
        return
    raise VLLMLoRAError(
        f"unload_lora_adapter({lora_name}) failed: HTTP {status}: {text}"
    )


def list_loaded_models(*, base_url: str, timeout_s: float = 30.0) -> list[str]:
    """Return the list of model IDs currently registered with vLLM (base + adapters)."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise VLLMLoRAError(f"GET /v1/models failed: {exc}") from exc
    return [m["id"] for m in data.get("data", [])]


# ──────────────────────────────────────────────────────────────────────────
# docker compose lifecycle
# ──────────────────────────────────────────────────────────────────────────

def _run_compose(
    args: Sequence[str], *, cwd: Path | None = None, check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", *args]
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(  # noqa: S603 — explicit args, no shell
        cmd, cwd=str(cwd) if cwd else None, check=check,
    )


def compose_up_vllm(
    *,
    base_compose: Path,
    override: Path | None,
    cwd: Path,
    profile: str = "serve",
) -> None:
    """``docker compose up -d vllm`` with optional override file."""
    args = ["-f", str(base_compose)]
    if override is not None:
        args.extend(["-f", str(override)])
    args.extend(["--profile", profile, "up", "-d", "vllm"])
    _run_compose(args, cwd=cwd)


def compose_down_vllm(
    *,
    base_compose: Path,
    override: Path | None,
    cwd: Path,
    profile: str = "serve",
) -> None:
    """``docker compose stop`` + ``rm -f`` for the vllm service.

    We use ``stop`` + ``rm`` instead of ``down`` so other services in the
    same compose file (notably any ``training`` or ``litellm-proxy``
    containers from concurrent runs) are not torn down.
    """
    args = ["-f", str(base_compose)]
    if override is not None:
        args.extend(["-f", str(override)])
    args.extend(["--profile", profile, "stop", "vllm"])
    _run_compose(args, cwd=cwd, check=False)
    args2 = ["-f", str(base_compose)]
    if override is not None:
        args2.extend(["-f", str(override)])
    args2.extend(["--profile", profile, "rm", "-f", "vllm"])
    _run_compose(args2, cwd=cwd, check=False)


# ──────────────────────────────────────────────────────────────────────────
# /v1/models readiness probe
# ──────────────────────────────────────────────────────────────────────────

def wait_for_vllm(
    base_url: str,
    *,
    timeout_s: float = _PROBE_TIMEOUT_S,
    interval_s: float = _PROBE_INTERVAL_S,
) -> bool:
    """Poll ``base_url/models`` until 200 or timeout.

    Returns True on success, False on timeout. ``base_url`` should include
    the OpenAI-API ``/v1`` suffix (we'll append ``/models``).
    """
    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310
                if resp.status == 200:
                    logger.info("vLLM ready at %s", url)
                    return True
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        time.sleep(interval_s)
    logger.error("vLLM probe of %s timed out after %.1fs (last=%s)",
                 url, timeout_s, last_err)
    return False


# ──────────────────────────────────────────────────────────────────────────
# Adapter merge: LoRA → bf16
# ──────────────────────────────────────────────────────────────────────────

# Tiny one-shot merge script that runs *inside the training container*. It
# loads the base checkpoint + LoRA adapter, runs ``merge_and_unload``, and
# saves the merged bf16 weights to disk. We embed it as a string so the
# orchestrator can pipe it via ``docker compose run --rm training bash -c``
# without depending on the script existing in the image.
_MERGE_SCRIPT = r"""
import argparse, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(args.base)
        tok.save_pretrained(out)
    except Exception as exc:
        print(f"  (skipping tokenizer copy: {exc})", file=sys.stderr)
    try:
        proc = AutoProcessor.from_pretrained(args.base)
        proc.save_pretrained(out)
    except Exception as exc:
        print(f"  (skipping processor copy: {exc})", file=sys.stderr)
    print(f"merged adapter saved to {out}")

if __name__ == "__main__":
    main()
"""


def _write_merge_script(workdir: Path) -> Path:
    """Persist the merge driver alongside the run so docker can mount it."""
    p = workdir / "_merge_lora.py"
    p.write_text(_MERGE_SCRIPT, encoding="utf-8")
    return p


def merge_lora_to_bf16(
    *,
    base_checkpoint: str,
    adapter_path: str,
    output_path: str,
    repo_root: Path,
    training_container_name: str,
    workdir: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run the merge script inside the persistent training container.

    Writes the merge driver into ``workdir`` (visible to the container via
    the bind mount at /workspace/LangSlice), then `docker exec`s into the
    container named ``training_container_name`` to execute it. The
    container is the same one used by iSFT.iterate for train_sft +
    slicebench eval; reusing it skips Unsloth's ~5-7 min cold-start tax.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    script_path = _write_merge_script(workdir)
    # Convert host path to its in-container equivalent (the bind mount is
    # always ``/workspace/LangSlice``).
    try:
        script_rel = script_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        script_rel = Path(script_path)
    script_in_container = f"/workspace/LangSlice/{str(script_rel).replace(os.sep, '/')}"
    shell_cmd = (
        "set -euo pipefail && "
        "cd /workspace/LangSlice && "
        # Guarded editable install — no-op in the persistent container
        # since langslice_harness is installed once at dev_up time.
        "( python -c 'import langslice_harness' >/dev/null 2>&1 "
        "  || python -m pip install --no-deps -e . >/dev/null 2>&1 ) && "
        f"python {script_in_container} "
        f"--base {base_checkpoint} "
        f"--adapter {adapter_path} "
        f"--out {output_path}"
    )
    cmd = ["docker", "exec"]
    for k, v in (extra_env or {}).items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.extend([training_container_name, "bash", "-lc", shell_cmd])
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(  # noqa: S603 — explicit args, no shell
        cmd, cwd=str(repo_root), env=env, check=True,
    )
