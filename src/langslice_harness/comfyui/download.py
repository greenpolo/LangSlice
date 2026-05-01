"""Generic Hugging Face snapshot downloader for gated diffusion weights.

Sidesteps the deprecated ``huggingface-cli`` and the newer ``hf`` CLI when
neither is conveniently available — calls ``huggingface_hub.snapshot_download``
directly. Useful for fetching ComfyUI model weights from gated repositories.

Usage as a script:
    python -m langslice_harness.comfyui.download \\
        --repo-id black-forest-labs/FLUX.2-klein-9b-nvfp4 \\
        --local-dir C:\\LabSoftware\\ComfyUI\\models\\diffusion_models

Or programmatically:
    from langslice_harness.comfyui import download_hf_snapshot
    download_hf_snapshot("repo/id", local_dir="...", allow_patterns=["*.safetensors"])
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Iterable
from pathlib import Path


def _resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    return getpass.getpass("HF token (read scope): ").strip()


def download_hf_snapshot(
    repo_id: str,
    *,
    local_dir: str | Path,
    token: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    max_workers: int = 4,
) -> str:
    """Download a Hugging Face repo snapshot.

    Returns the local path of the snapshot. Raises if the token is missing for
    a gated repo or if HF returns an error.
    """
    from huggingface_hub import login, snapshot_download

    resolved_token = token if token else _resolve_token(None)
    login(token=resolved_token, add_to_git_credential=False)

    local_dir = Path(local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        max_workers=max_workers,
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        token=resolved_token,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)

    print(f"Downloading {args.repo_id} -> {args.local_dir} ...", flush=True)
    path = download_hf_snapshot(
        args.repo_id,
        local_dir=args.local_dir,
        token=args.token,
        allow_patterns=args.allow_patterns,
        max_workers=args.max_workers,
    )
    print(f"\nSnapshot at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
