"""Local ComfyUI integration utilities.

A small, general-purpose HTTP client for talking to a running ComfyUI server
plus a thin wrapper around `huggingface_hub.snapshot_download` for fetching
gated diffusion weights without touching the deprecated `huggingface-cli`.

Independent of any particular workflow or model — caller supplies the
workflow template and prompt. ComfyUI is treated as a black-box GPU server.
"""

from langslice_harness.comfyui.client import (
    ComfyUIClient,
    ComfyUIConfig,
    ComfyUIPromptError,
    ComfyUIServerUnreachable,
)
from langslice_harness.comfyui.download import (
    download_hf_snapshot,
)

__all__ = [
    "ComfyUIClient",
    "ComfyUIConfig",
    "ComfyUIPromptError",
    "ComfyUIServerUnreachable",
    "download_hf_snapshot",
]
