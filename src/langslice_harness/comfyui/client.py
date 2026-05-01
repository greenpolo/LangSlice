"""ComfyUI HTTP client.

A general-purpose Python wrapper around ComfyUI's REST API. Submits a workflow
graph (API-format JSON), polls for completion, and fetches the output image(s).
Caller supplies the workflow template and any per-call parameter overrides;
this module knows nothing about specific models or use cases.

Setup:
    1. Run ComfyUI 0.20+ at the base URL (default http://127.0.0.1:8000).
    2. Place model weights in ComfyUI/models/ as the workflow expects.
    3. Construct ``ComfyUIClient(config=...)`` with the workflow template.

Output contract:
    ``submit(...)`` returns HWC float32 in [0, 1] for the first SaveImage
    output node. Multi-image and non-image outputs are not supported here —
    extend if needed.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

__all__ = [
    "ComfyUIClient",
    "ComfyUIConfig",
    "ComfyUIPromptError",
    "ComfyUIServerUnreachable",
]


class ComfyUIServerUnreachable(RuntimeError):
    """Raised when the HTTP endpoint is not reachable (server down, port shift, ...)."""


class ComfyUIPromptError(RuntimeError):
    """Raised when ComfyUI returns a node-level error or times out the workflow."""


@dataclass
class ComfyUIConfig:
    base_url: str = "http://127.0.0.1:8000"
    comfy_input_dir: Path | None = None
    """Where to stage uploaded reference images. None = disable disk staging
    (caller is expected to upload via /upload/image themselves)."""

    request_timeout_s: int = 30
    inference_timeout_s: int = 300
    poll_interval_s: float = 1.0
    verify_template_sha: bool = True


def _np_from_png_bytes(data: bytes) -> np.ndarray:
    from PIL import Image

    pil = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0


def _png_bytes_from_np(image: np.ndarray) -> bytes:
    from PIL import Image

    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


class ComfyUIClient:
    """Generic ComfyUI workflow runner over HTTP.

    Workflow templates are API-format JSON dicts (node_id → {class_type, inputs}).
    Callers parameterize by passing an `overrides` mapping that is shallow-merged
    onto the template's `inputs` per node ID.
    """

    def __init__(self, config: ComfyUIConfig | None = None) -> None:
        self.config = config or ComfyUIConfig()
        if self.config.comfy_input_dir is not None:
            self.config.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        self.ping()

    # ---------- public API ----------

    def ping(self) -> dict:
        """Check the server is reachable and return its system_stats payload."""
        url = f"{self.config.base_url}/system_stats"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:
            raise ComfyUIServerUnreachable(
                f"ComfyUI not reachable at {self.config.base_url} ({e})."
            ) from e

    def stage_image(self, image: np.ndarray, name: str) -> Path:
        """Save *image* (HWC float32 [0,1] or uint8) into ComfyUI's input dir."""
        if self.config.comfy_input_dir is None:
            raise RuntimeError("comfy_input_dir not set; cannot stage images.")
        path = self.config.comfy_input_dir / name
        path.write_bytes(_png_bytes_from_np(image))
        return path

    def submit(
        self,
        workflow: Mapping[str, dict],
        *,
        timeout_s: int | None = None,
    ) -> np.ndarray:
        """Submit workflow, poll, return first image as HWC float32 [0, 1]."""
        prompt_id = self._post_prompt(dict(workflow))
        history_entry = self._poll(prompt_id, timeout_s=timeout_s)
        return _np_from_png_bytes(self._fetch_first_image(history_entry))

    def submit_raw(
        self,
        workflow: Mapping[str, dict],
        *,
        timeout_s: int | None = None,
    ) -> bytes:
        """Same as ``submit`` but returns the raw PNG bytes."""
        prompt_id = self._post_prompt(dict(workflow))
        history_entry = self._poll(prompt_id, timeout_s=timeout_s)
        return self._fetch_first_image(history_entry)

    # ---------- workflow-template helpers ----------

    @staticmethod
    def load_template(
        path: Path,
        *,
        sha256_path: Path | None = None,
        verify_sha: bool = True,
    ) -> dict:
        """Load and (optionally) checksum-verify an API-format workflow JSON."""
        raw = path.read_bytes()
        if verify_sha and sha256_path is not None and sha256_path.exists():
            expected = sha256_path.read_text().strip().split()[0]
            actual = hashlib.sha256(raw).hexdigest()
            if expected != actual:
                raise RuntimeError(
                    f"Workflow SHA256 mismatch for {path.name} "
                    f"(expected {expected[:16]}…, got {actual[:16]}…)."
                )
        return json.loads(raw)

    @staticmethod
    def apply_overrides(
        template: dict,
        overrides: Mapping[str, Mapping[str, object]],
    ) -> dict:
        """Return a deep-copy of *template* with per-node-id ``inputs`` updates applied."""
        wf = json.loads(json.dumps(template))
        for node_id, node_overrides in overrides.items():
            if node_id not in wf:
                raise KeyError(f"workflow has no node id {node_id!r}")
            wf[node_id].setdefault("inputs", {}).update(dict(node_overrides))
        return wf

    # ---------- internals ----------

    def _post_prompt(self, workflow: dict) -> str:
        body = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
        req = urllib.request.Request(
            f"{self.config.base_url}/prompt",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout_s) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:1500]
            raise ComfyUIPromptError(f"submit failed HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ComfyUIServerUnreachable(f"submit failed: {e}") from e
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise ComfyUIPromptError(f"no prompt_id in response: {resp}")
        if resp.get("node_errors"):
            raise ComfyUIPromptError(f"node_errors: {resp['node_errors']}")
        return prompt_id

    def _poll(self, prompt_id: str, *, timeout_s: int | None = None) -> dict:
        deadline = time.time() + (timeout_s or self.config.inference_timeout_s)
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.config.base_url}/history/{prompt_id}",
                    timeout=10,
                ) as r:
                    history = json.loads(r.read())
            except (urllib.error.URLError, OSError) as e:
                raise ComfyUIServerUnreachable(f"history poll failed: {e}") from e
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False):
                    return entry
                if status.get("status_str") == "error":
                    raise ComfyUIPromptError(
                        f"comfy execution error: {json.dumps(status)[:2000]}"
                    )
            time.sleep(self.config.poll_interval_s)
        raise ComfyUIPromptError(
            f"prompt {prompt_id} timed out"
        )

    def _fetch_first_image(self, history_entry: dict) -> bytes:
        outputs = history_entry.get("outputs", {})
        for _node_id, node_out in outputs.items():
            for img in node_out.get("images", []):
                qs = urllib.parse.urlencode({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
                with urllib.request.urlopen(
                    f"{self.config.base_url}/view?{qs}",
                    timeout=self.config.request_timeout_s,
                ) as r:
                    return r.read()
        raise ComfyUIPromptError(f"no images in outputs: {outputs}")
