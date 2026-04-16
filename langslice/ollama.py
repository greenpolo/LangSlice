"""Ollama model management client.

Wraps the Ollama REST API for model listing, pulling, and deletion.
Used by the CLI (``langslice ollama ...``) and can be imported directly.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"


@dataclass
class ModelInfo:
    """Metadata for a locally-installed Ollama model."""

    name: str
    size: int  # bytes
    modified_at: str
    digest: str
    parameter_size: str = ""
    quantization_level: str = ""
    family: str = ""


class OllamaManager:
    """Thin client for the Ollama REST API.

    Parameters
    ----------
    host
        Base URL of the Ollama server (default ``http://localhost:11434``).
    timeout
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        timeout: float = 10.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return ``True`` if the Ollama server responds to a health check."""
        try:
            req = urllib.request.Request(f"{self.host}/", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def version(self) -> str | None:
        """Return the Ollama server version, or ``None`` if unreachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                return data.get("version")
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """List locally-installed models."""
        req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        models: list[ModelInfo] = []
        for m in data.get("models", []):
            models.append(
                ModelInfo(
                    name=m.get("name", ""),
                    size=m.get("size", 0),
                    modified_at=m.get("modified_at", ""),
                    digest=m.get("digest", ""),
                )
            )
        return models

    def model_info(self, name: str) -> dict[str, Any]:
        """Return detailed info for a model via ``POST /api/show``."""
        body = json.dumps({"model": name}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------
    # Pull / delete
    # ------------------------------------------------------------------

    def pull_model(
        self,
        name: str,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Download a model, streaming progress.

        Parameters
        ----------
        name
            Model tag to pull (e.g. ``gemma4:31b``).
        on_progress
            Callback ``(status, completed_bytes, total_bytes)`` invoked
            for each progress line from the server.
        """
        body = json.dumps({"model": name, "stream": True}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/pull",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = msg.get("status", "")
                completed = msg.get("completed", 0)
                total = msg.get("total", 0)
                if on_progress:
                    on_progress(status, completed, total)

    def delete_model(self, name: str) -> None:
        """Delete a locally-installed model."""
        body = json.dumps({"model": name}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/delete",
            data=body,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            _ = resp.read()

    # ------------------------------------------------------------------
    # Running models
    # ------------------------------------------------------------------

    def running_models(self) -> list[dict[str, Any]]:
        """List models currently loaded in VRAM."""
        req = urllib.request.Request(f"{self.host}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data.get("models", [])


# ======================================================================
# CLI helpers
# ======================================================================


def _format_size(nbytes: int) -> str:
    """Human-readable byte size."""
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1_000_000_000:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.0f} MB"
    return f"{nbytes} B"


def cli_status(host: str) -> None:
    """``langslice ollama status`` implementation."""
    mgr = OllamaManager(host=host)
    if mgr.is_running():
        ver = mgr.version() or "unknown"
        print(f"Ollama is running at {host} (version {ver})")
        running = mgr.running_models()
        if running:
            print(f"  Loaded models: {len(running)}")
            for m in running:
                print(f"    - {m.get('name', '?')}")
    else:
        print(f"Ollama is not running at {host}")
        sys.exit(1)


def cli_list(host: str) -> None:
    """``langslice ollama list`` implementation."""
    mgr = OllamaManager(host=host)
    models = mgr.list_models()
    if not models:
        print("No models installed.")
        return
    print(f"{'NAME':<30} {'SIZE':>10}   MODIFIED")
    print("-" * 65)
    for m in models:
        print(f"{m.name:<30} {_format_size(m.size):>10}   {m.modified_at[:10]}")


def cli_pull(host: str, name: str) -> None:
    """``langslice ollama pull`` implementation with progress bar."""
    mgr = OllamaManager(host=host)
    print(f"Pulling {name}...")

    last_status = ""

    def _on_progress(status: str, completed: int, total: int) -> None:
        nonlocal last_status
        if total > 0:
            pct = completed / total * 100
            bar_len = 30
            filled = int(bar_len * completed // total)
            bar = "█" * filled + "░" * (bar_len - filled)
            line = f"\r  {bar} {pct:5.1f}%  {_format_size(completed)} / {_format_size(total)}"
            sys.stdout.write(f"{line}  {status}")
            sys.stdout.flush()
        elif status != last_status:
            print(f"  {status}")
            last_status = status

    mgr.pull_model(name, on_progress=_on_progress)
    print()  # newline after progress bar
    print(f"Successfully pulled {name}")


def cli_remove(host: str, name: str) -> None:
    """``langslice ollama remove`` implementation."""
    mgr = OllamaManager(host=host)
    mgr.delete_model(name)
    print(f"Deleted {name}")
