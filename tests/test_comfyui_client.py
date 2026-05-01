"""Tests for the general-purpose ComfyUIClient using mocked HTTP."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _png_bytes(color: tuple[int, int, int] = (50, 100, 200), size: int = 32) -> bytes:
    from PIL import Image

    arr = np.full((size, size, 3), color, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class _MockResp:
    def __init__(self, payload: bytes | dict) -> None:
        self._payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._payload


def _mock_urlopen(url, *_args, **_kwargs):
    url_str = url.full_url if hasattr(url, "full_url") else str(url)

    if url_str.endswith("/system_stats"):
        return _MockResp({
            "system": {"comfyui_version": "0.20.1"},
            "devices": [{"vram_total": 34_000_000_000, "vram_free": 30_000_000_000}],
        })
    if url_str.endswith("/prompt"):
        return _MockResp({"prompt_id": "fake-prompt-123", "node_errors": {}})
    if "/history/" in url_str:
        return _MockResp({
            "fake-prompt-123": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    "99": {
                        "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
                    }
                },
            }
        })
    if "/view?" in url_str:
        return _MockResp(_png_bytes())
    raise RuntimeError(f"unexpected URL in mock: {url_str}")


@pytest.fixture
def comfy_client(tmp_path: Path):
    from langslice_harness.comfyui import ComfyUIClient, ComfyUIConfig

    config = ComfyUIConfig(comfy_input_dir=tmp_path / "comfy_in")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
        client = ComfyUIClient(config=config)
    return client, tmp_path


def _trivial_workflow() -> dict:
    return {
        "20": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "60": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "hello", "clip": ["10", 0]},
        },
        "99": {
            "class_type": "SaveImage",
            "inputs": {"images": ["91", 0], "filename_prefix": "out"},
        },
    }


def test_ping_succeeds(comfy_client) -> None:
    client, _ = comfy_client
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
        stats = client.ping()
    assert stats["system"]["comfyui_version"] == "0.20.1"


def test_unreachable_server_raises() -> None:
    from langslice_harness.comfyui import ComfyUIClient, ComfyUIConfig, ComfyUIServerUnreachable

    cfg = ComfyUIConfig(base_url="http://127.0.0.1:1")
    with pytest.raises(ComfyUIServerUnreachable):
        ComfyUIClient(config=cfg)


def test_submit_returns_hwc_float32_in_unit_range(comfy_client) -> None:
    client, _ = comfy_client
    wf = _trivial_workflow()
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
        out = client.submit(wf)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 3 and out.shape[2] == 3
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_submit_raw_returns_png_bytes(comfy_client) -> None:
    client, _ = comfy_client
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
        data = client.submit_raw(_trivial_workflow())
    assert isinstance(data, bytes)
    assert data.startswith(b"\x89PNG")


def test_apply_overrides_replaces_inputs() -> None:
    from langslice_harness.comfyui import ComfyUIClient

    template = _trivial_workflow()
    out = ComfyUIClient.apply_overrides(template, {
        "20": {"image": "atlas.png"},
        "60": {"text": "new prompt"},
    })
    assert out["20"]["inputs"]["image"] == "atlas.png"
    assert out["60"]["inputs"]["text"] == "new prompt"
    # original template is not mutated
    assert template["20"]["inputs"]["image"] == "x.png"


def test_apply_overrides_unknown_node_raises() -> None:
    from langslice_harness.comfyui import ComfyUIClient

    template = _trivial_workflow()
    with pytest.raises(KeyError, match="42"):
        ComfyUIClient.apply_overrides(template, {"42": {"text": "nope"}})


def test_load_template_sha_mismatch(tmp_path: Path) -> None:
    from langslice_harness.comfyui import ComfyUIClient

    wf_path = tmp_path / "wf.json"
    sha_path = tmp_path / "wf.json.sha256"
    wf_path.write_text(json.dumps(_trivial_workflow()))
    sha_path.write_text("0" * 64 + "\n")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        ComfyUIClient.load_template(wf_path, sha256_path=sha_path, verify_sha=True)


def test_load_template_skips_sha_when_disabled(tmp_path: Path) -> None:
    from langslice_harness.comfyui import ComfyUIClient

    wf_path = tmp_path / "wf.json"
    wf_path.write_text(json.dumps(_trivial_workflow()))
    out = ComfyUIClient.load_template(wf_path, verify_sha=False)
    assert "20" in out


def test_stage_image_writes_png(comfy_client) -> None:
    client, tmp_path = comfy_client
    img = np.zeros((16, 16, 3), dtype=np.float32)
    img[..., 0] = 0.5
    p = client.stage_image(img, "test.png")
    assert p.exists()
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_stage_image_no_input_dir_raises() -> None:
    from langslice_harness.comfyui import ComfyUIClient, ComfyUIConfig

    cfg = ComfyUIConfig(comfy_input_dir=None)
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
        client = ComfyUIClient(config=cfg)
    with pytest.raises(RuntimeError, match="comfy_input_dir"):
        client.stage_image(np.zeros((4, 4, 3), dtype=np.float32), "x.png")
