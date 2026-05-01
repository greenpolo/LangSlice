"""CLI tests for the augmentation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from augmentation.cli import _build_parser, main

# ---------------------------------------------------------------------------
# Fake atlas fixture
# ---------------------------------------------------------------------------


def _make_fake_atlas() -> Any:
    """Build a minimal object matching the BrainGlobe atlas duck-type."""
    atlas = MagicMock()
    atlas.atlas_name = "fake_atlas"
    atlas.orientation = "asr"

    # 5 coronal slices, 32 DV, 32 ML
    ref = np.random.default_rng(0).integers(0, 255, (5, 32, 32), dtype=np.uint8)
    atlas.reference = ref

    ann = np.zeros((5, 32, 32), dtype=np.int32)
    ann[:, 8:24, 8:24] = 315  # non-zero region in the middle
    atlas.annotation = ann

    atlas.resolution = (200.0, 200.0, 200.0)  # µm, gives 5 * 0.2mm = 1.0mm range
    atlas.metadata = {"species": "fake"}

    atlas.structures = {315: {"acronym": "CTX", "name": "Cortex", "rgb_triplet": [100, 150, 200]}}

    return atlas


@pytest.fixture
def fake_atlas() -> Any:
    return _make_fake_atlas()


# ---------------------------------------------------------------------------
# Helpers to patch atlas loading
# ---------------------------------------------------------------------------

_LOAD_ATLAS_TARGET = "augmentation.cli._load_atlas"


# ---------------------------------------------------------------------------
# render-grid tests
# ---------------------------------------------------------------------------


def test_render_grid_produces_png_and_manifest(tmp_path: Path, fake_atlas: Any) -> None:
    with patch(_LOAD_ATLAS_TARGET, return_value=fake_atlas):
        exit_code = main(
            [
                "--seed",
                "42",
                "render-grid",
                "--atlas",
                "fake_atlas",
                "--modalities",
                "dapi",
                "--count",
                "2",
                "--out",
                str(tmp_path),
            ]
        )

    assert exit_code == 0, f"CLI returned non-zero exit code: {exit_code}"
    png_path = tmp_path / "dapi.png"
    assert png_path.exists(), "Expected dapi.png in output dir"
    manifest_path = tmp_path / "dapi_manifest.json"
    assert manifest_path.exists(), "Expected dapi_manifest.json in output dir"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["modality"] == "dapi"
    assert manifest["count"] == 2
    assert len(manifest["cells"]) == 2


def test_render_grid_multiple_modalities(tmp_path: Path, fake_atlas: Any) -> None:
    with patch(_LOAD_ATLAS_TARGET, return_value=fake_atlas):
        exit_code = main(
            [
                "render-grid",
                "--atlas",
                "fake_atlas",
                "--modalities",
                "nissl",
                "brightfield",
                "--count",
                "2",
                "--out",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert (tmp_path / "nissl.png").exists()
    assert (tmp_path / "brightfield.png").exists()


# ---------------------------------------------------------------------------
# generate tests
# ---------------------------------------------------------------------------


def test_generate_produces_correct_count(tmp_path: Path, fake_atlas: Any) -> None:
    out_dir = tmp_path / "gen"
    with patch(_LOAD_ATLAS_TARGET, return_value=fake_atlas):
        exit_code = main(
            [
                "--seed",
                "7",
                "generate",
                "--atlas",
                "fake_atlas",
                "--modality",
                "nissl",
                "--count",
                "2",
                "--out",
                str(out_dir),
            ]
        )

    assert exit_code == 0
    pngs = list(out_dir.glob("nissl_*.png"))
    assert len(pngs) == 2, f"Expected 2 PNGs, got {len(pngs)}"


# ---------------------------------------------------------------------------
# inspect tests
# ---------------------------------------------------------------------------


def test_inspect_produces_png(tmp_path: Path, fake_atlas: Any) -> None:
    out_png = tmp_path / "inspect.png"
    with patch(_LOAD_ATLAS_TARGET, return_value=fake_atlas):
        exit_code = main(
            [
                "--seed",
                "0",
                "inspect",
                "--atlas",
                "fake_atlas",
                "--ap-mm",
                "0.2",
                "--modality",
                "nissl",
                "--out",
                str(out_png),
            ]
        )

    assert exit_code == 0
    assert out_png.exists(), "Expected output PNG from inspect"

    from PIL import Image

    img = np.array(Image.open(out_png))
    h, w = img.shape[:2]
    assert w >= h, f"Expected wide side-by-side image, got {h}x{w}"


# ---------------------------------------------------------------------------
# Argument parsing smoke tests
# ---------------------------------------------------------------------------


def test_parser_render_grid_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["render-grid", "--out", "/tmp/x"])
    assert args.command == "render-grid"
    assert args.count == 50
    assert args.seed == 42


def test_parser_generate_modality_required() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--out", "/tmp/x"])


def test_parser_inspect_ap_mm_required() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["inspect", "--modality", "dapi", "--out", "/tmp/x"])
