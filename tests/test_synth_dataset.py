from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth_dataset as sd


class _FakeAtlas:
    def __init__(self, atlas_name: str, version: str) -> None:
        self.atlas_name = atlas_name
        self.metadata = {"version": version}
        self.resolution = (25.0, 12.5, 10.0)
        self.reference = np.zeros((4, 8, 8), dtype=np.uint8)
        self.annotation = np.zeros((4, 8, 8), dtype=np.int32)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows


@pytest.fixture
def patched_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeAtlas]:
    atlas_a = _FakeAtlas("atlas_a", "v1")
    atlas_b = _FakeAtlas("atlas_b", "v2")
    atlas_map = {"atlas_a": atlas_a, "atlas_b": atlas_b}

    def fake_load_atlas(name: str) -> _FakeAtlas:
        return atlas_map[str(name)]

    def fake_get_position_range_mm(
        atlas: _FakeAtlas, *, plane: str = "coronal"
    ) -> tuple[float, float]:
        _ = plane
        return 0.0, 9.0

    def fake_get_oblique_slice(
        atlas: _FakeAtlas,
        *,
        base_position_mm: float,
        plane: str = "coronal",
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = plane
        h, w = output_size if output_size is not None else (24, 32)
        base = int(round(base_position_mm * 10.0 + abs(yaw_deg) + abs(pitch_deg) + abs(roll_deg)))
        atlas_bias = 7 if atlas.atlas_name == "atlas_a" else 13
        ref = np.full((h, w), (base + atlas_bias) % 255, dtype=np.uint8)
        ann = np.full((h, w), 1 if atlas.atlas_name == "atlas_a" else 2, dtype=np.int32)
        return ref, ann

    def make_renderer(offset: int):
        def _renderer(
            reference_slice: np.ndarray,
            annotation_slice: np.ndarray,
            atlas: object,
            *,
            seed: int,
            pixel_size_um: float,
            **kwargs: Any,
        ) -> np.ndarray:
            _ = annotation_slice, atlas, pixel_size_um, kwargs
            rng = np.random.default_rng(seed + offset)
            h, w = reference_slice.shape[:2]
            return rng.random((h, w, 3), dtype=np.float32)

        return _renderer

    monkeypatch.setattr(sd, "load_atlas", fake_load_atlas)
    monkeypatch.setattr(sd, "get_position_range_mm", fake_get_position_range_mm)
    monkeypatch.setattr(sd, "get_oblique_slice", fake_get_oblique_slice)
    monkeypatch.setattr(sd, "render_dapi_section", make_renderer(11))
    monkeypatch.setattr(sd, "render_nissl_section", make_renderer(23))
    monkeypatch.setattr(sd, "render_brightfield_section", make_renderer(37))
    monkeypatch.setattr(sd, "render_fluorescence_section", make_renderer(41))
    monkeypatch.setattr(sd, "render_ish_section", make_renderer(53))
    return atlas_map


def test_determinism_same_seed_same_png_hashes(
    tmp_path: Path, patched_env: dict[str, _FakeAtlas]
) -> None:
    _ = patched_env
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    sd.write_dataset(out1, 40, 123, ["atlas_a", "atlas_b"])
    sd.write_dataset(out2, 40, 123, ["atlas_a", "atlas_b"])

    m1 = _read_manifest(out1 / "manifest.jsonl")
    m2 = _read_manifest(out2 / "manifest.jsonl")
    seeds1 = [int(row["seed"]) for row in m1]
    seeds2 = [int(row["seed"]) for row in m2]
    assert seeds1 == seeds2

    hashes1 = [_sha256_file(out1 / "images" / f"{seed:08d}.png") for seed in seeds1]
    hashes2 = [_sha256_file(out2 / "images" / f"{seed:08d}.png") for seed in seeds2]
    assert hashes1 == hashes2


def test_mix_weight_empirical_match(patched_env: dict[str, _FakeAtlas]) -> None:
    _ = patched_env
    weights = {
        "dapi": 0.45,
        "nissl": 0.22,
        "fluorescence": 0.18,
        "ish": 0.10,
        "brightfield": 0.05,
    }
    rng = np.random.default_rng(7)
    counts = {k: 0 for k in weights}
    n = 500
    for _ in range(n):
        spec = sd.sample_spec(rng, atlases=["atlas_a", "atlas_b"], mix_weights=weights)
        counts[spec.modality] += 1

    for modality, weight in weights.items():
        frac = counts[modality] / n
        assert abs(frac - weight) <= 0.05, (
            f"{modality}: frac={frac:.3f}, weight={weight:.3f}"
        )


@pytest.mark.parametrize(
    ("modality", "mode", "counterstain"),
    [
        ("dapi", None, None),
        ("nissl", None, None),
        ("brightfield", "pan_neuronal", "auto"),
        ("fluorescence", sd.FLUORESCENCE_MODES[0].name, None),
        ("ish", sd.ISH_MODES[0].name, None),
    ],
)
def test_render_dispatch_outputs_hwc_float32(
    patched_env: dict[str, _FakeAtlas],
    modality: str,
    mode: str | None,
    counterstain: str | None,
) -> None:
    _ = patched_env
    spec = sd.SynthSpec(
        atlas_name="atlas_a",
        atlas_version="v1",
        plane="coronal",
        position_mm=4.5,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        modality=modality,
        mode=mode,
        counterstain=counterstain,
        damage_intensity="medium",
        apply_damage=True,
        seed=99,
    )
    image, meta = sd.render(spec)

    assert image.dtype == np.float32
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0

    expected = {f.name for f in fields(sd.SynthSpec)} | {"shape", "generator_version"}
    assert set(meta.keys()) == expected


def test_holdout_atlases_exclusion(patched_env: dict[str, _FakeAtlas]) -> None:
    _ = patched_env
    rng = np.random.default_rng(5)
    for _ in range(200):
        spec = sd.sample_spec(
            rng,
            atlases=["atlas_a", "atlas_b"],
            holdout_atlases=("atlas_b",),
        )
        assert spec.atlas_name == "atlas_a"


def test_manifest_roundtrip_keys_and_types(
    tmp_path: Path, patched_env: dict[str, _FakeAtlas]
) -> None:
    _ = patched_env
    out_dir = tmp_path / "roundtrip"
    sd.write_dataset(out_dir, 12, 17, ["atlas_a", "atlas_b"])

    rows = _read_manifest(out_dir / "manifest.jsonl")
    assert len(rows) == 12

    expected = {f.name for f in fields(sd.SynthSpec)} | {"shape", "generator_version"}
    for row in rows:
        assert set(row.keys()) == expected
        assert isinstance(row["atlas_name"], str)
        assert isinstance(row["atlas_version"], str)
        assert isinstance(row["plane"], str)
        assert isinstance(row["position_mm"], float)
        assert isinstance(row["yaw_deg"], float)
        assert isinstance(row["pitch_deg"], float)
        assert isinstance(row["roll_deg"], float)
        assert isinstance(row["modality"], str)
        assert row["mode"] is None or isinstance(row["mode"], str)
        assert row["counterstain"] is None or isinstance(row["counterstain"], str)
        assert isinstance(row["damage_intensity"], str)
        assert isinstance(row["apply_damage"], bool)
        assert isinstance(row["seed"], int)
        assert isinstance(row["shape"], list)
        assert len(row["shape"]) == 3
        assert all(isinstance(x, int) for x in row["shape"])
        assert row["generator_version"] == sd.__version__


def test_position_strata_thirds_empirical_balance(patched_env: dict[str, _FakeAtlas]) -> None:
    _ = patched_env
    rng = np.random.default_rng(42)
    counts = [0, 0, 0]
    n = 300
    for _ in range(n):
        spec = sd.sample_spec(
            rng,
            atlases=["atlas_a"],
            position_strata="thirds",
            plane_weights={"coronal": 1.0, "sagittal": 0.0, "horizontal": 0.0},
        )
        if spec.position_mm < 3.0:
            counts[0] += 1
        elif spec.position_mm < 6.0:
            counts[1] += 1
        else:
            counts[2] += 1

    for c in counts:
        frac = c / n
        assert abs(frac - (1.0 / 3.0)) <= 0.10, f"third fraction={frac:.3f}"
