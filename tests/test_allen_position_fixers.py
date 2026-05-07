from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO / "_local" / "eval" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_connectivity_alignment_uses_each_anchor_image_dimensions(monkeypatch):
    mod = _load_script("fix_allen_connectivity_positions.py")
    seen: list[tuple[int, float, float]] = []

    def fake_dims(image_id: int) -> tuple[int, int]:
        return {
            100: (200, 400),
            200: (800, 1000),
        }[image_id]

    def fake_coords(image_id: int, x_px: float, y_px: float) -> dict[str, float]:
        seen.append((image_id, x_px, y_px))
        return {"x": 1000.0 if image_id == 100 else 2000.0}

    monkeypatch.setattr(mod, "fetch_section_image_dims", fake_dims)
    monkeypatch.setattr(mod, "fetch_image_to_reference", fake_coords)

    mod.fit_brain_alignment(1, [(1, 100), (9, 200)], "x", True)

    assert seen == [(100, 100.0, 200.0), (200, 400.0, 500.0)]


def test_dev_alignment_uses_each_anchor_image_dimensions(monkeypatch):
    mod = _load_script("fix_allen_dev_positions.py")
    seen: list[tuple[int, float, float]] = []

    def fake_dims(image_id: int) -> tuple[int, int]:
        return {
            100: (160, 300),
            200: (640, 900),
        }[image_id]

    def fake_coords(image_id: int, x_px: float, y_px: float) -> dict[str, float]:
        seen.append((image_id, x_px, y_px))
        return {"x": 1000.0 if image_id == 100 else 2000.0}

    monkeypatch.setattr(mod, "fetch_image_dims", fake_dims)
    monkeypatch.setattr(mod, "fetch_coords", fake_coords)

    mod.fit_brain([(1, 100), (9, 200)], "coronal", 9)

    assert seen == [(100, 80.0, 150.0), (200, 320.0, 450.0)]


def test_dev_apply_corrections_marks_failed_brain_records(monkeypatch, tmp_path):
    mod = _load_script("fix_allen_dev_positions.py")
    dataset = "allen_dev_coronal"
    meta_path = tmp_path / "metadata.json"
    meta = [
        {"dataset_id": 1, "section_number": 1, "section_image_id": 11},
        {"dataset_id": 2, "section_number": 1, "section_image_id": 21},
    ]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(mod, "DATASETS", [dataset])
    monkeypatch.setattr(
        mod,
        "load_metadata",
        lambda ds: (meta_path, json.loads(meta_path.read_text(encoding="utf-8"))),
    )

    pos_by_image, failed_by_image, n_meta = mod.apply_corrections(
        {dataset: {1: ("P56", "coronal", [(1, 11)]), 2: ("P56", "coronal", [(1, 21)])}},
        {dataset: {1: (1.0, 0.1, "admba_3d_p56_mouse_25um")}},
        {dataset: {2: "fit_failed"}},
        dry_run=True,
    )

    assert pos_by_image == {11: (1.1, "admba_3d_p56_mouse_25um")}
    assert failed_by_image == {21: "fit_failed"}
    assert n_meta == 2


def test_flip_ap_positions_skips_non_ap_rows_by_default(monkeypatch, tmp_path):
    mod = _load_script("flip_ap_positions.py")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "ap",
                        "dataset": "ebrains_tta_atlas",
                        "subject_id": "S1",
                        "atlas": "allen_mouse_25um",
                        "orientation": "coronal",
                        "slice_axis": "ap",
                        "position_mm": 2.0,
                    }
                ),
                json.dumps(
                    {
                        "id": "dv",
                        "dataset": "ebrains_tta_atlas",
                        "subject_id": "S1",
                        "atlas": "allen_mouse_25um",
                        "orientation": "horizontal",
                        "slice_axis": "dv",
                        "position_mm": 2.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result, affected = mod.fix_jsonl(
        path,
        {"ebrains_tta_atlas"},
        {},
        {"S1"},
        {"allen_mouse_25um": 13.175},
        False,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["position_mm"] == 11.175
    assert rows[1]["position_mm"] == 2.0
    assert affected == {"ap"}
    assert result["n_changed"] == 1


def test_silva_abba_conversion_applies_minus_one_anterior_shift():
    # Silva ABBA → BG-ASR shift: subtract 1.0 mm to undo ABBA's
    # anterior-positive offset (recorded in manifest as
    # silva_abba_raw_minus_1mm_anterior_shift).
    mod = _load_script("fix_silva_ieg_positions.py")

    assert mod._convert(8.9, 5.4) == 7.9


def test_brun_midline_distance_conversion_uses_hemisphere():
    mod = _load_script("fix_brun2024_p14rat_positions.py")

    assert mod.convert_midline_distance(3.4, "RH", 19.929) == 6.5645
    assert mod.convert_midline_distance(3.4, "LH", 19.929) == 13.3645


def test_download_silva_uses_same_raw_bg_asr_ap_conversion():
    mod = _load_script("download_silva_ieg.py")

    assert mod.abba_location_to_bg_asr_mm(8.9) == 8.9


def test_extract_brun_uses_same_hemisphere_conversion():
    mod = _load_script("extract_brun2024.py")

    assert mod.midline_distance_to_whs_ml(3.4, "RH") == 6.5645
    assert mod.midline_distance_to_whs_ml(3.4, "LH") == 13.3645
