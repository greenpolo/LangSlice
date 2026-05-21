"""Run the separability gate on procedural DAPI vs real Stæger DAPI.

End-to-end:
1. Inventory real DAPI PNGs at data/datasets/dapi_pure/staeger_2020/, extract
   per-file subject ID (mouse name, e.g. "BM03") from the filename.
2. Generate ~120 procedural DAPI sections via render_dapi_section at random
   APs (sampled from the atlas's valid range) and per-image seeds, saved as
   PNGs in tmp/gate_aug_dapi/.
3. Call run_separability_gate with explicit path lists; print result + write
   JSON report.

Pass condition: held-out accuracy ≤ 0.55. Anything higher means the augmented
set has a detectable signature.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import dataclasses

import numpy as np
from augmentation.dapi_pipeline import render_dapi_section
from augmentation.validate import run_separability_gate
from PIL import Image

from langslice_harness.atlas.core import (
    get_position_range_mm,
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

REAL_DIR = Path("data/datasets/dapi_pure/staeger_2020")
AUG_DIR = Path("tmp/gate_aug_dapi")
REPORT_PATH = Path("tmp/outputs/dapi/gate_report.json")
N_AUG = 120
TARGET_PX_UM = 10.0
MASTER_SEED = 20260428

_SUBJECT_RE = re.compile(r"^(BM\d+)_S\d+\.png$")


def list_real_pngs() -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    subjects: list[str] = []
    for p in sorted(REAL_DIR.glob("*.png")):
        m = _SUBJECT_RE.match(p.name)
        if not m:
            continue
        paths.append(p)
        subjects.append(m.group(1))
    return paths, subjects


def upsampled_inputs(atlas: object, ap_mm: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)  # type: ignore[arg-type]
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]

    src_um = atlas.resolution[0]  # type: ignore[attr-defined]
    scale = src_um / TARGET_PX_UM
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    ref_arr = np.asarray(pil.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )
    return ref_arr, ann_up


def generate_augmented_set(atlas: object, n: int, master_seed: int) -> list[Path]:
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    # Drop any prior renders so nothing stale leaks into the gate.
    for old in AUG_DIR.glob("*.png"):
        old.unlink()

    rng = np.random.default_rng(master_seed)
    ap_lo, ap_hi = get_position_range_mm(atlas)  # type: ignore[arg-type]
    # Trim 0.5mm off each end so we don't render frames too close to the
    # cerebellum / olfactory-bulb extremes where annotation gets thin.
    ap_lo += 0.5
    ap_hi -= 0.5

    out_paths: list[Path] = []
    for i in range(n):
        ap = float(rng.uniform(ap_lo, ap_hi))
        seed = int(rng.integers(0, 2**31))
        t0 = time.time()
        ref, ann = upsampled_inputs(atlas, ap)
        img = render_dapi_section(
            ref,
            ann,
            atlas,
            seed=seed,
            pixel_size_um=TARGET_PX_UM,
        )
        arr = np.clip(img * 255, 0, 255).astype(np.uint8)
        path = AUG_DIR / f"aug_{i:04d}_ap{ap:05.2f}_s{seed}.png"
        Image.fromarray(arr, "RGB").save(path)
        out_paths.append(path)
        elapsed = time.time() - t0
        if i % 10 == 0 or i == n - 1:
            print(f"  [{i + 1:>3}/{n}] AP={ap:5.2f}mm  {elapsed:5.1f}s -> {path.name}", flush=True)
    return out_paths


def main() -> int:
    print("Loading atlas...", flush=True)
    atlas = load_atlas("allen_mouse_25um")

    print(f"\nInventorying real DAPI from {REAL_DIR}", flush=True)
    real_paths, real_subjects = list_real_pngs()
    n_subjects = len(set(real_subjects))
    print(f"  {len(real_paths)} files across {n_subjects} subjects", flush=True)

    print(f"\nGenerating {N_AUG} augmented sections @ {TARGET_PX_UM} µm/px ...", flush=True)
    t0 = time.time()
    aug_paths = generate_augmented_set(atlas, N_AUG, MASTER_SEED)
    print(f"  generated {len(aug_paths)} in {(time.time() - t0) / 60:.1f} min", flush=True)

    print("\nRunning separability gate...", flush=True)
    t0 = time.time()
    result = run_separability_gate(
        modality="dapi",
        real_image_paths=real_paths,
        real_subject_ids=real_subjects,
        augmented_image_paths=aug_paths,
        n_train_real=80,
        n_test_real=40,
        n_train_aug=80,
        n_test_aug=40,
        seed=MASTER_SEED,
        device="auto",
    )
    print(f"  gate completed in {time.time() - t0:.1f}s", flush=True)

    print("\n" + "=" * 60)
    print(f"  modality:    {result.modality}")
    print(f"  accuracy:    {result.accuracy:.4f}")
    print(f"  auc:         {result.auc:.4f}")
    print(
        f"  ci 95%:      [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]"
    )
    print(f"  n_train:     {result.n_train}  ({result.n_real_subjects_train} real subjects)")
    print(f"  n_test:      {result.n_test}  ({result.n_real_subjects_test} real subjects)")
    print(f"  PASS GATE?:  {result.pass_gate}  (target ≤ 0.55)")
    print("=" * 60)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    print(f"\nReport written to {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
