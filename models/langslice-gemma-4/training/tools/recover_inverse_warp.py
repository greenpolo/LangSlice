"""Recompute missing inverse-warp artifacts for an old registration run.

Older candidate directories produced before the itk-elastix
``WriteParameterFile`` / ``InitialTransformParameterFileName`` fixes are
missing the inverse-warp outputs (``slice_warped_to_atlas.png`` +
``slice_atlas_border_overlay.png``) and the raw-model-output overlay
(``generated_border_overlay.png``). This script reuses the saved
``generated_segmentation.png`` and ``input_slice.png`` so no
image-generation tokens are spent — it only re-runs Elastix forward +
inverse on the existing inputs, writes the missing PNGs into the
candidate dir, and patches the sibling ``registration.json`` with the
new inverse status + artifact paths.

Usage::

    python models/langslice-gemma-4/training/tools/recover_inverse_warp.py <candidate_dir>

The candidate dir is the one containing ``generated_segmentation.png``
(e.g. ``langslice_output/<run>/registration/candidate-XXXXXXXXXXXX``).
Atlas name, AP position and plane are read from the sibling
``registration.json``; pass ``--atlas``, ``--position``, ``--plane`` to
override when the sidecar is missing or out of date.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Allow running the script from a checked-out repo without installing.
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "Could not locate LangSlice repo root from recover-inverse-warp tool path."
    raise RuntimeError(msg)


_REPO_ROOT = _find_repo_root()
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from langslice_harness.atlas import load_atlas  # noqa: E402
from langslice_harness.harness.registration.image_gen_helpers import (  # noqa: E402
    _build_atlas_root_mask,
    _classify_pixels_to_region_ids,
    _extract_borders_from_classified,
    _generate_colored_region_slice,
    _register_colored_images,
    _run_inverse_warp_for_slice,
    _warp_atlas_rgb,
)

_REQUIRED_INPUTS = ("generated_segmentation.png", "input_slice.png")
_OUTPUTS_TO_REGEN = (
    "warped_atlas.png",
    "warped_border_overlay.png",
    "generated_border_overlay.png",
    "slice_warped_to_atlas.png",
    "slice_atlas_border_overlay.png",
)


def _sidecar_key_for(filename: str) -> str:
    """Map ``foo.png`` to the ``foo_path`` key the sidecar / GUI consume."""
    return filename.removesuffix(".png") + "_path"


def _overlay_borders(base: Image.Image, borders: np.ndarray) -> Image.Image:
    overlay = np.asarray(base.convert("RGB"), dtype=np.uint8).copy()
    overlay[np.asarray(borders) > 0] = (0, 255, 255)
    return Image.fromarray(overlay, mode="RGB")


def _read_sidecar_metadata(registration_json: Path) -> dict[str, object]:
    if not registration_json.is_file():
        return {}
    try:
        data = json.loads(registration_json.read_text())
    except json.JSONDecodeError:
        return {}
    session = data.get("annotation_session") or {}
    meta = session.get("metadata") or {}
    return {
        "atlas_name": meta.get("atlas_name"),
        "position_mm": meta.get("position_mm"),
        "plane": meta.get("plane"),
        "previous_inverse_warp_status": meta.get("inverse_warp_status"),
        "_raw": data,
    }


def _patch_sidecar(
    registration_json: Path,
    *,
    inverse_warp_status: str,
    artifact_paths: dict[str, str],
) -> None:
    if not registration_json.is_file():
        return
    try:
        data = json.loads(registration_json.read_text())
    except json.JSONDecodeError:
        return

    session = data.get("annotation_session")
    if isinstance(session, dict):
        meta = session.setdefault("metadata", {})
        meta["inverse_warp_status"] = inverse_warp_status
        ap = meta.setdefault("artifact_paths", {})
        ap.update(artifact_paths)
        for key, value in artifact_paths.items():
            meta[key] = value

    backup = registration_json.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(registration_json.read_text())
    registration_json.write_text(json.dumps(data, indent=2))


def recover_inverse_warp(
    candidate_dir: Path,
    *,
    atlas_name: str,
    position_mm: float,
    plane: str,
) -> dict[str, str]:
    for name in _REQUIRED_INPUTS:
        if not (candidate_dir / name).is_file():
            raise FileNotFoundError(f"Missing input artifact: {candidate_dir / name}")

    print(f"  Loading saved inputs from {candidate_dir}")
    slice_image = Image.open(candidate_dir / "input_slice.png").convert("RGB")
    target_size = slice_image.size  # (width, height)

    generated_seg = Image.open(candidate_dir / "generated_segmentation.png").convert("RGB")
    model_output = generated_seg.resize(target_size, resample=Image.Resampling.LANCZOS)
    model_output_rgb = np.asarray(model_output, dtype=np.uint8)

    print(f"  Loading atlas: {atlas_name}")
    atlas = load_atlas(atlas_name)

    print(f"  Rendering atlas at position {position_mm:.2f} mm ({plane})")
    atlas_colored = _generate_colored_region_slice(
        atlas, position_mm, target_size, plane=plane
    )
    atlas_rgb = np.asarray(atlas_colored, dtype=np.uint8)

    print("  Running Elastix forward registration (atlas -> model output)")
    t0 = time.perf_counter()
    result_transform, _elastix_elapsed = _register_colored_images(atlas_rgb, model_output_rgb)
    print(f"    Forward Elastix complete ({time.perf_counter() - t0:.1f}s)")

    print("  Warping atlas + extracting forward borders")
    warped_atlas_rgb = _warp_atlas_rgb(atlas_rgb, result_transform)
    warped_atlas_img = Image.fromarray(warped_atlas_rgb, mode="RGB")
    warped_classified = _classify_pixels_to_region_ids(
        warped_atlas_rgb, atlas, position_mm, plane=plane
    )
    warped_borders = _extract_borders_from_classified(warped_classified)
    warped_border_overlay = _overlay_borders(slice_image, warped_borders)

    print("  Extracting raw-model-output borders (no Elastix warp)")
    generated_classified = _classify_pixels_to_region_ids(
        model_output_rgb, atlas, position_mm, plane=plane
    )
    generated_borders = _extract_borders_from_classified(generated_classified)
    generated_border_overlay = _overlay_borders(slice_image, generated_borders)

    print("  Running inverse warp (slice -> atlas)")
    slice_rgb = np.asarray(slice_image, dtype=np.uint8)
    forward_fixed_gray = cv2.cvtColor(model_output_rgb, cv2.COLOR_RGB2GRAY)
    t0 = time.perf_counter()
    warped_slice_rgb, _inverse_params = _run_inverse_warp_for_slice(
        slice_rgb,
        forward_fixed_gray=forward_fixed_gray,
        forward_result_transform=result_transform,
    )
    print(f"    Inverse Elastix complete ({time.perf_counter() - t0:.1f}s)")

    root_mask = _build_atlas_root_mask(atlas, position_mm, target_size, plane=plane)
    slice_warped_to_atlas_img = Image.fromarray(
        np.dstack([warped_slice_rgb, root_mask]), mode="RGBA"
    )
    atlas_classified = _classify_pixels_to_region_ids(
        atlas_rgb, atlas, position_mm, plane=plane
    )
    atlas_borders = _extract_borders_from_classified(atlas_classified)
    slice_atlas_border_overlay = _overlay_borders(
        Image.fromarray(warped_slice_rgb, mode="RGB"), atlas_borders
    )

    print("  Writing recovered artifacts to candidate dir")
    warped_atlas_img.save(candidate_dir / "warped_atlas.png")
    warped_border_overlay.save(candidate_dir / "warped_border_overlay.png")
    generated_border_overlay.save(candidate_dir / "generated_border_overlay.png")
    slice_warped_to_atlas_img.save(candidate_dir / "slice_warped_to_atlas.png")
    slice_atlas_border_overlay.save(candidate_dir / "slice_atlas_border_overlay.png")

    return {
        _sidecar_key_for(name): str((candidate_dir / name).resolve())
        for name in _OUTPUTS_TO_REGEN
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "candidate_dir",
        type=Path,
        help="Path to the candidate dir (contains generated_segmentation.png)",
    )
    parser.add_argument(
        "--atlas",
        help="BrainGlobe atlas name (defaults to value from sibling registration.json)",
    )
    parser.add_argument(
        "--position",
        type=float,
        help="AP/ML/DV position in mm (defaults to value from sibling registration.json)",
    )
    parser.add_argument(
        "--plane",
        choices=("coronal", "sagittal", "horizontal"),
        help="Slicing plane (defaults to value from sibling registration.json or 'coronal')",
    )
    args = parser.parse_args()

    candidate_dir: Path = args.candidate_dir.expanduser().resolve()
    if not candidate_dir.is_dir():
        parser.error(f"candidate_dir is not a directory: {candidate_dir}")

    sidecar_path = candidate_dir.parent / "registration.json"
    sidecar = _read_sidecar_metadata(sidecar_path)

    atlas_name = args.atlas or sidecar.get("atlas_name")
    position_mm = args.position if args.position is not None else sidecar.get("position_mm")
    plane = args.plane or sidecar.get("plane") or "coronal"

    if not atlas_name:
        parser.error("Could not determine atlas name; pass --atlas")
    if position_mm is None:
        parser.error("Could not determine position; pass --position")

    print(f"Recovering inverse-warp artifacts for {candidate_dir.name}")
    print(f"  atlas: {atlas_name}  position: {float(position_mm):.2f} mm  plane: {plane}")
    if prev := sidecar.get("previous_inverse_warp_status"):
        print(f"  previous status: {prev}")

    try:
        artifact_paths = recover_inverse_warp(
            candidate_dir,
            atlas_name=str(atlas_name),
            position_mm=float(position_mm),
            plane=str(plane),
        )
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        print(f"ERROR: {status}")
        if sidecar_path.is_file():
            _patch_sidecar(sidecar_path, inverse_warp_status=status, artifact_paths={})
        return 1

    if sidecar_path.is_file():
        _patch_sidecar(
            sidecar_path,
            inverse_warp_status="ok",
            artifact_paths=artifact_paths,
        )
        print(f"  Patched {sidecar_path.name} (backup at registration.json.bak)")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
