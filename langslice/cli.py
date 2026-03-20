"""LangSlice CLI entry point."""
import argparse
import sys

import langslice


def _add_register_parser(subparsers: argparse._SubParsersAction) -> None:
    reg = subparsers.add_parser(
        "register",
        help="Run registration pipeline on a single slice image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reg.add_argument("image", help="Path to slice image (PNG, TIFF, JPEG)")
    reg.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    reg.add_argument("--position", type=float, required=True, help="AP position in mm")
    reg.add_argument(
        "--workflow",
        default=None,
        help="Registration workflow (single_pass, image_gen_two_shot, multimodal_tool_loop). "
        "Default: auto-select based on model.",
    )
    reg.add_argument("--model", default=None, help="Gemini model name")
    reg.add_argument("--landmarks", type=int, default=14, help="Target landmark count")
    reg.add_argument("--vlm-resolution", type=int, default=2048, help="Max long-edge pixels for VLM")
    reg.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    reg.add_argument("--no-borders", action="store_true", help="Disable atlas region borders")
    reg.add_argument(
        "--out",
        default=None,
        help="Output directory for results. Default: ./langslice_output/<timestamp>",
    )
    reg.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def _run_register(args: argparse.Namespace) -> None:
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    from PIL import Image

    from langslice.image_prep import normalize_image, prepare_image_for_vlm
    from langslice.registration.core import estimate_registration_runtime
    from langslice.registration.runtime import _write_debug_artifacts
    from langslice.registration.types import (
        annotation_session_to_dict,
        build_annotation_session_from_correspondences,
    )
    from langslice.vlm import config as vlm_config

    # Configure model before anything touches the client.
    if args.model:
        vlm_config.set_model_name(args.model)
    if args.temperature is not None:
        vlm_config.set_temperature(args.temperature)

    workflow = args.workflow
    if workflow is None:
        workflow = vlm_config.default_registration_workflow(vlm_config.MODEL_NAME)

    # Load and downscale image.
    print(f"Loading {args.image} ...")
    raw_image = Image.open(args.image)
    canonical = normalize_image(raw_image)
    original_size = canonical.size
    prep = prepare_image_for_vlm(canonical, max_long_edge=args.vlm_resolution)
    image = prep.image
    print(
        f"  Original: {original_size[0]}x{original_size[1]} -> "
        f"VLM input: {image.size[0]}x{image.size[1]}  "
        f"(scale={prep.scale_factor:.3f}, max_edge={args.vlm_resolution})"
    )

    # Set up output directory.
    if args.out:
        out_dir = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("langslice_output") / f"{timestamp}_{args.atlas}"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = str(out_dir)

    print(f"Atlas: {args.atlas}  Position: {args.position:.2f} mm")
    print(f"Workflow: {workflow}  Model: {vlm_config.MODEL_NAME}")
    print(f"Landmarks: {args.landmarks}  Borders: {not args.no_borders}")
    print(f"Output: {out_dir}")
    print()

    def on_progress(msg: str) -> None:
        print(f"  {msg}")

    # Run registration.
    result = estimate_registration_runtime(
        image=image,
        atlas_name=args.atlas,
        position_mm=args.position,
        target_landmark_count=args.landmarks,
        workflow=workflow,
        show_atlas_borders=not args.no_borders,
        on_progress=on_progress,
        debug_dir=debug_dir,
    )

    # Summary.
    affine = result.affine_result
    tx, ty = affine.translation_px
    sx, sy = affine.scale
    print()
    print(f"Registration complete ({result.qc_state})")
    print(f"  Accepted pairs: {len(result.accepted_correspondences)}")
    print(f"  Rotation: {affine.rotation_deg:.2f} deg")
    print(f"  Translation: ({tx:.1f}, {ty:.1f}) px")
    print(f"  Scale: ({sx:.3f}, {sy:.3f})")
    print(f"  Shear: {affine.shear:.3f}")
    mean_res = affine.provenance.get("mean_residual_px")
    max_res = affine.provenance.get("max_residual_px")
    if mean_res is not None:
        print(f"  Mean residual: {float(mean_res):.1f} px  Max: {float(max_res or 0):.1f} px")
    print(f"  Artifacts: {out_dir}")

    if args.json:
        session = result.annotation_session or build_annotation_session_from_correspondences(
            result.accepted_correspondences
        )
        payload = {
            "qc_state": result.qc_state,
            "accepted_correspondences": [
                {
                    "label": c.label,
                    "slice_xy": list(c.slice_xy),
                    "atlas_xy": list(c.atlas_xy),
                    "confidence": c.confidence,
                    "rationale": c.rationale,
                }
                for c in result.accepted_correspondences
            ],
            "affine": {
                "backend": affine.backend,
                "rotation_deg": affine.rotation_deg,
                "translation_px": list(affine.translation_px),
                "scale": list(affine.scale),
                "shear": affine.shear,
                "provenance": affine.provenance,
            },
            "annotation_session": annotation_session_to_dict(session),
        }
        print()
        print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="langslice",
        description="VLM-based brain slice registration using Gemini and BrainGlobe atlases",
    )
    subparsers = parser.add_subparsers(dest="command")

    # langslice gui
    subparsers.add_parser("gui", help="Launch the PySide6 desktop application")

    # langslice version
    subparsers.add_parser("version", help="Print version info")

    # langslice register
    _add_register_parser(subparsers)

    args = parser.parse_args()

    if args.command == "gui":
        from langslice.gui import launch
        launch()
    elif args.command == "version":
        print(f"langslice {langslice.__version__}")
    elif args.command == "register":
        _run_register(args)
    else:
        parser.print_help()
        sys.exit(1)
