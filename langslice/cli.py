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
        help="Registration workflow (colored_segmentation, image_gen_two_shot, "
        "multimodal_tool_loop). Default: auto-select based on model.",
    )
    reg.add_argument("--model", default=None, help="Gemini model name")
    reg.add_argument("--landmarks", type=int, default=14, help="Target landmark count")
    reg.add_argument(
        "--vlm-resolution",
        type=int,
        default=2048,
        help="Max long-edge pixels for VLM",
    )
    reg.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    reg.add_argument(
        "--thinking",
        default=None,
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        help="Gemini thinking level",
    )
    reg.add_argument("--no-borders", action="store_true", help="Disable atlas region borders")
    reg.add_argument(
        "--border-count",
        type=int,
        default=None,
        help="Number of border landmarks (default: half of --landmarks, rounded up)",
    )
    reg.add_argument(
        "--interior-count",
        type=int,
        default=None,
        help="Number of interior landmarks (default: half of --landmarks, rounded down)",
    )
    reg.add_argument(
        "--out",
        default=None,
        help="Output directory for results. Default: ./langslice_output/<timestamp>",
    )
    reg.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def _run_register(args: argparse.Namespace) -> None:
    import json
    from datetime import datetime
    from pathlib import Path

    from PIL import Image

    import langslice.vlm_config as vlm_config
    from langslice.image_prep import normalize_image, prepare_image_for_vlm
    from langslice.registration.core import estimate_registration_runtime
    from langslice.registration.types import (
        annotation_session_to_dict,
        build_annotation_session_from_correspondences,
    )

    # Configure model before anything touches the client.
    if args.model:
        vlm_config.set_model_name(args.model)
    if args.temperature is not None:
        vlm_config.set_temperature(args.temperature)
    if args.thinking:
        vlm_config.set_thinking_level(args.thinking)

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
    border_label = (
        f"border={args.border_count}"
        if args.border_count is not None
        else "auto"
    )
    interior_label = (
        f"interior={args.interior_count}"
        if args.interior_count is not None
        else "auto"
    )
    print(
        f"Landmarks: {args.landmarks} "
        f"({border_label}/{interior_label})  "
        f"Borders: {not args.no_borders}"
    )
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
        border_count=args.border_count,
        interior_count=args.interior_count,
    )

    # Summary.
    affine = result.affine_result
    tx, ty = affine.translation_px
    sx, sy = affine.scale
    print()
    print("Registration complete")
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


def _add_estimate_parser(subparsers: argparse._SubParsersAction) -> None:
    est = subparsers.add_parser(
        "estimate",
        help="Estimate the AP position of a brain slice image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    est.add_argument("image", help="Path to slice image (PNG, TIFF, JPEG)")
    est.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    est.add_argument(
        "--workflow",
        default=None,
        choices=["tool_use", "image_gen"],
        help="AP estimation workflow. Default: auto-select based on model.",
    )
    est.add_argument("--model", default=None, help="Gemini model name")
    est.add_argument(
        "--thinking",
        default=None,
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        help="Gemini thinking level",
    )
    est.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    est.add_argument(
        "--media-resolution",
        default="ultra_high",
        choices=["low", "medium", "high", "ultra_high"],
        help="Gemini media resolution for input images",
    )
    est.add_argument(
        "--vlm-resolution",
        type=int,
        default=2048,
        help="Max long-edge pixels for VLM",
    )
    est.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max tool-loop iterations",
    )
    est.add_argument(
        "--preprocess",
        default="auto",
        choices=["auto", "none"],
        help="Image preprocessing: 'auto' applies adaptive CLAHE + brightness normalization, "
        "'none' sends the raw image",
    )
    est.add_argument(
        "--borders",
        action="store_true",
        help="Enable atlas region borders (off by default)",
    )
    est.add_argument(
        "--out",
        default=None,
        help="Output directory for debug artifacts",
    )
    est.add_argument("--json", action="store_true", help="Print result JSON to stdout")
    est.add_argument(
        "--grid",
        action="store_true",
        help="Send atlas slices as a composite grid instead of individually (default: individual)",
    )
    est.add_argument(
        "--atlas-resolution",
        type=int,
        default=512,
        help="Max long-edge pixels for atlas slices (individual mode)",
    )


def _add_estimate_group_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "estimate-group",
        help="Estimate AP positions for a group of consecutive brain slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "images",
        nargs="+",
        help="Slice images in anterior-to-posterior order (2-8 images)",
    )
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument(
        "--interval", type=int, default=200, help="Section interval in microns (center-to-center)"
    )
    p.add_argument("--thickness", type=int, default=50, help="Slice thickness in microns")
    p.add_argument("--model", default=None, help="Gemini model name")
    p.add_argument(
        "--thinking",
        default=None,
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        help="Gemini thinking level",
    )
    p.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    p.add_argument(
        "--media-resolution",
        default="ultra_high",
        choices=["low", "medium", "high", "ultra_high"],
        help="Gemini media resolution for input images",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Max tool-loop iterations",
    )
    p.add_argument(
        "--preprocess",
        default="auto",
        choices=["auto", "none"],
        help="Image preprocessing: 'auto' applies adaptive CLAHE + brightness normalization",
    )
    p.add_argument(
        "--borders",
        action="store_true",
        help="Enable atlas region borders (off by default)",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Send atlas slices as a composite grid instead of individually (default: individual)",
    )
    p.add_argument(
        "--atlas-resolution",
        type=int,
        default=1024,
        help="Max long-edge pixels for atlas slices (individual mode)",
    )
    p.add_argument(
        "--vlm-resolution",
        type=int,
        default=2048,
        help="Max long-edge pixels for VLM",
    )
    p.add_argument("--out", default=None, help="Output directory for debug artifacts")
    p.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def _run_estimate_group(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from PIL import Image

    import langslice.vlm_config as vlm_config
    from langslice.estimation import estimate_group
    from langslice.image_prep import adaptive_preprocess, normalize_image, prepare_image_for_vlm

    # Configure model.
    if args.model:
        vlm_config.set_model_name(args.model)
    if args.temperature is not None:
        vlm_config.set_temperature(args.temperature)
    if args.thinking:
        vlm_config.set_thinking_level(args.thinking)

    # Load and preprocess images: normalize → downscale → CLAHE
    # (same order as single-slice CLI for experimental consistency).
    images: list[Image.Image] = []
    for path in args.images:
        raw = Image.open(path)
        canonical = normalize_image(raw)
        original_size = canonical.size
        prep = prepare_image_for_vlm(canonical, max_long_edge=args.vlm_resolution)
        img = prep.image
        if args.preprocess == "auto":
            img = adaptive_preprocess(img)
        images.append(img)
        print(
            f"  Loaded {path}: {original_size[0]}x{original_size[1]} -> "
            f"{img.size[0]}x{img.size[1]}px (scale={prep.scale_factor:.3f})"
        )

    n = len(images)
    if not 2 <= n <= 8:
        print(f"Error: expected 2-8 images, got {n}")
        sys.exit(1)

    interval_mm = args.interval / 1000.0
    total_span = (n - 1) * interval_mm

    print("\nGroup estimation:")
    print(f"  {n} slices, interval {args.interval}\u00b5m ({interval_mm:.3f}mm)")
    print(f"  Expected span: {total_span:.2f}mm")
    print(f"  Atlas: {args.atlas}")
    from langslice.estimation.google.ap_multi_slice import _DEFAULT_MODEL
    effective_model = args.model or _DEFAULT_MODEL
    print(f"  Model: {effective_model}")
    print(f"  Max iterations: {args.max_iterations}")
    print()

    debug_dir = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = str(out_dir)

    def on_progress(msg: str) -> None:
        print(f"  {msg}")

    result = estimate_group(
        images=images,
        atlas_name=args.atlas,
        interval_um=args.interval,
        thickness_um=args.thickness,
        max_iterations=args.max_iterations,
        media_resolution=args.media_resolution,
        model_name=args.model,
        show_borders=args.borders,
        send_individually=not args.grid,
        atlas_resolution=args.atlas_resolution,
        on_progress=on_progress,
        debug_dir=debug_dir,
    )

    # Summary.
    print()
    print("Group estimation complete:")
    for i, ap in enumerate(result.positions):
        print(f"  Slice {i + 1}: {ap.position_mm:.3f} mm")
    if len(result.positions) > 1:
        intervals = [
            abs(result.positions[i + 1].position_mm - result.positions[i].position_mm)
            for i in range(len(result.positions) - 1)
        ]
        mean_interval = sum(intervals) / len(intervals)
        print(f"  Mean interval: {mean_interval:.3f} mm (expected: {interval_mm:.3f} mm)")
    print(f"  Reasoning: {result.group_reasoning}")
    if result.debug_dir:
        print(f"  Artifacts: {result.debug_dir}")

    if args.json:
        payload = {
            "slices": [
                {
                    "slice_index": i + 1,
                    "image": args.images[i],
                    "position_mm": ap.position_mm,
                }
                for i, ap in enumerate(result.positions)
            ],
            "group_reasoning": result.group_reasoning,
            "interval_um": args.interval,
            "thickness_um": args.thickness,
        }
        print()
        print(json.dumps(payload, indent=2))


def _add_estimate_brain_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "estimate-brain",
        help="Estimate AP positions for a folder of brain slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing slice images")
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument("--thickness", type=int, default=50, help="Slice thickness in microns")
    p.add_argument("--interval", type=int, default=200, help="Average slice interval in microns")
    p.add_argument("--anchors", type=int, default=4, help="Number of anchor agents")
    p.add_argument("--parallel", type=int, default=4, help="Max concurrent Gemini calls")
    p.add_argument(
        "--z-axis",
        choices=["AP", "PA"],
        default="AP",
        help="Z-axis orientation of the slice series",
    )
    p.add_argument("--out", help="Output JSON path (default: <folder>/brain_estimate.json)")


def _run_estimate_brain(args: argparse.Namespace) -> None:
    import asyncio
    import json

    from langslice.whole_brain.discovery import discover_slices
    from langslice.whole_brain.pipeline import run_brain_estimation
    from langslice.whole_brain.types import BrainEstimationConfig

    config = BrainEstimationConfig(
        image_folder=args.image_folder,
        atlas_name=args.atlas,
        thickness_um=args.thickness,
        interval_um=args.interval,
        n_anchors=args.anchors,
        max_parallel=args.parallel,
        z_axis=args.z_axis,
    )

    # Cost estimate
    n_images = len(discover_slices(config.image_folder))
    n_non_anchors = n_images - config.n_anchors
    print("\nBrain estimation plan:")
    print(f"  {n_images} slices, {config.n_anchors} anchors")
    print()
    print(f"  Phase 1:  {config.n_anchors} anchor estimations (coarse + nano-banana fine)")
    print("  Phase 2:  interpolation")
    print(f"  Phase 3:  {n_non_anchors} non-anchor estimations (2-pass nano-banana)")
    print("  Phase 4:  isotonic regression (Huber loss)")
    print(f"  --parallel {config.max_parallel}")
    print()

    def on_progress(msg: str) -> None:
        print(msg)

    result = asyncio.run(
        run_brain_estimation(
            config,
            checkpoint_path=args.out,
            on_progress=on_progress,
        )
    )

    import os
    out_path = args.out or os.path.join(args.image_folder, "brain_estimate.json")
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"  {result.summary.n_slices} slices, {result.summary.n_anchors} anchors")
    print(f"  Mean interval: {result.summary.mean_interval_mm:.3f}mm")
    print(f"  Std interval:  {result.summary.std_interval_mm:.3f}mm")


def _run_estimate(args: argparse.Namespace) -> None:
    import json
    import os
    from pathlib import Path

    from PIL import Image

    import langslice.vlm_config as vlm_config
    from langslice.estimation import estimate_position, estimate_position_image_gen
    from langslice.image_prep import normalize_image, prepare_image_for_vlm

    # Configure model before anything touches the client.
    if args.model:
        vlm_config.set_model_name(args.model)
    if args.temperature is not None:
        vlm_config.set_temperature(args.temperature)
    if args.thinking:
        vlm_config.set_thinking_level(args.thinking)

    # Load and downscale image.
    print(f"Loading {args.image} ...")
    raw_image = Image.open(args.image)
    canonical = normalize_image(raw_image)
    original_size = canonical.size
    prep = prepare_image_for_vlm(canonical, max_long_edge=args.vlm_resolution)
    image = prep.image
    if args.preprocess == "auto":
        from langslice.image_prep import adaptive_preprocess
        image = adaptive_preprocess(image)
        preprocess_label = "adaptive (CLAHE + brightness)"
    else:
        preprocess_label = "none"
    print(
        f"  Original: {original_size[0]}x{original_size[1]} -> "
        f"VLM input: {image.size[0]}x{image.size[1]}  "
        f"(scale={prep.scale_factor:.3f}, max_edge={args.vlm_resolution}, "
        f"preprocess={preprocess_label})"
    )

    # Set up output directory.
    debug_dir = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = str(out_dir)
        os.environ["LANGSLICE_VLM_DEBUG_DIR"] = debug_dir

    print(f"Atlas: {args.atlas}")
    print(
        f"Model: {vlm_config.MODEL_NAME}  "
        f"Thinking: {vlm_config.THINKING_LEVEL}  "
        f"Temp: {vlm_config.TEMPERATURE}"
    )
    print(f"Max iterations: {args.max_iterations}")
    if debug_dir:
        print(f"Output: {debug_dir}")
    print()

    def on_progress(msg: str) -> None:
        print(f"  {msg}")

    # Resolve workflow: explicit flag > auto-detect from model.
    workflow = args.workflow
    if workflow is None:
        is_img_model = vlm_config.is_image_generation_model(vlm_config.MODEL_NAME)
        workflow = "image_gen" if is_img_model else "tool_use"
    print(f"Workflow: {workflow}")

    # Run AP estimation.
    if workflow == "image_gen":
        result = estimate_position_image_gen(
            image=image,
            atlas_name=args.atlas,
            on_progress=on_progress,
            show_borders=args.borders,
            send_individually=not args.grid,
            atlas_resolution=args.atlas_resolution,
        )
    else:
        result = estimate_position(
            image=image,
            atlas_name=args.atlas,
            on_progress=on_progress,
            max_iterations=args.max_iterations,
            media_resolution=args.media_resolution,
            show_borders=args.borders,
            send_individually=not args.grid,
            atlas_resolution=args.atlas_resolution,
        )

    # Summary.
    print()
    print("AP Estimation complete")
    print(f"  Position: {result.position_mm:.3f} mm")
    print(f"  Reasoning: {result.reasoning}")
    if result.debug_dir:
        print(f"  Artifacts: {result.debug_dir}")

    if args.json:
        payload = {
            "position_mm": result.position_mm,
            "reasoning": result.reasoning,
            "debug_dir": result.debug_dir,
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
    subparsers.add_parser("gui", help="Launch the Tauri desktop application")

    # langslice version
    subparsers.add_parser("version", help="Print version info")

    # langslice register
    _add_register_parser(subparsers)

    # langslice estimate
    _add_estimate_parser(subparsers)

    # langslice estimate-group
    _add_estimate_group_parser(subparsers)

    # langslice estimate-brain
    _add_estimate_brain_parser(subparsers)

    args = parser.parse_args()

    if args.command == "gui":
        print("The PySide6 GUI has been replaced by the Tauri desktop app.")
        print("To launch the Tauri GUI:")
        print("  cd tauri-gui && pnpm tauri dev")
        print()
        print("For headless operation, use the CLI commands:")
        print("  langslice estimate <image>")
        print("  langslice register <image> --position <mm>")
    elif args.command == "version":
        print(f"langslice {langslice.__version__}")
    elif args.command == "register":
        _run_register(args)
    elif args.command == "estimate":
        _run_estimate(args)
    elif args.command == "estimate-group":
        _run_estimate_group(args)
    elif args.command == "estimate-brain":
        _run_estimate_brain(args)
    else:
        parser.print_help()
        sys.exit(1)
