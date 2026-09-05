"""LangSlice CLI entry point."""
import argparse
import sys

import langslice

_PLANE_HELP = (
    "Slicing plane (normal axis). Position is interpreted along this axis "
    "(AP for coronal, ML for sagittal, DV for horizontal)."
)


def _resolve_register_models(
    *,
    default_image_model: str,
    default_review_model: str,
    image_model: str | None,
    review_model: str | None,
) -> tuple[str, str]:
    resolved_image_model = image_model or default_image_model
    resolved_review_model = review_model or default_review_model
    return resolved_image_model, resolved_review_model


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
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=_PLANE_HELP,
    )
    reg.add_argument("--model", default=None, help="Gemini model name")
    reg.add_argument("--image-model", default=None, help="Image generation model name")
    reg.add_argument(
        "--openai-image-route",
        default="images",
        choices=["images", "responses"],
        help="OpenAI image-generation route for registration candidates",
    )
    reg.add_argument("--review-model", default=None, help="Registration review agent model name")
    reg.add_argument(
        "--canvas-pad",
        type=float,
        default=0.0,
        help="Pad every canvas side by this fraction of the long edge with a "
        "background-colored margin, so a fragment or hemibrain's complete "
        "painted anatomy can exceed the original image bounds (0 to 1.5)",
    )
    reg.add_argument(
        "--pitch-deg",
        type=float,
        default=0.0,
        help="Block cutting angle about the plane's column axis (degrees). Every "
        "atlas render is resliced on that oblique plane instead of taken flat.",
    )
    reg.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="Block cutting angle about the plane's row axis (degrees) — the "
        "component that samples a different level on each side.",
    )
    reg.add_argument(
        "--vlm-resolution",
        type=int,
        default=2048,
        help="Max long-edge pixels for VLM",
    )
    reg.add_argument(
        "--clahe",
        action="store_true",
        help="Apply adaptive CLAHE + DAPI-weighted grayscale preprocessing to the slice "
        "before sending it to the image-gen registration model. Useful when the red "
        "fluorescence channel dominates and washes out structural detail.",
    )
    reg.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    reg.add_argument(
        "--thinking",
        default=None,
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        help="Gemini thinking level",
    )
    reg.add_argument(
        "--out",
        default=None,
        help="Output directory for results. Default: ./langslice_output/<timestamp>",
    )
    reg.add_argument(
        "--provider",
        default="google",
        choices=[
            "gemini-api", "openai-api", "openai-oauth",
            "google", "openai", "chatgpt",  # legacy aliases
        ],
        help=(
            "Access method: 'gemini-api' (Google API key), 'openai-api' "
            "(OpenAI-compatible API key / --endpoint), 'openai-oauth' "
            "(ChatGPT subscription via `langslice login`). Old spellings "
            "google/openai/chatgpt still work as aliases."
        ),
    )
    reg.add_argument(
        "--endpoint",
        default=None,
        help=(
            "OpenAI-compatible base URL (e.g. http://127.0.0.1:1234/v1). When "
            "set, the model name is used verbatim and the request is sent here, "
            "bypassing prefix dispatch."
        ),
    )
    reg.add_argument(
        "--image-axes",
        default=None,
        help=(
            "Anatomical directions of the image's rows,cols (ABBA-style axis "
            "mapping), e.g. 'ap,lr' for a horizontal image with anterior at "
            "the top and the animal's left on the image's left; 'si,ap' for "
            "a sagittal image with dorsal up and anterior left. Default: the "
            "atlas render's native orientation."
        ),
    )
    reg.add_argument(
        "--palette",
        default="family",
        choices=["family", "leaf-borders", "family-flat"],
        help=(
            "How the atlas is drawn for the image model. 'family' is flat "
            "regions, one color per palette unit; 'leaf-borders' adds "
            "Allen-Reference-Atlas-style hairlines at every leaf boundary, in "
            "a darker shade of the region's own color; 'family-flat' "
            "(experimental) paints one flat color per registration family, "
            "delineated by the same hairlines. Colors, the Elastix pair and "
            "everything classified from it are identical whichever is used."
        ),
    )
    reg.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help=(
            "Physical pixel size of the input image in micrometers. When "
            "given, the atlas references are rendered at TRUE physical "
            "scale relative to the image — the single most direct "
            "calibration between image and atlas."
        ),
    )
    reg.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def _run_register(args: argparse.Namespace) -> None:
    import json
    import os

    from langslice.api.models import RegisterRequest
    from langslice.api.runtime import run_register

    # Optional endpoint override for a local-engine model. The harness's
    # model resolver picks this up via env var.
    endpoint = getattr(args, "endpoint", None)
    if endpoint:
        os.environ["LANGSLICE_ENDPOINT"] = endpoint

    # Palette is a process-wide setting (see atlas.recolor.active_palette): the
    # render, the classifier and the family merge must not disagree about it.
    from langslice.atlas.recolor import PALETTE_ENV

    os.environ[PALETTE_ENV] = getattr(args, "palette", "family")

    image_model_arg = args.image_model

    from langslice.providers.registry import canonical_provider

    if canonical_provider(args.provider) == "openai-oauth":
        from langslice.providers import openai_oauth

        default_image_model = args.image_model or openai_oauth.DEFAULT_IMAGE_MODEL
        default_review_model = args.model or openai_oauth.DEFAULT_REVIEW_MODEL
        effective_model = default_image_model
    elif canonical_provider(args.provider) == "openai-api":
        import langslice.providers.openai_config as openai_config

        default_image_model = args.image_model or openai_config.get_openai_image_model()
        default_review_model = args.model or openai_config.get_openai_model()
        effective_model = default_image_model
    else:
        import langslice.providers.vlm_config as vlm_config

        default_image_model = args.image_model or args.model or vlm_config.MODEL_NAME
        default_review_model = args.model or vlm_config.MODEL_NAME
        # Configure model before anything touches the client. This is a
        # one-shot CLI process, so mutating the global runtime config here
        # (unlike api.runtime.run_register, which is called from the
        # long-lived `serve --stdio` engine and must not leak state across
        # requests) is harmless.
        if default_image_model:
            vlm_config.set_model_name(default_image_model)
        if args.temperature is not None:
            vlm_config.set_temperature(args.temperature)
        if args.thinking:
            vlm_config.set_thinking_level(args.thinking)

        effective_model = vlm_config.MODEL_NAME

    image_model, review_model = _resolve_register_models(
        default_image_model=default_image_model,
        default_review_model=default_review_model,
        image_model=image_model_arg,
        review_model=args.review_model,
    )

    out_dir = _register_output_dir(args.out, args.atlas)

    print(f"Atlas: {args.atlas}  Plane: {args.plane}  Position: {args.position:.2f} mm")
    print(f"Registration: image-gen  Model: {effective_model}  Provider: {args.provider}")
    print(f"Output: {out_dir}")
    print()

    def emit(event: object) -> None:
        print(f"  {getattr(event, 'message', event)}")

    request = RegisterRequest(
        image_path=args.image,
        atlas=args.atlas,
        position_mm=args.position,
        plane=args.plane,
        image_model=image_model,
        review_model=review_model,
        preprocess="auto" if getattr(args, "clahe", False) else "none",
        provider=args.provider,
        output_dir=str(out_dir),
        openai_image_route=args.openai_image_route,
        canvas_pad=args.canvas_pad,
        image_axes=getattr(args, "image_axes", None),
        pixel_size_um=getattr(args, "pixel_size_um", None),
        pitch_deg=getattr(args, "pitch_deg", 0.0),
        yaw_deg=getattr(args, "yaw_deg", 0.0),
        vlm_resolution=args.vlm_resolution,
    )
    result = run_register(request, emit=emit)

    session_dict = result.annotation_session or {}
    metadata_raw = session_dict.get("metadata", {})
    metadata: dict[str, object] = metadata_raw if isinstance(metadata_raw, dict) else {}
    dense_marker_count = metadata.get("n_markers")
    candidate_metadata = metadata.get("candidate_metadata")

    # Summary.
    print()
    print("Registration complete")
    print(f"  Accepted pairs: {result.accepted_correspondence_count}")
    if dense_marker_count is not None:
        print(f"  Dense markers: {dense_marker_count}")
    print(f"  Rotation: {result.rotation_deg:.2f} deg")
    print(f"  Translation: ({result.translation_px[0]:.1f}, {result.translation_px[1]:.1f}) px")
    print(f"  Scale: ({result.scale[0]:.3f}, {result.scale[1]:.3f})")
    print(f"  Shear: {result.shear:.3f}")
    print(f"  Artifacts: {out_dir}")

    if args.json:
        # Flat artifact-path keys (and inverse_warp_status) at the top level
        # so callers don't have to dig into annotation_session metadata.
        payload = {
            "accepted_correspondences": [],
            "dense_marker_count": dense_marker_count,
            "candidate_metadata": candidate_metadata,
            "affine": {
                "rotation_deg": result.rotation_deg,
                "translation_px": list(result.translation_px),
                "scale": list(result.scale),
                "shear": result.shear,
            },
            "annotation_session": result.annotation_session,
            "inverse_warp_status": result.inverse_warp_status,
            "warped_atlas_path": result.warped_atlas_path,
            "warped_border_overlay_path": result.warped_border_overlay_path,
            "generated_segmentation_path": result.generated_segmentation_path,
            "generated_border_overlay_path": result.generated_border_overlay_path,
            "slice_warped_to_atlas_path": result.slice_warped_to_atlas_path,
            "slice_atlas_border_overlay_path": result.slice_atlas_border_overlay_path,
        }
        print()
        print(json.dumps(payload, indent=2))


def _register_output_dir(out: str | None, atlas: str):
    from datetime import datetime
    from pathlib import Path

    if out:
        out_dir = Path(out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("langslice_output") / f"{timestamp}_{atlas}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _add_estimate_parser(subparsers: argparse._SubParsersAction) -> None:
    est = subparsers.add_parser(
        "estimate",
        help="Estimate the AP position of a brain slice image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    est.add_argument("image", help="Path to slice image (PNG, TIFF, JPEG)")
    est.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    est.add_argument(
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=_PLANE_HELP,
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
        default="medium",
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
        "--out",
        default=None,
        help="Output directory for debug artifacts",
    )
    est.add_argument("--json", action="store_true", help="Print result JSON to stdout")
    est.add_argument(
        "--provider",
        default="google",
        choices=["gemini-api", "openai-api", "google", "openai"],
        help="Model provider: 'google' for Gemini, 'openai' for OpenAI-compatible (Ollama, etc.)",
    )
    est.add_argument(
        "--endpoint",
        default=None,
        help=(
            "OpenAI-compatible base URL (e.g. http://127.0.0.1:1234/v1). When "
            "set, the model name is used verbatim and the request is sent here, "
            "bypassing prefix dispatch."
        ),
    )


def _add_collect_traces_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "collect-traces",
        help="Collect Gemini teacher traces for SFT training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True, help="JSONL manifest of trace jobs")
    p.add_argument("--out", required=True, help="Output directory for trace artifacts")
    p.add_argument("--model", default="gemini-3.1-pro-preview", help="Teacher model")
    p.add_argument(
        "--thinking",
        default="MEDIUM",
        choices=["LOW", "MEDIUM", "HIGH"],
        help="Gemini thinking level for trace collection",
    )
    p.add_argument(
        "--media-resolution",
        default="medium",
        choices=["low", "medium", "high", "ultra_high"],
        help="Gemini media resolution for input images",
    )
    p.add_argument("--max-iterations", type=int, default=20, help="Max tool-loop iterations")
    p.add_argument("--limit", type=int, default=None, help="Limit manifest rows for a test run")
    p.add_argument(
        "--kind",
        default="all",
        choices=["all", "single"],
        help="Run only one manifest kind",
    )
    p.add_argument("--resume", action="store_true", help="Skip runs with existing raw traces")
    p.add_argument(
        "--include-thought-summaries",
        dest="include_thought_summaries",
        action="store_true",
        default=True,
        help="Request Gemini thought summaries and store them in raw traces",
    )
    p.add_argument(
        "--no-include-thought-summaries",
        dest="include_thought_summaries",
        action="store_false",
        help="Do not request Gemini thought summaries",
    )
    p.add_argument(
        "--sft-export",
        default="both",
        choices=["deployment", "rationale", "both"],
        help="Which SFT trace export variants to write",
    )
    p.add_argument(
        "--persist-tool-images",
        dest="persist_tool_images",
        action="store_true",
        default=True,
        help="Persist multimodal atlas tool-result images beside each trace",
    )
    p.add_argument(
        "--no-persist-tool-images",
        dest="persist_tool_images",
        action="store_false",
        help="Do not persist multimodal atlas tool-result images",
    )


def _run_collect_traces(args: argparse.Namespace) -> None:
    from pathlib import Path

    from langslice.linear.trace_collection import (
        collect_manifest_traces,
    )

    kind_filter = None if args.kind == "all" else args.kind
    results = collect_manifest_traces(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out),
        model=args.model,
        thinking_level=args.thinking,
        media_resolution=args.media_resolution,
        max_iterations=args.max_iterations,
        include_thought_summaries=args.include_thought_summaries,
        sft_export=args.sft_export,
        persist_tool_images=args.persist_tool_images,
        limit=args.limit,
        kind_filter=kind_filter,
        resume=args.resume,
    )
    accepted = sum(
        1
        for result in results
        if isinstance(result.get("category"), dict)
        and result["category"].get("accepted") is True
    )
    print("Trace collection complete")
    print(f"  Runs: {len(results)}")
    print(f"  Accepted: {accepted}")
    print(f"  Output: {Path(args.out)}")


def _add_estimate_brain_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "estimate-brain",
        help="Run the whole-brain estimation engine on a folder of slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing slice images")
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument(
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=_PLANE_HELP,
    )
    p.add_argument("--thickness", type=int, default=50, help="Slice thickness in microns")
    p.add_argument("--interval", type=int, default=200, help="Section interval in microns")
    p.add_argument(
        "--keep-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust the discovered slice order (--no-keep-order lets the engine reorder)",
    )
    p.add_argument("--model", default=None, help="Model name for the engine's agent steps")
    p.add_argument(
        "--preprocess",
        default="auto",
        choices=["auto", "none"],
        help="Image preprocessing for everything the agents look at: 'auto' applies "
        "adaptive CLAHE + brightness normalization, 'none' shows the raw sections. "
        "Display only — the image files are never modified",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Results JSON path. Default: <image_folder>/brain_results.json",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        metavar="PATH",
        help="Write a full-content JSONL trace of every agent session here "
        "(what the agent was shown, said, called, and got back). Overrides "
        "LANGSLICE_TRACE_DIR",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from the folder checkpoint, skipping completed nodes",
    )
    p.add_argument(
        "--stop-after",
        default=None,
        metavar="NODE",
        choices=[
            "ingest", "survey", "fix", "seed",
            "position", "transforms", "review", "emit",
        ],
        help="Run up to and including NODE, checkpoint, and stop "
             "(re-run to continue from there)",
    )
    fresh_group = p.add_mutually_exclusive_group()
    fresh_group.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        help="Ignore any existing checkpoint and start over",
    )
    fresh_group.add_argument(
        "--rerun-from",
        default=None,
        metavar="NODE",
        choices=["position", "transforms", "review"],
        help="Rewind an existing checkpoint's NODE and everything after it "
             "(clearing what they wrote), then resume from there. Requires "
             "an existing checkpoint; mutually exclusive with --fresh. "
             "Composable with --stop-after",
    )


def _run_estimate_brain(args: argparse.Namespace) -> None:
    import asyncio
    import os

    from langslice.linear.whole_brain import BrainConfig, run_brain
    from langslice.linear.whole_brain.checkpoint import (
        default_checkpoint_path,
        load_checkpoint,
        save_checkpoint,
    )
    from langslice.linear.whole_brain.engine import rewind_state
    from langslice.linear.whole_brain.trace import TRACE_DIR_ENV

    if args.trace_dir:
        os.environ[TRACE_DIR_ENV] = args.trace_dir

    config = BrainConfig(
        image_folder=args.image_folder,
        atlas=args.atlas,
        plane=args.plane,
        thickness_um=args.thickness,
        interval_um=args.interval,
        keep_order=args.keep_order,
        model=args.model,
        out=args.out,
        resume=args.resume,
        preprocess=args.preprocess,
    )

    if args.rerun_from:
        checkpoint_path = default_checkpoint_path(os.path.abspath(config.image_folder))
        state = load_checkpoint(checkpoint_path)
        if state is None:
            raise SystemExit(
                f"--rerun-from requires an existing checkpoint; none found at "
                f"{checkpoint_path}"
            )
        rewind_state(state, args.rerun_from)
        save_checkpoint(state, checkpoint_path)
        config.resume = True
        print(f"Rewound checkpoint from '{args.rerun_from}' -> {checkpoint_path}")

    print(f"Atlas: {config.atlas}  Plane: {config.plane}")
    print(f"Interval: {config.interval_um}um  Thickness: {config.thickness_um}um")
    print(f"Folder: {config.image_folder}  Resume: {config.resume}")
    print(f"Preprocess: {config.preprocess}")
    if os.environ.get(TRACE_DIR_ENV):
        print(f"Agent traces: {os.environ[TRACE_DIR_ENV]}")
    print()

    state = asyncio.run(run_brain(config, emit=print, stop_after=args.stop_after))

    positioned = [s for s in state.slices if s.position_mm is not None]
    print()
    print(
        f"Whole-brain estimation stopped after {args.stop_after}"
        if args.stop_after
        else "Whole-brain estimation complete"
    )
    print(f"  Slices: {len(state.slices)}  Positioned: {len(positioned)}")
    print(f"  Nodes run: {', '.join(state.completed_nodes)}")
    if state.contact_sheet:
        print(f"  Contact sheet: {state.contact_sheet}")


def _run_estimate(args: argparse.Namespace) -> None:
    import json

    import langslice.providers.vlm_config as vlm_config
    from langslice.api.models import EstimateRequest
    from langslice.api.runtime import run_estimate
    from langslice.providers.registry import canonical_provider

    if canonical_provider(args.provider) == "openai-api":
        import langslice.providers.openai_config as openai_config

        effective_model = args.model or openai_config.get_openai_model()
        provider_label = "openai"
    else:
        effective_model = args.model or vlm_config.MODEL_NAME
        provider_label = "google"

    print(f"Atlas: {args.atlas}  Plane: {args.plane}")
    print(f"Model: {effective_model}  Provider: {provider_label}")
    print(f"Max iterations: {args.max_iterations}")
    if args.out:
        print(f"Output: {args.out}")
    print()

    def emit(event: object) -> None:
        print(f"  {getattr(event, 'message', event)}")

    request = EstimateRequest(
        image_path=args.image,
        atlas=args.atlas,
        plane=args.plane,
        model=args.model,
        thinking=args.thinking,
        temperature=args.temperature,
        media_resolution=args.media_resolution,
        max_iterations=args.max_iterations,
        preprocess=args.preprocess,
        provider=args.provider,
        endpoint=args.endpoint,
        output_dir=args.out,
    )
    result = run_estimate(request, emit=emit)

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langslice",
        description="VLM-based brain slice registration using Gemini and BrainGlobe atlases",
    )
    subparsers = parser.add_subparsers(dest="command")

    # langslice version
    subparsers.add_parser("version", help="Print version info")

    # langslice login
    subparsers.add_parser(
        "login",
        help="Sign in with ChatGPT (OAuth) so LangSlice can use your subscription",
    )

    # langslice linear <cmd> — position / affine estimation
    linear = subparsers.add_parser(
        "linear",
        help="Linear methods: slice position and affine estimation",
    )
    linear_sub = linear.add_subparsers(dest="subcommand", required=True)
    _add_estimate_parser(linear_sub)
    _add_estimate_brain_parser(linear_sub)
    _add_quick_affine_parser(linear_sub)

    # langslice nonlinear <cmd> — image-gen registration
    nonlinear = subparsers.add_parser(
        "nonlinear",
        help="Nonlinear methods: image-gen registration",
    )
    nonlinear_sub = nonlinear.add_subparsers(dest="subcommand", required=True)
    _add_register_parser(nonlinear_sub)

    # langslice abba
    _add_abba_parser(subparsers)

    # langslice collect-traces
    _add_collect_traces_parser(subparsers)

    # langslice serve
    _add_serve_parser(subparsers)

    return parser


def _add_quick_affine_parser(subparsers: argparse._SubParsersAction) -> None:
    """Fast affine-only Elastix preview, no image-gen. Produces a placeholder
    warped slice the moment a position is locked, before the full pipeline
    finishes."""
    p = subparsers.add_parser(
        "quick-affine",
        help="Affine-only Elastix preview registration (no image-gen, ~2s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", help="Path to slice image")
    p.add_argument("--atlas", required=True, help="BrainGlobe atlas name")
    p.add_argument("--position", type=float, required=True, help="Position in mm (per plane)")
    p.add_argument(
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=_PLANE_HELP,
    )
    p.add_argument("--out", required=True, help="Path to write the warped slice PNG")
    p.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def _run_quick_affine(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from PIL import Image

    from langslice.nonlinear.quick_affine import quick_affine_register

    image = Image.open(args.image)
    result = quick_affine_register(
        image,
        atlas_name=args.atlas,
        position_mm=args.position,
        plane=args.plane,
        out_path=Path(args.out),
    )
    print(f"Quick affine complete: {result['warped_slice_path']} ({result['elapsed_s']}s)")
    if args.json:
        print(json.dumps(result, indent=2))


def _add_abba_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "abba",
        help="Launch ABBA (abba-python GUI) with LangSlice registration installed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--abba-atlas",
        default="Adult Mouse Brain - Allen Brain Atlas V3p1",
        help="Atlas name passed to ABBA",
    )
    p.add_argument(
        "--atlas",
        default="allen_mouse_10um",
        help="BrainGlobe atlas LangSlice samples for region maps",
    )
    p.add_argument(
        "--provider",
        default="google",
        choices=[
            "gemini-api", "openai-api", "openai-oauth",
            "google", "openai", "chatgpt",  # legacy aliases
        ],
        help="Image-gen provider for the registration",
    )
    p.add_argument("--model", default=None, help="Image-gen model override")


def _run_abba(args: argparse.Namespace) -> None:
    try:
        import abba_python  # noqa: F401  # pyright: ignore[reportMissingImports]

        from langslice.integrations.abba import run_gui_session
    except ImportError as exc:
        raise SystemExit(
            "abba-python is not installed in this environment. Install it with\n"
            '  pip install "langslice[abba]"\n'
            "inside a conda env that provides OpenJDK 11 and Maven "
            f"(see the abba-python installation docs). ({exc})"
        ) from exc

    run_gui_session(
        abba_atlas=args.abba_atlas,
        atlas_name=args.atlas,
        provider=args.provider,
        model=args.model,
    )


def _add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "serve",
        help="Run LangSlice engine service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stdio",
        action="store_true",
        help="Run newline-delimited JSON service over stdin/stdout",
    )


def _run_serve(args: argparse.Namespace) -> None:
    if not args.stdio:
        raise SystemExit("serve currently requires --stdio")
    from langslice.api.service import run_stdio

    raise SystemExit(run_stdio())


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Group commands (`linear`, `nonlinear`) carry the leaf name in
    # `subcommand`; top-level commands only set `command`. Leaf names are
    # unique across groups, so one dispatch chain covers both.
    command = getattr(args, "subcommand", None) or args.command

    if command == "version":
        print(f"langslice {langslice.__version__}")
    elif command == "login":
        from langslice.providers.openai_oauth import login

        print(f"Signed in. Credentials saved to {login()}")
    elif command == "abba":
        _run_abba(args)
    elif command == "register":
        _run_register(args)
    elif command == "quick-affine":
        _run_quick_affine(args)
    elif command == "estimate":
        _run_estimate(args)
    elif command == "collect-traces":
        _run_collect_traces(args)
    elif command == "estimate-brain":
        _run_estimate_brain(args)
    elif command == "serve":
        _run_serve(args)
    else:
        parser.print_help()
        sys.exit(1)
