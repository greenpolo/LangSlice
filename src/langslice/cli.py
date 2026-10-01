"""LangSlice CLI entry point."""
import argparse
import sys
import textwrap
from typing import TYPE_CHECKING, cast

import langslice

if TYPE_CHECKING:
    from langslice.linear.spec import JobSpec
    from langslice.nonlinear.types import Deformation

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
        "--initial-alignment", default=None, metavar="JSON",
        help="JSON 3x3 affine (or object with atlas_to_slice): native atlas pixels to "
        "original image pixels. Selects route 'supplied' (one border-correction "
        "call); omit it to use route 'atlas' (draw from nothing against the "
        "outlined atlas template, see --passes).",
    )
    reg.add_argument(
        "--mirror-atlas-lr", action="store_true",
        help="Mirror sampled atlas left-right before applying the supplied alignment",
    )
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
        "--deformation",
        default="none",
        choices=["none", "bspline", "affine"],
        help="Residual fit after the model call. 'none' (default) runs no "
        "Elastix: outputs are the model's lines on the original plus the "
        "rough placement with an identity residual. 'bspline' is affine + "
        "B-spline; 'affine' fits the affine stage alone.",
    )
    reg.add_argument(
        "--passes",
        type=int,
        choices=[1, 2],
        default=1,
        help="Route 'atlas' (no supplied placement) only: draw boundaries once, "
        "or twice with a second call that corrects the first pass against the "
        "template. Ignored on route 'supplied' (--initial-alignment given), "
        "which always makes exactly one call.",
    )
    reg.add_argument(
        "--vlm-resolution",
        type=int,
        default=2048,
        help="Max long-edge pixels for VLM",
    )
    reg.add_argument(
        "--preprocess",
        default="auto",
        choices=["auto", "none"],
        help="Image preprocessing: 'auto' (default) applies the shared adaptive CLAHE + "
        "structural-channel-weighted blend before sending the slice to the image-gen "
        "model — the same preprocessing the linear path uses; 'none' sends the raw image",
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
            "gemini-api", "openai-api", "openai-oauth", "none",
            "google", "openai", "chatgpt",  # legacy aliases
        ],
        help=(
            "Access method: 'gemini-api' (Google API key), 'openai-api' "
            "(OpenAI-compatible API key / --endpoint), 'openai-oauth' "
            "(ChatGPT subscription via `langslice login`), 'none' (no model "
            "at all — registers the silhouette prior, which alone scores "
            "0.82 family dice against hand registrations). Old spellings "
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
    reg.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def _run_register(args: argparse.Namespace) -> None:
    import json
    import os

    from langslice.api.models import RegisterRequest
    from langslice.api.runtime import run_register

    initial_alignment = None
    alignment_path = getattr(args, "initial_alignment", None)
    if alignment_path:
        from pathlib import Path

        initial_alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
        if isinstance(initial_alignment, dict):
            if "atlas_to_slice" not in initial_alignment:
                raise ValueError("Initial-alignment JSON object must contain atlas_to_slice")
            initial_alignment = initial_alignment["atlas_to_slice"]
        if initial_alignment is None:
            raise ValueError("Initial-alignment JSON must contain a 3x3 affine matrix")

    # Optional endpoint override for a local-engine model. The harness's
    # model resolver picks this up via env var.
    endpoint = getattr(args, "endpoint", None)
    if endpoint:
        os.environ["LANGSLICE_ENDPOINT"] = endpoint

    image_model_arg = args.image_model

    from langslice.providers.registry import canonical_provider

    if canonical_provider(args.provider) == "none":
        # Model-free backbone: nothing to name, nothing to configure.
        default_image_model = default_review_model = ""
        effective_model = "none (silhouette prior)"
    elif canonical_provider(args.provider) == "openai-oauth":
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
        preprocess=getattr(args, "preprocess", "auto"),
        provider=args.provider,
        output_dir=str(out_dir),
        openai_image_route=args.openai_image_route,
        canvas_pad=args.canvas_pad,
        image_axes=getattr(args, "image_axes", None),
        pitch_deg=getattr(args, "pitch_deg", 0.0),
        yaw_deg=getattr(args, "yaw_deg", 0.0),
        deformation=cast("Deformation", getattr(args, "deformation", "none")),
        passes=getattr(args, "passes", 1),
        vlm_resolution=args.vlm_resolution,
        initial_atlas_to_slice=initial_alignment,
        initial_alignment_source=str(alignment_path) if alignment_path else "supplied",
        atlas_mirror_lr=getattr(args, "mirror_atlas_lr", False),
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
            "raw_correction_path": result.raw_correction_path,
            "rough_border_overlay_path": result.rough_border_overlay_path,
            "corrected_border_overlay_path": result.corrected_border_overlay_path,
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


def _add_linear_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Order, position and transform a folder of sections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing the section images")
    _add_linear_arguments(p)


def _add_linear_arguments(p: argparse.ArgumentParser) -> None:
    """Every ``linear run`` flag but the folder itself — shared with
    ``abba --linear FOLDER``, which takes the same job, plus ``--save-state``,
    inside a live ABBA session instead of headless."""
    p.add_argument(
        "--tasks",
        default="reorder,position,transform",
        help="Comma-separated subset of reorder,position,transform,nonlinear; "
        "nonlinear adds one image-model border correction per linearly aligned slice",
    )
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument(
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=_PLANE_HELP,
    )
    p.add_argument("--model", default=None, help="Model name for the agent session")
    p.add_argument(
        "--image-provider", default="openai-oauth",
        help="Image provider used only when the nonlinear task is enabled",
    )
    p.add_argument(
        "--image-model", default=None,
        help="Image model for the nonlinear tool; default is the provider's image model",
    )
    p.add_argument(
        "--reasoning",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort for models that expose one. Default: the "
        "provider's own",
    )
    p.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        metavar="UM",
        help="Micrometres per pixel of the section images. Default: read from "
        "each file's TIFF/OME tags",
    )
    p.add_argument(
        "--preprocess",
        default="auto",
        choices=["auto", "none"],
        help="Image preprocessing for everything the agent looks at. Display "
        "only - the image files are never modified",
    )
    p.add_argument(
        "--agent-preprocessing",
        action="store_true",
        help="Offer the preprocess tool: the agent may set channel weights, CLAHE, "
        "N4 and denoising for what it views and what a fit reads",
    )
    p.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        default=True,
        help="Do not let the agent mirror sections across the midline",
    )
    p.add_argument(
        "--hemisphere-cue",
        default="",
        metavar="TEXT",
        help="What marks a hemisphere in these sections (a notch, an injection...)",
    )
    p.add_argument("--thickness", type=int, default=50, help="Section thickness in microns")
    p.add_argument("--interval", type=int, default=200, help="Section interval in microns")
    p.add_argument(
        "--strict-interval",
        action="store_true",
        help="Sections must sit exactly one interval apart",
    )
    p.add_argument(
        "--deepslice", action="store_true", help="Offer the run_deepslice tool"
    )
    p.add_argument(
        "--bayesian", action="store_true", help="Offer the search_position tool"
    )
    p.add_argument(
        "--angles", action="store_true", help="Let the agent set the cutting angles"
    )
    p.add_argument("--pitch", type=float, default=None, help="Host-given cutting pitch, degrees")
    p.add_argument("--yaw", type=float, default=None, help="Host-given cutting yaw, degrees")
    p.add_argument(
        "--elastix", action="store_true", help="Let fit_affine use the Elastix affine"
    )
    p.add_argument(
        "--fact",
        dest="facts",
        action="append",
        default=[],
        metavar="TEXT",
        help="A fact about this stack, passed to the agent verbatim (repeatable)",
    )
    p.add_argument(
        "--positions",
        default=None,
        metavar="JSON",
        help='Host-supplied positions: a JSON file path or inline JSON mapping '
        'filename -> mm',
    )
    p.add_argument(
        "--order",
        default=None,
        metavar="JSON",
        help="Host-supplied order: a JSON file path or inline JSON list of filenames",
    )
    p.add_argument(
        "--transforms", default=None, metavar="JSON",
        help="Host-supplied calibrated linear transforms: JSON file or inline mapping "
        "of filenames to saved transform records",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Results JSON path. Default: <image_folder>/linear_results.json",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        metavar="PATH",
        help="Write a full-content JSONL trace of every agent session here. "
        "Overrides LANGSLICE_TRACE_DIR",
    )
    p.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        default=True,
        help="Ignore any existing checkpoint and start over",
    )
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Stop after one request reports more than N input tokens, cached included, "
        "then allow one submit call (default: disabled; checked after the response)",
    )
    p.add_argument(
        "--gates",
        action="store_true",
        help="Refuse set_positions for a section not compared since its last "
        "write, and submit until view_stack has run after the last write",
    )
    p.add_argument(
        "--playbook",
        action="store_true",
        help="Put GPT-6 Astra's own method in the job statement (hypothesise "
        "everything, confirm, write, re-check, review); for the cheaper models",
    )
    p.add_argument(
        "--max-quota-percent",
        type=int,
        default=None,
        help="Hard stop on the share of the provider's usage window one run "
        "may spend (default: JobSpec.max_quota_percent)",
    )
    p.add_argument(
        "--image-retention", choices=("legacy",), default="legacy",
        help="Compatibility option: legacy working set only; completion retirement was removed",
    )
    p.add_argument(
        "--no-debrief",
        dest="debrief",
        action="store_false",
        help="Skip the post-submit debrief question (what tools the agent missed)",
    )


def _load_json_arg(value: str | None) -> object | None:
    """A JSON argument, given inline or as a path to a JSON file."""
    import json
    from pathlib import Path

    if not value:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _apply_trace_dir(args: argparse.Namespace) -> None:
    """``--trace-dir`` overrides ``LANGSLICE_TRACE_DIR`` for this process."""
    import os

    from langslice.linear.trace import TRACE_DIR_ENV

    if args.trace_dir:
        os.environ[TRACE_DIR_ENV] = args.trace_dir


def _build_linear_spec(args: argparse.Namespace, image_folder: str) -> "JobSpec":
    """Args -> :class:`~langslice.linear.spec.JobSpec`.

    Shared by ``linear run`` (*image_folder* is the positional) and
    ``abba --linear FOLDER`` (*image_folder* is that flag's value) — both
    parsers add the same flags via :func:`_add_linear_arguments`.
    """
    from langslice.linear import JobSpec
    from langslice.linear.spec import NonlinearSpec, PositionSpec, ReorderSpec, TransformSpec

    inputs: dict[str, object] = {}
    positions = _load_json_arg(args.positions)
    if positions is not None:
        inputs["positions"] = positions
    order = _load_json_arg(args.order)
    if order is not None:
        inputs["order"] = order
    transforms = _load_json_arg(args.transforms)
    if transforms is not None:
        inputs["transforms"] = transforms
    if args.pixel_size_um:
        inputs["pixel_size_um"] = float(args.pixel_size_um)
    if args.pitch is not None or args.yaw is not None:
        inputs["angles"] = {"pitch": args.pitch or 0.0, "yaw": args.yaw or 0.0}

    return JobSpec(
        image_folder=image_folder,
        atlas=args.atlas,
        plane=args.plane,
        model=args.model,
        reasoning=args.reasoning,
        out=args.out,
        preprocess=args.preprocess,
        agent_preprocessing=bool(getattr(args, "agent_preprocessing", False)),
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        reorder=ReorderSpec(flip=args.flip, hemisphere_cue=args.hemisphere_cue),
        position=PositionSpec(
            thickness_um=args.thickness,
            interval_um=args.interval,
            strict_interval=args.strict_interval,
            deepslice=args.deepslice,
            bayesian=args.bayesian,
            gated=args.gates,
            playbook=args.playbook,
        ),
        transform=TransformSpec(angles=args.angles, elastix=args.elastix),
        nonlinear=NonlinearSpec(provider=args.image_provider, image_model=args.image_model),
        facts=list(args.facts),
        inputs=inputs,
        resume=args.resume,
        debrief=args.debrief,
        image_retention=getattr(args, "image_retention", "legacy"),
        **({"max_input_tokens": args.max_input_tokens} if args.max_input_tokens else {}),
        **({"max_quota_percent": args.max_quota_percent} if args.max_quota_percent else {}),
    )


def _run_linear(args: argparse.Namespace) -> None:
    import asyncio
    import os

    from langslice.linear import run
    from langslice.linear.trace import TRACE_DIR_ENV

    _apply_trace_dir(args)
    spec = _build_linear_spec(args, args.image_folder)

    print(f"Atlas: {spec.atlas}  Plane: {spec.plane}")
    print(f"Tasks: {', '.join(spec.tasks) or '(none)'}")
    print(f"Interval: {spec.position.interval_um}um  "
          f"Thickness: {spec.position.thickness_um}um")
    print(f"Folder: {spec.image_folder}  Resume: {spec.resume}")
    if os.environ.get(TRACE_DIR_ENV):
        print(f"Agent traces: {os.environ[TRACE_DIR_ENV]}")
    print()

    state = asyncio.run(run(spec, emit=print))

    positioned = [s for s in state.slices if s.position_mm is not None]
    transformed = [s for s in state.slices if s.transform is not None]
    print()
    print("Linear run complete" if state.submitted else "Linear run ended without a submission")
    print(f"  Sections: {len(state.slices)}  Positioned: {len(positioned)}  "
          f"Transformed: {len(transformed)}")
    if state.debrief:
        print("\nAgent debrief (what it reached for that was not there):")
        print(textwrap.indent(state.debrief, "  "))
    if state.interval_breaks:
        print(f"  Interval breaks: {state.interval_breaks}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langslice",
        description="Register brain sections to BrainGlobe atlases with VLM agents and image-gen",
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
        help="Linear methods: section order, position and in-plane alignment",
    )
    linear_sub = linear.add_subparsers(dest="subcommand", required=True)
    _add_linear_run_parser(linear_sub)
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

    # langslice serve
    _add_serve_parser(subparsers)

    # langslice mcp
    _add_mcp_parser(subparsers)

    # langslice claude <cmd> — jobs for the Claude connector
    claude = subparsers.add_parser(
        "claude",
        help="Claude connector: prepare a job to paste into Claude Desktop or Claude Code",
    )
    claude_sub = claude.add_subparsers(dest="subcommand", required=True)
    _add_claude_prepare_parser(claude_sub)

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
        "--nonlinear-atlas",
        default="allen_mouse_10um",
        help="BrainGlobe atlas the nonlinear registration plugin samples for "
        "region maps (independent of the linear agent's --atlas)",
    )
    p.add_argument(
        "--provider",
        default="google",
        choices=[
            "gemini-api", "openai-api", "openai-oauth",
            "google", "openai", "chatgpt",  # legacy aliases
        ],
        help="Image-gen provider for the nonlinear registration",
    )
    p.add_argument(
        "--linear",
        default=None,
        metavar="FOLDER",
        help="Also run the linear agent (order/position/transform) on FOLDER "
        "inside this ABBA session, so you watch it move sections in "
        "BigDataViewer as it works. Takes the same flags as `linear run` "
        "(below); --atlas and --model govern the linear agent, while the "
        "nonlinear plugin uses --nonlinear-atlas / --image-model",
    )
    p.add_argument(
        "--save-state",
        default=None,
        metavar="PATH",
        help="Write an ABBA .abba state file here once the linear agent's "
        "session ends (only with --linear)",
    )
    # Everything `linear run` takes — image_folder is `--linear` here instead
    # of a positional. --atlas / --model belong to the linear agent only; the
    # nonlinear plugin has its own --nonlinear-atlas / --image-model.
    _add_linear_arguments(p)


def _run_abba(args: argparse.Namespace) -> None:
    try:
        import abba_python  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "abba-python is not installed in this environment. Install it with\n"
            '  pip install "langslice[abba]"\n'
            "inside a conda env that provides OpenJDK 11 and Maven "
            f"(see the abba-python installation docs). ({exc})"
        ) from exc

    if args.linear:
        from langslice.integrations.abba_linear import run_linear_in_abba

        _apply_trace_dir(args)
        spec = _build_linear_spec(args, args.linear)
        run_linear_in_abba(
            spec,
            abba_atlas=args.abba_atlas,
            save_state=args.save_state,
            atlas_name=args.nonlinear_atlas,
            provider=args.provider,
            model=args.image_model,
        )
        return

    from langslice.integrations.abba import run_gui_session

    run_gui_session(
        abba_atlas=args.abba_atlas,
        atlas_name=args.nonlinear_atlas,
        provider=args.provider,
        model=args.image_model,
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


def _add_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "mcp",
        help="Serve the linear tools over MCP (stdio) to a host that brings its "
        "own model, such as Claude Desktop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "image_folder",
        nargs="?",
        default=None,
        help="Open this folder at startup. Without it the host names the "
        "folder through the start_job tool",
    )
    p.add_argument(
        "--job",
        default=None,
        metavar="JOB_ID",
        help="Open this saved job at startup, so its tools are listed from the "
        "first request (for hosts that do not follow tool-list changes)",
    )
    # The same job flags as `linear run`; they apply to every folder this
    # server opens. --model, --reasoning and the budget flags are the host's
    # business here and are ignored.
    _add_linear_arguments(p)


def _run_mcp(args: argparse.Namespace) -> None:
    try:
        from langslice.mcp_server.server import serve
    except ImportError as exc:
        raise SystemExit(
            'The MCP SDK is not installed. Install it with\n  pip install "langslice[mcp]"'
            f"\n({exc})"
        ) from exc

    _apply_trace_dir(args)
    serve(lambda folder: _build_linear_spec(args, folder), args.image_folder, args.job)


def _add_claude_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prepare",
        help="Save a job for a folder of sections and print the prompt to paste "
        "into Claude (the command-line Copy prompt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing the section images")
    p.add_argument(
        "--notes", default="", metavar="TEXT",
        help="The user's notes for this job, passed to Claude verbatim",
    )
    # The same job flags as `linear run`; --model, --reasoning and the budget
    # flags belong to LangSlice's own agent and do not reach Claude.
    _add_linear_arguments(p)


def _run_claude_prepare(args: argparse.Namespace) -> None:
    from langslice.api.claude_jobs import prepare_folder

    spec = _build_linear_spec(args, args.image_folder)
    job = prepare_folder(spec, args.notes, trace_dir=args.trace_dir)
    print(f"Saved job {job['job_id']} in {job['job_dir']}", file=sys.stderr)
    print(job["prompt"])


def main(argv: list[str] | None = None):
    # `.env` holds the API keys (GEMINI_API_KEY, OPENAI_API_KEY); every lane
    # reads it, not only the one whose module happens to be imported.
    from langslice.providers.openai_config import _load_dotenv

    _load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command not in {"serve", "login", "version"}:
        from langslice.api.setup import apply_saved_credentials

        apply_saved_credentials()

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
    elif command == "run":
        _run_linear(args)
    elif command == "serve":
        _run_serve(args)
    elif command == "mcp":
        _run_mcp(args)
    elif command == "prepare":
        _run_claude_prepare(args)
    else:
        parser.print_help()
        sys.exit(1)
