"""``langslice nonlinear register``: image-model registration of one section."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

from langslice.doors.cli.linear import PLANE_HELP

if TYPE_CHECKING:
    from langslice.core.nonlinear.types import Deformation


def resolve_register_models(
    *,
    default_image_model: str,
    default_review_model: str,
    image_model: str | None,
    review_model: str | None,
) -> tuple[str, str]:
    resolved_image_model = image_model or default_image_model
    resolved_review_model = review_model or default_review_model
    return resolved_image_model, resolved_review_model


def add_register_parser(subparsers: argparse._SubParsersAction) -> None:
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
        help=PLANE_HELP,
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
        choices=["none", "deformable"],
        help="Fit after the model call. 'none' (default) fits nothing: outputs "
        "are the model's lines on the original plus the rough placement with "
        "an identity residual. 'deformable' fits the model's lines against the "
        "atlas family borders with the deformable package (Elastix B-spline; "
        "fit_deformable's traced_lines fit).",
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


def run_register(args: argparse.Namespace) -> None:
    import json
    import os

    from langslice.doors.api.models import RegisterRequest
    from langslice.doors.api.runtime import run_register

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

    image_model, review_model = resolve_register_models(
        default_image_model=default_image_model,
        default_review_model=default_review_model,
        image_model=image_model_arg,
        review_model=args.review_model,
    )

    out_dir = register_output_dir(args.out, args.atlas)

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


def register_output_dir(out: str | None, atlas: str):
    from datetime import datetime
    from pathlib import Path

    if out:
        out_dir = Path(out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("langslice_output") / f"{timestamp}_{atlas}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
