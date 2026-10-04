"""``langslice linear``: the agent run and the quick affine preview, and the
job flags every command that opens a stack shares (``linear run``, ``abba
--linear``, ``mcp``, ``claude prepare``, ``job FOLDER init``)."""

from __future__ import annotations

import argparse
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langslice.core.spec import JobSpec

PLANE_HELP = (
    "Slicing plane (normal axis). Position is interpreted along this axis "
    "(AP for coronal, ML for sagittal, DV for horizontal)."
)



def add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Order, position and transform a folder of sections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing the section images")
    add_linear_arguments(p)


def add_linear_arguments(p: argparse.ArgumentParser) -> None:
    """Every ``linear run`` flag but the folder itself — shared with
    ``abba --linear FOLDER``, which takes the same job, plus ``--save-state``,
    inside a live ABBA session instead of headless."""
    p.add_argument(
        "--tasks",
        default="reorder,position,transform",
        help="Comma-separated subset of reorder,position,transform,nonlinear; "
        "nonlinear gives every linearly aligned slice a deformation (fit_deformable), "
        "with image-model border tracing unless --image-provider none",
    )
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument(
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=PLANE_HELP,
    )
    p.add_argument("--model", default=None, help="Model name for the agent session")
    p.add_argument(
        "--image-provider", default="openai-oauth",
        help="Image provider for the nonlinear task's trace_borders (openai-oauth, "
        "openai-api, gemini-api), or none: the nonlinear task then fits deformations "
        "to the stain alone, with no image model",
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
        "--image-resolution",
        default="low",
        choices=["low", "medium", "high", "auto"],
        help="Size of the pictures the agent is shown; auto lets it choose per call",
    )
    p.add_argument(
        "--engine",
        default="either",
        choices=["ants", "elastix", "either"],
        help="Deformable-fit engine for fit_deformable; either lets the agent choose",
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
        "--orientation", default=None, metavar="JSON",
        help='Host-supplied orientation: JSON file or inline mapping of filenames to '
        '{"flip": true|false, "rotation_deg": 0|90|180|270}',
    )
    p.add_argument(
        "--locked", default=None, metavar="JSON",
        help="Sections already aligned in-plane by the user: JSON file or inline list "
        "of filenames; their flip, rotation and transform cannot change",
    )
    p.add_argument(
        "--damaged", default=None, metavar="JSON",
        help="Sections the user marked damaged: JSON file or inline mapping of "
        "filenames to a note; the flags cannot be cleared",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Results JSON path. Default: <job folder>/exports/linear_results.json",
    )
    p.add_argument(
        "--job-dir",
        default=None,
        metavar="PATH",
        help="Put the job folder (checkpoint, undo history, pictures, results) here "
        "instead of <image_folder>/langslice; e.g. one per benchmark arm. A folder "
        "holding the job of another image folder is refused",
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


def load_json_arg(value: str | None, flag: str = "JSON argument") -> object | None:
    """A JSON argument, given inline or as a path to a JSON file.

    An inline object or list (``{...}``, ``[...]``) is parsed as JSON whatever
    its length (never tried as a file name: a real ``--transforms`` map is
    longer than any). Anything else is a path when that file exists, else
    inline JSON. Neither is a ``ValueError`` naming *flag*.
    """
    import json
    from pathlib import Path

    if not value:
        return None
    path: Path | None = None
    if not value.lstrip().startswith(("{", "[")):
        try:
            candidate = Path(value).expanduser()
            path = candidate if candidate.is_file() else None
        except OSError:  # not usable as a file name: inline JSON
            path = None
    if path is not None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{flag}: could not read JSON from {path}: {exc}") from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag}: neither an existing JSON file nor inline JSON ({exc})"
                         ) from exc


def apply_trace_dir(args: argparse.Namespace) -> None:
    """``--trace-dir`` overrides ``LANGSLICE_TRACE_DIR`` for this process."""
    import os

    from langslice.agent.trace import TRACE_DIR_ENV

    if args.trace_dir:
        os.environ[TRACE_DIR_ENV] = args.trace_dir


def build_linear_spec(args: argparse.Namespace, image_folder: str) -> JobSpec:
    """Args -> :class:`~langslice.core.spec.JobSpec`.

    Shared by ``linear run`` (*image_folder* is the positional) and
    ``abba --linear FOLDER`` (*image_folder* is that flag's value) — both
    parsers add the same flags via :func:`add_linear_arguments`.
    """
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, ReorderSpec, TransformSpec

    inputs: dict[str, object] = {}
    positions = load_json_arg(args.positions, "--positions")
    if positions is not None:
        inputs["positions"] = positions
    order = load_json_arg(args.order, "--order")
    if order is not None:
        inputs["order"] = order
    transforms = load_json_arg(args.transforms, "--transforms")
    if transforms is not None:
        inputs["transforms"] = transforms
    for key, flag in (("orientation", "--orientation"), ("locked", "--locked"),
                      ("damaged", "--damaged")):
        value = load_json_arg(getattr(args, key, None), flag)
        if value is not None:
            inputs[key] = value
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
        job_dir=getattr(args, "job_dir", None),
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
        transform=TransformSpec(angles=args.angles),
        nonlinear=NonlinearSpec(
            provider=args.image_provider,
            image_model=args.image_model,
            engine=getattr(args, "engine", "either"),
        ),
        image_resolution=getattr(args, "image_resolution", "low"),
        facts=list(args.facts),
        inputs=inputs,
        resume=args.resume,
        debrief=args.debrief,
        image_retention=getattr(args, "image_retention", "legacy"),
        **({"max_input_tokens": args.max_input_tokens} if args.max_input_tokens else {}),
        **({"max_quota_percent": args.max_quota_percent} if args.max_quota_percent else {}),
    )


def run_linear(args: argparse.Namespace) -> None:
    import asyncio
    import os

    from langslice.agent.engine import run
    from langslice.agent.trace import TRACE_DIR_ENV

    apply_trace_dir(args)
    spec = build_linear_spec(args, args.image_folder)

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


def add_quick_affine_parser(subparsers: argparse._SubParsersAction) -> None:
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
        help=PLANE_HELP,
    )
    p.add_argument("--out", required=True, help="Path to write the warped slice PNG")
    p.add_argument("--json", action="store_true", help="Print result JSON to stdout")


def run_quick_affine(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from PIL import Image

    from langslice.core.nonlinear.quick_affine import quick_affine_register

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
