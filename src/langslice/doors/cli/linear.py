"""``langslice linear``: the agent run, and the job flags every command
that opens a stack shares (``linear run``, ``mcp``, ``job FOLDER init``)."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.spec import DEFAULT_TASKS as SPEC_DEFAULT_TASKS
from langslice.core.toolset import valid_names
from langslice.doors import plugins as _plugins  # noqa: F401 - registers the plugin loader

if TYPE_CHECKING:
    from langslice.core.spec import JobSpec

#: The tasks a stack-opening command runs without ``--tasks``: the job's own default.
DEFAULT_TASKS = ",".join(SPEC_DEFAULT_TASKS)
#: ... and with ``--registration``: only the nonlinear step, on top of the
#: imported linear registration.
REGISTRATION_TASKS = "nonlinear"
#: The flags ``--registration`` cannot be combined with (it supplies what
#: they would), by argparse destination.
REGISTRATION_CLASHES = {
    "positions": "--positions", "transforms": "--transforms",
    "orientation": "--orientation", "pitch": "--pitch", "yaw": "--yaw",
    "section_angles": "--section-angles", "angles": "--angles",
}

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
    p.add_argument(
        "--tool-plugin", dest="tool_plugins", action="append", default=[],
        metavar="MODULE:FACTORY",
        help="Add the tools of a tool plugin to the agent's toolbox (repeatable): a "
        "Python module and the function in it that returns the tools. Importing it runs "
        "its code; the job records which plugin ran (docs/plugins.md)",
    )
    p.add_argument(
        "--tool-plugin-config", dest="tool_plugin_configs", action="append", default=[],
        metavar="MODULE:FACTORY=JSON",
        help="Options for a plugin's factory (repeatable): the plugin's reference, '=', "
        "then a JSON object or the path of a JSON file",
    )
    add_session_arguments(p)
    add_trace_argument(p)
    p.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        default=True,
        help="Ignore any existing checkpoint and start over",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open the LangSlice window on this job (`langslice gui`) to watch the run; "
        "closing it does not stop the run",
    )


def add_linear_arguments(p: argparse.ArgumentParser) -> None:
    """Every ``linear run`` flag but the folder itself, shared by every
    command that opens a stack."""
    p.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated subset of position,transform,nonlinear: position "
        "places and orders the sections, transform aligns them in-plane, nonlinear "
        "deforms each linearly aligned section (fireants_syn, and trace_borders with an "
        "image model). A task left out takes its answer from the flags that supply "
        "it (--positions, --transforms ...). Default: "
        f"{DEFAULT_TASKS}; with --registration, {REGISTRATION_TASKS} (the imported "
        "linear registration kept as it is)",
    )
    p.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name")
    p.add_argument(
        "--plane",
        default="coronal",
        choices=["coronal", "sagittal", "horizontal"],
        help=PLANE_HELP,
    )
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
        "--image-prompt", default=None, metavar="FILE",
        help="A UTF-8 text file holding the base prompt trace_borders sends the image "
        "model, in place of LangSlice's own (NonlinearSpec.image_prompt); {plane} in it "
        "becomes the section plane. It must describe the provider's attachment order, "
        "and its traces are recorded as an untested profile. The agent's per-section "
        "edit starts from it. Default: LangSlice's prompt for the provider",
    )
    p.add_argument(
        "--pixel-size-um",
        default=None,
        metavar="UM|JSON",
        help="Micrometres per pixel of the section images: one number for every "
        "file, or a JSON file or inline mapping of filenames to micrometres per "
        "pixel. Default (and for a file the mapping does not name): read from "
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
        "--image-resolution",
        default="low",
        choices=["low", "medium", "high", "auto"],
        help="Size of the pictures the agent is shown; auto lets it choose per call",
    )
    p.add_argument(
        "--disable",
        dest="disabled",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="Tools and tool options the agent does not get, on top of what the tasks "
        "give (repeatable): a tool (zoom), or tool:option (look:positioning, "
        "interactive_transform:flip, which keeps the agent from mirroring sections). "
        "Valid names: " + ", ".join(valid_names()),
    )
    p.add_argument(
        "--hemisphere-cue",
        default="",
        metavar="TEXT",
        help="What marks a hemisphere in these sections (a notch, an injection...)",
    )
    p.add_argument("--thickness", type=int,
                   help="Known section thickness in microns; omit if unknown")
    p.add_argument("--interval", type=int,
                   help="Known section interval in microns; omit to infer spacing")
    p.add_argument(
        "--strict-interval",
        action="store_true",
        help="Sections must sit exactly one interval apart",
    )
    p.add_argument(
        "--angles", action="store_true", help="Let the agent set the cutting angles"
    )
    p.add_argument("--pitch", type=float, default=None, help="Host-given cutting pitch, degrees")
    p.add_argument("--yaw", type=float, default=None, help="Host-given cutting yaw, degrees")
    p.add_argument(
        "--section-angles", default=None, metavar="JSON",
        help='Host-given cutting angles per section (a registration made elsewhere): JSON '
        'file or inline mapping of filenames to {"pitch": deg, "yaw": deg}; not with '
        "--pitch/--yaw",
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
        "--registration", default=None, metavar="FILE",
        help="A linear registration made elsewhere, imported as every section's "
        "position, cutting angles, orientation and in-plane transform: QuickNII or "
        "VisuAlign JSON/XML, DeepSlice CSV/JSON/XML, or a LangSlice registration.json. "
        "Not with --positions, --transforms, --orientation, --pitch/--yaw, "
        "--section-angles or --angles. VisuAlign markers are not imported",
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
        "--frozen", default=None, metavar="JSON",
        help='What the agent may not change: JSON file or inline mapping of a filename '
        '(or "*" for every section) to a list of properties, any of position, angles, '
        'orientation, affine, deformation; a frozen property keeps the value the '
        'supplied inputs give it',
    )
    p.add_argument(
        "--damaged", default=None, metavar="JSON",
        help="Notes on damaged tissue: JSON file or inline mapping of filenames "
        "to notes; notes alone do not exclude atlas regions from fits",
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
        "instead of <image_folder>/langslice. A folder holding the job of another "
        "image folder is refused",
    )
    p.add_argument(
        "--playbook",
        action="store_true",
        help="Add optional positioning guidance: compare candidate planes and regions, "
        "write supported placements, revisit uncertainty and review the stack",
    )
    for task in ("position", "transform", "nonlinear"):
        p.add_argument(
            f"--{task}-notes", dest=f"{task}_notes", default="", metavar="TEXT",
            help=f"The user's notes for the {task} task, shown to the agent with the task "
            f"(one line each; {task.capitalize()}Spec.notes)",
        )


def add_session_arguments(p: argparse.ArgumentParser) -> None:
    """The flags of the built-in agent's session (``linear run`` only)."""
    p.add_argument("--model", default=None, help="Model name for the agent session")
    p.add_argument(
        "--reasoning",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort for models that expose one. Default: the "
        "provider's own",
    )
    p.add_argument(
        "--no-debrief",
        dest="debrief",
        action="store_false",
        help="Skip the post-submit debrief question (what tools the agent missed)",
    )
    p.add_argument(
        "--force-view",
        dest="force_view",
        action="store_true",
        help="The change and fitting tools always return their picture: their view "
        "argument is not offered (JobSpec.force_view)",
    )
    p.add_argument(
        "--require-deformation",
        dest="require_deformation",
        action="store_true",
        help="Nonlinear task: submit is refused until every section carries a deformation "
        "(NonlinearSpec.require_deformation); otherwise a deformation is the agent's choice",
    )


def add_trace_argument(p: argparse.ArgumentParser) -> None:
    """``--trace-dir`` (``linear run`` and ``mcp``, :func:`apply_trace_dir`)."""
    p.add_argument(
        "--trace-dir",
        default=None,
        metavar="PATH",
        help="Write a full-content JSONL trace here. Overrides LANGSLICE_TRACE_DIR "
        "(linear run without either: <job folder>/trace)",
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


def read_text_arg(value: str | None, flag: str) -> str | None:
    """The text of the UTF-8 file *value* names (None without one); a file that
    cannot be read is a ``ValueError`` naming *flag*."""
    if not value:
        return None
    path = Path(value).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{flag}: could not read {path}: {exc}") from exc


def apply_trace_dir(args: argparse.Namespace) -> None:
    """``--trace-dir`` overrides ``LANGSLICE_TRACE_DIR`` for this process."""
    import os

    from langslice.agent.trace import TRACE_DIR_ENV

    if args.trace_dir:
        os.environ[TRACE_DIR_ENV] = args.trace_dir


def build_linear_spec(
    args: argparse.Namespace, image_folder: str, *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> JobSpec:
    """Args -> :class:`~langslice.core.spec.JobSpec`.

    Shared by every command that opens a stack (``linear run``, ``mcp``,
    ``job FOLDER init``):
    their parsers add the same flags via :func:`add_linear_arguments`. A
    ``--registration`` file is imported here (:func:`build_job_spec`; its
    warnings said through *emit*).
    """
    spec, _report = build_job_spec(args, image_folder, atlas_loader=atlas_loader, emit=emit)
    return spec


def build_job_spec(
    args: argparse.Namespace, image_folder: str, *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> tuple[JobSpec, dict[str, Any] | None]:
    """:func:`build_linear_spec` and the ``--registration`` import report
    (None without one: ``doors.jobs.with_registration``). ``ValueError`` for
    flags that cannot be combined, an unreadable or ambiguous registration,
    or one that places no section."""
    spec = spec_from_args(args, image_folder)
    registration = getattr(args, "registration", None)
    if not registration:
        return spec, None
    from langslice.doors.jobs import with_registration

    return with_registration(spec, registration, atlas_loader=atlas_loader, emit=emit)


def spec_from_args(args: argparse.Namespace, image_folder: str) -> JobSpec:
    """The spec the flags give, ``--registration`` not yet imported (its
    clashes with other flags refused, its default tasks applied)."""
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec

    registration = getattr(args, "registration", None)
    if registration:
        given = [flag for name, flag in REGISTRATION_CLASHES.items()
                 if getattr(args, name, None) not in (None, False)]
        if given:
            raise ValueError(f"--registration supplies every section's position, cutting "
                             f"angles, orientation and transform; it cannot be combined "
                             f"with {', '.join(given)}")
    tasks = args.tasks if args.tasks is not None else (
        REGISTRATION_TASKS if registration else DEFAULT_TASKS)

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
    for key, flag in (("orientation", "--orientation"), ("damaged", "--damaged")):
        value = load_json_arg(getattr(args, key, None), flag)
        if value is not None:
            inputs[key] = value
    if args.pixel_size_um:  # a number, or a JSON object / file of filename: um
        inputs["pixel_size_um"] = load_json_arg(args.pixel_size_um, "--pixel-size-um")
    if args.pitch is not None or args.yaw is not None:
        inputs["angles"] = {"pitch": args.pitch or 0.0, "yaw": args.yaw or 0.0}
    section_angles = load_json_arg(getattr(args, "section_angles", None), "--section-angles")
    if section_angles is not None:
        if "angles" in inputs:
            raise ValueError("--section-angles gives each section its own plane; it cannot "
                             "be combined with the stack-wide --pitch/--yaw")
        if not isinstance(section_angles, dict) or not all(
                isinstance(value, dict) for value in section_angles.values()):
            raise ValueError('--section-angles must map filenames to {"pitch": deg, '
                             '"yaw": deg}')
        inputs["angles"] = section_angles

    frozen: Any = load_json_arg(getattr(args, "frozen", None), "--frozen")
    return JobSpec(
        image_folder=image_folder,
        atlas=args.atlas,
        plane=args.plane,
        model=getattr(args, "model", None),
        reasoning=getattr(args, "reasoning", None),
        out=args.out,
        job_dir=getattr(args, "job_dir", None),
        preprocess=args.preprocess,
        tasks=[task.strip() for task in tasks.split(",") if task.strip()],
        position=PositionSpec(
            thickness_um=args.thickness,
            interval_um=args.interval,
            strict_interval=args.strict_interval,
            playbook=args.playbook,
            notes=getattr(args, "position_notes", ""),
        ),
        transform=TransformSpec(angles=args.angles, hemisphere_cue=args.hemisphere_cue,
                                notes=getattr(args, "transform_notes", "")),
        disabled=[name.strip() for item in getattr(args, "disabled", []) for name in item.split(",")
                  if name.strip()],
        nonlinear=NonlinearSpec(
            provider=args.image_provider,
            image_model=args.image_model,
            require_deformation=bool(getattr(args, "require_deformation", False)),
            notes=getattr(args, "nonlinear_notes", ""),
            image_prompt=read_text_arg(getattr(args, "image_prompt", None), "--image-prompt"),
        ),
        image_resolution=getattr(args, "image_resolution", "low"),
        facts=list(args.facts),
        inputs=inputs,
        frozen=frozen or {},
        resume=args.resume,
        debrief=getattr(args, "debrief", True),
        force_view=bool(getattr(args, "force_view", False)),
    )


def with_tool_plugins(spec: JobSpec, args: argparse.Namespace) -> JobSpec:
    """*spec* with the tool plugins ``--tool-plugin`` names (and their
    ``--tool-plugin-config`` options): the plugins are loaded, so a plugin
    that cannot be loaded, or whose tools clash, is a ``ValueError`` here."""
    refs = [ref.strip() for item in getattr(args, "tool_plugins", []) for ref in item.split(",")
            if ref.strip()]
    if not refs:
        if getattr(args, "tool_plugin_configs", []):
            raise ValueError("--tool-plugin-config needs the --tool-plugin it configures")
        return spec
    import dataclasses

    configs: dict[str, Any] = {}
    for item in getattr(args, "tool_plugin_configs", []):
        ref, equals, value = item.partition("=")
        if not equals or ref.strip() not in refs:
            raise ValueError(f"--tool-plugin-config {item[:40]!r}: expected MODULE:FACTORY=JSON "
                             "of a --tool-plugin")
        loaded = load_json_arg(value, "--tool-plugin-config")
        if not isinstance(loaded, dict):
            raise ValueError("--tool-plugin-config options are a JSON object")
        configs[ref.strip()] = loaded
    return dataclasses.replace(spec, plugins=[*spec.plugins, *refs],
                               plugin_config={**spec.plugin_config, **configs})


def open_gui(spec: Any) -> None:
    """Start ``langslice gui`` on the run's job folder as a process of its own
    (the hosts layer, reached by command, never imported); it outlives nothing
    of the run, and the run never waits for it."""
    import subprocess
    import sys

    from langslice.job.layout import locate_job_folder

    folder, _ = locate_job_folder(spec.image_folder, spec.job_dir, register=False)
    command = [sys.executable, "-c", "from langslice.cli import main; main()", "gui",
               str(folder)]
    try:
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        print(f"The LangSlice window could not be started: {exc}")


def run_linear(args: argparse.Namespace) -> None:
    import asyncio
    import os

    from langslice.agent.engine import run
    from langslice.agent.trace import TRACE_DIR_ENV

    apply_trace_dir(args)
    try:
        spec = with_tool_plugins(build_linear_spec(args, args.image_folder), args)
    except ValueError as exc:
        raise SystemExit(f"langslice linear run: {exc}") from exc
    if not os.environ.get(TRACE_DIR_ENV):
        # Every run keeps its full agent trace: by default in the job folder.
        from langslice.agent.trace import job_trace_dir
        from langslice.job.layout import job_folder_for

        folder = Path(spec.job_dir) if spec.job_dir else job_folder_for(spec.image_folder)
        os.environ[TRACE_DIR_ENV] = str(job_trace_dir(folder))
    if getattr(args, "gui", False):
        open_gui(spec)

    print(f"Atlas: {spec.atlas}  Plane: {spec.plane}")
    print(f"Tasks: {', '.join(spec.tasks) or '(none)'}")
    interval = (f"{spec.position.interval_um}um"
                if spec.position.interval_um is not None else "not supplied")
    thickness = (f"{spec.position.thickness_um}um"
                 if spec.position.thickness_um is not None else "not supplied")
    print(f"Interval: {interval}  Thickness: {thickness}")
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
