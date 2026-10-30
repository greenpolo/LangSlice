"""``langslice linear``: the agent run, and the job spec every command that
opens a stack makes from the job flags (``linear run``, ``mcp``, ``job
FOLDER init``; the flags are :mod:`langslice.doors.commands.flags`)."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.doors import plugins as _plugins  # noqa: F401 - registers the plugin loader
from langslice.doors.commands.flags import add_flags
from langslice.job.settings.schema import CLI_TASKS, REGISTRATION_TASKS

if TYPE_CHECKING:
    from langslice.job.settings.spec import JobSpec

#: The flags ``--registration`` cannot be combined with (it supplies what
#: they would), by argparse destination.
REGISTRATION_CLASHES = {
    "positions": "--positions", "transforms": "--transforms",
    "orientation": "--orientation", "pitch": "--pitch", "yaw": "--yaw",
    "section_angles": "--section-angles",
    "transform_angles": "--transform-angles",
}

def add_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "run",
        help="Order, position and transform a folder of sections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_folder", help="Folder containing the section images")
    add_flags(p, "job")
    add_flags(p, "plugins")
    add_flags(p, "session")
    add_trace_argument(p)
    add_flags(p, "resume")
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open the LangSlice window on this job (`langslice gui`) to watch the run; "
        "closing it does not stop the run",
    )


def add_trace_argument(p: argparse.ArgumentParser) -> None:
    """``--trace-dir`` (``linear run`` and ``mcp``, :func:`apply_trace_dir`)."""
    p.add_argument(
        "--trace-dir",
        default=None,
        metavar="PATH",
        help="Write a full-content JSONL trace here. Overrides LANGSLICE_TRACE_DIR "
        "(linear run also writes it to <job folder>/trace, and only there without "
        "either)",
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

    from langslice.doors.agent.trace import TRACE_DIR_ENV

    if args.trace_dir:
        os.environ[TRACE_DIR_ENV] = args.trace_dir


def build_linear_spec(
    args: argparse.Namespace, image_folder: str, *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> JobSpec:
    """Args -> :class:`~langslice.job.settings.spec.JobSpec`.

    Shared by every command that opens a stack (``linear run``, ``mcp``,
    ``job FOLDER init``): their parsers add the same job flags
    (:mod:`langslice.doors.commands.flags`). A ``--registration`` file is
    imported here (:func:`build_job_spec`; its warnings said through
    *emit*).
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
    from langslice.job.settings.spec import (
        JobSpec,
        NonlinearSpec,
        PositionSpec,
        TransformSpec,
    )

    registration = getattr(args, "registration", None)
    if registration:
        given = [flag for name, flag in REGISTRATION_CLASHES.items()
                 if getattr(args, name, None) not in (None, False)]
        if given:
            raise ValueError(f"--registration supplies every section's position, cutting "
                             f"angles, orientation and transform; it cannot be combined "
                             f"with {', '.join(given)}")
    tasks = args.tasks if args.tasks is not None else (
        REGISTRATION_TASKS if registration else CLI_TASKS)

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
            thickness_um=getattr(args, "position_thickness_um", None),
            interval_um=getattr(args, "position_interval_um", None),
            strict_interval=bool(
                getattr(args, "position_strict_interval", False)),
            playbook=bool(getattr(args, "position_playbook", False)),
            notes=getattr(args, "position_notes", ""),
        ),
        transform=TransformSpec(
            angles=bool(getattr(args, "transform_angles", False)),
            hemisphere_cue=getattr(args, "transform_hemisphere_cue", ""),
            notes=getattr(args, "transform_notes", "")),
        disabled=[name.strip() for item in getattr(args, "disabled", []) for name in item.split(",")
                  if name.strip()],
        nonlinear=NonlinearSpec(
            provider=getattr(
                args, "nonlinear_provider", NonlinearSpec.provider),
            image_model=getattr(args, "nonlinear_image_model", None),
            require_deformation=bool(
                getattr(args, "nonlinear_require_deformation", False)),
            notes=getattr(args, "nonlinear_notes", ""),
            image_prompt=read_text_arg(
                getattr(args, "nonlinear_image_prompt", None),
                "--nonlinear-image-prompt"),
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
    """*spec* with the tool plugins ``--plugins`` names (and their
    ``--plugin-config`` options): the plugins are loaded, so a plugin
    that cannot be loaded, or whose tools clash, is a ``ValueError`` here."""
    named: list[str] = getattr(args, "plugins", [])
    refs = [ref.strip() for item in named for ref in item.split(",")
            if ref.strip()]
    if not refs:
        if getattr(args, "plugin_config", []):
            raise ValueError(
                "--plugin-config needs the --plugins it configures")
        return spec
    import dataclasses

    configs: dict[str, Any] = {}
    for item in getattr(args, "plugin_config", []):
        ref, equals, value = item.partition("=")
        if not equals or ref.strip() not in refs:
            raise ValueError(f"--plugin-config {item[:40]!r}: expected "
                             "MODULE:FACTORY=JSON of a --plugins")
        loaded = load_json_arg(value, "--plugin-config")
        if not isinstance(loaded, dict):
            raise ValueError("--plugin-config options are a JSON object")
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


def apply_saved_agent_choice(args: argparse.Namespace) -> bool:
    """Fill ``--model`` (and ``--reasoning`` when it too is unset) from the
    user's settings file (:func:`langslice.doors.user_settings.agent_choice`)
    when the command names no model; True when it did. Without the file,
    a chosen connection or a model saved for it, *args* is left as given."""
    if getattr(args, "model", None):
        return False
    from langslice.doors.user_settings import agent_choice

    choice = agent_choice()
    if choice is None:
        return False
    args.model = choice[0]
    if not getattr(args, "reasoning", None):
        args.reasoning = choice[1]
    return True


def run_linear(args: argparse.Namespace) -> None:
    import asyncio
    import os

    from langslice.doors.agent.engine import run
    from langslice.doors.agent.trace import TRACE_DIR_ENV

    apply_trace_dir(args)
    chosen = apply_saved_agent_choice(args)
    if chosen:
        print(f"Model: {args.model}"
              + (f"  Reasoning: {args.reasoning}" if args.reasoning else "")
              + " (from the LangSlice window's Settings)")
    try:
        spec = with_tool_plugins(build_linear_spec(args, args.image_folder), args)
    except ValueError as exc:
        raise SystemExit(f"langslice linear run: {exc}") from exc
    if not os.environ.get(TRACE_DIR_ENV):
        # Every run keeps its full agent trace: by default in the job folder.
        from langslice.doors.agent.trace import job_trace_dir
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
