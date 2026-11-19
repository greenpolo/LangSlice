"""``langslice linear run``: the registration run of an agent LangSlice drives
itself, and the job spec every command that opens a stack makes from the job
flags (``linear run``, ``mcp``, ``langslice-job FOLDER init``; the flags are
:mod:`langslice.doors.commands.flags`)."""

from __future__ import annotations

import argparse
import importlib
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.stack.section import INITIAL
from langslice.doors import credentials
from langslice.doors.agent_cli.background import DETACHED
from langslice.doors.commands.flags import add_flags, load_json_arg, values_from_args
from langslice.job.settings.defaults import choose_provider, default_tasks

if TYPE_CHECKING:
    from langslice.job.settings.spec import JobSpec

REGISTRATION_CLASHES = {
    "positions": "--positions",
    "transforms": "--transforms",
    "orientation": "--orientation",
    "pitch": "--pitch",
    "yaw": "--yaw",
    "section_angles": "--section-angles",
    "transform_angles": "--transform-angles",
}

importlib.import_module("langslice.plugins")


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


def apply_trace_dir(args: argparse.Namespace) -> None:
    """``--trace-dir`` overrides ``LANGSLICE_TRACE_DIR`` for this process."""
    import os

    from langslice.doors.agent.trace import TRACE_DIR_ENV

    if args.trace_dir:
        os.environ[TRACE_DIR_ENV] = args.trace_dir


def build_linear_spec(
    args: argparse.Namespace,
    image_folder: str,
    *,
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
    args: argparse.Namespace,
    image_folder: str,
    *,
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
    clashes with other flags refused). A setting no flag gives is the spec's
    default; the tasks and the image provider are those of
    :mod:`langslice.job.settings.defaults`."""
    from langslice.job.settings.spec import JobSpec

    registration = getattr(args, "registration", None)
    if registration:
        given = [
            flag
            for name, flag in REGISTRATION_CLASHES.items()
            if getattr(args, name, None) not in (None, False)
        ]
        if given:
            raise ValueError(
                f"--registration supplies every section's position, cutting "
                f"angles, orientation and transform; it cannot be combined "
                f"with {', '.join(given)}"
            )
    values = values_from_args(args)
    values["image_folder"] = image_folder
    inputs: dict[str, Any] = values.get("inputs", {})
    if "tasks" not in values:
        values["tasks"] = default_tasks(image_folder, inputs, registration=bool(registration))
    nonlinear: dict[str, Any] = values.setdefault("nonlinear", {})
    nonlinear["provider"] = choose_provider(
        getattr(args, "nonlinear_provider", None),
        enabled=values.get("enabled", []),
        saved=credentials.saved_image_provider,
    )
    return JobSpec.from_dict(values)


def with_tool_plugins(spec: JobSpec, args: argparse.Namespace) -> JobSpec:
    """*spec* with the tool plugins ``--plugins`` names (and their
    ``--plugin-config`` options): the plugins are loaded, so a plugin
    that cannot be loaded, or whose tools clash, is a ``ValueError`` here."""
    named: list[str] = getattr(args, "plugins", [])
    refs = [ref.strip() for item in named for ref in item.split(",") if ref.strip()]
    if not refs:
        if getattr(args, "plugin_config", []):
            raise ValueError("--plugin-config needs the --plugins it configures")
        return spec
    configs: dict[str, Any] = {}
    for item in getattr(args, "plugin_config", []):
        ref, equals, value = item.partition("=")
        if not equals or ref.strip() not in refs:
            raise ValueError(
                f"--plugin-config {item[:40]!r}: expected MODULE:FACTORY=JSON of a --plugins"
            )
        loaded = load_json_arg(value, "--plugin-config")
        if not isinstance(loaded, dict):
            raise ValueError("--plugin-config options are a JSON object")
        configs[ref.strip()] = loaded
    return spec.with_changes(
        plugins=[*spec.plugins, *refs], plugin_config={**spec.plugin_config, **configs}
    )


def open_gui(spec: Any) -> None:
    """Start ``langslice gui`` on the run's job folder as a process of its own
    (the hosts layer, reached by command, never imported); it outlives nothing
    of the run, and the run never waits for it."""
    import subprocess
    import sys

    from langslice.job.folder.layout import locate_job_folder

    folder = locate_job_folder(spec.image_folder, spec.job_dir, register=False)
    command = [sys.executable, "-c", "from langslice.cli import main; main()", "gui", str(folder)]
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=DETACHED.creationflags,
            start_new_session=DETACHED.start_new_session,
        )
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
        print(
            f"Model: {args.model}"
            + (f"  Reasoning: {args.reasoning}" if args.reasoning else "")
            + " (from the LangSlice window's Settings)"
        )
    try:
        spec = with_tool_plugins(build_linear_spec(args, args.image_folder), args)
    except ValueError as exc:
        raise SystemExit(f"langslice linear run: {exc}") from exc
    if not os.environ.get(TRACE_DIR_ENV):
        from langslice.doors.agent.trace import job_trace_dir
        from langslice.job.folder.layout import job_folder_for

        folder = Path(spec.job_dir) if spec.job_dir else job_folder_for(spec.image_folder)
        os.environ[TRACE_DIR_ENV] = str(job_trace_dir(folder))
    if getattr(args, "gui", False):
        open_gui(spec)

    print(f"Atlas: {spec.atlas}  Plane: {spec.plane}")
    print(f"Tasks: {', '.join(spec.tasks) or '(none)'}")
    interval = (
        f"{spec.position.interval_um}um"
        if spec.position.interval_um is not None
        else "not supplied"
    )
    thickness = (
        f"{spec.position.thickness_um}um"
        if spec.position.thickness_um is not None
        else "not supplied"
    )
    print(f"Interval: {interval}  Thickness: {thickness}")
    print(f"Folder: {spec.image_folder}  Resume: {spec.resume}")
    if os.environ.get(TRACE_DIR_ENV):
        print(f"Agent traces: {os.environ[TRACE_DIR_ENV]}")
    print()

    state = asyncio.run(run(spec, emit=print))

    positioned = [s for s in state.slices if s.position_source != INITIAL]
    transformed = [s for s in state.slices if s.transform is not None]
    print()
    print("Linear run complete" if state.submitted else "Linear run ended without a submission")
    print(
        f"  Sections: {len(state.slices)}  Positioned: {len(positioned)}  "
        f"Transformed: {len(transformed)}"
    )
    if state.debrief:
        print("\nAgent debrief (what it reached for that was not there):")
        print(textwrap.indent(state.debrief, "  "))
    if state.interval_breaks:
        print(f"  Interval breaks: {state.interval_breaks}")
