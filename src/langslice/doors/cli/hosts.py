"""The commands that serve a host: ``abba``, ``serve``, ``mcp`` and
``claude prepare``."""

from __future__ import annotations

import argparse
import sys

from langslice.doors.cli.linear import add_linear_arguments, apply_trace_dir, build_linear_spec


def add_abba_parser(subparsers: argparse._SubParsersAction) -> None:
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
    add_linear_arguments(p)


def run_abba(args: argparse.Namespace) -> None:
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
        from langslice.hosts.integrations.abba_linear import run_linear_in_abba

        apply_trace_dir(args)
        spec = build_linear_spec(args, args.linear)
        run_linear_in_abba(
            spec,
            abba_atlas=args.abba_atlas,
            save_state=args.save_state,
            atlas_name=args.nonlinear_atlas,
            provider=args.provider,
            model=args.image_model,
        )
        return

    from langslice.hosts.integrations.abba import run_gui_session

    run_gui_session(
        abba_atlas=args.abba_atlas,
        atlas_name=args.nonlinear_atlas,
        provider=args.provider,
        model=args.image_model,
    )


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
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


def run_serve(args: argparse.Namespace) -> None:
    if not args.stdio:
        raise SystemExit("serve currently requires --stdio")
    from langslice.hosts.api.service import run_stdio

    raise SystemExit(run_stdio())


def add_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
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
    add_linear_arguments(p)


def run_mcp(args: argparse.Namespace) -> None:
    try:
        from langslice.doors.mcp.server import serve
    except ImportError as exc:
        raise SystemExit(
            'The MCP SDK is not installed. Install it with\n  pip install "langslice[mcp]"'
            f"\n({exc})"
        ) from exc

    apply_trace_dir(args)
    serve(lambda folder: build_linear_spec(args, folder), args.image_folder, args.job)


def add_claude_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
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
    add_linear_arguments(p)


def run_claude_prepare(args: argparse.Namespace) -> None:
    from langslice.hosts.api.claude_jobs import prepare_folder

    spec = build_linear_spec(args, args.image_folder)
    job = prepare_folder(spec, args.notes, trace_dir=args.trace_dir)
    print(f"Saved job {job['job_id']} in {job['job_dir']}", file=sys.stderr)
    print(job["prompt"])
