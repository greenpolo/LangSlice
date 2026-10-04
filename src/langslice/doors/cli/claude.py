"""The commands for a host that brings its own model: ``mcp`` (the linear
tools over MCP) and ``claude prepare`` (a saved job to paste into Claude)."""

from __future__ import annotations

import argparse
import sys

from langslice.doors.cli.linear import add_linear_arguments, apply_trace_dir, build_linear_spec


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
    from langslice.doors.api.claude_jobs import prepare_folder

    spec = build_linear_spec(args, args.image_folder)
    job = prepare_folder(spec, args.notes, trace_dir=args.trace_dir)
    print(f"Saved job {job['job_id']} in {job['job_dir']}", file=sys.stderr)
    print(job["prompt"])
