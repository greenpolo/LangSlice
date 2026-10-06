"""``langslice mcp``: the job's tools over MCP, for a host that brings its
own model. A host opens a job folder made with ``langslice-job FOLDER init``
as it stands, or a saved ABBA job by id."""

from __future__ import annotations

import argparse

from langslice.doors.cli.linear import (
    add_linear_arguments,
    apply_trace_dir,
    build_linear_spec,
    spec_from_args,
)


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
        help="Open this folder at startup: its job (langslice-job FOLDER init) "
        "as saved, else a new one. Without it the host names the folder "
        "through the start_job tool",
    )
    p.add_argument(
        "--job",
        default=None,
        metavar="JOB_ID",
        help="Open this saved job at startup, so its tools are listed from the "
        "first request (for hosts that do not follow tool-list changes)",
    )
    # The same job flags as `linear run`; they make the job of a folder that
    # holds none (a folder's own job keeps its saved settings; with --fresh
    # they start every folder over). --model,
    # --reasoning and the budget flags are the host's business here and are
    # ignored.
    add_linear_arguments(p)


def run_mcp(args: argparse.Namespace) -> None:
    try:
        from langslice.doors.mcp.server import serve
    except ImportError as exc:
        raise SystemExit(
            'The MCP SDK is not installed. Install it with\n  pip install "langslice[mcp]"'
            f"\n({exc})"
        ) from exc

    try:  # the job flags, checked once before serving
        spec_from_args(args, args.image_folder or ".")
    except ValueError as exc:
        raise SystemExit(f"langslice mcp: {exc}") from exc
    apply_trace_dir(args)
    serve(lambda folder: build_linear_spec(args, folder), args.image_folder, args.job,
          fresh=args.fresh)
