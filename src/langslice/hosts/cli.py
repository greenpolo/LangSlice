"""The commands that drive a host: ``abba`` (ABBA with LangSlice installed)
and ``serve`` (the engine service the Fiji connector starts).

The ``langslice`` command (:mod:`langslice.doors.cli`) registers them by
module path and loads this module only when it builds its parser
(``HOST_COMMANDS``), so the doors never import a host.
"""

from __future__ import annotations

import argparse

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
