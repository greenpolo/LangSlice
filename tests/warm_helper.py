"""The agent CLI and its warm process on the synthetic atlas, in processes of their own.

``python -m tests.warm_helper FOLDER VERB ...`` is ``langslice-job FOLDER
VERB ...`` as installed (the warm process first,
:func:`langslice.doors.cli.warm.run`; a cold call here otherwise), with the
shared synthetic atlas and fit settings (``tests/synthetic_stack.py``); the
warm process it starts is :func:`serve`, the same server with that atlas.
``tests/test_warm.py`` runs both. A module ``warmprobe`` on the import path
is loaded by the server, so a test can change a file the server's code came
from.
"""

from __future__ import annotations

import os
import sys

#: How the client starts its server here (``warm.BOOT`` for the tests).
BOOT = ("import sys\nif sys.path and sys.path[0] == '': del sys.path[0]\n"
        "from tests.warm_helper import serve\nraise SystemExit(serve(sys.argv[1:]))")


def _synthetic() -> None:
    from tests.cli_child import install
    from tests.synthetic_stack import apply_patches, atlas_loader

    apply_patches()
    install(atlas_loader())


def serve(args: list[str]) -> int:
    """The warm process (:func:`langslice.doors.cli.warm.serve_main`) on the
    synthetic atlas. The environment is put back as it came: the imports
    may set variables (OpenCV's ``LD_LIBRARY_PATH``) that the server compares
    with its callers'."""
    held = dict(os.environ)
    _synthetic()
    os.environ.clear()
    os.environ.update(held)
    try:
        import warmprobe  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    from langslice.doors.cli import warm

    warm.BOOT = BOOT
    return warm.serve_main(args)


def main() -> int:
    from langslice.doors.cli import jobcli, warm

    warm.BOOT = BOOT
    code = warm.run(sys.argv[1:])
    if code is not None:
        return code
    _synthetic()
    return jobcli.run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
