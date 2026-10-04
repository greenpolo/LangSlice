"""The agent CLI on the synthetic atlas, in a process of its own.

``python -m tests.cli_child job FOLDER VERB ...`` is ``langslice job FOLDER
VERB ...`` with the golden recorder's synthetic atlas and fit settings
(``tests/golden/record.py``): what a background run starts in the tests
(``langslice.doors.cli.background.CHILD_COMMAND``), where BrainGlobe's
atlases are not available.
"""

from __future__ import annotations

import sys
from typing import Any


def install(loader: Any) -> None:
    """Every job the CLI and the library open reads *loader*'s atlas."""
    import langslice.doors.jobs as jobs

    original = jobs.context

    def context(*args: Any, **kwargs: Any) -> Any:
        kwargs["atlas_loader"] = loader
        return original(*args, **kwargs)

    jobs.context = context  # type: ignore[assignment]


def main() -> int:
    from tests.golden.record import apply_patches, atlas_loader

    apply_patches()
    install(atlas_loader())
    from langslice.cli import main as cli_main

    return int(cli_main(sys.argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
