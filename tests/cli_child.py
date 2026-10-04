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


def install(loader: Any, put: Any = setattr) -> None:
    """Every job the CLI and the library open reads *loader*'s atlas, and a
    background run starts this module, not the real CLI (whose BrainGlobe
    atlas would be downloaded). *put* sets each attribute (a test passes
    ``monkeypatch.setattr``, so both are undone after it)."""
    import langslice.doors.jobs as jobs
    from langslice.doors.cli import background

    original = jobs.context

    def context(*args: Any, **kwargs: Any) -> Any:
        kwargs["atlas_loader"] = loader
        return original(*args, **kwargs)

    put(jobs, "context", context)
    put(background, "CHILD_COMMAND", [sys.executable, "-m", "tests.cli_child"])


#: Seconds every commit waits before writing (tests widen the window in
#: which two processes writing at once would overwrite each other).
COMMIT_DELAY_ENV = "LANGSLICE_TEST_COMMIT_DELAY"


def main() -> int:
    import os
    import time

    from tests.golden.record import apply_patches, atlas_loader

    apply_patches()
    install(atlas_loader())
    delay = float(os.environ.get(COMMIT_DELAY_ENV) or 0)
    if delay:
        from langslice.linear.job import Job

        commit = Job.commit

        def slow(self: Any, before: Any) -> None:
            time.sleep(delay)
            commit(self, before)

        Job.commit = slow  # type: ignore[method-assign]
    from langslice.cli import main as cli_main

    return int(cli_main(sys.argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
