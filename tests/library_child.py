"""One library call on the synthetic atlas, in a process of its own.

``python -m tests.library_child FOLDER VERB JSON`` opens the job as a script
does (``langslice.open_job``), calls the verb with the JSON object's
arguments and prints the reply as one JSON line: a second writer on a job
folder for the tests of shared editing (``tests/cli_child.py`` prepares the
process).
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    from tests.cli_child import prepare

    prepare()
    import langslice

    folder, verb, arguments = sys.argv[1:4]
    with langslice.open_job(folder) as job:
        reply = getattr(job, verb)(**json.loads(arguments))
    print(json.dumps(reply, default=str))
    return 0 if reply.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
