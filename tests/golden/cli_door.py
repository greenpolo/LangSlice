"""Run the golden scenario's agent-CLI door: ``langslice-job`` calls on a job.

Every call is the installed ``langslice-job`` command (the console script
beside this interpreter) run as a process of its own, as a coding agent runs
it: the catalog (``ops``, ``schema``), ``init``, reads and writes of the
tools, a dry run, a long fit, refusals and a launch prompt. Each call's exit
code and the JSON envelope it printed on stdout are kept; stderr holds
progress lines and library notices, which are not outputs.
``tests/test_golden_outputs.py`` runs it with the job's environment (calls
cold, long verbs in the calling process) and compares the job folder.

Run it by hand as ``python tests/golden/cli_door.py IMAGES CALLS_JSON``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from stack import (
    ATLAS,
    CUTTING_ANGLES,
    FRONT,
    HAND_TRANSFORMS,
    MIDDLE,
    MISSING,
    PIXEL_SIZE_UM,
    TASKS,
    ZOOM_BOX,
    placements,
)

#: The agent CLI as installed: the console script of this environment.
COMMAND = Path(sys.executable).with_name("langslice-job")
#: The calls that name no job folder.
CATALOG = ("ops", "schema")


def commands(images: str) -> list[list[str]]:
    """Return the scenario's calls, each the arguments after the command.

    Parameters
    ----------
    images
        The image folder the job is made for.

    Returns
    -------
    list of list of str
        One argument list per call, in order.
    """
    positions = {"sections": placements(), "cutting_angles": CUTTING_ANGLES}
    transforms = {"sections": HAND_TRANSFORMS}
    box = ",".join(str(value) for value in ZOOM_BOX)
    return [
        ["ops"],
        ["schema", "look", "position_sections"],
        [
            images,
            "init",
            "--atlas",
            ATLAS,
            "--tasks",
            ",".join(TASKS),
            "--image-provider",
            "none",
            "--pixel-size-um",
            str(PIXEL_SIZE_UM),
        ],
        [images, "ops"],
        [images, "status"],
        [images, "position_sections", "--args", json.dumps(positions)],
        [images, "look", "--mode", "overlay", "--sections", FRONT],
        [images, "zoom", "--box", box],
        [
            images,
            "interactive-transform",
            "--dry-run",
            "--args",
            json.dumps(transforms),
        ],
        [images, "interactive_transform", "--args", json.dumps(transforms)],
        [images, "elastix_affine", "--sections", MIDDLE],
        [images, "grep-atlas", "--query", "CA1", "--section", MIDDLE],
        [images, "note", "--text", "Golden scenario, agent CLI."],
        [images, "undo"],
        [images, "look", "--mode", "overlay", "--sections", MISSING],
        [images, "status", "--bogus", "1"],
        [images, "launch", "--role", "register-scripting"],
        [images, "status", "--verbose"],
    ]


def call(arguments: list[str], cwd: Path) -> dict[str, object]:
    """Run one ``langslice-job`` call and keep its exit code and envelope.

    Parameters
    ----------
    arguments
        The arguments after the command.
    cwd
        The folder the call runs in.

    Returns
    -------
    dict
        ``verb`` (the command or verb, for the diff's labels),
        ``arguments``, ``exit`` and ``envelope`` (the parsed stdout, or its
        text when it is not JSON).
    """
    finished = subprocess.run(
        [str(COMMAND), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    try:
        envelope: object = json.loads(finished.stdout)
    except ValueError:
        envelope = finished.stdout
    return {
        "verb": arguments[0] if arguments[0] in CATALOG else arguments[1],
        "arguments": arguments,
        "exit": finished.returncode,
        "envelope": envelope,
    }


def main() -> None:
    """Run the calls on the first folder; write them to the second."""
    images, calls_path = Path(sys.argv[1]), Path(sys.argv[2])
    calls = [
        call(arguments, images.parent) for arguments in commands(str(images))
    ]
    calls_path.write_text(json.dumps(calls, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
