"""``langslice-job FOLDER env``: the job's own Python environment.

A coding agent that wants a package LangSlice does not ship installs it here,
never into LangSlice's environment: ``env/`` in the job folder, a virtual
environment that also sees every package of the environment LangSlice runs in
(a ``.pth`` file adds that environment's site-packages, so ``import
langslice`` works in the agent's scripts), with its own packages first.
Installing goes through ``env install`` (uv when it is on the PATH, else the
environment's pip); uv links each package from its cache, so the same package
in many jobs is stored once.

The packages installed are listed in ``scripts/requirements.txt`` after every
install, and ``env remove`` deletes ``env/`` and keeps that list, from which
the next ``env`` (or ``env install``) rebuilds it. The harness plugins remove
it when the agent's session ends (:mod:`langslice.doors.cli.transcripts`).

Subcommands: ``env`` (make it when missing; its paths and packages),
``env install PACKAGE [PACKAGE ...]`` (pip requirement strings) and
``env remove``.

``langslice-job FOLDER python SCRIPT [ARGS ...]`` (:func:`python`) runs the
agent's script as the environment's python would (made first when missing):
in the job's warm process when one answers (:mod:`langslice.doors.cli.warm`,
with that python's import path), else that python itself.
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
from email.parser import HeaderParser
from pathlib import Path
from typing import Any

from langslice.doors.cli.envelope import Envelope
from langslice.job.layout import ENV_DIR, JobLayout

#: The list of the packages installed in the job's environment.
REQUIREMENTS = Path("scripts") / "requirements.txt"
#: The ``.pth`` file that lets the job's environment import LangSlice's packages.
PARENT_PTH = "_langslice_parent.pth"
#: Packages a new environment brings itself, never listed.
SEEDED = frozenset({"pip", "setuptools", "wheel"})
#: How long one install may take.
INSTALL_TIMEOUT_S = 1800


def command(layout: JobLayout, positional: list[str]) -> Envelope:
    """``env``, ``env install PACKAGE ...`` or ``env remove`` on the job in *layout*."""
    action, packages = (positional[0], positional[1:]) if positional else ("", [])
    if action == "remove" and not packages:
        removed = remove(layout)
        return Envelope(result={"removed": removed,
                                "requirements": str(layout.folder / REQUIREMENTS)})
    if action not in ("", "install") or (action == "install" and not packages):
        return Envelope.failure(
            "BAD_ARGUMENTS", "env takes nothing, `install PACKAGE [PACKAGE ...]` or `remove`.")
    try:
        built = ensure(layout)
        if packages:
            install(layout, packages)
    except (OSError, subprocess.SubprocessError) as exc:
        return Envelope.failure("ENV_FAILED", str(exc))
    return Envelope(result={**describe(layout), "rebuilt_from_requirements": built})


def env_dir(layout: JobLayout) -> Path:
    return layout.folder / ENV_DIR


def python_of(folder: Path) -> Path:
    """The interpreter of the environment in *folder*."""
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


#: Held while the environment is made (``logs/env.lock``): calls started
#: together on a job without one make it once.
ENV_LOCK = "env.lock"
#: Written in ``env/`` once it is made, its packages installed.
READY = ".langslice-ready"


def ensure(layout: JobLayout) -> bool:
    """Make the job's environment when it is missing (once, under
    :data:`ENV_LOCK`, however many calls ask at once); True when it was
    rebuilt from ``scripts/requirements.txt``."""
    from filelock import FileLock

    if (env_dir(layout) / READY).exists():
        return False
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(layout.logs_dir / ENV_LOCK), timeout=INSTALL_TIMEOUT_S):
        built = _make(layout)
        (env_dir(layout) / READY).touch()
        return built


def _make(layout: JobLayout) -> bool:
    folder = env_dir(layout)
    if python_of(folder).exists():
        return False
    uv = shutil.which("uv")
    command = ([uv, "venv", "--quiet", "--python", sys.executable, str(folder)] if uv
               else [sys.executable, "-m", "venv", str(folder)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    (_site_packages(folder) / PARENT_PTH).write_text(
        "".join(f"import site; site.addsitedir({str(path)!r})\n" for path in _parent_sites()),
        encoding="utf-8")
    requirements = layout.folder / REQUIREMENTS
    if requirements.is_file() and requirements.read_text(encoding="utf-8").strip():
        _pip(folder, ["-r", str(requirements)])
        return True
    return False


def install(layout: JobLayout, packages: list[str]) -> None:
    """Install *packages* into the job's environment and list what it holds."""
    _pip(env_dir(layout), list(packages))
    write_requirements(layout)


def remove(layout: JobLayout) -> bool:
    """Delete the job's environment, its package list kept; False when there
    was none."""
    folder = env_dir(layout)
    if not folder.exists():
        return False
    write_requirements(layout)
    shutil.rmtree(folder)
    return True


def installed(folder: Path) -> list[tuple[str, str]]:
    """The packages installed in the environment in *folder* itself (not the
    ones it sees through LangSlice's), as (name, version)."""
    try:
        site_dir = _site_packages(folder)
    except OSError:
        return []
    found = []
    for meta in sorted(site_dir.glob("*.dist-info/METADATA")):
        headers = HeaderParser().parsestr(meta.read_text(encoding="utf-8", errors="replace"))
        name, version = headers.get("Name"), headers.get("Version")
        if name and version and name.lower() not in SEEDED:
            found.append((name, version))
    return found


def write_requirements(layout: JobLayout) -> None:
    """``scripts/requirements.txt``: the environment's own packages, pinned."""
    packages = installed(env_dir(layout))
    if not packages:
        return
    path = layout.folder / REQUIREMENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}=={version}\n" for name, version in packages),
                    encoding="utf-8")


def describe(layout: JobLayout) -> dict[str, Any]:
    folder = env_dir(layout)
    python = python_of(folder)
    return {
        "python": str(python),
        "run": f"langslice-job {layout.folder} python scripts/<script>.py [ARGS ...]",
        "install": f"langslice-job {layout.folder} env install PACKAGE [PACKAGE ...]",
        "packages": [f"{name}=={version}" for name, version in installed(folder)],
        "requirements": str(layout.folder / REQUIREMENTS),
    }


def _pip(folder: Path, arguments: list[str]) -> None:
    python = python_of(folder)
    uv = shutil.which("uv")
    command = ([uv, "pip", "install", "--quiet", "--python", str(python), *arguments] if uv
               else [str(python), "-m", "pip", "install", "--quiet", *arguments])
    done = subprocess.run(command, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
    if done.returncode != 0:
        raise subprocess.SubprocessError(
            (done.stderr or done.stdout).strip()[-4000:] or "the install failed")


def _site_packages(folder: Path) -> Path:
    if os.name == "nt":
        return folder / "Lib" / "site-packages"
    found = sorted((folder / "lib").glob("python*/site-packages"))
    if not found:
        raise OSError(f"No site-packages in {folder}")
    return found[0]


def _parent_sites() -> list[str]:
    """The site-packages of the environment LangSlice runs in."""
    return [path for path in site.getsitepackages() if Path(path).is_dir()]


#: ``python``'s usage, in its refusals.
PYTHON_USAGE = "langslice-job FOLDER python SCRIPT [ARGS ...]"


def python(folder: str, rest: list[str]) -> int:
    """``langslice-job FOLDER python SCRIPT [ARGS ...]``: *rest* (the script
    and its arguments, verbatim) run by the job's environment's python, the
    environment made first when missing (:func:`ensure`). Its stdout, stderr
    and exit code are the script's, as ``env/bin/python SCRIPT ARGS`` gives
    them; in the job's warm process (:func:`langslice.doors.cli.warm.run_script`)
    the script runs in this process with that python's import path, else this
    process becomes that python. A call that cannot start the script prints
    the envelope of its refusal (``NO_JOB``, ``BAD_ARGUMENTS``,
    ``ENV_FAILED``) and returns its exit code."""
    from langslice.doors.cli import warm
    from langslice.doors.cli.envelope import emit
    from langslice.doors.jobs import NoJob, find
    from langslice.doors.trace import log_call

    try:
        layout = JobLayout(find(folder))
    except NoJob as exc:
        return emit(Envelope.failure("NO_JOB", str(exc)))
    if not rest or rest[0].startswith("-"):
        return emit(Envelope.failure(
            "BAD_ARGUMENTS", f"python takes a script and its arguments: {PYTHON_USAGE} "
            "(no interpreter options).", verb="python"))
    try:
        ensure(layout)
    except (OSError, subprocess.SubprocessError) as exc:
        return emit(Envelope.failure("ENV_FAILED", str(exc), verb="python"))
    log_call(layout.logs_dir, {"verb": "python", "arguments": list(rest),
                               "warm": warm.in_child()})
    interpreter = python_of(env_dir(layout))
    if warm.in_child():
        return warm.run_script(interpreter, rest)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":  # no exec there: the script's exit code is passed on
        return subprocess.call([str(interpreter), *rest])
    os.execv(interpreter, [str(interpreter), *rest])
