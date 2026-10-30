"""Run an agent's script in a warm child as the job's python would."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from langslice.doors.agent_cli.warm.child import (
    ExitBySignal,
    as_python,
    set_executable,
)

#: What the job environment's python is asked: its import path and prefixes.
PROBE = ("import json, site, sys\n"
         "print(json.dumps({'path': sys.path[1:] if sys.path[:1] == [''] else sys.path,"
         " 'executable': sys.executable,"
         " 'prefix': sys.prefix, 'exec_prefix': sys.exec_prefix,"
         " 'base_prefix': sys.base_prefix, 'version': list(sys.version_info),"
         " 'safe_path': bool(getattr(sys.flags, 'safe_path', False)),"
         " 'site': site.getsitepackages()}))")
#: Packages every environment brings itself, never taken for one of the agent's.
SEEDED = frozenset({"pip", "setuptools", "wheel", "_distutils_hack", "pkg_resources",
                    "distutils-precedence", "_virtualenv"})


def _own_modules(folders: list[str]) -> set[str]:
    """The top-level names the environment's own site-packages provide."""
    names: set[str] = set()
    for folder in folders:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name == "__pycache__" or name.endswith((".dist-info", ".egg-info", ".data")):
                    continue
                names.add(name)
            elif name.endswith((".py", ".so", ".pyd")):
                names.add(name.split(".", 1)[0])
    return names - SEEDED


def _cannot_run_here(info: dict[str, Any], env: str) -> str | None:
    """Why the environment's python must run the script itself: another
    Python, or a package of its own that this process loaded from LangSlice's
    environment already."""
    if list(info.get("version") or []) != list(sys.version_info):
        return "another Python version"
    if os.path.realpath(str(info.get("base_prefix"))) != os.path.realpath(sys.base_prefix):
        return "another Python"
    own = [path for path in info.get("site") or []
           if os.path.realpath(path).startswith(os.path.join(os.path.realpath(env), ""))]
    loaded = {name.split(".", 1)[0] for name in sys.modules}
    clash = _own_modules(own) & loaded
    if clash:
        return f"its own {', '.join(sorted(clash))}"
    return None


def run_script(interpreter: Path, rest: list[str]) -> int:
    """In a warm child: run the script ``rest[0]`` with arguments ``rest[1:]``
    as *interpreter* (the job environment's python) would: its import path
    (the script's folder first, then that python's own), its ``.pth`` files,
    ``sys.executable``, ``sys.argv``, ``__main__``; the exit code, a
    traceback, ``SystemExit`` and Ctrl-C as Python gives them. When that
    python must run it itself (:func:`_cannot_run_here`, a script that is
    not a file) this process becomes it."""
    import site
    import subprocess

    script = rest[0]
    env = str(Path(interpreter).parent.parent)
    try:
        probe = subprocess.run([str(interpreter), "-c", PROBE], capture_output=True,
                               text=True, timeout=120, check=True)
        info = json.loads(probe.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        info = None
    if (info is None or not os.path.isfile(script) or script.endswith(".pyc")
            or _cannot_run_here(info, env) is not None):
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(interpreter, [str(interpreter), *rest])
    first = [] if info.get("safe_path") else [os.path.dirname(os.path.realpath(script))]
    sys.path[:] = [*first, *info["path"]]
    known = site._init_pathinfo()  # type: ignore[attr-defined]
    from langslice.doors.agent_cli.environment import PARENT_PTH

    for folder in info["site"]:
        if not os.path.realpath(folder).startswith(os.path.join(os.path.realpath(env), "")):
            continue
        try:
            names = sorted(name for name in os.listdir(folder)
                           if name.endswith(".pth") and not name.startswith("."))
        except OSError:
            continue
        for name in names:
            if name != PARENT_PTH:
                site.addpackage(folder, name, known)  # type: ignore[attr-defined]
    set_executable(info["executable"])
    sys.prefix, sys.exec_prefix = info["prefix"], info["exec_prefix"]
    sys.argv = list(rest)
    path = os.path.abspath(script)
    code, signum = as_python(lambda: _run_main(path), script=path)
    if signum:
        raise ExitBySignal(signum)
    return code


def _run_main(path: str) -> None:
    """Run the file *path* as ``python FILE`` runs it: compiled with its
    absolute path, in a fresh ``__main__`` module (``__file__``, a source
    loader, no spec or package) that stands in ``sys.modules`` while it runs."""
    import builtins
    import importlib.machinery
    import types

    with open(path, "rb") as handle:
        source = handle.read()
    code = compile(source, path, "exec", dont_inherit=True)
    main = types.ModuleType("__main__")
    main.__dict__.update({
        "__file__": path, "__cached__": None, "__package__": None, "__spec__": None,
        "__loader__": importlib.machinery.SourceFileLoader("__main__", path),
        "__builtins__": builtins, "__annotations__": {},
    })
    held = sys.modules.get("__main__")
    sys.modules["__main__"] = main
    try:
        exec(code, main.__dict__)  # noqa: S102 - the agent's own script, as python runs it
    finally:
        if held is not None:
            sys.modules["__main__"] = held
