"""Find an engine's interpreter, and refuse a call when it is missing.

An engine run as a subprocess (STalign, emlddmm) runs under the Python of
its own conda environment (``stalign2``, ``rigid_emlddmm``). A plugin's
``python`` option names that interpreter; without one, the environment is
looked up by name in the folders conda keeps environments in, and the
interpreter is the one inside it for this platform. An option the engine
needs that is unset or names a path that does not exist is refused at call
time with ``ENGINE_MISSING``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langslice.plugin_api import RefusalCode, ToolError

#: Where Claude Science makes conda environments: the folder searched last,
#: and the one an interpreter is named under when no folder holds it.
CLAUDE_SCIENCE_ENVS = Path("~/.claude-science/conda/envs")


def env_folders() -> list[Path]:
    """List the folders that may hold conda environments, in search order.

    Returns
    -------
    list of Path
        ``SLICEBENCH_CONDA_ENVS`` (where SliceBench keeps the engines'
        environments), each folder of conda's ``CONDA_ENVS_PATH``, the
        ``envs`` folder of the conda that ``CONDA_EXE`` names, the user's
        ``~/.conda/envs``, then Claude Science's.
    """
    named = [os.environ.get("SLICEBENCH_CONDA_ENVS", "")]
    named += os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep)
    folders = [Path(folder).expanduser() for folder in named if folder]

    conda = os.environ.get("CONDA_EXE")
    if conda:
        folders.append(Path(conda).parent.parent / "envs")
    folders.append(Path.home() / ".conda" / "envs")
    folders.append(CLAUDE_SCIENCE_ENVS.expanduser())
    return folders


def interpreter_in(env_folder: Path) -> Path:
    """Name the Python interpreter of one environment on this platform.

    Parameters
    ----------
    env_folder
        The environment's root folder, a conda environment or a venv.

    Returns
    -------
    Path
        On Windows ``python.exe`` at the root (a conda environment), else
        the ``Scripts`` folder's ``python.exe`` when only that exists (a
        venv); elsewhere ``bin/python``.
    """
    if sys.platform != "win32":
        return env_folder / "bin" / "python"
    venv_python = env_folder / "Scripts" / "python.exe"
    conda_python = env_folder / "python.exe"
    if venv_python.exists() and not conda_python.exists():
        return venv_python
    return conda_python


def env_python(env_name: str) -> str:
    """Name the interpreter of the conda environment ``env_name``.

    Parameters
    ----------
    env_name
        The environment's name, such as ``stalign2``.

    Returns
    -------
    str
        The interpreter inside the first folder of :func:`env_folders` that
        holds the environment; when none does, the one it would have under
        Claude Science's folder, which the call then refuses as missing.
    """
    folders = env_folders()
    holders = [folder for folder in folders if (folder / env_name).is_dir()]
    if holders:
        return str(interpreter_in(holders[0] / env_name))
    return str(interpreter_in(folders[-1] / env_name))


def require_path(
    engine: str, ref: str, key: str, value: str, *, variable: str = ""
) -> None:
    """Refuse a call whose engine needs a path that is unset or missing.

    Parameters
    ----------
    engine
        The engine's name, as the refusal says it.
    ref
        The plugin's reference, whose ``plugin_config`` sets the option.
    key
        The option that holds the path, such as ``python`` or ``src``.
    value
        The option's value; empty when nothing set it.
    variable
        The environment variable that also sets the option, named in the
        refusal; empty when there is none.

    Raises
    ------
    ToolError
        ``ENGINE_MISSING``, naming the option and where to set it.
    """
    where = f"the plugin_config of {ref}"
    if variable:
        where += f" or the environment variable {variable}"
    if not value:
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"{engine} cannot run here: {key} is not set (set it in {where})",
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise ToolError(
            RefusalCode.ENGINE_MISSING,
            f"{engine} cannot run here: {key} {str(path)!r} does not exist "
            f"(set it in {where})",
        )
