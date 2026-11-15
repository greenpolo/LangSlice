"""On Windows, ``langslice-job FOLDER env``, run as the installed command,
makes the job's environment (``Scripts\\python.exe``, ``Lib\\site-packages``)
and that environment's python imports LangSlice through its ``.pth`` file
alone. Offline: the environment is made from this Python (uv's or venv's)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import langslice
from langslice.doors.agent_cli.main import main
from tests.cli_child import install
from tests.synthetic_stack import PIXEL_SIZE_UM, atlas_loader, write_sections

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows' job environment")


def _command() -> str | None:
    beside = Path(sys.executable).parent / "langslice-job.exe"
    return str(beside) if beside.is_file() else shutil.which("langslice-job")


def test_the_installed_command_makes_an_environment_that_imports_langslice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    command = _command()
    if command is None:
        pytest.skip("langslice-job is not installed beside this Python")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    install(atlas_loader(), monkeypatch.setattr)
    folder = tmp_path / "stack"
    write_sections(folder)
    code = main([str(folder), "init", "--tasks", "position,transform",
                 "--nonlinear-provider", "none", "--preprocess", "none",
                 "--pixel-size-um", str(PIXEL_SIZE_UM)])
    assert code == 0, capsys.readouterr().out
    job = Path(json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]["job_folder"])

    env = dict(os.environ)
    # The command runs the langslice under test.
    source = str(Path(langslice.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [env.get("PYTHONPATH")])])
    done = subprocess.run([command, str(job), "env"], capture_output=True, text=True,
                          encoding="utf-8", stdin=subprocess.DEVNULL, env=env, timeout=600,
                          check=False)
    envelope = json.loads(done.stdout.strip().splitlines()[-1])
    assert done.returncode == 0 and envelope["ok"], (envelope, done.stderr[-2000:])
    python = Path(envelope["result"]["python"])
    assert python == job / "env" / "Scripts" / "python.exe" and python.is_file()

    # Without PYTHONPATH, only the environment's .pth file can make it import LangSlice.
    env.pop("PYTHONPATH")
    probe = subprocess.run(
        [str(python), "-c", "import sys, langslice, numpy; print(sys.prefix)"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env, timeout=300,
        check=False)
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert Path(probe.stdout.strip()) == job / "env"
