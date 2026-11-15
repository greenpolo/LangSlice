"""The in-tree engines on every platform: the GPU queue without ``fcntl``, the engine
interpreter's default path, the job paths a reply names, and a timed-out engine killed with
every process it started.

No engine environment is needed: the engines' own code is not run here.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from langslice.engines import _blocks, _environment, _gpu_queue, _process_tree
from langslice.plugin_api import ToolError

ENGINES = Path(_gpu_queue.__file__).resolve().parent


# --- the GPU queue ------------------------------------------------------------------------------


def test_every_engine_takes_the_shared_queue_and_none_imports_fcntl():
    for name in ("stalign", "emlddmm"):
        tree = ast.parse((ENGINES / name / "engine.py").read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        assert "fcntl" not in imported, name
        assert any(isinstance(node, ast.ImportFrom) and node.module == "_gpu_queue"
                   for node in ast.walk(tree)), name


def test_the_queue_loads_as_an_engine_loads_it_without_langslice(tmp_path: Path):
    lock = tmp_path / "queue" / "gpu.lock"
    code = (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    sys.modules['fcntl'] = None\n"
        f"sys.path.append({str(ENGINES)!r})\n"
        "from pathlib import Path\n"
        "from _gpu_queue import gpu_turn\n"
        f"with gpu_turn('cuda', path=Path({str(lock)!r})) as turn:\n"
        "    print(turn.waited_s >= 0.0)\n"
        "print(any(name.startswith('langslice') for name in sys.modules))\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", "False"]
    assert lock.exists()


def test_the_queue_follows_langslice_gpu_lock(tmp_path: Path):
    lock = tmp_path / "elsewhere.lock"
    code = "import langslice.engines._gpu_queue as q; print(q.GPU_LOCK)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, check=False, timeout=60,
                          env={**os.environ, "LANGSLICE_GPU_LOCK": str(lock)})
    assert done.stdout.strip() == str(lock), done.stderr


def test_a_turn_waits_for_a_filelock_holder_and_the_cpu_takes_none(tmp_path: Path):
    from filelock import FileLock

    lock = tmp_path / "gpu.lock"
    with _gpu_queue.gpu_turn("cpu", path=lock) as turn:
        assert turn.waited_s == 0.0
    assert not lock.exists()

    held = threading.Event()

    def hold() -> None:
        with FileLock(str(lock)):
            held.set()
            time.sleep(0.5)

    other = threading.Thread(target=hold)
    other.start()
    assert held.wait(10)
    with _gpu_queue.gpu_turn("cuda:0", path=lock) as turn:
        assert turn.waited_s >= 0.2
    other.join()


# --- the engine's interpreter -------------------------------------------------------------------


@pytest.mark.parametrize(("platform", "wanted"), [
    ("win32", ("python.exe",)), ("linux", ("bin", "python")), ("darwin", ("bin", "python"))])
def test_a_conda_environment_s_interpreter_is_the_platform_s(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, wanted: tuple[str, ...]):
    monkeypatch.setattr(sys, "platform", platform)
    assert _environment.interpreter_in(tmp_path) == tmp_path.joinpath(*wanted)


def test_a_venv_on_windows_is_read_from_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "Scripts" / "python.exe").touch()
    assert _environment.interpreter_in(tmp_path) == tmp_path / "Scripts" / "python.exe"
    (tmp_path / "python.exe").touch()  # a conda environment has both; its root one runs
    assert _environment.interpreter_in(tmp_path) == tmp_path / "python.exe"


def test_the_environment_is_found_by_name_and_otherwise_named_under_claude_science(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    envs = tmp_path / "envs"
    (envs / "stalign2").mkdir(parents=True)
    monkeypatch.setenv("SLICEBENCH_CONDA_ENVS", str(envs))
    monkeypatch.delenv("CONDA_ENVS_PATH", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    found = Path(_environment.env_python("stalign2"))
    assert found == _environment.interpreter_in(envs / "stalign2")
    missing = Path(_environment.env_python("no_such_env_here"))
    assert missing.is_relative_to(_environment.CLAUDE_SCIENCE_ENVS.expanduser())
    assert "Software" not in missing.parts


def test_an_unset_source_is_refused_naming_where_to_set_it():
    with pytest.raises(ToolError) as refused:
        _environment.require_path("emlddmm", "ref:tools", "emlddmm_src", "",
                                  variable="EMLDDMM_SRC")
    assert refused.value.code == "ENGINE_MISSING"
    assert "emlddmm_src is not set" in str(refused.value)
    assert "EMLDDMM_SRC" in str(refused.value)


def test_no_engine_default_names_a_machine_folder():
    from langslice.engines.emlddmm import plugin as emlddmm
    from langslice.engines.spomialign import plugin as spomialign
    from langslice.engines.stalign import plugin as stalign

    for plugin in (emlddmm, spomialign, stalign):
        for key, value in plugin.DEFAULTS.items():
            assert "Software" not in str(value), (plugin.NAME, key)
    assert emlddmm.DEFAULTS["emlddmm_src"] == ""


# --- paths in replies ---------------------------------------------------------------------------


def test_a_reply_names_a_job_path_with_forward_slashes(tmp_path: Path):
    context = SimpleNamespace(job_dir=tmp_path)
    folder = tmp_path / "plugins" / "stalign_lddmm" / "calls" / "0001_s0"
    relative = _blocks.job_path(context, folder)  # pyright: ignore[reportArgumentType]
    assert relative == "plugins/stalign_lddmm/calls/0001_s0"
    outside = tmp_path.parent / "other"
    assert _blocks.job_path(context, outside) == str(outside)  # pyright: ignore[reportArgumentType]


# --- a timed-out engine -------------------------------------------------------------------------


def test_an_engine_s_output_comes_back_as_from_subprocess_run(tmp_path: Path):
    done = _process_tree.run_engine(
        [sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err'); sys.exit(3)"],
        timeout_s=60, cwd=tmp_path)
    assert (done.returncode, done.stdout.strip(), done.stderr) == (3, "out", "err")


def test_a_timeout_kills_the_engine_and_the_process_it_started(tmp_path: Path):
    psutil = pytest.importorskip("psutil")
    grandchild_pid = tmp_path / "grandchild.pid"
    child = (
        "import subprocess, sys, time\n"
        "sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(grandchild_pid)!r}, 'w').write(str(sleeper.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(120)\n"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as expired:
        _process_tree.run_engine([sys.executable, "-c", child], timeout_s=5)
    assert time.monotonic() - started < 60  # the sleeper held the pipes: no wait for it
    assert "started" in (expired.value.output or "")
    pid = int(grandchild_pid.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(pid)
