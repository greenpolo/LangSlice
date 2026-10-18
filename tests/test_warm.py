"""The agent CLI's warm process (``langslice.doors.cli.warm``): a call it
answers is the cold call, byte for byte.

Every call here is a process of its own (``python -m tests.warm_helper``,
``langslice-job`` on the synthetic atlas), as an agent's shell makes it. The
same commands run cold on one copy of a job and warm on another; their
stdout, stderr, exit codes and the job files they write must match. Then:
a changed source file stops the server (that call runs cold and a new
server starts), a killed client takes its call with it, calls run in
parallel, Ctrl-C reaches the call, and ``python SCRIPT`` runs the agent's
script as the job environment's python would.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from langslice.doors.cli.jobcli import main
from tests.cli_child import install
from tests.synthetic_stack import ID0, ID1, ID2, PIXEL_SIZE_UM, atlas_loader, write_sections

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="the warm process is Linux only")

SCRIPT = """\
import json, os, sys
import langslice
print(json.dumps({"argv": sys.argv, "cwd": os.getcwd(), "path0": sys.path[0],
                  "file": __file__, "name": __name__, "exe": sys.executable,
                  "prefix": sys.prefix}))
from tests.synthetic_stack import atlas_loader
job = langslice.open_job(sys.argv[1], atlas_loader=atlas_loader())
print(sorted(row["id"] for row in job.status()["rows"]))
mode = sys.argv[2] if len(sys.argv) > 2 else ""
if mode == "raise":
    raise ValueError("boom")
if mode == "exit":
    sys.exit(int(sys.argv[3]))
if mode == "message":
    sys.exit("a message")
if mode == "lsdemo":
    import lsdemo
    print("lsdemo", lsdemo.VALUE)
"""
SLEEPER = """\
import os, sys, time
open(sys.argv[1], "w").write(str(os.getpid()))
try:
    time.sleep(float(sys.argv[2]))
except KeyboardInterrupt:
    print("interrupted", flush=True)
    raise
print("slept", flush=True)
"""


# --- the jobs and the processes ----------------------------------------------------------


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two copies of one placed job (cold, warm), a short private runtime
    folder for the sockets, a home, a module the server loads, scripts."""
    root = tmp_path_factory.mktemp("warm")
    images = root / "stack"
    write_sections(images)
    home = root / "home"
    home.mkdir()
    probe = root / "probe"
    probe.mkdir()
    (probe / "warmprobe.py").write_text("VALUE = 1\n")
    runtime = Path(tempfile.mkdtemp(prefix="lsw", dir="/tmp"))
    os.chmod(runtime, 0o700)
    held = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    mp = pytest.MonkeyPatch()
    install(atlas_loader(), mp.setattr)
    try:
        assert main([str(images), "init", "--tasks", "position,transform,nonlinear",
                     "--image-provider", "none", "--preprocess", "none",
                     "--pixel-size-um", str(PIXEL_SIZE_UM)]) == 0
        assert main([str(images), "position_sections", "--sections", json.dumps(
            [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
             {"id": ID2, "position_mm": 0.2}]), "--view", "false"]) == 0
    finally:
        mp.undo()
        if held is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = held
    folders = {}
    for name in ("cold", "warm"):
        folders[name] = root / name / "stack"
        shutil.copytree(images, folders[name])
        text = (folders[name] / "langslice" / "job.json").read_text()
        (folders[name] / "langslice" / "job.json").write_text(
            text.replace(str(images), str(folders[name])))
        scripts = folders[name] / "langslice" / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "probe.py").write_text(SCRIPT)
        (scripts / "sleep.py").write_text(SLEEPER)
    yield {**folders, "home": home, "probe": probe, "runtime": runtime, "root": root}
    stop_servers(root)
    shutil.rmtree(runtime, ignore_errors=True)


def environment(place: dict[str, Path], **extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTEST", "LANGSLICE_WARM"))}
    env.update(HOME=str(place["home"]), XDG_RUNTIME_DIR=str(place["runtime"]),
               PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "src"), str(place["probe"])]),
               LANGSLICE_ATLAS_CACHE="off", **extra)
    return env


def command(*argv: str) -> list[str]:
    return [sys.executable, "-m", "tests.warm_helper", *argv]


def run(place: dict[str, Path], *argv: str, warm: bool = True, cwd: Path | None = None,
        timeout: float = 300) -> subprocess.CompletedProcess[str]:
    env = environment(place, **({} if warm else {"LANGSLICE_WARM": "0"}))
    return subprocess.run(command(*argv), capture_output=True, text=True, env=env,
                          cwd=str(cwd or REPO), timeout=timeout, check=False)


def sockets(place: dict[str, Path]) -> list[Path]:
    return sorted((place["runtime"] / "langslice-warm").glob("*.sock"))


def logs(place: dict[str, Path]) -> str:
    return "".join(path.read_text() for path in
                   sorted((place["runtime"] / "langslice-warm").glob("*.log")))


def calls(folder: Path) -> list[dict[str, Any]]:
    path = folder / "langslice" / "logs" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wait_warm(place: dict[str, Path], timeout: float = 240) -> None:
    """Until the warm copy's server answers (a call it served logs ``warm``)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sockets(place):
            done = run(place, str(place["warm"]), "status")
            assert done.returncode == 0, done.stderr
            if calls(place["warm"])[-1].get("warm"):
                return
        time.sleep(0.5)
    raise AssertionError(f"no warm process came up:\n{logs(place)}")


def stop_servers(root: Path) -> None:
    """End the servers of the jobs under *root* (they would idle for half an hour)."""
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if "tests.warm_helper import serve" in args and str(root) in args:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.fixture(scope="module")
def warm(place: dict[str, Path]) -> dict[str, Path]:
    """The warm copy's server, up."""
    first = run(place, str(place["warm"]), "status")
    assert first.returncode == 0, first.stderr
    wait_warm(place)
    return place


def same(place: dict[str, Path], *argv: str, cwd_name: str | None = None) -> tuple[
        subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    """Run ``FOLDER *argv`` cold on the cold copy and warm on the warm one:
    the same exit code, stdout and stderr (the folders' names aside)."""
    results = []
    for name in ("cold", "warm"):
        folder = place[name]
        args = [str(folder) if arg == "{folder}" else arg for arg in argv]
        cwd = folder / "langslice" if cwd_name == "job" else None
        results.append(run(place, str(folder), *args, warm=name == "warm", cwd=cwd))
    cold, warm = results
    for stream in ("stdout", "stderr"):
        texts = [getattr(result, stream).replace(str(place[name]), "<FOLDER>")
                 for name, result in zip(("cold", "warm"), results, strict=True)]
        assert texts[0] == texts[1], (argv, stream, texts)
    assert cold.returncode == warm.returncode, (argv, cold.returncode, warm.returncode)
    assert calls(place["warm"])[-1].get("warm") is True, (argv, logs(place))
    return cold, warm


def files_alike(place: dict[str, Path]) -> None:
    for name in ("state.json", "registration.json"):
        held = [json.loads((place[copy] / "langslice" / name).read_text()
                           .replace(str(place[copy]), "<FOLDER>")) for copy in ("cold", "warm")]
        for data in held:
            data.pop("source", None)  # the code's checkout, the same here
        assert held[0] == held[1], name


# --- a warm call is the cold call ----------------------------------------------------------


def test_warm_calls_answer_as_cold_ones(warm):
    place = warm
    cold, hot = same(place, "status")
    assert hot.returncode == 0 and json.loads(hot.stdout)["ok"] is True
    cold, hot = same(place, "look", "--mode", "overlay", "--sections", ID0)
    assert hot.returncode == 0
    for artifact in json.loads(hot.stdout)["artifacts"]:  # the pictures, byte for byte
        mine = Path(artifact["path"])
        theirs = Path(str(mine).replace(str(place["warm"]), str(place["cold"])))
        assert mine.read_bytes() == theirs.read_bytes()
    same(place, "interactive_transform", "--sections",
         json.dumps([{"id": ID0, "rotation_deg": 3.0, "scale_x": 1.05}]))
    same(place, "position_sections", "--dry-run", "--sections",
         json.dumps([{"id": ID1, "position_mm": 0.12}]))
    _cold, refused = same(place, "position_sections", "--sections",
                          json.dumps([{"id": "nope.png", "position_mm": 0.1}]))
    assert refused.returncode != 0 and json.loads(refused.stdout)["ok"] is False
    _cold, bad = same(place, "align_everything")
    assert bad.returncode == 2 and json.loads(bad.stdout)["error"]["code"] == "UNKNOWN_VERB"
    same(place, "undo")
    files_alike(place)


def test_python_runs_the_script_as_the_job_environments_python(warm):
    place = warm
    for mode in ([], ["raise"], ["exit", "3"], ["message"]):
        cold, hot = same(place, "python", "scripts/probe.py", "{folder}", *mode, cwd_name="job")
        shown = json.loads(hot.stdout.splitlines()[0])
        assert shown["argv"][0] == "scripts/probe.py" and shown["name"] == "__main__"
        assert shown["file"].endswith("/langslice/scripts/probe.py")
        assert shown["path0"].endswith("/langslice/scripts")
        assert shown["exe"].endswith("/langslice/env/bin/python")
    assert hot.returncode == 1 and hot.stderr == "a message\n"
    _cold, raised = same(place, "python", "scripts/probe.py", "{folder}", "raise",
                         cwd_name="job")
    assert raised.returncode == 1
    assert raised.stderr.startswith("Traceback (most recent call last):\n  File \"")
    assert raised.stderr.endswith("ValueError: boom\n") and "warm.py" not in raised.stderr
    _cold, three = same(place, "python", "scripts/probe.py", "{folder}", "exit", "3",
                        cwd_name="job")
    assert three.returncode == 3
    # A package of the job's own environment imports as it would there.
    from tests.test_job_env_and_transcripts import wheel

    for copy in ("cold", "warm"):
        done = run(place, str(place[copy]), "env", "install", str(wheel(place["root"])),
                   warm=False)
        assert done.returncode == 0, done.stdout
    _cold, demo = same(place, "python", "scripts/probe.py", "{folder}", "lsdemo",
                       cwd_name="job")
    assert demo.stdout.splitlines()[-1] == "lsdemo 42"


def test_calls_run_in_parallel_and_all_land(warm):
    place = warm
    started = time.monotonic()
    env = environment(place)
    folder = place["warm"]
    jobs = [subprocess.Popen(command(str(folder), "python", "scripts/sleep.py",
                                     str(place["root"] / f"pid{index}"), "2"),
                             cwd=str(folder / "langslice"), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True) for index in range(3)]
    writes = [subprocess.Popen(command(str(folder), "note", "--text", f"note {index}"),
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True) for index in range(3)]
    for process in jobs + writes:
        out, err = process.communicate(timeout=120)
        assert process.returncode == 0, (out, err)
    assert time.monotonic() - started < 5.5  # three 2 s sleeps side by side
    notes = json.loads((folder / "langslice" / "state.json").read_text()).get("notes") or []
    texts = json.dumps(notes)
    assert all(f"note {index}" in texts for index in range(3))


def _pid_of(path: Path, timeout: float = 60) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"no pid in {path}")


def _gone(pid: int, timeout: float = 15) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_killed_client_takes_its_call_along(warm):
    place = warm
    folder = place["warm"]
    marker = place["root"] / "killed.pid"
    client = subprocess.Popen(command(str(folder), "python", "scripts/sleep.py", str(marker),
                                      "120"), cwd=str(folder / "langslice"),
                              env=environment(place), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    pid = _pid_of(marker)
    assert pid != client.pid  # the script runs in the warm process's child
    client.kill()
    client.wait(timeout=30)
    assert _gone(pid), "the call outlived its client"


def test_ctrl_c_reaches_the_call(warm):
    place = warm
    results = {}
    for name in ("cold", "warm"):
        folder = place[name]
        marker = place["root"] / f"interrupted-{name}.pid"
        client = subprocess.Popen(
            command(str(folder), "python", "scripts/sleep.py", str(marker), "120"),
            cwd=str(folder / "langslice"),
            env=environment(place, **({} if name == "warm" else {"LANGSLICE_WARM": "0"})),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _pid_of(marker)
        client.send_signal(signal.SIGINT)
        out, err = client.communicate(timeout=60)
        results[name] = (client.returncode, out, err.splitlines()[-1] if err else "")
    assert results["warm"] == results["cold"]
    assert results["warm"][0] == -signal.SIGINT and results["warm"][1] == "interrupted\n"


def test_a_changed_source_file_stops_the_server(warm):
    """Last: it replaces the module's server."""
    place = warm
    (place["probe"] / "warmprobe.py").write_text("VALUE = 2  # changed\n")
    done = run(place, str(place["warm"]), "status")
    assert done.returncode == 0 and json.loads(done.stdout)["ok"] is True
    assert calls(place["warm"])[-1].get("warm") is False  # this call ran cold
    deadline = time.monotonic() + 30
    while "warmprobe.py changed" not in logs(place) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert "warmprobe.py changed" in logs(place)
    wait_warm(place)  # a new server, started by that call
    assert logs(place).count("warm in ") >= 2
