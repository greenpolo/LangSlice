"""Live shared editing: writers on one job folder never overwrite each other.

Every write holds the job folder's lock (``job.lock``) and reloads what
others saved first (``Job.writing``); a long fit computes outside the lock
and, when it applies, refuses a section whose inputs changed meanwhile
(``STALE_INPUT``) while the others apply. Here an "agent" (a second job on
the folder, through the library) writes while a CLI fit computes
(``ants_syn``, ``elastix_affine``: long verbs), and several CLI processes
write at once.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import langslice
from langslice.core import deformation
from langslice.doors.cli.jobcli import main
from tests.cli_child import install
from tests.golden.record import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    apply_patches,
    atlas_loader,
    write_sections,
)

REPO = Path(__file__).resolve().parents[1]
IDENTITY = {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0, "translate_x_mm": 0.0,
            "translate_y_mm": 0.0}


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("concurrency") / "stack"
    write_sections(images)
    return images


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(list(argv))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1, lines
    return int(code or 0), json.loads(lines[0])


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           capsys: pytest.CaptureFixture[str]) -> Path:
    """A placed stack (positions; identity transforms on s0 and s1) and its job."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    install(atlas_loader(), monkeypatch.setattr)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    code, envelope = cli(capsys, str(folder), "init", "--tasks",
                         "position,transform,nonlinear", "--image-provider", "none",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM))
    assert code == 0, envelope
    code, envelope = cli(capsys, str(folder), "position_sections", "--sections", json.dumps(
        [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15}]),
        "--view", "false")
    assert code == 0, envelope
    code, envelope = cli(capsys, str(folder), "interactive_transform", "--sections",
                         json.dumps([{"id": ID0, **IDENTITY}, {"id": ID1, **IDENTITY}]),
                         "--view", "false")
    assert code == 0, envelope
    return folder


def saved(images: Path) -> dict[str, dict[str, Any]]:
    """The state as the job folder holds it, per section."""
    data = json.loads((images / "langslice" / "state.json").read_text())
    return {item["id"]: item for item in data["slices"]}


def during(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, write: Any) -> None:
    """Run *write* (another writer's call) once, as *module.name* starts."""
    original = getattr(module, name)
    done: list[bool] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not done:
            done.append(True)
            write()
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapped)


def _needs_ants() -> None:
    from langslice.ops.deformable import ants_ready

    if not ants_ready():
        pytest.skip("ants_syn needs antspyx")


def test_an_agent_write_while_a_fit_computes_survives(capsys, images, monkeypatch):
    _needs_ants()
    agent = langslice.open_job(images)
    during(monkeypatch, deformation, "run_jobs",
           lambda: agent.position_sections(sections=[{"id": ID2, "position_mm": 0.2}],
                                           view=False))
    code, envelope = cli(capsys, str(images), "ants_syn", "--sections", ID0,
                         "--view", "false")
    assert code == 0, envelope
    assert envelope["result"]["results"][0]["written"] is True
    held = saved(images)
    assert held[ID2]["position_mm"] == 0.2          # the agent's write
    assert held[ID0]["deformation"]["key"]          # the fit's
    # The agent's next call sees the fit.
    agent.status()
    assert agent.state.resolve(ID0).deformation["key"] == held[ID0]["deformation"]["key"]
    agent.close()


def test_a_fit_whose_section_moved_meanwhile_is_refused_for_that_section(
        capsys, images, monkeypatch):
    _needs_ants()
    agent = langslice.open_job(images)
    during(monkeypatch, deformation, "run_jobs",
           lambda: agent.position_sections(sections=[{"id": ID0, "position_mm": 0.11}],
                                           view=False))
    code, envelope = cli(capsys, str(images), "ants_syn", "--sections",
                         json.dumps([ID0, ID1]), "--view", "false")
    assert code == 0, envelope
    rows = {row["id"]: row for row in envelope["result"]["results"]}
    assert rows[ID0]["error"] == "STALE_INPUT" and "again" in rows[ID0]["message"]
    assert rows[ID1]["status"] == "ok" and rows[ID1]["written"] is True
    assert f"{ID0}: STALE_INPUT" in envelope["warnings"]
    held = saved(images)
    assert held[ID0]["position_mm"] == 0.11 and not held[ID0].get("deformation")
    assert held[ID1]["deformation"]["key"]
    agent.close()


def test_an_affine_fit_whose_section_moved_meanwhile_is_refused(capsys, images, monkeypatch):
    from langslice.core import transform

    agent = langslice.open_job(images)
    during(monkeypatch, transform, "fit_elastix",
           lambda: agent.interactive_transform(sections=[{"id": ID1, **IDENTITY,
                                                          "rotation_deg": 3.0}], view=False))
    code, envelope = cli(capsys, str(images), "elastix_affine", "--sections",
                         json.dumps([ID0, ID1]), "--view", "false")
    assert code == 0, envelope
    rows = {row["id"]: row for row in envelope["result"]["results"]}
    assert rows[ID0]["status"] == "ok" and rows[ID1]["error"] == "STALE_INPUT"
    held = saved(images)
    assert held[ID0]["transform"]["kind"] == "elastix"
    assert held[ID1]["transform"]["physical"]["rotation_deg"] == pytest.approx(3.0)
    agent.close()


def test_cli_processes_writing_at_once_all_land(images):
    # Each commit waits 0.4 s inside its write: without the lock, writers
    # that read the state before another's commit would overwrite it.
    # This checkout's src first: the venv's editable install may be another's.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO / "src"), str(REPO))),
           "LANGSLICE_TEST_COMMIT_DELAY": "0.4"}

    def start(*argv: str) -> subprocess.Popen[str]:
        return subprocess.Popen([sys.executable, "-m", "tests.cli_child", str(images),
                                 *argv], cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    started = [start("position_sections", "--sections",
                     json.dumps([{"id": ID2, "position_mm": 0.2}]), "--view", "false"),
               start("position_sections", "--sections",
                     json.dumps([{"id": ID0, "position_mm": 0.12}]), "--view", "false")]
    started += [start("note", "--text", f"note {number}") for number in range(4)]
    for process in started:
        out, err = process.communicate(timeout=300)
        assert process.returncode == 0, err[-2000:]
        assert json.loads(out.strip().splitlines()[-1])["ok"] is True
    held = saved(images)
    assert held[ID2]["position_mm"] == 0.2 and held[ID0]["position_mm"] == 0.12
    notes = json.loads((images / "langslice" / "state.json").read_text())["notes"]
    assert sorted(note for note in notes if note.startswith("note ")) == [
        f"note {number}" for number in range(4)]


def test_two_stores_never_number_two_pictures_alike(tmp_path: Path, monkeypatch: Any):
    """Two view stores on one job folder (a running agent and a CLI call, two
    processes) queue pictures before either is written: each picture still
    gets its own number, so neither overwrites the other's files (review
    finding 11)."""
    import threading

    from PIL import Image

    from langslice.job import views
    from langslice.job.layout import JobLayout

    layout = JobLayout(tmp_path / "job")
    layout.ensure()
    release = threading.Event()
    real = views.ViewStore._write_call

    def held(self: Any, call: Any) -> None:
        release.wait(10)
        real(self, call)

    monkeypatch.setattr(views.ViewStore, "_write_call", held)
    first, second = views.ViewStore(layout), views.ViewStore(layout)
    picture = Image.new("RGB", (8, 8), "red")
    names = (first.save(tool="look", pictures=[(picture, None)])
             + second.save(tool="look", pictures=[(picture, None), (picture, None)])
             + first.save(tool="zoom", pictures=[(picture, None)]))
    release.set()
    first.flush()
    second.flush()
    seqs = [int(name.split("_")[0]) for name in names]
    assert len(set(seqs)) == len(seqs) == 4, names
    entries = [json.loads(line) for line in layout.views_index.read_text().splitlines()]
    assert sorted(entry["seq"] for entry in entries) == sorted(seqs)
    assert len({entry["path"] for entry in entries}) == 4
    # A reopened store continues after every number handed out.
    third = views.ViewStore(layout)
    assert int(third.save(tool="note", pictures=[(picture, None)])[0][:6]) == max(seqs) + 1
    third.flush()
