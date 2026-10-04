"""Live shared editing: writers on one job folder never overwrite each other.

Every write holds the job folder's lock (``job.lock``) and reloads what
others saved first (``Job.writing``); a long fit computes outside the lock
and, when it applies, refuses a section whose inputs changed meanwhile
(``STALE_INPUT``) while the others apply. Here an "agent" (a second job on
the folder, through the library) writes while a CLI fit computes, and
several CLI processes write at once.
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
from langslice.cli import main
from langslice.linear import deformation
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
    code, envelope = cli(capsys, "job", str(folder), "init", "--tasks",
                         "position,transform,nonlinear", "--image-provider", "none",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM))
    assert code == 0, envelope
    code, envelope = cli(capsys, "job", str(folder), "set_positions", "--entries", json.dumps(
        [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15}]))
    assert code == 0, envelope
    code, envelope = cli(capsys, "job", str(folder), "adjust_transforms", "--entries",
                         json.dumps([{"id": ID0, **IDENTITY}, {"id": ID1, **IDENTITY}]))
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


def test_an_agent_write_while_a_fit_computes_survives(capsys, images, monkeypatch):
    agent = langslice.open_job(images)
    during(monkeypatch, deformation, "run_jobs",
           lambda: agent.set_positions(entries=[{"id": ID2, "position_mm": 0.2}]))
    code, envelope = cli(capsys, "job", str(images), "fit_deformable", "--slices", ID0)
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
    agent = langslice.open_job(images)
    during(monkeypatch, deformation, "run_jobs",
           lambda: agent.set_positions(entries=[{"id": ID0, "position_mm": 0.11}]))
    code, envelope = cli(capsys, "job", str(images), "fit_deformable", "--slices",
                         json.dumps([ID0, ID1]))
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
    from langslice.linear import transform

    agent = langslice.open_job(images)
    during(monkeypatch, transform, "fit_silhouette",
           lambda: agent.adjust_transforms(entries=[{"id": ID1, **IDENTITY,
                                                     "rotation_deg": 3.0}]))
    code, envelope = cli(capsys, "job", str(images), "fit_affine", "--slices",
                         json.dumps([ID0, ID1]), "--method", "silhouette")
    assert code == 0, envelope
    rows = {row["id"]: row for row in envelope["result"]["results"]}
    assert rows[ID0]["status"] == "ok" and rows[ID1]["error"] == "STALE_INPUT"
    held = saved(images)
    assert held[ID0]["transform"]["kind"] == "silhouette"
    assert held[ID1]["transform"]["physical"]["rotation_deg"] == pytest.approx(3.0)
    agent.close()


def test_cli_processes_writing_at_once_all_land(images):
    # Each commit waits 0.4 s inside its write: without the lock, writers
    # that read the state before another's commit would overwrite it.
    env = {**os.environ, "PYTHONPATH": str(REPO), "LANGSLICE_TEST_COMMIT_DELAY": "0.4"}

    def start(*argv: str) -> subprocess.Popen[str]:
        return subprocess.Popen([sys.executable, "-m", "tests.cli_child", "job", str(images),
                                 *argv], cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    started = [start("set_positions", "--entries", json.dumps([{"id": ID2,
                                                                "position_mm": 0.2}])),
               start("set_positions", "--entries", json.dumps([{"id": ID0,
                                                                "position_mm": 0.12}]))]
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
