"""Opening the GUI (``langslice.hosts.gui.window``): the browser found on each
platform, the app window, both fallbacks, a port in use, and the address
written at once."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from langslice.hosts.gui import window

REPO = Path(__file__).resolve().parents[1]


def _nothing_on_path(name: str) -> str | None:
    return None


def test_the_path_comes_first() -> None:
    found = window.find_browser(platform="linux", env={}, home=Path("/home/u"),
                                which=lambda n: "/usr/bin/chromium" if n == "chromium"
                                else None, is_file=lambda p: True)
    assert found == "/usr/bin/chromium"


def test_windows_install_folders() -> None:
    env = {"PROGRAMFILES": r"C:\Program Files", "LOCALAPPDATA": r"C:\Users\u\AppData\Local"}
    paths = window.candidates("win32", env, Path("/"))
    assert Path(env["PROGRAMFILES"]) / "Google/Chrome/Application/chrome.exe" in paths
    assert Path(env["LOCALAPPDATA"]) / "Microsoft/Edge/Application/msedge.exe" in paths
    edge = Path(env["LOCALAPPDATA"]) / "Microsoft/Edge/Application/msedge.exe"
    found = window.find_browser(platform="win32", env=env, home=Path("/"),
                                which=_nothing_on_path, is_file=lambda p: p == edge)
    assert found == str(edge)


def test_mac_applications() -> None:
    home = Path("/Users/u")
    brave = home / "Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    paths = window.candidates("darwin", {}, home)
    assert Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome") in paths
    assert brave in paths
    assert window.find_browser(platform="darwin", env={}, home=home, which=_nothing_on_path,
                               is_file=lambda p: p == brave) == str(brave)


def test_linux_without_a_browser_on_the_path() -> None:
    assert window.candidates("linux", {"PROGRAMFILES": "x"}, Path("/home/u")) == []
    assert window.find_browser(platform="linux", env={}, home=Path("/home/u"),
                               which=_nothing_on_path, is_file=lambda p: True) is None


class _Process:
    def __init__(self, code: int | None) -> None:
        self.code = code

    def wait(self, timeout: float | None = None) -> int:
        if self.code is None:
            raise subprocess.TimeoutExpired("browser", timeout or 0)
        return self.code

    def poll(self) -> int | None:
        return self.code


def test_an_app_window_with_a_private_profile() -> None:
    launched: list[list[str]] = []

    def popen(args: list[str], **kwargs: Any) -> Any:
        launched.append(args)
        return _Process(None)

    opened = window.open_window("http://127.0.0.1:1/t/", browser="/usr/bin/chrome",
                                popen=popen, open_tab=lambda url: False, say=lambda s: None)
    assert opened.how == "app" and opened.running()
    args = launched[0]
    assert args[0] == "/usr/bin/chrome" and "--app=http://127.0.0.1:1/t/" in args
    profile = next(a.split("=", 1)[1] for a in args if a.startswith("--user-data-dir="))
    assert Path(profile).is_dir() and opened.profile == profile
    assert any(a.startswith("--window-size=") for a in args)
    opened.process = None
    opened.close()
    assert not Path(profile).exists()


def test_no_browser_opens_a_tab_and_failing_that_prints_the_address() -> None:
    said: list[str] = []
    tabs: list[str] = []
    opened = window.open_window("http://u/", find=lambda: None,
                                open_tab=lambda url: tabs.append(url) or True, say=said.append)
    assert (opened.how, tabs, said) == ("tab", ["http://u/"], [])
    opened = window.open_window("http://u/", find=lambda: None, open_tab=lambda url: False,
                                say=said.append)
    assert opened.how == "none" and said == ["Open this address in a browser: http://u/"]


def test_a_browser_that_exits_at_once_falls_back_to_a_tab() -> None:
    said: list[str] = []
    opened = window.open_window("http://u/", browser="/usr/bin/chrome",
                                popen=cast(Any, lambda args, **kw: _Process(1)),
                                open_tab=lambda url: True, say=said.append)
    assert opened.how == "tab" and "closed at once" in said[0]


def test_linear_run_gui_starts_the_command_on_the_job_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from langslice.doors.commands.linear import open_gui

    started: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kwargs: started.append(command))
    open_gui(SimpleNamespace(image_folder=str(tmp_path), job_dir=None))
    assert started[0][-2:] == ["gui", str(tmp_path / "langslice")]
    open_gui(SimpleNamespace(image_folder=str(tmp_path), job_dir=str(tmp_path / "job")))
    assert started[1][-1] == str(tmp_path / "job")


def test_the_window_opens_on_the_tab_asked_for() -> None:
    url = "http://127.0.0.1:5000/token/"
    assert window.start_address(url, "viewer", folders=True) == url
    assert window.start_address(url, "prompt", folders=True) == url + "#tab=prompt&job=0"
    assert window.start_address(url, "prompt", folders=False) == url + "#tab=prompt"


def test_gui_tab_reaches_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    import langslice.hosts.gui as gui
    from langslice.hosts.cli import add_gui_parser, run_gui

    job = tmp_path / "images" / "langslice"
    job.mkdir(parents=True)
    (job / "job.json").write_text("{}")
    parser = argparse.ArgumentParser()
    add_gui_parser(parser.add_subparsers())
    calls: list[tuple[list[Path], str]] = []
    monkeypatch.setattr(gui, "serve", lambda folders, **kw: calls.append((folders, kw["tab"])))
    run_gui(parser.parse_args(["gui", str(tmp_path / "images"), "--tab", "prompt"]))
    run_gui(parser.parse_args(["gui"]))
    assert calls == [([job], "prompt"), ([], "viewer")]
    with pytest.raises(SystemExit):
        parser.parse_args(["gui", "--tab", "settings"])


def test_a_port_in_use_ends_with_one_sentence() -> None:
    import langslice.hosts.gui as gui

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(SystemExit) as ended:
            gui.serve([], port=port, window=False, say=lambda _text: None)
    assert str(ended.value) == (f"Port {port} is in use: choose another with --port, "
                                "or leave --port out.")


def test_the_address_is_written_at_once_when_the_output_is_a_file(tmp_path: Path) -> None:
    out = tmp_path / "gui.out"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO / "src"), str(REPO))),
           "LANGSLICE_RECENT_JOBS": "off", "LANGSLICE_SETTINGS": "off"}
    with out.open("w") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "langslice", "gui", "--no-window", "--port", "0"],
            stdout=handle, stderr=subprocess.DEVNULL, env=env, cwd=tmp_path)
    try:
        deadline = time.monotonic() + 60
        while not out.read_text() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.1)
        assert out.read_text().startswith("LangSlice GUI at http://127.0.0.1:")
    finally:
        process.terminate()
        process.wait(10)
