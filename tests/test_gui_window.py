"""Opening the GUI (``langslice.hosts.gui.window``): the browser found on each
platform, the app window, and both fallbacks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from langslice.hosts.gui import window


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

    from langslice.doors.cli.linear import open_gui

    started: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kwargs: started.append(command))
    open_gui(SimpleNamespace(image_folder=str(tmp_path), job_dir=None))
    assert started[0][-2:] == ["gui", str(tmp_path / "langslice")]
    open_gui(SimpleNamespace(image_folder=str(tmp_path), job_dir=str(tmp_path / "job")))
    assert started[1][-1] == str(tmp_path / "job")
