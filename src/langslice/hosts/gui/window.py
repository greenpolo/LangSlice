"""Opening the GUI: a browser app window without an address bar, else a tab.

The order: a Chrome, Chromium, Edge or Brave on the ``PATH``; on Windows the
usual install folders under ``PROGRAMFILES`` and ``LOCALAPPDATA``; on macOS
``/Applications`` and ``~/Applications``. The window is opened with
``--app=URL`` and a private profile, so it is a process of its own (closing it
ends that process, which is how the server knows the window is gone). Without
such a browser a normal tab is opened with :mod:`webbrowser`; failing that the
URL is printed and the server keeps serving.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: The window's size in pixels.
WINDOW_SIZE = (1440, 960)
#: Browser commands looked up on the PATH, in order.
PATH_NAMES = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "microsoft-edge-stable", "brave-browser", "brave",
    "chrome.exe", "msedge.exe", "brave.exe",
)
#: Under PROGRAMFILES, PROGRAMFILES(X86) and LOCALAPPDATA on Windows.
WINDOWS_PATHS = (
    "Google/Chrome/Application/chrome.exe",
    "Microsoft/Edge/Application/msedge.exe",
    "BraveSoftware/Brave-Browser/Application/brave.exe",
    "Chromium/Application/chrome.exe",
)
WINDOWS_ROOTS = ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
#: Under /Applications and ~/Applications on macOS.
MAC_APPS = (
    "Google Chrome.app/Contents/MacOS/Google Chrome",
    "Chromium.app/Contents/MacOS/Chromium",
    "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "Brave Browser.app/Contents/MacOS/Brave Browser",
)


def candidates(platform: str, env: dict[str, str], home: Path) -> list[Path]:
    """The install paths to try on *platform* (``sys.platform``) after the PATH."""
    found: list[Path] = []
    if platform.startswith("win"):
        for root in WINDOWS_ROOTS:
            if env.get(root):
                found += [Path(env[root]) / name for name in WINDOWS_PATHS]
    elif platform == "darwin":
        for root in (Path("/Applications"), home / "Applications"):
            found += [root / name for name in MAC_APPS]
    return found


def find_browser(*, platform: str = sys.platform, env: dict[str, str] | None = None,
                 home: Path | None = None,
                 which: Callable[[str], str | None] = shutil.which,
                 is_file: Callable[[Path], bool] = Path.is_file) -> str | None:
    """The first app-window browser found, or None."""
    for name in PATH_NAMES:
        path = which(name)
        if path:
            return path
    for path in candidates(platform, dict(os.environ) if env is None else env,
                           Path.home() if home is None else home):
        if is_file(path):
            return str(path)
    return None


def app_arguments(browser: str, url: str, profile: str,
                  size: tuple[int, int] = WINDOW_SIZE) -> list[str]:
    args = [
        browser, f"--app={url}", f"--user-data-dir={profile}",
        f"--window-size={size[0]},{size[1]}", "--no-first-run",
        "--no-default-browser-check", "--disable-session-crashed-bubble",
        "--disable-background-mode", "--disable-extensions",
    ]
    if os.name == "posix" and os.environ.get("DISPLAY"):
        args.append("--ozone-platform=x11")
    return args


@dataclass
class Opened:
    """How the GUI was opened: ``app`` (its own window, *process*), ``tab`` or
    ``none`` (the URL printed)."""

    how: str
    process: subprocess.Popen[bytes] | None = None
    profile: str | None = None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self) -> None:
        if self.running():
            assert self.process is not None
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.profile:
            shutil.rmtree(self.profile, ignore_errors=True)


def open_window(url: str, *, browser: str | None = None,
                find: Callable[[], str | None] = find_browser,
                popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
                open_tab: Callable[[str], bool] = webbrowser.open,
                say: Callable[[str], None] = print) -> Opened:
    """Open *url* as an app window, else a tab, else print it."""
    browser = browser or find()
    if browser:
        profile = tempfile.mkdtemp(prefix="langslice-gui-")
        try:
            process = popen(app_arguments(browser, url, profile),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            shutil.rmtree(profile, ignore_errors=True)
        else:
            try:
                code = process.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                return Opened("app", process, profile)
            shutil.rmtree(profile, ignore_errors=True)
            say(f"The browser window closed at once (status {code}); opening a tab.")
    try:
        if open_tab(url):
            return Opened("tab")
    except Exception:  # webbrowser raises its own errors on a broken setup
        pass
    say(f"Open this address in a browser: {url}")
    return Opened("none")
