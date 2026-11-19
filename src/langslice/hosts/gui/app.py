"""The LangSlice window as a desktop app of its own.

``python -m langslice.hosts.gui.app URL`` shows the page the GUI server
serves in a native window (pywebview: the system's web view, WebView2 on
Windows, WebKit on macOS and Linux), titled "LangSlice" with LangSlice's
icon, so it has its own entry in the taskbar or dock and no browser is
involved. It is a process of its own: closing the window ends it, which is
how the server knows the window is gone (:func:`langslice.hosts.gui.serve`).
:mod:`.window` starts it when pywebview is installed (the ``window`` extra),
and falls back to a browser's app window without it.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

TITLE = "LangSlice"
APP_ID = "LangSlice.Window"
ICON = Path(__file__).parent / "static" / "langslice.ico"
DARK_TITLE_BAR = 20
DEFAULT_SIZE = (1440, 960)
MIN_SIZE = (960, 640)
GROUND = "#121213"
SIZED_ARGUMENTS = 3
WINDOWS = "nt"
USAGE = "usage: python -m langslice.hosts.gui.app URL [WIDTH HEIGHT]"


class _Handle(Protocol):
    def ToInt64(self) -> int: ...


class _Native(Protocol):
    Handle: _Handle


class _Event(Protocol):
    def __iadd__(self, handler: Callable[[_Window], None]) -> _Event: ...


class _Events(Protocol):
    before_show: _Event


class _Window(Protocol):
    events: _Events
    native: _Native


class _Webview(Protocol):
    def create_window(
        self,
        title: str,
        url: str,
        *,
        width: int,
        height: int,
        min_size: tuple[int, int],
        background_color: str,
    ) -> _Window: ...

    def start(self, *, icon: str) -> None: ...


class _Shell(Protocol):
    def SetCurrentProcessExplicitAppUserModelID(self, app_id: str) -> int: ...


class _Dwm(Protocol):
    def DwmSetWindowAttribute(
        self, window: int, attribute: int, value: object, size: int
    ) -> int: ...


def _library(name: str) -> object:
    """Load one of Windows' libraries by name."""
    loader = cast("Callable[[str], object]", vars(ctypes)["WinDLL"])
    return loader(name)


def own_taskbar_identity() -> None:
    """Give this process its own taskbar identity (Windows only)."""
    if os.name == WINDOWS:
        shell = cast("_Shell", _library("shell32"))
        shell.SetCurrentProcessExplicitAppUserModelID(APP_ID)


def dark_title_bar(window: _Window) -> None:
    """Draw the window's title bar dark, as the page is (Windows only).

    Parameters
    ----------
    window
        The pywebview window, once its native window exists.
    """
    if os.name == WINDOWS:
        on = ctypes.c_int(1)
        dwm = cast("_Dwm", _library("dwmapi"))
        dwm.DwmSetWindowAttribute(
            window.native.Handle.ToInt64(),
            DARK_TITLE_BAR,
            ctypes.byref(on),
            ctypes.sizeof(on),
        )


def main() -> int:
    """Show the command line's address in LangSlice's own window.

    The arguments are the address, then optionally the window's width and
    height. The call returns when the window is closed.

    Returns
    -------
    int
        0, once the window was closed.
    """
    args = sys.argv[1:]
    if not args:
        raise SystemExit(USAGE)
    sized = len(args) >= SIZED_ARGUMENTS
    width, height = (int(args[1]), int(args[2])) if sized else DEFAULT_SIZE
    webview = cast("_Webview", importlib.import_module("webview"))
    own_taskbar_identity()
    window = webview.create_window(
        TITLE,
        args[0],
        width=width,
        height=height,
        min_size=MIN_SIZE,
        background_color=GROUND,
    )
    window.events.before_show += dark_title_bar
    webview.start(icon=str(ICON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
