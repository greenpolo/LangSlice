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
from pathlib import Path
from typing import Any

#: The window's title, and its name in the taskbar.
TITLE = "LangSlice"
#: Windows groups taskbar windows by this identity; without one of its own
#: the window would be filed under Python's, with Python's icon.
APP_ID = "LangSlice.Window"
#: The icon: LangSlice's mark (``assets/logo/logomark_dark_512.png``).
ICON = Path(__file__).parent / "static" / "langslice.ico"
#: ``DWMWA_USE_IMMERSIVE_DARK_MODE``: the attribute for a dark title bar.
DARK_TITLE_BAR = 20
#: The window's size without one given, and the smallest it can be made.
DEFAULT_SIZE = (1440, 960)
MIN_SIZE = (960, 640)
#: The page's ground (``static/tokens.css`` ``--ground``), until it loads.
GROUND = "#121213"
#: The arguments that carry a size: the address, the width, the height.
SIZED_ARGUMENTS = 3
#: ``os.name`` on Windows, where the taskbar and title bar are set.
WINDOWS = "nt"
USAGE = "usage: python -m langslice.hosts.gui.app URL [WIDTH HEIGHT]"


def own_taskbar_identity() -> None:
    """Give this process its own taskbar identity (Windows only)."""
    if os.name == WINDOWS:
        shell: Any = ctypes.windll.shell32  # type: ignore[attr-defined]
        shell.SetCurrentProcessExplicitAppUserModelID(APP_ID)


def dark_title_bar(window: object) -> None:
    """Draw the window's title bar dark, as the page is (Windows only).

    Parameters
    ----------
    window
        The pywebview window, once its native window exists.
    """
    if os.name == WINDOWS:
        native: Any = window.native  # type: ignore[attr-defined]
        on = ctypes.c_int(1)
        dwm: Any = ctypes.windll.dwmapi  # type: ignore[attr-defined]
        dwm.DwmSetWindowAttribute(
            int(native.Handle.ToInt64()),
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
    # pywebview comes with the `window` extra only.
    webview: Any = importlib.import_module("webview")
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
