"""The LangSlice GUI: one local window with the agent viewer (``langslice gui``).

LangSlice serves the page on this computer (:mod:`.server`) and opens it as a
browser window without an address bar (:mod:`.window`). The agent viewer
watches job folders (:mod:`.watch`); it is read-only, and closing the window
never stops a run.
"""

from __future__ import annotations

import errno
import sys
import time
from collections.abc import Callable
from pathlib import Path

#: How long the server outlives its window.
LINGER_S = 5.0


def _in_use(exc: OSError) -> bool:
    """Whether binding failed because another socket holds the port. Windows
    refuses a port another program listens on as forbidden (WSAEACCES), not
    as in use."""
    if sys.platform == "win32":
        return exc.winerror in (errno.WSAEADDRINUSE, errno.WSAEACCES)
    return exc.errno == errno.EADDRINUSE


def serve(folders: list[Path], *, port: int = 0, window: bool = True,
          tab: str = "viewer", say: Callable[[str], None] = print) -> None:
    """Serve the GUI for *folders* and the recent jobs until Ctrl-C, or until
    its app window has been closed for ``LINGER_S``; the window opens on
    *tab* (:data:`.window.TABS`). A port that cannot be served ends with
    one sentence (``SystemExit``); the address is written at once, also
    when the output is a file."""
    from langslice.hosts.gui.server import GuiServer
    from langslice.hosts.gui.watch import JobList
    from langslice.hosts.gui.window import Opened, open_window, start_address
    from langslice.job.recent import recent_file

    jobs = JobList(folders, recent=recent_file())
    try:
        server = GuiServer(jobs, port=port)
    except OSError as exc:
        reason = "is in use" if _in_use(exc) else f"cannot be used ({exc.strerror})"
        raise SystemExit(f"Port {port} {reason}: choose another with --port, "
                         "or leave --port out.") from None
    server.start()
    address = start_address(server.url, tab, folders=bool(folders))
    say(f"LangSlice GUI at {address}")
    sys.stdout.flush()
    opened = open_window(address, say=say) if window else Opened("none")
    sys.stdout.flush()
    closed_at: float | None = None
    try:
        while True:
            time.sleep(0.5)
            if opened.how != "app":
                continue
            if opened.running():
                closed_at = None
            elif closed_at is None:
                closed_at = time.monotonic()
            elif time.monotonic() - closed_at > LINGER_S:
                break
    except KeyboardInterrupt:
        pass
    finally:
        opened.close()
        server.close()
