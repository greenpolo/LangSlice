"""The LangSlice GUI: one local window with the agent viewer (``langslice gui``).

LangSlice serves the page on this computer (:mod:`.server`) and opens it as a
browser window without an address bar (:mod:`.window`). The agent viewer
watches job folders (:mod:`.watch`); it is read-only, and closing the window
never stops a run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

#: How long the server outlives its window.
LINGER_S = 5.0


def serve(folders: list[Path], *, port: int = 0, window: bool = True,
          say: Callable[[str], None] = print) -> None:
    """Serve the GUI for *folders* and the recent jobs until Ctrl-C, or until
    its app window has been closed for ``LINGER_S``."""
    from langslice.hosts.gui.server import GuiServer
    from langslice.hosts.gui.watch import JobList
    from langslice.hosts.gui.window import Opened, open_window
    from langslice.job.recent import recent_file

    jobs = JobList(folders, recent=recent_file())
    server = GuiServer(jobs, port=port)
    server.start()
    say(f"LangSlice GUI at {server.url}")
    opened = open_window(server.url, say=say) if window else Opened("none")
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
