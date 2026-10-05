"""``langslice abba``: ABBA 0.24.x started from Python, with the LangSlice connector.

The ONE ABBA integration is the Java Fiji connector (``connectors/fiji/``:
the dialog under ABBA's Register menu, which starts LangSlice as its own
Python worker). This module only starts ABBA from Python so that the
connector's runs can also be watched with the Python viewer and log:

- the JVM gets ABBA 0.24.1's Java dependencies (:data:`ABBA_JAVA_DEPENDENCIES`;
  abba-python 0.11.0's own pins are for ABBA 0.11) and the connector jar
  (:func:`connector_jar`: ``LANGSLICE_CONNECTOR_JAR``, else the connector's
  Maven build output ``connectors/fiji/target/*.jar`` of this checkout);
- ABBA's theme is initialized (abba-python builds the BigDataViewer view
  without the Java launcher that does it; null strokes crash the overlay
  painter as soon as sections are selected);
- the connector is pointed at this Python environment (``sys.prefix``) by
  default: the ``langslice.environment`` system property, and the
  connector's saved environment when none is saved yet;
- a :class:`RunListener` is registered with the connector's event registry
  (``org.langslice.fiji.LangSliceEvents``): every worker message of a run
  (``checkpoint``, ``agent_event``, ``log``) and the connector's own
  (``run_started`` with ``viewer``, ``job``, ``applied``, ``run_finished``)
  reaches it as JSON text, and it feeds the PASSIVE agent viewer
  (:class:`~langslice.hosts.integrations.abba_follow.AbbaFollower` with
  :class:`~langslice.hosts.integrations.abba_compare.ComparisonWindow`)
  and the browser activity log
  (:func:`~langslice.hosts.integrations.abba_chat.create_activity_window`),
  whose pictures are read from the events' ``views`` (saved view files in
  the job folder; events never carry image bytes). The viewer opens only
  for a run whose ``run_started`` asked for it. Neither writes anything to
  ABBA: the connector applies every change.

JPype, scyjava, pyimagej and abba-python are imported lazily, so the
listener plumbing is unit-tested with fakes in the plain environment.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ABBA 0.24.1's Java dependencies (``ch.epfl.biop:ImageToAtlasRegister:0.24.1``
#: pom: bdv-playground 0.23.1, image-loaders 0.23.3, biop-tools 0.24.1,
#: selector 0.23.0, ijl-utilities-wrappers 0.12.1, imglib2-realtransform
#: 4.0.6, atlas 0.23.4). ABBA 0.24 needs Java 21.
ABBA_VERSION = "0.24.1"
ABBA_JAVA_DEPENDENCIES: tuple[str, ...] = (
    f"ch.epfl.biop:ImageToAtlasRegister:{ABBA_VERSION}",
    "ch.epfl.biop:bigdataviewer-biop-tools:0.24.1",
    "sc.fiji:bigdataviewer-playground:0.23.1",
    "ch.epfl.biop:bigdataviewer-image-loaders:0.23.3",
    "ch.epfl.biop:bigdataviewer-selector:0.23.0",
    "ch.epfl.biop:ijl-utilities-wrappers:0.12.1",
    "net.imglib2:imglib2-realtransform:4.0.6",
    "ch.epfl.biop:atlas:0.23.4",
    "com.formdev:flatlaf:3.5.1",
)
JAVA_VERSION = "21"
#: Where the connector jar is looked for first.
CONNECTOR_JAR_ENV = "LANGSLICE_CONNECTOR_JAR"
#: The connector's Maven build output, relative to the repository root.
CONNECTOR_TARGET = Path("connectors") / "fiji" / "target"
#: The Java registry the connector forwards a run's worker messages through.
EVENTS_CLASS = "org.langslice.fiji.LangSliceEvents"
#: The Java system property naming the Python environment the connector
#: should start by default.
ENVIRONMENT_PROPERTY = "langslice.environment"
#: How long the listener waits for a saved view file to appear (the job's
#: view store writes in the background).
VIEW_WAIT_S = 3.0
#: Largest picture read into the log.
MAX_VIEW_BYTES = 16 * 1024 * 1024

MISSING_JAR = (
    "The LangSlice Fiji connector jar was not found. Build it (in "
    "connectors/fiji: `mvn package`, which writes connectors/fiji/target/*.jar), "
    f"or set {CONNECTOR_JAR_ENV} to the jar's path."
)


def repository_root() -> Path:
    """The checkout this package was installed from (editable installs)."""
    return Path(__file__).resolve().parents[4]


def connector_jar(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The connector jar: *explicit*, else ``$LANGSLICE_CONNECTOR_JAR``, else
    the newest ``connectors/fiji/target/*.jar`` of this checkout (sources,
    javadoc and test jars skipped). ``FileNotFoundError`` with what to do
    when there is none."""
    for given in (explicit, os.environ.get(CONNECTOR_JAR_ENV, "").strip() or None):
        if given:
            path = Path(given).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"The connector jar {path} does not exist. {MISSING_JAR}")
            return path.resolve()
    found = [Path(name) for name in glob.glob(str(repository_root() / CONNECTOR_TARGET / "*.jar"))
             if not name.endswith(("-sources.jar", "-javadoc.jar", "-tests.jar"))
             and not Path(name).name.startswith("original-")]
    if not found:
        raise FileNotFoundError(MISSING_JAR)
    return max(found, key=lambda path: path.stat().st_mtime).resolve()


# --- the listener ----------------------------------------------------------------------


def read_views(paths: Any, *, wait_s: float = VIEW_WAIT_S) -> list[dict[str, Any]]:
    """The log's pictures for an event's ``views``: each saved ``view.jpg``'s
    bytes (waiting up to *wait_s* for files still being written), labelled by
    its folder (``<seq>_<tool>``). Missing or oversized files are skipped."""
    images: list[dict[str, Any]] = []
    if not isinstance(paths, list):
        return images
    deadline = time.monotonic() + max(0.0, wait_s)
    for value in paths:
        if not isinstance(value, str):
            continue
        path = Path(value)
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            if path.stat().st_size > MAX_VIEW_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        images.append({"data": data, "mime_type": mime, "label": path.parent.name})
    return images


class RunListener:
    """Feeds the Python viewer and log from the connector's worker messages.

    *slices* returns the current run's snapshot filename -> ABBA slice map
    (``LangSliceEvents.slices()``); *follower_factory* builds the viewer's
    follower for a map; *log_factory* the activity log (None: no log).
    Messages are handled in order on one background thread, so the Java
    thread that delivers them never waits (pictures are read from disk).
    A failure in a viewer or the log is logged and never reaches the run.
    """

    def __init__(
        self, *, slices: Callable[[], Mapping[str, Any]],
        follower_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        log_factory: Callable[[], Any] | None = None,
        view_wait_s: float = VIEW_WAIT_S,
    ) -> None:
        self.slices = slices
        self.follower_factory = follower_factory
        self.log_factory = log_factory
        self.view_wait_s = view_wait_s
        self.follower: Any = None
        self.log: Any = None
        self._run: frozenset[str] = frozenset()
        #: The run's ``run_started`` says whether the user asked for the viewer.
        self.viewer_wanted = True
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    # The Java side calls this (through a JProxy Consumer<String>).
    def accept(self, text: Any) -> None:
        self._queue.put(str(text))
        if self._thread is None:
            self._thread = threading.Thread(target=self._drain, name="langslice-abba-events",
                                            daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Stop after the messages already received; finish the follower."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)
            self._thread = None
        self._finish_run()

    def _drain(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                self.handle(text)
            except Exception:
                logger.warning("The ABBA viewer or log could not take an event", exc_info=True)

    # --- one message ------------------------------------------------------------------

    def handle(self, text: str) -> None:
        """Route one worker message (JSON text) to the viewer and the log."""
        message = json.loads(text)
        if not isinstance(message, dict):
            return
        kind = message.get("kind")
        if kind == "run_started":
            self._finish_run()
            self.viewer_wanted = bool(message.get("viewer", True))
            self._to_log({"kind": "status", "text": "Running"})
            self._show_log()
            self._ensure_run(force=True)
        elif kind == "agent_event" and isinstance(message.get("event"), dict):
            self._agent_event(message["event"])
        elif kind == "log" and isinstance(message.get("message"), str):
            self._to_log({"kind": "progress", "text": message["message"]})
        elif kind == "applied":
            failed = message.get("failed") or {}
            if isinstance(failed, dict) and failed:
                self._to_log({"kind": "progress", "text": "ABBA could not apply: " + "; ".join(
                    f"{name}: {reason}" for name, reason in failed.items())})
            if message.get("angles_failed"):
                self._to_log({"kind": "progress",
                              "text": f"ABBA could not set the angles: {message['angles_failed']}"})
        elif kind == "run_finished":
            self._to_log({"kind": "status", "text": str(message.get("message") or "Finished")})
            self._finish_run()

    def _agent_event(self, event: dict[str, Any]) -> None:
        self._ensure_run()
        if self.follower is not None:
            try:
                self.follower.on_event(event)
            except Exception:
                logger.warning("ABBA agent viewer could not follow", exc_info=True)
        if self.log_factory is None:
            return
        shown = dict(event)
        if shown.get("views"):
            shown["images"] = read_views(shown["views"], wait_s=self.view_wait_s)
        self._to_log(shown)

    def _ensure_run(self, *, force: bool = False) -> None:
        """A follower for the current run's sections (a new one when the
        connector's section map changed: a new run)."""
        try:
            mapping = {str(key): value for key, value in dict(self.slices()).items()}
        except Exception:
            logger.warning("The connector's section map is unavailable", exc_info=True)
            mapping = {}
        names = frozenset(mapping)
        if not force and names == self._run and (self.follower is not None or not names):
            return
        self._finish_run()
        self._run = names
        if self.follower_factory is not None and mapping and self.viewer_wanted:
            try:
                self.follower = self.follower_factory(mapping)
            except Exception:
                logger.warning("ABBA agent viewer could not start", exc_info=True)

    def _finish_run(self) -> None:
        follower, self.follower = self.follower, None
        self._run = frozenset()
        if follower is not None:
            try:
                follower.finish()
            except Exception:
                logger.warning("ABBA agent viewer could not finish", exc_info=True)

    def _show_log(self) -> None:
        """Reopen the session's log window when the user closed it."""
        if self.log is None:
            return
        try:
            self.log.show()
        except Exception:
            logger.warning("Agent log could not reopen", exc_info=True)

    def _to_log(self, event: dict[str, Any]) -> None:
        if self.log_factory is None:
            return
        try:
            if self.log is None or self.log.closed:
                self.log = self.log_factory()
            self.log.on_event(event)
        except Exception:
            logger.warning("Agent log could not show an event", exc_info=True)


# --- the JVM ---------------------------------------------------------------------------


def init_theme() -> None:
    """Initialize ABBA's theme when abba-python left its strokes null."""
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    prefs = jimport("ch.epfl.biop.atlas.aligner.gui.bdv.ABBABdvViewPrefs")
    if prefs.line_between_selected_slices_stroke is None:
        theme = jimport("ch.epfl.biop.atlas.aligner.gui.bdv.ABBATheme")
        theme.setTheme(theme.createLightTheme())


def point_connector_at(prefix: str) -> None:
    """Make *prefix* the connector's default Python environment: the
    ``langslice.environment`` system property, and its saved environment
    when none is saved yet (a user's own choice is kept)."""
    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    jimport("java.lang.System").setProperty(ENVIRONMENT_PROPERTY, prefix)
    try:
        discovery = jimport("org.langslice.fiji.EnvironmentDiscovery")
        if discovery.saved() is None:
            discovery.save(jimport("java.nio.file.Paths").get(prefix))
    except Exception:
        logger.warning("Could not save the connector's environment", exc_info=True)


def _java_listener(listener: RunListener) -> Any:
    from jpype import JImplements, JOverride  # pyright: ignore[reportMissingImports]

    @JImplements("java.util.function.Consumer")
    class Forward:
        @JOverride
        def accept(self, text: Any) -> None:
            listener.accept(str(text))

    return Forward()


def start_abba(
    *, abba_atlas: str, jar: Path, viewer: bool = True, log: bool = True,
    prefix: str | None = None,
) -> tuple[Any, RunListener, Any]:
    """Start ABBA's GUI with the connector jar on the classpath and register
    the Python listener. Returns ``(abba, listener, java_listener)``; keep
    them referenced while ABBA runs."""
    try:
        import imagej  # pyright: ignore[reportMissingImports]
        import scyjava.config  # pyright: ignore[reportMissingImports]
        from abba_python.abba import Abba  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "abba-python is not installed in this environment. Install it with\n"
            '  pip install "langslice[abba]"\n'
            "inside a conda env that provides OpenJDK 21 and Maven. "
            f"({exc})") from exc

    scyjava.config.add_classpath(str(jar))
    if hasattr(scyjava.config, "set_java_constraints"):
        scyjava.config.set_java_constraints(fetch="auto", version=JAVA_VERSION)
    ij = imagej.init(list(ABBA_JAVA_DEPENDENCIES), mode="interactive")
    ij.ui().showUI()
    try:
        from abba_python.abba import add_brainglobe_atlases  # pyright: ignore

        add_brainglobe_atlases(ij)
    except Exception:
        logger.warning("BrainGlobe atlases were not registered with ABBA", exc_info=True)
    abba = Abba(abba_atlas, ij=ij)
    init_theme()
    abba.show_bdv_ui()
    point_connector_at(prefix or sys.prefix)

    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    swing = jimport("javax.swing.SwingUtilities")
    frame = swing.getWindowAncestor(abba.get_bdv_view().getBdvh().getViewerPanel())
    events = jimport(EVENTS_CLASS)

    def follower(mapping: Mapping[str, Any]) -> Any:
        from langslice.hosts.integrations.abba_compare import ComparisonWindow
        from langslice.hosts.integrations.abba_follow import AbbaFollower

        return AbbaFollower(abba, mapping,
                            comparison_factory=lambda: ComparisonWindow(abba, parent=frame))

    def activity() -> Any:
        from langslice.hosts.integrations.abba_chat import create_activity_window

        return create_activity_window(parent=frame)

    def slices() -> dict[str, Any]:
        held = events.slices()
        return {str(key): held.get(key) for key in held.keySet()}

    listener = RunListener(
        slices=slices,
        follower_factory=follower if viewer else None,
        log_factory=activity if log else None,
    )
    forward = _java_listener(listener)
    events.addListener(forward)
    return abba, listener, forward


def run_abba_session(
    *, abba_atlas: str, jar: Path, viewer: bool = True, log: bool = True,
) -> None:
    """``langslice abba``: start ABBA with the connector *jar*
    (:func:`connector_jar`) and block until its JVM shuts down."""
    _abba, listener, _forward = start_abba(abba_atlas=abba_atlas, jar=jar, viewer=viewer,
                                           log=log)
    try:
        wait_for_jvm_shutdown()
    finally:
        listener.close()


def wait_for_jvm_shutdown(poll_s: float = 0.5) -> None:
    """Block while the JVM (and with it ABBA's windows) is running."""
    import jpype  # pyright: ignore[reportMissingImports]

    while jpype.isJVMStarted():
        time.sleep(poll_s)
