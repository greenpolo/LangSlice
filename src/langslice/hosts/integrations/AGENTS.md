# LangSlice `hosts/integrations/` — `langslice abba` and its viewer and log

Package guide for `src/langslice/hosts/integrations/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

The ABBA integration is the Java Fiji connector (`connectors/fiji/`: the
dialog under ABBA's Register menu, Setup, the native actions) with
LangSlice as its own Python worker (`langslice serve --stdio`,
`hosts/api/service.py`; the worker itself is `doors/api/abba_worker.py`).
This package is the optional Python side of a session started with
`langslice abba`: it starts ABBA 0.24.x from Python and follows the
connector's runs in a passive agent viewer and an activity log. Nothing
here changes ABBA's registrations: the connector applies every change.
ABBA's sign table for the cutting angles is `core/abba_angles.py`.

## `abba_launch.py` — `langslice abba` (`hosts/cli.py`)

- ABBA starts from Python (pyimagej + abba-python's `Abba` class, `ij=`
  our own) with `ABBA_JAVA_DEPENDENCIES` (ImageToAtlasRegister 0.24.1 and
  the versions its pom pins; abba-python 0.11.0's own list is for ABBA
  0.11) on Java 21 (`scyjava.config.set_java_constraints(fetch="auto",
  version="21")`), and the connector jar on the classpath
  (`connector_jar`: `--connector-jar`, else `LANGSLICE_CONNECTOR_JAR`,
  else the newest `connectors/fiji/target/*.jar` of this checkout; a clear
  `mvn package` message when there is none).
- `init_theme` sets ABBA's theme: abba-python builds the BDV view without
  the Java launcher that sets it, and null strokes crash the overlay
  painter.
- `point_connector_at` sets the `langslice.environment` system property to
  this environment (the connector's `EnvironmentDiscovery.current()` runs
  that session's workers from it) and saves it as the connector's
  environment only when none is saved.
- `RunListener` is registered with `org.langslice.fiji.LangSliceEvents`
  (a `java.util.function.Consumer<String>` JPype proxy). The connector
  forwards every worker message of a run (`checkpoint`, `agent_event`,
  `log`) and its own (`run_started` with `mode`, `viewer` and `sections`,
  `job`, `applied`, `run_finished`) as JSON text. The Java thread only
  queues; one Python thread handles messages in order. A new section map
  (`LangSliceEvents.slices()`) or `run_started` is a new run (the old
  follower finishes); the follower opens only when the run asked for the
  viewer.
- One log window (`create_activity_window`) serves the session:
  `run_started` reopens it when the user closed it, and a disposed one
  (`closed`) is replaced on the next message. Agent events reach it with
  their pictures read from the event's `views` (saved `view.jpg` paths in
  the job folder, waiting up to `VIEW_WAIT_S` for the background writer);
  events never carry image bytes.
- JPype, scyjava, pyimagej and abba-python are imported lazily: the
  plumbing is tested with fakes (`tests/test_integrations_abba_launch.py`).
  `tests/test_abba_viewer_jvm.py` (marked `slow`; needs a Python with
  JPype and scyjava, `LANGSLICE_ABBA_PYTHON`) starts a headless ABBA
  0.24.1 JVM and checks every Java member the viewer and launcher use,
  copies a real source the way the viewer does, and delivers a connector
  message into `RunListener` when the jar is built.

## The agent viewer

`abba_follow.py::AbbaFollower` consumes seed and executed
`tool_start`/`tool_end` events (`doors.tools.toolbox._serialized` emits
them inside its lock, with corrected-index references resolved to stable
filenames). Model `tool_call` announcements may be batched ahead of
execution and never drive the viewer. `_FOLLOWED` names the tools whose
targets are shown; every one in `_WRITES` (all but the looks and
`preprocess`) refreshes its targets on its end event, after the connector
has applied the checkpoint's rows. Every followed event goes to the
comparison window its `comparison_factory` builds on first use; the main
ABBA view, its camera, selection and display modes are never touched, and
a viewer failure is logged and never reaches the run.

`abba_compare.py::ComparisonWindow` is a separate read-only frame: the
positioning overview on top, up to four independent BigDataViewer focus
panels per page below, paging targets in execution order. The panels borrow
the registered section and atlas sources, wrapped in their own
`TransformedSource` and converter (`copy_native_source`, through
bdv-playground's `sc.fiji.bdvpg.source.SourceHelper`), with their own
cameras; they never register or remove global sources or change ABBA's
display state. A 250 ms poll tracks source replacement, affine changes and
positions as ABBA's asynchronous actions finish, keeps each camera's
calibrated fit (`ensure_camera` rewrites only a changed camera) and
repaints only when content changed, so polling never restarts progressive
rendering. All methods run on the Swing EDT.

`abba_overview.py::NativeOverview` borrows ABBA's `extendedSlicedSources`
positioning mosaic at the host's `getStep`, with the host's active atlas
channels and copied converter settings. Sections keep native size. In
positioning mode their centres are the host's `getDisplayedCenter`
(overlap and stair placement included); in review mode the overview uses
ABBA 0.24's positioning formula (`native_position_x`, voxel-snapped) one
atlas height below, without switching the host's mode. The camera fits the
stack, keeps in-range framing and reframes when the host changes its
display step. Selection is drawn with ABBA's own `CircleGraphicalHandle`
and `SquareGraphicalHandle` (`HANDLE_PACKAGE`), with local suppliers and no
editing behaviours, and `ABBABdvViewPrefs` strokes and colours. Both areas
show committed ABBA registrations; candidate positions are never drawn
(the log's tool pictures show those).

## The activity log

`abba_chat.py::create_activity_window` prefers a local Chrome/Chromium app
window beside ABBA (`ChatWindow`: a private profile; `show` relaunches the
app window only when none is open), falling back to the Swing
`abba_activity.py::ActivityWindow` when no browser starts. The browser log
renders escaped Markdown, streamed assistant text and provider-exposed
reasoning summaries; compact expandable tool rows show tool names, targets,
status and image counts, and an image panel below supports history and
full-image inspection. Summary wording comes from the provider; encrypted
reasoning and signatures are excluded (`_public`). `ChatServer` is a
loopback GET server on a random path token serving the bundled `static/`
assets, sanitized events and exact image bytes, with no command or
registration endpoints. History is bounded to `MAX_EVENTS` events / 2 MiB of
event JSON and 64 MiB of images. Closing the log never cancels the run. The
Swing fallback keeps its adjustable split, bounded image history and
off-EDT image decoding.
