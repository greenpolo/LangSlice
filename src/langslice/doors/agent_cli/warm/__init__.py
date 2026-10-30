"""The agent CLI's warm process: one fork server per job folder.

A ``langslice-job FOLDER ...`` call is a process of its own, and most of what a
call costs is starting one: importing ITK's Elastix classes, torch and FireANTs
and LangSlice's toolbox, and loading the job's atlas. The warm process pays
that once per job folder and then forks a child for every call. The child runs
what a cold call runs (:func:`langslice.doors.agent_cli.main.run`) with the
caller's arguments, working folder, environment variables and standard streams,
so a call still behaves as a process of its own: its own memory, its own holds
of the job folder's lock, ``Job.load`` and ``Job.sync`` as before, and calls
run in parallel when they are made in parallel. What it writes on stdout and
stderr and its exit code are a cold call's.

The client (:func:`run`, the first thing ``langslice-job`` does; standard
library only, so a warm call starts in milliseconds) works out the job
folder of ``FOLDER VERB`` (``init``, ``ops`` and ``schema`` always run cold),
connects to that job's socket and hands over its argv, working folder,
environment and the file descriptors of its stdin, stdout and stderr
(``SCM_RIGHTS``); then it waits for the child's exit status and ends with it
(a child killed by a signal: the client kills itself with that signal).
Signals the client gets (Ctrl-C, ``SIGTERM``, ``SIGHUP``, ``SIGQUIT``) are
passed to the child's process group; a client that dies or is killed takes
the child with it (``SIGKILL`` to its process group). With no warm process
up, the call runs cold and starts one, detached, for the next calls.
``LANGSLICE_WARM=0`` turns it off; it is Linux only (fork with these
libraries is not safe on macOS, and Windows has none).

The server (:func:`serve_main`) holds ``<key>.lock`` for its life (one per job
folder and Python environment), loads the libraries and the job's atlas
(:func:`warm_up`; nothing touches the GPU, so a child can initialise CUDA),
computes every placed section's frame for the children to reuse
(:func:`langslice.core.maps.section_maps.share_frames`), then listens on
``<key>.sock`` in ``$XDG_RUNTIME_DIR/langslice-warm/`` (else
``/tmp/langslice-warm-<uid>/``, mode 0700; only the same user's processes are
served). It never runs stale code: before each call it checks the files of
every first-party module it has loaded (LangSlice's own, a plugin's: anything
outside the standard library and site-packages), the site-packages folders and
the atlas's folder; one changed, it answers that call ``stale`` (the call runs
cold and starts a new server), stops listening and exits once its running calls
end. A caller whose environment differs where the loaded libraries read it
(:data:`ENV_PREFIXES`, ``HOME``) runs cold. It exits after :data:`IDLE_S`
without a call, or :data:`SUBMITTED_IDLE_S` once the job is submitted. A
section written since the last call has its frame computed again while the
server is idle.

A child: its own process group, killed with the server, the caller's
streams on 0, 1 and 2 (Python's ``sys.stdin``/``stdout``/``stderr`` made
again with the caller's encodings and buffering), ``sys.argv``, ``sys.path``,
``os.environ`` and the working folder the caller's, then the cold call's
code, then Python's own ending (threads joined, ``atexit``, streams
flushed). ``langslice-job FOLDER python SCRIPT`` runs the agent's script in
the child (:func:`run_script`) as the job environment's python would.

The modules: ``protocol.py`` (where a job's warm process listens, the
messages), ``client.py``, ``server.py``, ``child.py``, ``script.py`` (an
agent's script) and ``pools.py`` (the threads stopped before a fork).
"""
