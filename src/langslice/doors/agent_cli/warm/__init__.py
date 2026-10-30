"""The agent CLI's warm process: one fork server per job folder.

What a user sees of it (which calls run cold, when it exits, where its
socket is) is ``docs/agents/cli.md``, "The warm process"; ``README.md`` lists
the private state of other libraries it touches. In the code:

- The client (:func:`langslice.doors.agent_cli.warm.client.run`, the first
  thing ``langslice-job`` does; standard library only) hands the call's argv,
  working folder, environment and the file descriptors of its stdin, stdout and
  stderr (``SCM_RIGHTS``) to the job's socket, then ends with the child's exit
  status (a child killed by a signal: the client kills itself with that
  signal). Signals it gets go to the child's process group; a client that dies
  takes the child with it.
- The server (:func:`langslice.doors.agent_cli.warm.server.serve_main`) holds
  ``<key>.lock`` for its life, loads the libraries and the atlas
  (:func:`langslice.doors.agent_cli.warm.server.warm_up`), shares every placed
  section's frame with the children, and answers a call ``stale`` once any
  first-party module, site-packages folder or the atlas changed.
- A child (``child.py``) becomes the caller's process (streams, argv,
  ``sys.path``, environment, working folder), runs the cold call's code
  (:func:`langslice.doors.agent_cli.main.run`) and ends as Python ends;
  ``script.py`` runs an agent's script in it. ``protocol.py`` is where a job's
  warm process listens and the messages; ``pools.py`` stops the threads loading
  left behind before a fork.
"""
