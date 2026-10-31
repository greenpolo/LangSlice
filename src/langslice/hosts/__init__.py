"""Host connectors that run in LangSlice's own environment.

The top layer of the layered core: the LangSlice window
(:mod:`langslice.hosts.gui`), the engine service the Fiji connector starts
(:mod:`langslice.hosts.service`: preparing a job from ABBA and following it),
and the host commands ``gui`` and ``serve`` (:mod:`langslice.hosts.cli`,
loaded by the ``langslice`` command by module path). Nothing below imports a
host.
"""
