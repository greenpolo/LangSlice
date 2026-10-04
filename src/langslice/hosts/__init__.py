"""Host connectors that run in LangSlice's own environment.

The top layer of the layered core: the ABBA integration
(:mod:`langslice.hosts.integrations`), the engine service the Fiji connector
starts and the ABBA plugin's worker (:mod:`langslice.hosts.api`), and the
host commands ``abba`` and ``serve`` (:mod:`langslice.hosts.cli`, loaded by
the ``langslice`` command by module path). Nothing below imports a host.
"""
