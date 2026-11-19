"""Tool plugins: the machinery behind ``langslice.plugin_api``.

A plugin is a Python module whose factory returns a list of
:class:`~langslice.core.plugins.ToolSpec`; :mod:`langslice.plugins.loader`
loads it, :mod:`langslice.plugins.context` gives its handlers their
``ToolContext`` and :mod:`langslice.plugins.runtime` makes each tool a
first-class tool of the job's toolbox. Importing the package registers the
loader the core calls for the plugins a ``JobSpec`` names. The names a plugin
author imports are listed in :mod:`langslice.plugin_api`.
"""

from __future__ import annotations

from langslice.plugins import loader

loader.install()
