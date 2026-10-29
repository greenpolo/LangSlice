"""The ``langslice`` command's entry point (``langslice.cli:main``).

The commands live in :mod:`langslice.doors.commands.parser`, one module per
command group.
"""

from langslice.doors.commands.parser import main

__all__ = ["main"]
