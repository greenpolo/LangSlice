"""The doors: thin translators between a caller and the operations.

Phase 5 of the layered refactor. :mod:`langslice.doors.declarations` is every
verb's one declaration (arguments and description); the agent tools and the
MCP server are built from it, and so is the agent CLI
(:mod:`langslice.doors.cli`), the library's job handle
(:mod:`langslice.doors.library`) and the job folder's reference card
(:mod:`langslice.doors.card`). The native agent tools are
:mod:`langslice.doors.tools`, the MCP server :mod:`langslice.doors.mcp`.
"""
