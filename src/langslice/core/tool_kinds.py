"""The kinds and groups a tool is listed under: plain names, no imports.

A plugin's tools (:mod:`langslice.core.plugins`) and the native ones share
them; the names are here alone so the tool table stays light to import.
"""

from __future__ import annotations

from typing import Literal

#: Whether a tool only reads the job or writes it.
Kind = Literal["read", "write"]
#: The groups the tools are listed in.
Group = Literal["Common", "Positioning", "Linear", "Nonlinear"]
#: The groups, in the order every door lists them.
GROUPS: tuple[Group, ...] = ("Common", "Positioning", "Linear", "Nonlinear")
