"""The kinds and groups a tool is listed under: plain names, no imports.

A plugin's tools (:mod:`langslice.core.plugins`) and the native ones share
them; the names are here alone so the tool table stays light to import.
"""

from __future__ import annotations

from typing import Literal

Kind = Literal["read", "write"]
Group = Literal["Common", "Positioning", "Linear", "Nonlinear"]
GROUPS: tuple[Group, ...] = ("Common", "Positioning", "Linear", "Nonlinear")
