"""Read an engine plugin's options (``plugin_config``) and refuse a misspelt one."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from langslice.engines._devices import AUTO
from langslice.plugin_api import PluginError


def known_options(
    plugin: str, config: Mapping[str, object] | None, names: Collection[str]
) -> Mapping[str, object]:
    """Return a plugin's configuration after checking its keys.

    Parameters
    ----------
    plugin
        The plugin's name, for the message.
    config
        The plugin's entry of ``plugin_config``; None is empty.
    names
        The options the plugin takes.

    Returns
    -------
    Mapping
        The configuration given.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    given = config or {}
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise PluginError(
            f"{plugin} takes the options {', '.join(sorted(names))}; got {', '.join(unknown)}"
        )
    return given


def device_option(value: object) -> str:
    """Read a ``device`` option: empty means ``auto``."""
    return str(value) or AUTO
