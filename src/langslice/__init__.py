"""LangSlice: register histology sections to BrainGlobe atlases.

The library a script uses (one import path, nothing of the agent framework
loaded)::

    import langslice

    job = langslice.open_job("/data/M04")      # a job folder, or its images
    job.status()                                # the verbs, as methods
    xyz = langslice.coordinate_map(".../view.json")   # atlas µm per pixel
    atlas = langslice.load_atlas("allen_mouse_25um")  # a BrainGlobe atlas

- :func:`open_job` (:mod:`langslice.doors.library`): the job's verbs as
  methods, the same names and arguments as the agent tools and ``langslice
  job FOLDER VERB`` (``langslice ops``, ``langslice schema VERB``).
- :func:`coordinate_map` (:mod:`langslice.core.layers`): a saved picture's
  pixels in atlas micrometres, ``(rows, cols, 3)`` float32, the atlas's own
  axis order, voxel ``i``'s centre at ``i * resolution``.
- :func:`load_atlas` (:mod:`langslice.core.atlas.core`): a cached BrainGlobe atlas.

Each is loaded on first use, so ``import langslice`` stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["__version__", "coordinate_map", "load_atlas", "open_job"]

if TYPE_CHECKING:
    from langslice.core.atlas.core import load_atlas as load_atlas
    from langslice.core.layers import coordinate_map as coordinate_map
    from langslice.doors.library import open_job as open_job

#: The public names, each loaded from its module on first use.
_PUBLIC = {
    "open_job": ("langslice.doors.library", "open_job"),
    "coordinate_map": ("langslice.core.layers", "coordinate_map"),
    "load_atlas": ("langslice.core.atlas.core", "load_atlas"),
}


def __getattr__(name: str) -> Any:
    if name in _PUBLIC:
        import importlib

        module, attribute = _PUBLIC[name]
        value = getattr(importlib.import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'langslice' has no attribute {name!r}")


# Tool-calling Gemini models (especially Flash) routinely return responses
# whose only part is a `function_call` — when google-genai's response.text
# property is accessed under those conditions, it emits a noisy WARNING via
# the `google_genai.types` logger. Our agent loop reads function calls via
# `event.get_function_calls()` directly so the warning is pure noise, but
# it leaks all the way out to the Tauri agent panel and looks like an
# error. Silence just that specific message; other google_genai warnings
# stay visible.
import logging as _logging  # noqa: E402


class _GeminiNonTextPartsFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


_logging.getLogger("google_genai.types").addFilter(_GeminiNonTextPartsFilter())
