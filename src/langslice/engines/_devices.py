"""Name the torch device an engine runs on in LangSlice's process.

STalign, emlddmm and Brainways take a ``device`` option: ``auto`` (the GPU
when torch sees one, else the CPU), ``cpu``, ``cuda`` or a numbered GPU such
as ``cuda:1``. A fit on a GPU takes the machine's GPU queue
(``langslice.plugin_api.gpu_turn``); whether a device is a GPU is the core's
:func:`~langslice.core.fits.deformable.gpu_queue.on_gpu`.
"""

from __future__ import annotations

import re

from langslice.core.fits.deformable.gpu_queue import on_gpu

__all__ = [
    "AUTO",
    "CPU",
    "DEVICE_OPTION",
    "FIRST_GPU",
    "GPU",
    "cuda_available",
    "gpu_name",
    "on_gpu",
    "resolve_device",
]

AUTO = "auto"
GPU = "cuda"
FIRST_GPU = "cuda:0"
CPU = "cpu"
DEVICE_OPTION = re.compile(r"auto|cpu|cuda(:\d+)?")


def cuda_available() -> bool:
    """Say whether torch sees a GPU.

    Returns
    -------
    bool
        True when torch can run on CUDA here.
    """
    import torch

    return torch.cuda.is_available()


def resolve_device(option: str) -> str:
    """Resolve a device option to the device a run takes.

    Parameters
    ----------
    option
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.

    Returns
    -------
    str
        ``cuda:0`` for ``cuda`` (and for ``auto`` when torch sees a GPU),
        a numbered GPU as given, else the option as given (``cpu`` for
        ``auto`` without a GPU).
    """
    if option == AUTO:
        option = GPU if cuda_available() else CPU
    if option == GPU:
        return FIRST_GPU
    return option


def gpu_name(device: str) -> str | None:
    """Name the GPU a device runs on.

    Parameters
    ----------
    device
        A device as :func:`resolve_device` names it.

    Returns
    -------
    str or None
        The GPU's name for a GPU device; None for the CPU.
    """
    import torch

    if not on_gpu(device):
        return None
    return torch.cuda.get_device_name(torch.device(device))
