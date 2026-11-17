"""Find a section's position along the slicing axis with Brainways' network.

Vendored from Brainways 0.1.16.3 (Kantor et al. 2025, Neuropsychopharmacology;
https://github.com/bkntr/brainways, commit
1c43b0b708564974ce057167b6f83ed967dd3b04). Copyright (C) Ben Kantor and the
Brainways contributors. This program is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. It is distributed WITHOUT ANY
WARRANTY; see the GNU General Public License in ``LICENSE`` beside this file.

Brainways' network (``brainways.model.siamese``) is a siamese pair: one
backbone embeds the section and an atlas plane, and a small head says from
the two embeddings whether the section lies before or after the plane
(``tanh`` of one number: its sign). ``SiameseModel.predict`` runs a binary
search over the atlas planes between Brainways' limits with it, and the
position is the plane the search ends on. :func:`find_ap` is that search.

The backbone is timm's ``tf_efficientnetv2_s`` (``features_only``, its last
stage, 256 channels), as Brainways builds it without pretrained weights;
here it is written as functional torch on the checkpoint's own tensors:
TensorFlow's "same" padding on the strided convolutions, batch norm with
epsilon 1e-3, SiLU, squeeze-and-excitation on the inverted residuals. The
head is Brainways': the two embeddings averaged over the plane,
concatenated, then a 512-256-1 perceptron.

The weights are Brainways' own checkpoint, read from the Hugging Face
repository ``brainways/siamese`` at a fixed revision and checked by its
SHA-256 (:func:`fetch_weights`).
"""

from __future__ import annotations

import enum
import functools
import hashlib
import logging
import math
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from langslice.engines.brainways.images import plane_input, section_input

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class Device(enum.StrEnum):
    """Name where the network runs: ``auto`` is the GPU when torch sees one."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


REPOSITORY = "brainways/siamese"
REVISION = "a0f3b1957b59b2ca2c02f7d094725c6dcb77783d"
MODEL_ID = "5redxpw6"
WEIGHTS_SHA256 = (
    "a3835674e54dac85b4a89533405a8ad7a90ca6f862599423c58747871689f0fd"
)
DEFAULT_CACHE = Path("~/.cache/langslice/brainways")
DOWNLOAD_TIMEOUT_S = 120
AP_LIMITS = {"whs_sd_rat_39um": (122, 800)}
BN_EPS = 1e-3
BN_MOMENTUM = 0.1
STAGES = (
    ("cn", 2, 1, 24),
    ("er", 4, 2, 48),
    ("er", 4, 2, 64),
    ("ir", 6, 2, 128),
    ("ir", 9, 1, 160),
    ("ir", 15, 2, 256),
)
STEM_CHANNELS = 24
BACKBONE = "_backbone._model."
BN_PARTS = ("running_mean", "running_var", "weight", "bias")


@dataclass(frozen=True)
class Network:
    """Brainways' network, loaded on one device.

    Attributes
    ----------
    weights
        The checkpoint's tensors by their names in it, on ``device``.
    device
        The torch device the network runs on, ``cuda`` or ``cpu``.
    sha256
        The checkpoint file's SHA-256.
    """

    weights: dict[str, torch.Tensor] = field(repr=False)
    device: Device
    sha256: str


@dataclass(frozen=True)
class ReferenceVolume:
    """Hold the atlas reference volume Brainways compares a section with.

    Attributes
    ----------
    voxels : array, shape (AP, DV, LR)
        The atlas's reference image, its axis 0 the slicing axis.
    scale
        The volume's largest value as float32, by which Brainways divides
        it (``BrainwaysAtlas.raw_numpy_reference``).
    """

    voxels: npt.NDArray[np.generic]
    scale: np.float32

    @classmethod
    def from_voxels(cls, voxels: npt.NDArray[np.generic]) -> ReferenceVolume:
        """Wrap an atlas reference volume with its scale.

        Parameters
        ----------
        voxels : array, shape (AP, DV, LR)
            The atlas's reference image, axis 0 along the slicing axis.

        Returns
        -------
        ReferenceVolume
            The volume and its largest value as float32.
        """
        return cls(voxels, np.float32(np.max(voxels)))

    def plane(self, ap: int) -> npt.NDArray[np.float32]:
        """Read one plane as Brainways' ``raw_numpy_reference`` holds it.

        Parameters
        ----------
        ap
            The plane's index along axis 0.

        Returns
        -------
        array, shape (DV, LR), float32
            The plane divided by the volume's largest value.
        """
        plane = cast("npt.NDArray[np.generic]", self.voxels[ap])
        divided = plane.astype(np.float32) / self.scale
        return np.asarray(divided, dtype=np.float32)


def resolve_device(option: Device) -> Device:
    """Choose the torch device for a ``device`` option.

    Parameters
    ----------
    option
        ``auto`` takes the GPU when torch sees one, as Brainways does.

    Returns
    -------
    Device
        ``cuda`` or ``cpu``.
    """
    import torch

    if option != Device.AUTO:
        return option
    if torch.cuda.is_available():
        return Device.CUDA
    return Device.CPU


def cuda_available() -> bool:
    """Say whether torch sees a GPU.

    Returns
    -------
    bool
        True when torch can run on CUDA here.
    """
    import torch

    return torch.cuda.is_available()


def gpu_name(device: Device) -> str | None:
    """Name the GPU a device runs on.

    Parameters
    ----------
    device
        The torch device the network runs on.

    Returns
    -------
    str or None
        The first GPU's name for ``cuda``; None for ``cpu``.
    """
    import torch

    if device != Device.CUDA:
        return None
    return torch.cuda.get_device_name(0)


def torch_version() -> str:
    """Give the version of torch the network runs on.

    Returns
    -------
    str
        ``torch.__version__``.
    """
    import torch

    return str(torch.__version__)


def fetch_weights(cache_folder: Path) -> Path:
    """Fetch Brainways' checkpoint into a folder, once.

    Parameters
    ----------
    cache_folder
        The folder that keeps it, under ``<revision>/<model id>/``; a
        download that fails leaves nothing there.

    Returns
    -------
    Path
        The checkpoint, ``state_dict.pt``, its SHA-256 checked.

    Raises
    ------
    ValueError
        When the file fetched is not the checkpoint (its digest differs).
    """
    path = cache_folder.expanduser() / REVISION / MODEL_ID / "state_dict.pt"
    if path.is_file() and _sha256(path) == WEIGHTS_SHA256:
        return path

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = (
        f"{endpoint}/{REPOSITORY}/resolve/{REVISION}/{MODEL_ID}/state_dict.pt"
    )
    logger.info("Fetching Brainways' weights from %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    opened = cast(
        "IO[bytes]", urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S)
    )
    with opened as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)

    digest = _sha256(partial)
    if digest != WEIGHTS_SHA256:
        partial.unlink()
        raise ValueError(
            f"the file fetched from {url} has SHA-256 {digest}, not "
            f"Brainways' checkpoint {WEIGHTS_SHA256}"
        )
    partial.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=2)
def load_network(device: Device, *, cache_folder: str = "") -> Network:
    """Load Brainways' network, fetching its weights on first use.

    The network stays loaded for the life of the process, one per device
    and folder.

    Parameters
    ----------
    device
        The torch device to run it on, ``cuda`` or ``cpu``.
    cache_folder
        The folder that keeps the checkpoint; empty is
        ``~/.cache/langslice/brainways``.

    Returns
    -------
    Network
        The checkpoint's tensors on ``device``, with its digest.

    Raises
    ------
    ValueError
        When the checkpoint does not hold the tensors this network reads.
    """
    import torch

    path = fetch_weights(Path(cache_folder or DEFAULT_CACHE))
    state = cast(
        "dict[str, torch.Tensor]",
        torch.load(path, map_location="cpu", weights_only=True),
    )
    names = tensor_names()
    missing = [name for name in names if name not in state]
    if missing:
        raise ValueError(
            f"Brainways' checkpoint {path} lacks {len(missing)} tensors of "
            f"the network, first {missing[0]}"
        )
    weights = {name: state[name].to(device) for name in names}
    return Network(weights, device, WEIGHTS_SHA256)


def tensor_names() -> list[str]:
    """List the name of every tensor the network reads, as stored.

    Returns
    -------
    list of str
        The backbone's (stem, then each block of each stage) and the
        head's names in Brainways' checkpoint, without the batch norms'
        ``num_batches_tracked``.
    """
    names = [BACKBONE + "conv_stem.weight", *_bn_names(BACKBONE + "bn1")]
    for stage, (kind, repeats, _stride, _channels) in enumerate(STAGES):
        for index in range(repeats):
            prefix = f"{BACKBONE}blocks.{stage}.{index}."
            names += _block_names(prefix, kind)
    for layer in ("classifier.0", "classifier.2"):
        names += [f"{layer}.weight", f"{layer}.bias"]
    return names


def _bn_names(prefix: str) -> list[str]:
    return [f"{prefix}.{part}" for part in BN_PARTS]


def _block_names(prefix: str, kind: str) -> list[str]:
    if kind == "cn":
        return [prefix + "conv.weight", *_bn_names(prefix + "bn1")]
    if kind == "er":
        return [
            prefix + "conv_exp.weight",
            *_bn_names(prefix + "bn1"),
            prefix + "conv_pwl.weight",
            *_bn_names(prefix + "bn2"),
        ]
    return [
        prefix + "conv_pw.weight",
        *_bn_names(prefix + "bn1"),
        prefix + "conv_dw.weight",
        *_bn_names(prefix + "bn2"),
        prefix + "se.conv_reduce.weight",
        prefix + "se.conv_reduce.bias",
        prefix + "se.conv_expand.weight",
        prefix + "se.conv_expand.bias",
        prefix + "conv_pwl.weight",
        *_bn_names(prefix + "bn3"),
    ]


def find_ap(
    network: Network,
    thumbnail: npt.NDArray[np.uint8],
    planes: AtlasEmbeddings,
    limits: tuple[int, int],
) -> int:
    """Search the atlas planes for a section's position, as Brainways does.

    Parameters
    ----------
    network
        Brainways' network, loaded on its device.
    thumbnail : array, uint8
        The section's thumbnail (``images.section_thumbnail``).
    planes
        The atlas planes' embeddings, made on ``network``'s device.
    limits
        The first and last plane the search may end on (axis-0 indices).

    Returns
    -------
    int
        The plane the search ends on, an index along axis 0.
    """
    section = embed(network, section_input(thumbnail))
    low, high = limits
    while low < high:
        mid = (low + high) // 2
        if relative_position(network, section, planes.at(mid)) < 0:
            high = mid - 1
        else:
            low = mid + 1
    return low


class AtlasEmbeddings:
    """Embed atlas planes on demand, each once.

    One object serves one tool call: its planes' embeddings stay in it
    until the call drops it.

    Parameters
    ----------
    network
        The network that embeds them.
    volume
        The atlas reference volume the planes are read from.
    """

    def __init__(self, network: Network, volume: ReferenceVolume) -> None:
        self._network = network
        self._volume = volume
        self._embedded: dict[int, torch.Tensor] = {}

    def at(self, ap: int) -> torch.Tensor:
        """Embed the plane at an index along the slicing axis.

        Parameters
        ----------
        ap
            The plane's index along axis 0 of the volume.

        Returns
        -------
        torch.Tensor, shape (1, 256)
            The plane's embedding.
        """
        if ap not in self._embedded:
            image = plane_input(self._volume.plane(ap))
            self._embedded[ap] = embed(self._network, image)
        return self._embedded[ap]


def embed(network: Network, image: npt.NDArray[np.float32]) -> torch.Tensor:
    """Embed one network input with the backbone.

    Parameters
    ----------
    network
        Brainways' network, loaded on its device.
    image : array, shape (3, 224, 224), float32
        The input (``images.section_input`` or ``images.plane_input``).

    Returns
    -------
    torch.Tensor, shape (1, 256)
        The backbone's last stage averaged over the plane.
    """
    import torch
    from torch.nn import functional

    batch = torch.as_tensor(image)[np.newaxis].to(network.device)
    with torch.no_grad():
        features = _backbone(network.weights, batch)
        return functional.adaptive_avg_pool2d(features, 1).flatten(1)


def relative_position(
    network: Network, section: torch.Tensor, plane: torch.Tensor
) -> float:
    """Say whether a section lies before or after an atlas plane.

    Parameters
    ----------
    network
        Brainways' network, loaded on its device.
    section : torch.Tensor, shape (1, 256)
        The section's embedding.
    plane : torch.Tensor, shape (1, 256)
        The plane's embedding.

    Returns
    -------
    float
        ``tanh`` of the head's output: negative when the section lies
        before the plane (a lower index), zero or positive otherwise.
    """
    import torch
    from torch.nn import functional

    weights = network.weights
    with torch.no_grad():
        pair = torch.cat([section, plane], dim=1)
        hidden = functional.linear(
            pair, weights["classifier.0.weight"], weights["classifier.0.bias"]
        )
        logit = functional.linear(
            functional.relu(hidden),
            weights["classifier.2.weight"],
            weights["classifier.2.bias"],
        )
        return float(torch.tanh(logit.squeeze(1))[0].item())


def _backbone(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    x = _bn_act(
        weights,
        BACKBONE + "bn1",
        _conv(x, weights[BACKBONE + "conv_stem.weight"], 2),
    )
    in_channels = STEM_CHANNELS
    for stage, (kind, repeats, stride, channels) in enumerate(STAGES):
        for index in range(repeats):
            prefix = f"{BACKBONE}blocks.{stage}.{index}."
            block_stride = stride if index == 0 else 1
            skip = block_stride == 1 and in_channels == channels
            x = BLOCKS[kind](weights, prefix, x, block_stride, skip)
            in_channels = channels
    return x


def _conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    *,
    bias: torch.Tensor | None = None,
    groups: int = 1,
) -> torch.Tensor:
    from torch.nn import functional

    kernel = weight.shape[-1]
    if stride == 1:
        return functional.conv2d(
            x, weight, bias, 1, (kernel - 1) // 2, 1, groups
        )
    pad_h = _same_padding(x.shape[-2], kernel, stride)
    pad_w = _same_padding(x.shape[-1], kernel, stride)
    padded = functional.pad(
        x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    )
    return functional.conv2d(padded, weight, bias, stride, 0, 1, groups)


def _same_padding(size: int, kernel: int, stride: int) -> int:
    return max((math.ceil(size / stride) - 1) * stride + kernel - size, 0)


def _bn(
    weights: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
) -> torch.Tensor:
    from torch.nn import functional

    mean, var, scale, shift = (
        weights[f"{prefix}.{part}"] for part in BN_PARTS
    )
    return functional.batch_norm(
        x, mean, var, scale, shift, False, BN_MOMENTUM, BN_EPS
    )


def _bn_act(
    weights: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
) -> torch.Tensor:
    from torch.nn import functional

    return functional.silu(_bn(weights, prefix, x))


def _conv_block(
    weights: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    stride: int,
    skip: bool,
) -> torch.Tensor:
    out = _bn_act(
        weights,
        prefix + "bn1",
        _conv(x, weights[prefix + "conv.weight"], stride),
    )
    if skip:
        return out + x
    return out


def _fused_block(
    weights: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    stride: int,
    skip: bool,
) -> torch.Tensor:
    out = _bn_act(
        weights,
        prefix + "bn1",
        _conv(x, weights[prefix + "conv_exp.weight"], stride),
    )
    out = _bn(
        weights,
        prefix + "bn2",
        _conv(out, weights[prefix + "conv_pwl.weight"], 1),
    )
    if skip:
        return out + x
    return out


def _inverted_block(
    weights: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    stride: int,
    skip: bool,
) -> torch.Tensor:
    out = _bn_act(
        weights,
        prefix + "bn1",
        _conv(x, weights[prefix + "conv_pw.weight"], 1),
    )
    depthwise = weights[prefix + "conv_dw.weight"]
    out = _bn_act(
        weights,
        prefix + "bn2",
        _conv(out, depthwise, stride, groups=depthwise.shape[0]),
    )
    out = _squeeze_excite(weights, prefix + "se.", out)
    out = _bn(
        weights,
        prefix + "bn3",
        _conv(out, weights[prefix + "conv_pwl.weight"], 1),
    )
    if skip:
        return out + x
    return out


def _squeeze_excite(
    weights: dict[str, torch.Tensor], prefix: str, x: torch.Tensor
) -> torch.Tensor:
    import torch
    from torch.nn import functional

    pooled = x.mean((2, 3), keepdim=True)
    reduced = functional.silu(
        _conv(
            pooled,
            weights[prefix + "conv_reduce.weight"],
            1,
            bias=weights[prefix + "conv_reduce.bias"],
        )
    )
    gate = _conv(
        reduced,
        weights[prefix + "conv_expand.weight"],
        1,
        bias=weights[prefix + "conv_expand.bias"],
    )
    return x * torch.sigmoid(gate)


BLOCKS = {"cn": _conv_block, "er": _fused_block, "ir": _inverted_block}
