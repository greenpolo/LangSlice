"""DeepSlice's mouse network: its weights, its model and its prediction.

Vendored from DeepSlice 1.2.8 (Carey et al. 2023, Nature Communications;
https://github.com/PolarBean/DeepSlice, commit
b0d6965ee1861403bc5686eccab4f226c8b15c8b), its ``neural_network`` and
``metadata`` packages and ``DSModel.predict``, mouse only. Copyright (c)
2023 Harry Carey; MIT License, in ``LICENSE`` beside this file.

The network is Keras's Xception (with its ImageNet top, as DeepSlice builds
it) followed by a 256-256-9 dense head that answers a QuickNII anchoring.
Its weights come in three files, fetched once from the EBRAINS bucket
DeepSlice's own metadata names, checked by SHA-256 and kept in a cache
folder (:func:`fetch_weights`): the ImageNet Xception it is built from, and
the two trained networks of the ensemble (``Allen_Mixed_Best.h5``, the
primary, and ``Synthetic_data_final.hdf5``), whose layers are read by name
into the model (:func:`_load_weights`). An image is read by Keras's own
generator at 299 x 299 (nearest neighbour), made grey with scikit-image's
weights in the image's precision and divided by its standard deviation,
one image a batch; with the ensemble the two networks' answers are
averaged.

TensorFlow, ``tf_keras`` and ``h5py`` (``langslice[engines]``) are imported
when a model is built, never when this module loads.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import shutil
import urllib.request
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

logger = logging.getLogger(__name__)

#: Where DeepSlice's metadata keeps its weights (its ``config.json``).
BUCKET = "https://data-proxy.ebrains.eu/api/v1/buckets/deepslice/weights"
#: The weight files, by role: name and SHA-256 of the file DeepSlice 1.2.8
#: downloads (checked against its own run, ``README.md``).
XCEPTION = (
    "xception_weights_tf_dim_ordering_tf_kernels.h5",
    "c5bf1c05b020c4177164039b854c3ef92b16b73384734647c1f0ad5cf79f6975",
)
PRIMARY = (
    "Allen_Mixed_Best.h5",
    "da34a5bca0245314a68daff30c1446758c0851b21370ef9797d5288018b08717",
)
SECONDARY = (
    "Synthetic_data_final.hdf5",
    "b85b7325158d117b2ac7559495c0b50dfcf3545aa29281a345f9ef1e33542e42",
)
#: Where the weights are kept unless the plugin's options say otherwise.
CACHE_FOLDER = Path("~/.cache/langslice/deepslice")
#: How long one download may stall, seconds.
DOWNLOAD_TIMEOUT_S = 600
#: The side, pixels, every image is read at (Xception's input).
SIDE_PX = 299
#: scikit-image's ``rgb2gray`` weights of red, green and blue.
GREY_WEIGHTS = (0.2125, 0.7154, 0.0721)
#: The suffixes DeepSlice reads an image by.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Weights:
    """Hold the paths of DeepSlice's three weight files.

    Parameters
    ----------
    xception
        The ImageNet Xception the model is built from.
    primary
        The ensemble's first network, ``Allen_Mixed_Best.h5``.
    secondary
        Its second, ``Synthetic_data_final.hdf5``.
    """

    xception: Path
    primary: Path
    secondary: Path


@dataclass(frozen=True)
class Prediction:
    """Hold DeepSlice's answer for a list of images.

    Parameters
    ----------
    anchorings
        One QuickNII anchoring per image, in the order given.
    widths, heights
        Each image's size in pixels.
    """

    anchorings: npt.NDArray[np.float64]
    widths: list[int]
    heights: list[int]


class _Variable(Protocol):
    @property
    def name(self) -> str: ...


class _Layer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def weights(self) -> Sequence[_Variable]: ...

    @property
    def layers(self) -> Sequence[_Layer]: ...

    def set_weights(self, weights: list[npt.NDArray[np.float32]]) -> None: ...


class _Model(Protocol):
    @property
    def layers(self) -> Sequence[_Layer]: ...

    def add(self, layer: object) -> None: ...

    def predict(
        self, x: object, *, steps: int, verbose: int
    ) -> npt.NDArray[np.float32]: ...


class _Node(Protocol):
    @property
    def attrs(self) -> dict[str, npt.NDArray[np.bytes_]]: ...

    def __getitem__(self, key: str) -> _Node: ...

    def __array__(self) -> npt.NDArray[np.float32]: ...


class _File(_Node, Protocol):
    def __enter__(self) -> _File: ...

    def __exit__(self, *details: object) -> None: ...


class _Generator(Protocol):
    @property
    def n(self) -> int: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class _Keras:
    """The parts of tf_keras and h5py the network is built and read with."""

    xception: Callable[..., _Layer]
    sequential: Callable[[], _Model]
    dense: Callable[..., object]
    image_generator: Callable[..., object]
    open_h5: Callable[[str, str], _File]


def fetch_weights(cache_folder: Path) -> Weights:
    """Fetch DeepSlice's three weight files into a folder, once.

    Parameters
    ----------
    cache_folder
        The folder that keeps them; a download that fails leaves nothing
        there.

    Returns
    -------
    Weights
        The three files, each checked by its SHA-256.

    Raises
    ------
    ValueError
        When a file fetched is not DeepSlice's (its digest differs).
    OSError
        When a file cannot be fetched or written.
    """
    folder = cache_folder.expanduser()
    return Weights(
        xception=_fetched(folder, XCEPTION),
        primary=_fetched(folder, PRIMARY),
        secondary=_fetched(folder, SECONDARY),
    )


def _fetched(folder: Path, file: tuple[str, str]) -> Path:
    """Return one weight file in the folder, fetching it when missing."""
    name, digest = file
    path = folder / name
    if path.is_file() and _sha256(path) == digest:
        return path

    url = f"{BUCKET}/{name}"
    logger.info("Fetching DeepSlice's weights from %s", url)
    folder.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(name + ".part")
    opened = cast(  # urlopen is untyped: a binary stream
        "IO[bytes]", urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S)
    )
    with opened as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)

    fetched = _sha256(partial)
    if fetched != digest:
        partial.unlink()
        raise ValueError(
            f"the file fetched from {url} has SHA-256 {fetched}, not "
            f"DeepSlice's {name} ({digest})"
        )
    partial.replace(path)
    return path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def predict(
    images: Sequence[Path], weights: Weights, *, ensemble: bool
) -> Prediction:
    """Answer one QuickNII anchoring per image with DeepSlice's network.

    The model is built anew for the call, in Keras's current session
    (the caller clears it first: see ``engine.fresh_keras_session``).

    Parameters
    ----------
    images
        The images, PNG or JPEG.
    weights
        DeepSlice's weight files.
    ensemble
        Whether to average the two networks' answers.

    Returns
    -------
    Prediction
        The anchorings (float64) and the images' sizes.
    """
    keras = _load_keras()
    model = _built(keras, weights)
    generator = _generator(keras, images)
    steps = generator.n
    primary = model.predict(generator, steps=steps, verbose=0)
    anchorings = primary.astype(np.float64)
    if ensemble:
        generator.reset()
        _load_weights(keras, model, weights.secondary)
        secondary = model.predict(generator, steps=steps, verbose=0)
        anchorings = cast(  # the two networks' answers averaged
            "npt.NDArray[np.float64]",
            np.mean([anchorings, secondary], axis=0),
        )

    sizes = [_size(path) for path in images]
    return Prediction(
        anchorings=np.asarray(anchorings, np.float64),
        widths=[width for width, _ in sizes],
        heights=[height for _, height in sizes],
    )


def _load_keras() -> _Keras:
    """Import tf_keras's model parts and h5py."""
    # tf_keras and h5py have no type information here: typed by hand.
    applications = importlib.import_module("tf_keras.applications")
    models = importlib.import_module("tf_keras.models")
    layers = importlib.import_module("tf_keras.layers")
    images = importlib.import_module("tf_keras.preprocessing.image")
    h5py = importlib.import_module("h5py")
    return _Keras(
        xception=cast("Callable[..., _Layer]", applications.Xception),
        sequential=cast("Callable[[], _Model]", models.Sequential),
        dense=cast("Callable[..., object]", layers.Dense),
        image_generator=cast(
            "Callable[..., object]", images.ImageDataGenerator
        ),
        open_h5=cast("Callable[[str, str], _File]", h5py.File),
    )


def _built(keras: _Keras, weights: Weights) -> _Model:
    """Build DeepSlice's mouse model with its primary network's weights."""
    model = keras.sequential()
    model.add(keras.xception(include_top=True, weights=str(weights.xception)))
    model.add(keras.dense(256, activation="relu"))
    model.add(keras.dense(256, activation="relu"))
    model.add(keras.dense(9, activation="linear"))
    _load_weights(keras, model, weights.primary)
    return model


def _load_weights(keras: _Keras, model: _Model, path: Path) -> None:
    """Set the model's layers from one of DeepSlice's weight files.

    The three dense layers by position; the Xception's layers by name,
    each of its weights by the name after the layer's.
    """
    with keras.open_h5(str(path), "r") as file:
        for index, name in enumerate(("dense", "dense_1", "dense_2")):
            group = file[name][name]
            kernel = np.asarray(group["kernel:0"])
            bias = np.asarray(group["bias:0"])
            model.layers[index + 1].set_weights([kernel, bias])

        saved = file["xception"]
        names = cast("list[bytes]", saved.attrs["weight_names"].tolist())
        saved_layers = {name.decode("utf-8").split("/")[0] for name in names}
        for layer in model.layers[0].layers:
            if layer.name in saved_layers:
                group = saved[layer.name]
                keys = [_weight_key(weight.name) for weight in layer.weights]
                layer.set_weights([np.asarray(group[key]) for key in keys])


def _weight_key(name: str) -> str:
    """Return the key a layer's weight is saved under in DeepSlice's file."""
    parts = name.split("/")
    return parts[1] if len(parts) > 1 else f"{name}:0"


def _generator(keras: _Keras, images: Sequence[Path]) -> _Generator:
    """Return Keras's reader of the images, as DeepSlice builds it."""
    pandas = importlib.import_module("pandas")
    frame = cast("Callable[..., object]", pandas.DataFrame)(
        {"Filenames": [str(path) for path in images]}
    )
    with warnings.catch_warnings():
        # Keras warns that the standard-deviation step runs without centring.
        warnings.simplefilter("ignore")
        generator = cast(
            "_DataGenerator",
            keras.image_generator(
                preprocessing_function=_grey,
                samplewise_std_normalization=True,
            ),
        )
        return generator.flow_from_dataframe(
            frame,
            x_col="Filenames",
            y_col=None,
            target_size=(SIDE_PX, SIDE_PX),
            batch_size=1,
            color_mode="rgb",
            shuffle=False,
            class_mode=None,
        )


class _DataGenerator(Protocol):
    @property
    def flow_from_dataframe(self) -> Callable[..., _Generator]: ...


def _grey(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Make an RGB image grey as scikit-image's ``rgb2gray`` does."""
    weights = np.array(GREY_WEIGHTS, dtype=image.dtype)
    return (image @ weights).reshape(SIDE_PX, SIDE_PX, 1)


def _size(path: Path) -> tuple[int, int]:
    """Return an image file's width and height in pixels."""
    with Image.open(path) as image:
        width, height = image.size
    return width, height
