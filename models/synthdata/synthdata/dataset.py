from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from PIL import Image

from .augmentation.brightfield_pipeline import render_brightfield_section
from .augmentation.dapi_pipeline import render_dapi_section
from .augmentation.fluorescence_pipeline import render_fluorescence_section
from .augmentation.ish_pipeline import render_ish_section
from .augmentation.modes import FLUORESCENCE_MODES, ISH_MODES
from .augmentation.nissl_pipeline import render_nissl_section
from .augmentation.oblique import get_oblique_slice, sample_oblique_angles
from langslice_harness.atlas.core import get_position_range_mm, load_atlas

__version__ = "0.1.0"


_DEFAULT_MODALITY_WEIGHTS: dict[str, float] = {
    "dapi": 0.45,
    "nissl": 0.22,
    "fluorescence": 0.18,
    "ish": 0.10,
    "brightfield": 0.05,
}

_DEFAULT_PLANE_WEIGHTS: dict[str, float] = {
    "coronal": 0.70,
    "sagittal": 0.20,
    "horizontal": 0.10,
}

_DEFAULT_DAMAGE_INTENSITY_WEIGHTS: dict[str, float] = {
    "light": 0.20,
    "medium": 0.60,
    "heavy": 0.20,
}

_BRIGHTFIELD_MODE_CHOICES: tuple[str | None, ...] = (
    "pan_neuronal",
    "sparse_interneuron",
    "myelin",
    None,
)

_BRIGHTFIELD_COUNTERSTAIN_CHOICES: tuple[str, ...] = (
    "none",
    "hematoxylin",
    "auto",
)

_FLUORESCENCE_MODE_BY_NAME = {mode.name: mode for mode in FLUORESCENCE_MODES}
_ISH_MODE_BY_NAME = {mode.name: mode for mode in ISH_MODES}


@dataclass(frozen=True)
class SynthSpec:
    atlas_name: str
    atlas_version: str
    plane: str
    position_mm: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    modality: str
    mode: str | None
    counterstain: str | None
    damage_intensity: str
    apply_damage: bool
    seed: int
    apply_geometry_warp: bool = True


def _coerce_weight_mapping(
    candidates: Iterable[str],
    weights: Mapping[str, float] | None,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    keys = [str(k) for k in candidates]
    if weights is None:
        raw = {k: float(defaults[k]) for k in keys}
    else:
        raw = {k: float(weights.get(k, 0.0)) for k in keys}
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return {k: v / total for k, v in raw.items()}


def _weighted_choice(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys], dtype=np.float64)
    probs /= probs.sum()
    return str(rng.choice(np.array(keys, dtype=object), p=probs))


def _sample_position_mm(
    rng: np.random.Generator,
    min_mm: float,
    max_mm: float,
    position_strata: str,
) -> float:
    if position_strata == "thirds":
        width = (max_mm - min_mm) / 3.0
        third = int(rng.choice(np.array([0, 1, 2], dtype=np.int32)))
        lo = min_mm + third * width
        hi = min_mm + (third + 1) * width
        return float(rng.uniform(lo, hi))
    if position_strata == "uniform":
        return float(rng.uniform(min_mm, max_mm))
    raise ValueError(
        f"Unsupported position_strata {position_strata!r}; expected 'thirds' or 'uniform'."
    )


def _sample_mode_and_counterstain(
    rng: np.random.Generator,
    modality: str,
) -> tuple[str | None, str | None]:
    if modality in {"dapi", "nissl"}:
        return None, None
    if modality == "brightfield":
        mode = rng.choice(np.array(_BRIGHTFIELD_MODE_CHOICES, dtype=object))
        counterstain = str(
            rng.choice(np.array(_BRIGHTFIELD_COUNTERSTAIN_CHOICES, dtype=object))
        )
        return mode, counterstain
    if modality == "fluorescence":
        mode_names = np.array([m.name for m in FLUORESCENCE_MODES], dtype=object)
        mode_weights = np.array([m.weight for m in FLUORESCENCE_MODES], dtype=np.float64)
        mode_weights /= mode_weights.sum()
        return str(rng.choice(mode_names, p=mode_weights)), None
    if modality == "ish":
        mode_names = np.array([m.name for m in ISH_MODES], dtype=object)
        mode_weights = np.array([m.weight for m in ISH_MODES], dtype=np.float64)
        mode_weights /= mode_weights.sum()
        return str(rng.choice(mode_names, p=mode_weights)), None
    raise ValueError(f"Unsupported modality: {modality!r}")


def _atlas_version(atlas: object) -> str:
    metadata = getattr(atlas, "metadata", {})
    if not isinstance(metadata, Mapping):
        return ""
    version = metadata.get("version") or metadata.get("atlas_version") or ""
    return str(version)


def sample_spec(
    rng,
    *,
    atlases,
    mix_weights=None,
    position_strata="thirds",
    oblique_prob=0.35,
    plane_weights=None,
    damage_intensity_weights=None,
    apply_damage_prob=0.85,
    holdout_atlases=(),
) -> SynthSpec:
    atlas_names = [str(name) for name in atlases]
    held_out = {str(name) for name in holdout_atlases}
    eligible = [name for name in atlas_names if name not in held_out]
    if not eligible:
        raise ValueError("No eligible atlases left after holdout filtering.")

    modality_weights = _coerce_weight_mapping(
        _DEFAULT_MODALITY_WEIGHTS.keys(),
        mix_weights,
        _DEFAULT_MODALITY_WEIGHTS,
    )
    plane_weights_norm = _coerce_weight_mapping(
        _DEFAULT_PLANE_WEIGHTS.keys(),
        plane_weights,
        _DEFAULT_PLANE_WEIGHTS,
    )
    damage_weights = _coerce_weight_mapping(
        _DEFAULT_DAMAGE_INTENSITY_WEIGHTS.keys(),
        damage_intensity_weights,
        _DEFAULT_DAMAGE_INTENSITY_WEIGHTS,
    )

    modality = _weighted_choice(rng, modality_weights)
    mode, counterstain = _sample_mode_and_counterstain(rng, modality)

    atlas_name = str(rng.choice(np.array(eligible, dtype=object)))
    atlas = load_atlas(atlas_name)
    plane = _weighted_choice(rng, plane_weights_norm)
    min_mm, max_mm = get_position_range_mm(atlas, plane=plane)
    position_mm = _sample_position_mm(
        rng,
        float(min_mm),
        float(max_mm),
        str(position_strata),
    )

    if float(rng.random()) < float(oblique_prob):
        yaw_deg, pitch_deg, roll_deg = sample_oblique_angles(rng)
    else:
        yaw_deg, pitch_deg, roll_deg = 0.0, 0.0, 0.0

    damage_intensity = _weighted_choice(rng, damage_weights)
    apply_damage = bool(float(rng.random()) < float(apply_damage_prob))
    seed = int(rng.integers(0, 2**31 - 1))

    return SynthSpec(
        atlas_name=atlas_name,
        atlas_version=_atlas_version(atlas),
        plane=plane,
        position_mm=float(position_mm),
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
        modality=modality,
        mode=mode,
        counterstain=counterstain,
        damage_intensity=damage_intensity,
        apply_damage=apply_damage,
        seed=seed,
    )


def _render_from_slices(
    spec: SynthSpec,
    atlas: object,
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
) -> np.ndarray:
    pixel_size_um = float(min(atlas.resolution[1], atlas.resolution[2]))

    if spec.modality == "dapi":
        return render_dapi_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=spec.seed,
            pixel_size_um=pixel_size_um,
            apply_damage=spec.apply_damage,
            damage_intensity=spec.damage_intensity,
            apply_geometry_warp=spec.apply_geometry_warp,
            plane=spec.plane,
            position_mm=spec.position_mm,
        )

    if spec.modality == "nissl":
        return render_nissl_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=spec.seed,
            pixel_size_um=pixel_size_um,
            cream_base=None,
            apply_damage=spec.apply_damage,
            damage_intensity=spec.damage_intensity,
            apply_geometry_warp=spec.apply_geometry_warp,
            plane=spec.plane,
            position_mm=spec.position_mm,
        )

    if spec.modality == "brightfield":
        counterstain = spec.counterstain if spec.counterstain is not None else "auto"
        return render_brightfield_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=spec.seed,
            pixel_size_um=pixel_size_um,
            mode=spec.mode,
            counterstain=counterstain,
            apply_damage=spec.apply_damage,
            damage_intensity=spec.damage_intensity,
            apply_geometry_warp=spec.apply_geometry_warp,
            plane=spec.plane,
            position_mm=spec.position_mm,
        )

    if spec.modality == "fluorescence":
        if spec.mode is None:
            mode = None
        else:
            mode = _FLUORESCENCE_MODE_BY_NAME.get(spec.mode)
            if mode is None:
                raise ValueError(f"Unknown fluorescence mode {spec.mode!r}")
        return render_fluorescence_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=spec.seed,
            pixel_size_um=pixel_size_um,
            mode=mode,
            apply_damage=spec.apply_damage,
            damage_intensity=spec.damage_intensity,
            apply_geometry_warp=spec.apply_geometry_warp,
            plane=spec.plane,
            position_mm=spec.position_mm,
        )

    if spec.modality == "ish":
        if spec.mode is None:
            mode = None
        else:
            mode = _ISH_MODE_BY_NAME.get(spec.mode)
            if mode is None:
                raise ValueError(f"Unknown ISH mode {spec.mode!r}")
        return render_ish_section(
            reference_slice,
            annotation_slice,
            atlas,
            seed=spec.seed,
            pixel_size_um=pixel_size_um,
            mode=mode,
            apply_damage=spec.apply_damage,
            damage_intensity=spec.damage_intensity,
            apply_geometry_warp=spec.apply_geometry_warp,
            plane=spec.plane,
            position_mm=spec.position_mm,
        )

    raise ValueError(f"Unsupported modality {spec.modality!r}")


def render(spec) -> tuple[np.ndarray, dict]:
    atlas = load_atlas(spec.atlas_name)
    ref_u8, ann_i32 = get_oblique_slice(
        atlas,
        base_position_mm=spec.position_mm,
        plane=spec.plane,
        yaw_deg=spec.yaw_deg,
        pitch_deg=spec.pitch_deg,
        roll_deg=spec.roll_deg,
    )

    try:
        image = _render_from_slices(spec, atlas, ref_u8, ann_i32)
    except Exception:
        if spec.plane != "coronal":
            # Some renderers can fail on non-coronal slices in edge cases.
            # Fall back silently to coronal while preserving oblique angles.
            ref_u8, ann_i32 = get_oblique_slice(
                atlas,
                base_position_mm=spec.position_mm,
                plane="coronal",
                yaw_deg=spec.yaw_deg,
                pitch_deg=spec.pitch_deg,
                roll_deg=spec.roll_deg,
            )
            image = _render_from_slices(spec, atlas, ref_u8, ann_i32)
        else:
            raise

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    metadata = asdict(spec)
    metadata.update(
        {
            "shape": tuple(int(v) for v in image.shape),
            "generator_version": __version__,
        }
    )
    return image, metadata


class SynthIterator:
    def __init__(self, master_seed, atlases, **same_kwargs):
        self._rng = np.random.default_rng(master_seed)
        self._atlases = tuple(str(a) for a in atlases)
        self._kwargs = dict(same_kwargs)

    def __iter__(self) -> Iterator[tuple[np.ndarray, dict]]:
        return self

    def __next__(self) -> tuple[np.ndarray, dict]:
        spec = sample_spec(self._rng, atlases=self._atlases, **self._kwargs)
        return render(spec)


def _save_png(image: np.ndarray, path: Path) -> None:
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def write_dataset(out_dir, n_samples, master_seed, atlases, **same_kwargs) -> None:
    out_root = Path(out_dir)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    dataset_info_path = out_root / "dataset_info.json"
    iterator = SynthIterator(master_seed, atlases, **same_kwargs)

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for _ in range(int(n_samples)):
            image, metadata = next(iterator)
            seed = int(metadata["seed"])
            img_path = images_dir / f"{seed:08d}.png"
            _save_png(image, img_path)
            manifest_f.write(json.dumps(metadata, ensure_ascii=True) + "\n")

    dataset_info = {
        "generator_version": __version__,
        "n_samples": int(n_samples),
        "master_seed": int(master_seed),
        "atlases": [str(a) for a in atlases],
        "sampling_kwargs": _jsonable(same_kwargs),
    }
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")


def _parse_weights_csv(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    out: dict[str, float] = {}
    for chunk in text.split(","):
        name, _, num = chunk.partition("=")
        key = name.strip()
        if not key:
            continue
        out[key] = float(num.strip())
    return out or None


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mix-weights",
        type=str,
        default=None,
        help="CSV weights, e.g. dapi=0.45,nissl=0.22,fluorescence=0.18,ish=0.10,brightfield=0.05",
    )
    parser.add_argument(
        "--position-strata",
        choices=("thirds", "uniform"),
        default="thirds",
        help="Position sampling strategy inside each atlas range",
    )
    parser.add_argument(
        "--oblique-prob",
        type=float,
        default=0.35,
        help="Probability of non-zero oblique tilt",
    )
    parser.add_argument(
        "--plane-weights",
        type=str,
        default=None,
        help="CSV weights, e.g. coronal=0.7,sagittal=0.2,horizontal=0.1",
    )
    parser.add_argument(
        "--damage-intensity-weights",
        type=str,
        default=None,
        help="CSV weights, e.g. light=0.2,medium=0.6,heavy=0.2",
    )
    parser.add_argument(
        "--apply-damage-prob",
        type=float,
        default=0.85,
        help="Probability of apply_damage=True",
    )
    parser.add_argument(
        "--holdout-atlases",
        nargs="*",
        default=(),
        help="Atlas names excluded from sampling",
    )


def _sampling_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mix_weights": _parse_weights_csv(args.mix_weights),
        "position_strata": args.position_strata,
        "oblique_prob": args.oblique_prob,
        "plane_weights": _parse_weights_csv(args.plane_weights),
        "damage_intensity_weights": _parse_weights_csv(args.damage_intensity_weights),
        "apply_damage_prob": args.apply_damage_prob,
        "holdout_atlases": tuple(args.holdout_atlases),
    }


def _cmd_write(args: argparse.Namespace) -> int:
    write_dataset(
        args.out,
        args.n,
        args.seed,
        args.atlases,
        **_sampling_kwargs_from_args(args),
    )
    print(f"Wrote {args.n} samples to {args.out}", flush=True)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    kwargs = _sampling_kwargs_from_args(args)
    spec = sample_spec(rng, atlases=args.atlases, **kwargs)
    image, metadata = render(spec)

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_png(image, out_path)

    print(json.dumps(metadata, indent=2), flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synthdata.dataset",
        description="Synthetic slice dataset generator",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser(
        "write",
        help="Write a synthetic dataset to disk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    write_parser.add_argument("--out", required=True, help="Output directory")
    write_parser.add_argument("--n", type=int, required=True, help="Number of samples")
    write_parser.add_argument("--seed", type=int, required=True, help="Master seed")
    write_parser.add_argument(
        "--atlases",
        nargs="+",
        required=True,
        help="Atlas names to sample from",
    )
    _add_sampling_args(write_parser)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Sample and render one synthetic section",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inspect_parser.add_argument("--seed", type=int, required=True, help="Master seed")
    inspect_parser.add_argument(
        "--atlases",
        nargs="+",
        required=True,
        help="Atlas names to sample from",
    )
    inspect_parser.add_argument("--out", default=None, help="Optional output PNG path")
    _add_sampling_args(inspect_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "write":
            return _cmd_write(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
