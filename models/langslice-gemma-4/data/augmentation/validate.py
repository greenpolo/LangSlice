"""Separability gate: frozen-ResNet50 features + logistic regression classifier.

Pass condition per modality: held-out accuracy <= 0.55.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = ["SeparabilityResult", "run_separability_gate", "run_validate_cli"]

log = logging.getLogger(__name__)

Modality = Literal["dapi", "nissl", "brightfield", "fluorescence", "ish"]

_CACHE_DIR = Path.home() / ".cache" / "langslice" / "aug_features"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PASS_THRESHOLD = 0.55
_BOOTSTRAP_REPS = 200
_MIN_TEST_IMAGES = 30

_MODALITY_TAGS: dict[str, list[str]] = {
    "dapi": ["dapi", "nuclear"],
    "nissl": ["nissl", "cresyl violet", "thionine"],
    "brightfield": ["brightfield ihc", "dab", "ihc"],
    "fluorescence": ["fluorescence"],
    "ish": ["ish", "in situ"],
}
_EXCLUDE_TAGS: dict[str, list[str]] = {
    "brightfield": ["ish", "in situ"],
    "fluorescence": ["dapi"],
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SeparabilityResult:
    modality: str
    accuracy: float
    auc: float
    n_train: int
    n_test: int
    n_real_subjects_train: int
    n_real_subjects_test: int
    pass_gate: bool
    confidence_interval: tuple[float, float]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _cache_key(image_path: Path) -> str:
    stat = image_path.stat()
    raw = f"{image_path!s}:{stat.st_mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_model(device: str) -> tuple[object, object]:
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as tvt

    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()  # type: ignore[assignment]
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    transform = tvt.Compose(
        [
            tvt.Resize(224),
            tvt.CenterCrop(224),
            tvt.ToTensor(),
            tvt.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
        ]
    )
    return model, transform


def _extract_features(
    paths: list[Path],
    device: str,
    model: object,
    transform: object,
) -> np.ndarray:
    import torch
    from PIL import Image

    use_cache = os.environ.get("LANGSLICE_DISABLE_FEATURE_CACHE", "") not in ("1", "true", "True")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    feats: list[np.ndarray] = []
    for p in paths:
        cache_path = _CACHE_DIR / (_cache_key(p) + ".npy")
        if use_cache and cache_path.exists():
            feats.append(np.load(str(cache_path)))
            continue

        img = Image.open(p).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)  # type: ignore[operator]
        with torch.no_grad():
            feat = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)  # type: ignore[operator]

        if use_cache:
            np.save(str(cache_path), feat)
        feats.append(feat)

    return np.stack(feats, axis=0)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_reps: int = _BOOTSTRAP_REPS,
) -> tuple[float, float]:
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_reps):
        idx = rng.integers(0, n, size=n)
        scores.append(float(np.mean(y_true[idx] == y_pred[idx])))
    arr = np.array(scores)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


# ---------------------------------------------------------------------------
# Subject-level split
# ---------------------------------------------------------------------------


def _subject_split(
    paths: list[Path],
    subject_ids: list[str],
    n_train: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[list[Path], list[Path], list[str], list[str]]:
    from collections import defaultdict

    grouped: dict[str, list[Path]] = defaultdict(list)
    for p, s in zip(paths, subject_ids, strict=False):
        grouped[s].append(p)

    subjects = list(grouped.keys())
    rng.shuffle(subjects)  # type: ignore[arg-type]

    test_paths: list[Path] = []
    test_subjects: list[str] = []
    train_paths: list[Path] = []
    train_subjects_used: list[str] = []

    for s in subjects:
        imgs = grouped[s]
        if len(test_paths) < n_test and s not in test_subjects:
            take = imgs[: n_test - len(test_paths)]
            test_paths.extend(take)
            if s not in test_subjects:
                test_subjects.append(s)

    remaining = [s for s in subjects if s not in test_subjects]
    for s in remaining:
        imgs = grouped[s]
        if len(train_paths) >= n_train:
            break
        take = imgs[: n_train - len(train_paths)]
        train_paths.extend(take)
        if s not in train_subjects_used:
            train_subjects_used.append(s)

    if len(test_paths) < _MIN_TEST_IMAGES:
        raise RuntimeError(
            f"Test set has only {len(test_paths)} images (minimum {_MIN_TEST_IMAGES}). "
            "Provide more real-histology data or reduce n_test_real."
        )

    if len(test_paths) < n_test:
        warnings.warn(
            f"Requested {n_test} test images but only {len(test_paths)} available.",
            stacklevel=2,
        )
    if len(train_paths) < n_train:
        warnings.warn(
            f"Requested {n_train} train images but only {len(train_paths)} available.",
            stacklevel=2,
        )

    return train_paths, test_paths, train_subjects_used, test_subjects


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Public gate API
# ---------------------------------------------------------------------------


def run_separability_gate(
    modality: Modality,
    *,
    real_image_paths: list[Path],
    real_subject_ids: list[str],
    augmented_image_paths: list[Path],
    n_train_real: int = 200,
    n_test_real: int = 100,
    n_train_aug: int = 200,
    n_test_aug: int = 100,
    seed: int = 0,
    device: str = "auto",
) -> SeparabilityResult:
    """Train a frozen-ResNet50 + LR classifier to discriminate augmented from real.

    Labels: 0 = real, 1 = augmented.
    Pass condition: held-out accuracy <= 0.55.
    """
    from sklearn.linear_model import LogisticRegression

    dev = _resolve_device(device)
    rng = np.random.default_rng(seed)
    model, transform = _load_model(dev)

    train_real, test_real, train_subj, test_subj = _subject_split(
        real_image_paths, real_subject_ids, n_train_real, n_test_real, rng
    )

    aug_paths = list(augmented_image_paths)
    rng.shuffle(aug_paths)  # type: ignore[arg-type]
    train_aug = aug_paths[: n_train_aug]
    test_aug = aug_paths[n_train_aug : n_train_aug + n_test_aug]

    if len(test_aug) < _MIN_TEST_IMAGES:
        raise RuntimeError(
            f"Augmented test set has only {len(test_aug)} images (minimum {_MIN_TEST_IMAGES})."
        )

    log.info(
        "Extracting features: %d train-real, %d test-real, %d train-aug, %d test-aug",
        len(train_real),
        len(test_real),
        len(train_aug),
        len(test_aug),
    )

    feat_train_real = _extract_features(train_real, dev, model, transform)
    feat_test_real = _extract_features(test_real, dev, model, transform)
    feat_train_aug = _extract_features(train_aug, dev, model, transform)
    feat_test_aug = _extract_features(test_aug, dev, model, transform)

    X_train = np.concatenate([feat_train_real, feat_train_aug], axis=0)
    y_train = np.concatenate(
        [np.zeros(len(feat_train_real)), np.ones(len(feat_train_aug))]
    )
    X_test = np.concatenate([feat_test_real, feat_test_aug], axis=0)
    y_test = np.concatenate(
        [np.zeros(len(feat_test_real)), np.ones(len(feat_test_aug))]
    )

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))

    try:
        from sklearn.metrics import roc_auc_score

        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float("nan")

    ci = _bootstrap_accuracy_ci(y_test, y_pred, rng)

    return SeparabilityResult(
        modality=modality,
        accuracy=accuracy,
        auc=auc,
        n_train=len(X_train),
        n_test=len(X_test),
        n_real_subjects_train=len(train_subj),
        n_real_subjects_test=len(test_subj),
        pass_gate=accuracy <= _PASS_THRESHOLD,
        confidence_interval=ci,
    )


# ---------------------------------------------------------------------------
# Data inventory parser
# ---------------------------------------------------------------------------


def _parse_inventory(inv_path: Path, modality: str) -> list[tuple[Path, str]]:
    """Return (image_path, subject_id) pairs for the given modality.

    Parses the markdown data_inventory.md: each dataset block has a ``Path:``
    line, a ``Staining:`` line, and an ``Imaging:`` line. We match modality tags
    against those fields, then glob for images under the dataset path.

    Subject ID = immediate parent directory name under the dataset root.
    Cap at 500 images to avoid enormous pulls.
    """
    text = inv_path.read_text(encoding="utf-8")
    # The inventory's `Path:` fields are relative to the repo root (e.g.,
    # `data/datasets/foo/`); resolve them from the current working directory.
    base_data = Path.cwd()

    include_tags = _MODALITY_TAGS[modality]
    exclude_tags = _EXCLUDE_TAGS.get(modality, [])

    block_pat = re.compile(
        r"(?m)^####.*?\n(.*?)(?=^####|\Z)",
        re.DOTALL,
    )
    path_pat = re.compile(r"- \*\*Path:\*\*\s+`([^`]+)`")
    staining_pat = re.compile(r"- \*\*Staining:\*\*\s+(.+)")
    imaging_pat = re.compile(r"- \*\*Imaging:\*\*\s+(.+)")

    results: list[tuple[Path, str]] = []

    for match in block_pat.finditer(text):
        block = match.group(1)
        path_m = path_pat.search(block)
        staining_m = staining_pat.search(block)
        imaging_m = imaging_pat.search(block)

        if not path_m:
            continue

        combined = " ".join(
            filter(
                None,
                [
                    staining_m.group(1) if staining_m else "",
                    imaging_m.group(1) if imaging_m else "",
                ],
            )
        ).lower()

        if not any(t.lower() in combined for t in include_tags):
            continue
        if any(t.lower() in combined for t in exclude_tags):
            continue

        raw_path = path_m.group(1).strip()
        # Inventory paths sometimes have `_local/` prefix from older entries;
        # strip it so we resolve from the repo root.
        stripped = re.sub(r"^_local[/\\]", "", raw_path)
        dataset_path = base_data / Path(stripped)

        if not dataset_path.exists():
            log.debug("Dataset path does not exist, skipping: %s", dataset_path)
            continue

        for img_path in dataset_path.rglob("*"):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            subject_id = img_path.parent.name
            results.append((img_path, subject_id))
            if len(results) >= 500:
                break

        if len(results) >= 500:
            break

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_validate_cli(args: object) -> None:
    """Entry point callable from the sister cli.py wave-3 agent."""
    import dataclasses
    import json

    _a = vars(args)
    inv_path = Path(_a["real_source"])
    modality: str = _a["modality"]
    aug_dir = Path(_a["augmented_dir"])
    report_path = Path(_a["report"]) if _a.get("report") else None
    seed: int = int(_a.get("seed", 0))  # type: ignore[arg-type]
    device: str = _a.get("device", "auto")  # type: ignore[assignment]

    pairs = _parse_inventory(inv_path, modality)
    if not pairs:
        raise RuntimeError(
            f"No datasets matched modality '{modality}' in {inv_path}. "
            "Check Staining/Imaging tags in the inventory."
        )

    real_paths = [p for p, _ in pairs]
    subject_ids = [s for _, s in pairs]
    log.info("Found %d real images for modality '%s'", len(real_paths), modality)

    aug_paths = sorted(
        p for p in aug_dir.glob("**/*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not aug_paths:
        log.warning(
            "No augmented images found in %s — smoke test only; results will not be meaningful.",
            aug_dir,
        )
        aug_paths = []

    result = run_separability_gate(
        modality,  # type: ignore[arg-type]
        real_image_paths=real_paths,
        real_subject_ids=subject_ids,
        augmented_image_paths=aug_paths,
        seed=seed,
        device=device,
    )

    status = "PASS" if result.pass_gate else "FAIL"
    print(
        f"[{status}] modality={result.modality} accuracy={result.accuracy:.4f} "
        f"auc={result.auc:.4f} ci=({result.confidence_interval[0]:.4f}, "
        f"{result.confidence_interval[1]:.4f}) "
        f"n_train={result.n_train} n_test={result.n_test}"
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(result), fh, indent=2)
        print(f"Report written to {report_path}")


def _build_argparser() -> object:
    import argparse

    p = argparse.ArgumentParser(
        description="Run the separability gate for one augmentation modality."
    )
    p.add_argument(
        "--modality",
        required=True,
        choices=["dapi", "nissl", "brightfield", "fluorescence", "ish"],
    )
    p.add_argument(
        "--real-source",
        required=True,
        help="Path to _local/eval/data_inventory.md",
    )
    p.add_argument(
        "--augmented-dir",
        required=True,
        help="Directory containing augmented atlas images.",
    )
    p.add_argument("--report", default=None, help="Optional path to write JSON report.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = _build_argparser()
    run_validate_cli(parser.parse_args())  # type: ignore[arg-type]
