"""Tests for the separability gate in augmentation/validate.py."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

OFFLINE = os.environ.get("OFFLINE", "") in ("1", "true", "True")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        OFFLINE,
        reason="Skipped in OFFLINE mode (requires ResNet50 download)",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solid_color_image(path: Path, color: tuple[int, int, int], size: int = 32) -> None:
    img = Image.fromarray(
        np.full((size, size, 3), color, dtype=np.uint8)
    )
    img.save(str(path))


def _write_images(
    directory: Path,
    color: tuple[int, int, int],
    n: int,
    prefix: str = "img",
    subject_prefix: str = "subj",
    subjects_per_group: int = 1,
) -> tuple[list[Path], list[str]]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    subjects: list[str] = []
    for i in range(n):
        subj_id = f"{subject_prefix}_{i // (n // max(subjects_per_group, 1))}"
        p = directory / f"{prefix}_{i:04d}.png"
        _make_solid_color_image(p, color)
        paths.append(p)
        subjects.append(subj_id)
    return paths, subjects


# ---------------------------------------------------------------------------
# Distinguishable distributions: red vs blue
# Should produce accuracy well above 0.55, gate fails.
# ---------------------------------------------------------------------------


def test_distinguishable_distributions_fail_gate(tmp_path: Path) -> None:
    from augmentation.validate import run_separability_gate

    real_dir = tmp_path / "real"
    aug_dir = tmp_path / "aug"
    real_paths, real_subjects = _write_images(
        real_dir, (220, 10, 10), 120, prefix="real", subject_prefix="mouse", subjects_per_group=6
    )
    aug_paths, _ = _write_images(aug_dir, (10, 10, 220), 120, prefix="aug")

    result = run_separability_gate(
        "nissl",
        real_image_paths=real_paths,
        real_subject_ids=real_subjects,
        augmented_image_paths=aug_paths,
        n_train_real=50,
        n_test_real=30,
        n_train_aug=50,
        n_test_aug=30,
        seed=7,
    )

    assert result.accuracy > 0.90, f"Expected >0.90 for red vs blue, got {result.accuracy}"
    assert result.pass_gate is False


# ---------------------------------------------------------------------------
# Indistinguishable distributions: both random grey noise
# Should produce accuracy near 0.5, gate passes.
# ---------------------------------------------------------------------------


def test_indistinguishable_distributions_pass_gate(tmp_path: Path) -> None:
    from augmentation.validate import run_separability_gate

    real_dir = tmp_path / "real"
    aug_dir = tmp_path / "aug"

    rng = np.random.default_rng(0)

    real_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)

    real_paths: list[Path] = []
    real_subjects: list[str] = []
    for i in range(120):
        noise = rng.integers(100, 150, size=(32, 32, 3), dtype=np.uint8)
        p = real_dir / f"real_{i:04d}.png"
        Image.fromarray(noise).save(str(p))
        real_paths.append(p)
        real_subjects.append(f"mouse_{i // 20}")

    aug_paths: list[Path] = []
    for i in range(120):
        noise = rng.integers(100, 150, size=(32, 32, 3), dtype=np.uint8)
        p = aug_dir / f"aug_{i:04d}.png"
        Image.fromarray(noise).save(str(p))
        aug_paths.append(p)

    result = run_separability_gate(
        "dapi",
        real_image_paths=real_paths,
        real_subject_ids=real_subjects,
        augmented_image_paths=aug_paths,
        n_train_real=50,
        n_test_real=30,
        n_train_aug=50,
        n_test_aug=30,
        seed=0,
    )

    assert result.accuracy < 0.70, (
        f"Expected near-chance accuracy for identical noise distributions, got {result.accuracy}"
    )
    assert result.pass_gate is True


# ---------------------------------------------------------------------------
# Subject-level split: 10 subjects, assert no overlap
# ---------------------------------------------------------------------------


def test_subject_level_split_no_overlap(tmp_path: Path) -> None:
    from augmentation.validate import _subject_split

    rng = np.random.default_rng(42)
    n_subjects = 10
    imgs_per_subject = 15

    all_paths: list[Path] = []
    all_subjects: list[str] = []
    for s in range(n_subjects):
        for i in range(imgs_per_subject):
            p = tmp_path / f"s{s}" / f"img_{i}.png"
            all_paths.append(p)
            all_subjects.append(f"subject_{s}")

    train_paths, test_paths, train_subjects, test_subjects = _subject_split(
        all_paths, all_subjects, n_train=60, n_test=30, rng=rng
    )

    train_subject_set = set(train_subjects)
    test_subject_set = set(test_subjects)
    overlap = train_subject_set & test_subject_set
    assert len(overlap) == 0, f"Subjects appear in both train and test: {overlap}"


# ---------------------------------------------------------------------------
# Bootstrap CI: lower < accuracy < upper
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_accuracy(tmp_path: Path) -> None:
    from augmentation.validate import run_separability_gate

    real_dir = tmp_path / "real"
    aug_dir = tmp_path / "aug"
    real_paths, real_subjects = _write_images(
        real_dir, (200, 50, 50), 80, subjects_per_group=4
    )
    aug_paths, _ = _write_images(aug_dir, (50, 50, 200), 80)

    result = run_separability_gate(
        "fluorescence",
        real_image_paths=real_paths,
        real_subject_ids=real_subjects,
        augmented_image_paths=aug_paths,
        n_train_real=30,
        n_test_real=30,
        n_train_aug=30,
        n_test_aug=30,
        seed=1,
    )

    lo, hi = result.confidence_interval
    assert lo <= result.accuracy <= hi, (
        f"Accuracy {result.accuracy} outside CI [{lo}, {hi}]"
    )


# ---------------------------------------------------------------------------
# Cache: second extraction call hits cache (no re-forward pass)
# ---------------------------------------------------------------------------


def test_feature_cache_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from augmentation.validate import _extract_features, _load_model

    monkeypatch.setenv("LANGSLICE_DISABLE_FEATURE_CACHE", "0")

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    p = img_dir / "test.png"
    _make_solid_color_image(p, (128, 64, 32))

    cache_dir = tmp_path / "cache"
    import augmentation.validate as av

    monkeypatch.setattr(av, "_CACHE_DIR", cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = _load_model(device)

    feats1 = _extract_features([p], device, model, transform)
    assert (cache_dir / (av._cache_key(p) + ".npy")).exists(), "Cache file not created"

    forward_count = [0]
    original_forward = cast(Callable[..., object], model.__call__)  # type: ignore[attr-defined]

    def counting_forward(*args: object, **kwargs: object) -> object:
        forward_count[0] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "__call__", counting_forward)

    feats2 = _extract_features([p], device, model, transform)
    assert forward_count[0] == 0, "Model was called again despite cache hit"
    np.testing.assert_array_equal(feats1, feats2)


# ---------------------------------------------------------------------------
# Minimum test set guard
# ---------------------------------------------------------------------------


def test_raises_if_test_set_too_small(tmp_path: Path) -> None:
    from augmentation.validate import run_separability_gate

    real_dir = tmp_path / "real"
    aug_dir = tmp_path / "aug"

    real_paths, real_subjects = _write_images(real_dir, (128, 128, 128), 10)
    aug_paths, _ = _write_images(aug_dir, (64, 64, 64), 40)

    with pytest.raises(RuntimeError, match="minimum"):
        run_separability_gate(
            "ish",
            real_image_paths=real_paths,
            real_subject_ids=real_subjects,
            augmented_image_paths=aug_paths,
            n_train_real=5,
            n_test_real=5,
            n_train_aug=20,
            n_test_aug=20,
            seed=0,
        )
