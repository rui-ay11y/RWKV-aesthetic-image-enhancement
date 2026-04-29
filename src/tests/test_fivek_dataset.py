from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

pytest.importorskip("torchvision")

from src.data.fivek_dataset import FiveKDataModule, FiveKDataset
from src.data.transforms import to_image_range


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _build_fivek_dirs(
    root: Path,
    *,
    input_dir: str,
    target_dir: str,
    count: int,
) -> None:
    for idx in range(count):
        name = f"a{idx:04d}.jpg"
        _write_image(root / input_dir / name, (idx, idx, idx))
        _write_image(root / target_dir / name, (255 - idx, 0, idx))


def test_fivek_dataset_supports_raw_and_letter_expert_dirs(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="raw", target_dir="c", count=6)

    dataset = FiveKDataset(tmp_path, split="train", expert="C", max_items=3)

    assert dataset.input_dir.name == "raw"
    assert dataset.target_dir.name == "c"
    assert len(dataset) == 3

    inputs, targets = dataset[0]
    assert tuple(inputs.shape) == (3, 8, 8)
    assert tuple(targets.shape) == (3, 8, 8)


def test_fivek_datamodule_respects_split_limits(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=30)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_subset_size=5,
        val_subset_size=2,
        test_subset_size=2,
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    assert len(datamodule._train) == 5
    assert len(datamodule._val) == 2

    datamodule.setup(stage="test")
    assert len(datamodule._test) == 2


def test_fivek_datamodule_keeps_legacy_subset_args(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=50)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_split=4,
        test_split=3,
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    assert len(datamodule._train) == 4
    assert len(datamodule._val) == 3

    datamodule.setup(stage="test")
    assert len(datamodule._test) == 3


def test_fivek_datamodule_uses_three_way_ratio_split(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=20)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    assert len(datamodule._train) == 16
    assert len(datamodule._val) == 2
    assert len(datamodule._test) == 2


def test_to_image_range_uses_explicit_normalization_flag() -> None:
    x = torch.tensor([[[[-1.0, 0.0, 1.0]]]])

    converted = to_image_range(x, normalized_to_neg_one_one=True)
    assert torch.allclose(converted, torch.tensor([[[[0.0, 0.5, 1.0]]]]))

    untouched = to_image_range(converted, normalized_to_neg_one_one=False)
    assert torch.allclose(untouched, converted)


def test_fivek_datamodule_can_disable_minus1_to_1_normalization(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=10)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_subset_size=2,
        val_subset_size=1,
        test_subset_size=1,
        image_size=8,
        crop_size=8,
        normalize_to_neg_one_one=False,
        use_augmentation=False,
        batch_size=1,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    inputs, targets = datamodule._train[0]

    assert 0.0 <= float(inputs.min()) <= 1.0
    assert 0.0 <= float(targets.min()) <= 1.0
    assert 0.0 <= float(inputs.max()) <= 1.0
    assert 0.0 <= float(targets.max()) <= 1.0


def test_fivek_datamodule_can_overfit_on_train_subset(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=20)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_subset_size=4,
        val_subset_size=2,
        test_subset_size=2,
        image_size=8,
        crop_size=8,
        normalize_to_neg_one_one=False,
        overfit_on_train_subset=True,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    assert len(datamodule._train) == 4
    assert len(datamodule._val) == 4
    assert len(datamodule._test) == 4
    assert datamodule._train.file_names == datamodule._val.file_names
    assert datamodule._train.file_names == datamodule._test.file_names
