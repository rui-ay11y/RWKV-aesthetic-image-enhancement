"""MIT-Adobe FiveK dataset and PyTorch Lightning DataModule."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import get_train_transforms, get_val_transforms

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - fallback for lightweight tests
    class _LightningDataModule:
        pass

    class _PLNamespace:
        LightningDataModule = _LightningDataModule

    pl = _PLNamespace()


def _pick_existing_dir(
    data_dir: Path,
    candidates: list[str],
    description: str,
) -> Path:
    for name in candidates:
        path = data_dir / name
        if path.is_dir():
            return path

    joined = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not find {description} under {data_dir}. Tried: {joined}"
    )


def _resolve_input_dir(data_dir: Path) -> Path:
    return _pick_existing_dir(
        data_dir,
        ["input", "raw"],
        "an input image directory",
    )


def _resolve_target_dir(data_dir: Path, expert: str) -> Path:
    expert_upper = expert.upper()
    expert_lower = expert.lower()
    return _pick_existing_dir(
        data_dir,
        [
            f"expert{expert_lower}",
            expert_lower,
            f"expert{expert_upper}",
            expert_upper,
        ],
        f"a target directory for expert {expert_upper}",
    )


def _scan_all_names(data_dir: Path) -> list[str]:
    input_dir = _resolve_input_dir(data_dir)
    return sorted(
        p.stem
        for p in input_dir.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _load_split_file(path: Path) -> Optional[list[str]]:
    if not path.exists():
        return None
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


class FiveKDataset(Dataset):
    """MIT-Adobe FiveK single-split dataset.

    Supported directory layouts::

        data_dir/
        ├── input/ or raw/  # Raw unprocessed images
        │   ├── a0001.png
        │   └── ...
        ├── expertC/ or c/  # Expert C retouched images (ground truth)
        │   ├── a0001.png
        │   └── ...
        └── splits/
            ├── train.txt   # 4500 filenames (no extension)
            └── test.txt    # 500  filenames (no extension)

    When splits/ does not exist the dataset auto-splits by index order.

    Args:
        data_dir: Root directory of the FiveK dataset.
        split: One of 'train' or 'test'.
        expert: Expert identifier letter, default 'C'.
        transform: Transform applied to both input and target images.
                   Must be deterministic OR the caller handles paired seeding.
        max_items: Optional cap on the number of filenames loaded.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        expert: str = "C",
        transform=None,
        max_items: Optional[int] = None,
        file_names: Optional[list[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.expert = expert.upper()
        self.transform = transform

        self.input_dir = _resolve_input_dir(self.data_dir)
        self.target_dir = _resolve_target_dir(self.data_dir, self.expert)
        if file_names is not None:
            self.file_names = file_names[:max_items] if max_items is not None else file_names
        else:
            self.file_names = self._load_file_list(split, max_items=max_items)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file_list(
        self,
        split: str,
        max_items: Optional[int] = None,
    ) -> list[str]:
        split_file = self.data_dir / "splits" / f"{split}.txt"
        if split_file.exists():
            names = _load_split_file(split_file) or []
        else:
            # Fallback: scan input/ and split by index. The canonical FiveK
            # release contains 5000 images. When explicit split files are not
            # provided, keep the dataset fallback simple and deterministic.
            all_names = _scan_all_names(self.data_dir)
            if len(all_names) < 3:
                names = all_names
            else:
                train_end = int(len(all_names) * 0.8)
                val_end = train_end + int(len(all_names) * 0.1)
                if split == "train":
                    names = all_names[:train_end]
                elif split == "val":
                    names = all_names[train_end:val_end]
                else:
                    names = all_names[val_end:]

        if max_items is not None:
            return names[:max_items]
        return names

    def _find_image(self, directory: Path, stem: str) -> Path:
        """Return path to image file, trying .png then .jpg."""
        for ext in (".png", ".jpg", ".jpeg"):
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found: {directory / stem}.*")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        name = self.file_names[idx]
        input_path = self._find_image(self.input_dir, name)
        target_path = self._find_image(self.target_dir, name)

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            # Use a shared seed so random crop & flip are identical for
            # both input and target (preserving spatial correspondence).
            seed = torch.randint(0, 2 ** 31, (1,)).item()
            torch.manual_seed(seed)
            input_tensor: Tensor = self.transform(input_img)
            torch.manual_seed(seed)
            target_tensor: Tensor = self.transform(target_img)
        else:
            import torchvision.transforms.functional as TF
            input_tensor = TF.to_tensor(input_img)
            target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor


class FiveKDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MIT-Adobe FiveK.

    Args:
        data_dir: Path to FiveK dataset root. Accepts env-var via Hydra.
        expert: Expert letter used as ground truth (default 'C').
        train_ratio: Fraction of images assigned to the training split.
        val_ratio: Fraction of images assigned to the validation split.
        test_ratio: Fraction of images assigned to the test split.
        train_split: Legacy cap on the number of training images to use.
        val_split: Legacy cap on the number of validation images to use.
        test_split: Legacy cap on the number of test images to use.
        train_subset_size: Preferred alias for the training subset size.
        val_subset_size: Preferred alias for the validation subset size.
        test_subset_size: Preferred alias for the test subset size.
        image_size: Resize shorter edge to this value before cropping.
        crop_size: Random crop size during training; also val resize target.
        overfit_on_train_subset: Reuse the training subset for val/test to run
            a true small-set overfit sanity check.
        use_augmentation: Enable random crop + horizontal flip during training.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin CPU memory for faster GPU transfer.
    """

    def __init__(
        self,
        data_dir: str = "data/fivek",
        expert: str = "C",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        train_split: Optional[int] = None,
        val_split: Optional[int] = None,
        test_split: Optional[int] = None,
        train_subset_size: Optional[int] = None,
        val_subset_size: Optional[int] = None,
        test_subset_size: Optional[int] = None,
        image_size: int = 480,
        crop_size: int = 256,
        normalize_to_neg_one_one: bool = True,
        overfit_on_train_subset: bool = False,
        use_augmentation: bool = True,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.expert = expert
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio <= 0:
            raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
        self.train_ratio = train_ratio / total_ratio
        self.val_ratio = val_ratio / total_ratio
        self.test_ratio = test_ratio / total_ratio
        # Keep the legacy split cap fields for backward
        # compatibility, but prefer the explicit subset_size names.
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.train_subset_size = train_subset_size if train_subset_size is not None else train_split
        legacy_val_cap = val_split if val_split is not None else test_split
        self.val_subset_size = val_subset_size if val_subset_size is not None else legacy_val_cap
        self.test_subset_size = (
            test_subset_size if test_subset_size is not None else test_split
        )
        self.image_size = image_size
        self.crop_size = crop_size
        self.normalize_to_neg_one_one = normalize_to_neg_one_one
        self.overfit_on_train_subset = overfit_on_train_subset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._train: Optional[FiveKDataset] = None
        self._val: Optional[FiveKDataset] = None
        self._test: Optional[FiveKDataset] = None

    def _compute_split_names(self) -> tuple[list[str], list[str], list[str]]:
        splits_dir = self.data_dir / "splits"
        train_names = _load_split_file(splits_dir / "train.txt")
        val_names = _load_split_file(splits_dir / "val.txt")
        test_names = _load_split_file(splits_dir / "test.txt")

        if train_names is not None and val_names is not None and test_names is not None:
            return train_names, val_names, test_names

        all_names = _scan_all_names(self.data_dir)
        num_items = len(all_names)
        if num_items < 3:
            return all_names, all_names, all_names

        train_end = int(num_items * self.train_ratio)
        val_end = train_end + int(num_items * self.val_ratio)

        # Ensure each split gets at least one sample when possible.
        train_end = max(1, min(train_end, num_items - 2))
        val_end = max(train_end + 1, min(val_end, num_items - 1))

        return (
            all_names[:train_end],
            all_names[train_end:val_end],
            all_names[val_end:],
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_tfm = (
            get_train_transforms(
                self.image_size,
                self.crop_size,
                normalize_to_neg_one_one=self.normalize_to_neg_one_one,
            )
            if self.use_augmentation
            else get_val_transforms(
                self.crop_size,
                normalize_to_neg_one_one=self.normalize_to_neg_one_one,
            )
        )
        val_tfm = get_val_transforms(
            self.crop_size,
            normalize_to_neg_one_one=self.normalize_to_neg_one_one,
        )
        train_names, val_names, test_names = self._compute_split_names()

        if stage in ("fit", None):
            fit_train_names = (
                train_names[:self.train_subset_size]
                if self.train_subset_size is not None
                else train_names
            )
            fit_val_names = (
                fit_train_names
                if self.overfit_on_train_subset
                else (
                    val_names[:self.val_subset_size]
                    if self.val_subset_size is not None
                    else val_names
                )
            )
            self._train = FiveKDataset(
                self.data_dir,
                "train",
                self.expert,
                train_tfm,
                file_names=fit_train_names,
            )
            self._val = FiveKDataset(
                self.data_dir,
                "train" if self.overfit_on_train_subset else "val",
                self.expert,
                val_tfm,
                file_names=fit_val_names,
            )
        if stage in ("test", None):
            fit_train_names = (
                train_names[:self.train_subset_size]
                if self.train_subset_size is not None
                else train_names
            )
            fit_test_names = (
                fit_train_names
                if self.overfit_on_train_subset
                else (
                    test_names[:self.test_subset_size]
                    if self.test_subset_size is not None
                    else test_names
                )
            )
            self._test = FiveKDataset(
                self.data_dir,
                "train" if self.overfit_on_train_subset else "test",
                self.expert,
                val_tfm,
                file_names=fit_test_names,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
