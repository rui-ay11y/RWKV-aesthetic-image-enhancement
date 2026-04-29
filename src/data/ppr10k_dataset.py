"""PPR10K dataset — placeholder for future experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PPR10KDataset(Dataset):
    """PPR10K dataset placeholder.

    TODO: Implement when PPR10K data is available.
          Reference: https://github.com/csjliang/PPR10K
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        target_type: str = "a",
        transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.file_names: list[str] = []
        # TODO: load file list from data_dir/splits/{split}.txt

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        raise NotImplementedError("PPR10KDataset not yet implemented.")


class PPR10KDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for PPR10K. Placeholder.

    TODO: Implement setup() / train_dataloader() / val_dataloader() /
          test_dataloader() following the FiveKDataModule pattern.
    """

    def __init__(
        self,
        data_dir: str = "data/ppr10k",
        target_type: str = "a",
        image_size: int = 480,
        crop_size: int = 256,
        use_augmentation: bool = True,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        # TODO: store remaining args

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError("PPR10KDataModule not yet implemented.")

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
