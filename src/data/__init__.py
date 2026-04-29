__all__ = ["FiveKDataModule", "get_train_transforms", "get_val_transforms", "denormalize"]


def __getattr__(name: str):
    if name == "FiveKDataModule":
        from src.data.fivek_dataset import FiveKDataModule

        return FiveKDataModule
    if name == "get_train_transforms":
        from src.data.transforms import get_train_transforms

        return get_train_transforms
    if name == "get_val_transforms":
        from src.data.transforms import get_val_transforms

        return get_val_transforms
    if name == "denormalize":
        from src.data.transforms import denormalize

        return denormalize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
