from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

from Utils.data.cv.config import CVDatasetConfig
from Utils.config.path import DATASETS_DIR


@dataclass
class CVDatasetBundle:
    train_dataset: Dataset
    train_loader: DataLoader

    val_dataset: Dataset | None = None
    val_loader: DataLoader | None = None

    test_dataset: Dataset | None = None
    test_loader: DataLoader | None = None

    num_classes: int | None = None
    class_names: list[str] | None = None

    channels: int | None = None
    input_shape: tuple[int, ...] | None = None

    mean: tuple | None = None
    std: tuple | None = None

    config: CVDatasetConfig | None = None


class CVBaseDatasetLoader(ABC):
    dataset_name: str = ""

    def __init__(self, config: CVDatasetConfig | None = None):
        self.config = config or CVDatasetConfig()
        self.root = DATASETS_DIR / self.dataset_name

    def build_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def build_test_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    @abstractmethod
    def load(self):
        pass
