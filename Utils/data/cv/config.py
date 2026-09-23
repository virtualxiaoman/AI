from dataclasses import dataclass
from typing import Callable


@dataclass
class CVDatasetConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True

    train_transform: Callable | None = None
    test_transform: Callable | None = None

    val_ratio: float = 0
    random_seed: int = 42

    download: bool = True
