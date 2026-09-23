from torch.utils.data import random_split


def split_train_val(dataset, ratio: float, seed: int):
    if ratio <= 0:
        return dataset, None

    length = len(dataset)

    val_size = int(length * ratio)
    train_size = length - val_size

    generator = None

    if seed is not None:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    return train_dataset, val_dataset
