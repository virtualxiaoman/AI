from Utils.data.cv.config import CVDatasetConfig
from Utils.data.cv.load_data import CVDatasetFactory


def main():
    config = CVDatasetConfig(
        batch_size=128,
        val_ratio=0.1
    )

    data = CVDatasetFactory.create(
        "CIFAR10",
        config
    )

    print(data.num_classes)
    print(data.input_shape)
    print(data.mean)

    for x, y in data.train_loader:
        print(x.shape)
        break


if __name__ == "__main__":
    main()
