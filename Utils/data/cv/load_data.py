from Utils.data.cv.datasets.cifar10 import CVCIFAR10Loader
from Utils.data.cv.datasets.mnist import CVMNISTLoader


class CVDatasetFactory:
    registry = {
        "CIFAR10": CVCIFAR10Loader,
        "MNIST": CVMNISTLoader,
    }

    @classmethod
    def create(cls, name, config=None):
        loader_cls = cls.registry[name]

        return loader_cls(config).load()
