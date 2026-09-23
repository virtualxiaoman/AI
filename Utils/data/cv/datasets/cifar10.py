from torchvision import datasets, transforms

from Utils.data.base import CVBaseDatasetLoader, CVDatasetBundle
from Utils.data.cv.utils.metadata import extract_metadata
from Utils.data.cv.utils.split import split_train_val


class CVCIFAR10Loader(CVBaseDatasetLoader):
    dataset_name = "CIFAR10"

    def load(self):

        mean = (
            0.4914,
            0.4822,
            0.4465
        )

        std = (
            0.2470,
            0.2435,
            0.2616
        )

        train_transform = self.config.train_transform

        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        test_transform = self.config.test_transform

        if test_transform is None:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_dataset = datasets.CIFAR10(
            self.root,
            train=True,
            transform=train_transform,
            download=self.config.download
        )

        test_dataset = datasets.CIFAR10(
            self.root,
            train=False,
            transform=test_transform,
            download=self.config.download
        )

        train_dataset, val_dataset = split_train_val(
            train_dataset,
            self.config.val_ratio,
            self.config.random_seed
        )

        meta = extract_metadata(train_dataset)

        return CVDatasetBundle(
            train_dataset=train_dataset,
            train_loader=self.build_loader(train_dataset),
            val_dataset=val_dataset,
            val_loader=self.build_test_loader(val_dataset) if val_dataset else None,
            test_dataset=test_dataset,
            test_loader=self.build_test_loader(test_dataset),
            mean=mean,
            std=std,
            config=self.config,
            **meta
        )
