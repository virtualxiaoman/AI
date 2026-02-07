import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models

from Utils.loss_fn import ArcFaceLoss
from Utils.train_net import NetTrainerArcFace


def per_class_split(dataset: datasets.ImageFolder, val_ratio: float = 0.2, seed: int = 42) -> Tuple[list, list]:
    """
    Return (train_indices, val_indices) splitting indices of the ImageFolder dataset such that
    for each class the split ratio is approximately val_ratio.
    """
    random.seed(seed)
    targets = dataset.targets  # list of class indices per sample
    class_to_indices = {}
    for idx, cls in enumerate(targets):
        class_to_indices.setdefault(cls, []).append(idx)

    train_idx, val_idx = [], []
    for cls, idxs in class_to_indices.items():
        n = len(idxs)
        n_val = max(1, int(round(n * val_ratio)))  # ensure at least 1 sample in val if possible
        shuffled = idxs.copy()
        random.shuffle(shuffled)
        val_subset = shuffled[:n_val]
        train_subset = shuffled[n_val:]
        # if class is tiny and train becomes empty, move one from val to train
        if len(train_subset) == 0 and len(val_subset) > 1:
            train_subset.append(val_subset.pop())
        train_idx.extend(train_subset)
        val_idx.extend(val_subset)
    return train_idx, val_idx


class ResNet18Embedding(nn.Module):
    """
    ResNet-18 backbone for embedding extraction.
    Output: (B, embedding_dim)
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        return: (B, embedding_dim)
        """
        x = self.backbone(x)  # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings (important for ArcFace)
        return x


def main():
    data_root = "../../../Datasets/ATTfaces/raw"
    epochs = 50
    batch_size = 64
    lr = 0.01
    weight_decay = 1e-4
    embedding_dim = 512
    s = 30.0
    m = 0.5
    val_ratio = 0.2
    num_workers = 4

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(str(data_root))
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images of {num_classes} classes in {data_root}")

    train_idx, val_idx = per_class_split(full_dataset, val_ratio=val_ratio, seed=42)
    train_dataset = Subset(datasets.ImageFolder(str(data_root), transform=train_transform), train_idx)
    val_dataset = Subset(datasets.ImageFolder(str(data_root), transform=val_transform), val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = ResNet18Embedding(embedding_dim=embedding_dim)
    loss_fn = ArcFaceLoss(num_classes=num_classes, embedding_dim=embedding_dim, s=s, m=m, easy_margin=False)
    optimizer = torch.optim.SGD(list(net.parameters()) + list(loss_fn.parameters()),
                                lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = NetTrainerArcFace(
        net=net,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        eval_type='acc',
        eval_interval=1
    )
    trainer.train_net(net_save_path="../../../Models/ATTfaces/resnet18_arcface.pth")


if __name__ == "__main__":
    main()
    # Epoch 49/50, Train Loss: 0.000291, Train Acc: 1.000000, Test Acc: 1.000000, Time: 4.00s, LR: 0.000010, GPU: 设备0：U0.05+R1.50/T15.92GB
    # Epoch 50/50, Train Loss: 0.000197, Train Acc: 1.000000, Test Acc: 1.000000, Time: 3.86s, LR: 0.000000, GPU: 设备0：U0.05+R1.50/T15.92GB
    # >>> [train_net] (*^w^*) Congratulations！训练结束，总共花费时间: 181.29072618484497秒
    # [train_net] 最佳结果 epoch = 11, acc = 1.0
