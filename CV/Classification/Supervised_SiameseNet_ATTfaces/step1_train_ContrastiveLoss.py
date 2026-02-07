import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets

from Utils.train_net import NetTrainerPair
from Utils.loss_fn import ContrastiveLoss


class SiameseDataset(Dataset):
    """
    从 ImageFolder 构造正样本对 / 负样本对
    """

    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            self.class_to_indices.setdefault(label, []).append(idx)
        self.labels = self.dataset.targets
        self.num_classes = len(self.dataset.classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        # 50% 同类，50% 异类
        if random.random() < 0.5:
            # 正样本对
            idx2 = random.choice(self.class_to_indices[label1])
            label = 1
        else:
            # 负样本对
            label2 = random.choice([l for l in range(self.num_classes) if l != label1])
            idx2 = random.choice(self.class_to_indices[label2])
            label = 0
        img2, _ = self.dataset[idx2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # 去掉分类头
        self.backbone = backbone
        self.embedding = nn.Linear(512, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # 非常重要
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = SiameseDataset("../../../Datasets/ATTfaces/split/train", transform)
    val_dataset = SiameseDataset("../../../Datasets/ATTfaces/split/test", transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    epochs = 50
    model = SiameseNet(embedding_dim=256)
    criterion = ContrastiveLoss(margin=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = NetTrainerPair(
        net=model,
        train_loader=train_loader,
        test_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        loss_fn=criterion,
        scheduler=scheduler,
        epochs=epochs,
        eval_type='acc',
        eval_interval=1
    )
    trainer.train_net(net_save_path="../../../Models/ATTfaces/siamese_net.pth")


if __name__ == "__main__":
    main()
    # Epoch 49/50, Train Loss: 0.006101, Train Acc: 1.000000, Test Acc: 0.940000, Time: 3.79s, LR: 0.000000, GPU: 设备0：U0.20+R1.82/T15.92GB
    # Epoch 50/50, Train Loss: 0.005725, Train Acc: 1.000000, Test Acc: 0.930000, Time: 3.82s, LR: 0.000000, GPU: 设备0：U0.20+R1.82/T15.92GB
    # >>> [train_net] (*^w^*) Congratulations！训练结束，总共花费时间: 212.17000651359558秒
    # [train_net] 最佳结果 epoch = 5, acc = 0.98
