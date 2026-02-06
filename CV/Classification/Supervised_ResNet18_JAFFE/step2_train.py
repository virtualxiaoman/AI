import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from Utils.train_net import NetTrainerFNN


def main():
    # 1. 路径设置 (请确保这里的路径指向你实际的文件夹)
    data_dir = "../../../Datasets/JAFFE/jaffe_split"

    # 2. 定义数据增强与预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. 创建 Dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    # 4. 创建你要求的 train_loader 和 test_loader
    # 注意：JAFFE 数据集很小，batch_size 建议设为 16 或 32
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 获取类别信息
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"7分类: {class_names}")

    # 5. 模型初始化
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 替换输出层

    epochs = 50

    # 6. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 7. 训练模型
    trainer = NetTrainerFNN(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        loss_fn=criterion,
        net=model,
        scheduler=scheduler,
        epochs=epochs,
        eval_interval=1,
        use_amp=True
    )
    trainer.train_net(net_save_path="../../../Models/JAFFE/supervised_resnet18_jaffe.pth")


if __name__ == '__main__':
    main()
    # Epoch 50/50, Train Loss: 0.020368, Train Acc: 1.000000, Test Acc: 0.903226, Time: 2.94s, LR: 0.000000, GPU: 设备0：U0.19+R0.47/T15.92GB
    # >>> [train_net] (*^w^*) Congratulations！训练结束，总共花费时间: 145.0529305934906秒
    # [train_net] 最佳结果 epoch = 24, acc = 0.9354838709677419
