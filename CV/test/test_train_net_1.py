# test_mnist_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.train_net import NetTrainerFNN


# 1. 简单的 CNN 定义（适用于 MNIST）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # -> 32 x 28 x 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64 x 14 x 14 after pool
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 类（MNIST）

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # -> B x 32 x 14 x 14
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # -> B x 64 x 7 x 7
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logits，不要在这里加 softmax（CrossEntropyLoss 要求 logits）
        return x


# 2. 数据准备（MNIST）
def get_mnist_loaders(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的常用归一化
    ])

    train_dataset = datasets.MNIST(root='../../Datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../../Datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# 3. 运行训练
def main():
    # 超参数
    batch_size = 512
    epochs = 200
    lr = 1e-4
    eval_interval = 10
    num_workers = 0  # windows上通常设为0以避免子进程问题

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, num_workers=num_workers)
    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()  # 适用于多分类（labels 为 long tensor）
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainer = NetTrainerFNN(
        train_loader=train_loader,
        test_loader=test_loader,
        net=net,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=epochs,
        eval_type=None,  # 测试不显示给出eval_type
        eval_during_training=True,
        eval_interval=eval_interval,
        use_amp=True
    )
    trainer.view_parameters(view_net_struct=True, view_params_count=True, view_params_details=True)
    trainer.train_net(net_save_path="../../Models/MNIST/simple_cnn_mnist.pth")


if __name__ == "__main__":
    # 某些测试结果如下：
    # [__init__] 当前设备为cuda
    # [__init__] 根据loss_fn自动检测到目前为分类任务，eval_type=acc
    # G:\Projects\py\AI\Utils\train_net.py:537: UserWarning: [train_net] 最佳模型保存文件夹dir_path='../../Models/MNIST'不存在，已自动创建
    #   warnings.warn(f"[train_net] 最佳模型保存文件夹dir_path='{dir_path}'不存在，已自动创建")
    # [log] 第一批次的shape如下：
    #       X: torch.Size([256, 1, 28, 28]), y: torch.Size([256]), outputs: torch.Size([256, 10])
    # [log] 网络的总参数量: 421642
    # >>> [train_net] (^v^)开始训练模型，参数epochs=10。当前设备为cuda，网络类型为FNN，评估类型为acc。
    # Epoch 1/10, Train Loss: 0.216620, Train Acc: 0.972517, Test Acc: 0.972400, Time: 17.44s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 2/10, Train Loss: 0.057524, Train Acc: 0.986767, Test Acc: 0.986100, Time: 16.37s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 3/10, Train Loss: 0.039310, Train Acc: 0.989850, Test Acc: 0.985900, Time: 16.08s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 4/10, Train Loss: 0.030767, Train Acc: 0.993317, Test Acc: 0.990600, Time: 16.31s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 5/10, Train Loss: 0.023402, Train Acc: 0.994267, Test Acc: 0.988700, Time: 16.82s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 6/10, Train Loss: 0.017407, Train Acc: 0.995717, Test Acc: 0.990800, Time: 16.92s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 7/10, Train Loss: 0.014254, Train Acc: 0.996583, Test Acc: 0.990900, Time: 16.26s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 8/10, Train Loss: 0.013076, Train Acc: 0.997617, Test Acc: 0.991500, Time: 16.39s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 9/10, Train Loss: 0.010360, Train Acc: 0.997233, Test Acc: 0.990400, Time: 17.04s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # Epoch 10/10, Train Loss: 0.009777, Train Acc: 0.998033, Test Acc: 0.990500, Time: 17.07s, GPU: 设备0：U0.02+R0.18/T4.00GB
    # >>> [train_net] (*^w^*) Congratulations！训练结束，总共花费时间: 166.67874908447266秒
    # [train_net] 最佳结果 epoch = 8, acc = 0.9915

    # [__init__] 当前设备为cuda
    # [__init__] 根据loss_fn自动检测到目前为分类任务，eval_type=acc
    # [log] 数据总量：Train: 60000, Test: 10000
    # [log] 第一批次的shape如下：
    #       X: torch.Size([512, 1, 28, 28]), y: torch.Size([512]), outputs: torch.Size([512, 10])
    # [log] 网络的总参数量: 421642
    # >>> [train_net] (^v^)开始训练模型，参数epochs=200。当前设备为cuda，网络类型为FNN，评估类型为acc。
    # [train_net] 最佳模型保存地址net_save_path=../../Models/MNIST/simple_cnn_mnist.pth
    # Epoch 1/200, Train Loss: 1.007206, Train Acc: 0.900733, Test Acc: 0.905800, Time: 33.60s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 11/200, Train Loss: 0.056050, Train Acc: 0.984550, Test Acc: 0.983800, Time: 34.18s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 21/200, Train Loss: 0.033034, Train Acc: 0.991483, Test Acc: 0.988200, Time: 21.51s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 31/200, Train Loss: 0.022220, Train Acc: 0.993917, Test Acc: 0.988300, Time: 20.20s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 41/200, Train Loss: 0.015127, Train Acc: 0.995983, Test Acc: 0.989000, Time: 21.50s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 51/200, Train Loss: 0.009922, Train Acc: 0.997917, Test Acc: 0.988300, Time: 28.32s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 61/200, Train Loss: 0.007265, Train Acc: 0.998567, Test Acc: 0.988300, Time: 39.25s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 71/200, Train Loss: 0.004421, Train Acc: 0.999250, Test Acc: 0.988400, Time: 19.88s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 81/200, Train Loss: 0.002463, Train Acc: 0.999767, Test Acc: 0.988400, Time: 39.41s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 91/200, Train Loss: 0.001252, Train Acc: 0.999933, Test Acc: 0.988600, Time: 38.88s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 101/200, Train Loss: 0.000709, Train Acc: 0.999950, Test Acc: 0.988600, Time: 35.07s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 111/200, Train Loss: 0.000469, Train Acc: 1.000000, Test Acc: 0.989200, Time: 36.24s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 121/200, Train Loss: 0.000269, Train Acc: 1.000000, Test Acc: 0.988600, Time: 39.21s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 131/200, Train Loss: 0.008548, Train Acc: 0.999267, Test Acc: 0.989000, Time: 19.87s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 141/200, Train Loss: 0.000172, Train Acc: 1.000000, Test Acc: 0.988700, Time: 39.25s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 151/200, Train Loss: 0.000117, Train Acc: 1.000000, Test Acc: 0.988800, Time: 39.24s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 161/200, Train Loss: 0.000099, Train Acc: 1.000000, Test Acc: 0.988900, Time: 39.24s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 171/200, Train Loss: 0.000160, Train Acc: 1.000000, Test Acc: 0.990000, Time: 39.25s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 181/200, Train Loss: 0.000079, Train Acc: 1.000000, Test Acc: 0.989600, Time: 39.25s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # Epoch 191/200, Train Loss: 0.000054, Train Acc: 1.000000, Test Acc: 0.989400, Time: 39.24s, GPU: 设备0：U0.02+R0.63/T4.00GB
    # >>> [train_net] (*^w^*) Congratulations！训练结束，总共花费时间: 6787.191251516342秒
    # [train_net] 最佳结果 epoch = 171, acc = 0.99

    main()
