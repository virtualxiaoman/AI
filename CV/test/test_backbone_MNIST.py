# mnist_train_all.py
# 与您原风格一致的 MNIST 训练脚本
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

from Utils.train_net import NetTrainerFNN

# ---- 超参数（可按需调整） ----
DATA_ROOT = "../../Datasets/MNIST"
MODEL_ROOT = "../../Models/MNIST"
os.makedirs(MODEL_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128        # 若显存较小，可改为 32 或 64
NUM_WORKERS = 2
EPOCHS = 10              # 快速验证用 1-2 epoch 即可
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PRINT_INTERVAL = 50

# 学习率（建议低一些，MNIST 容易过拟合）
LR_RESNET = 0.01
LR_CONVNEXT = 0.01
LR_VIT = 3e-4

# MNIST normalization (灰度)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

# ---- Transforms（针对 MNIST: 1x28x28） ----
train_transform = transforms.Compose([
    # 保持中心、轻微随机裁剪以增加鲁棒性（padding 可选）
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

# ---- Datasets / Loaders ----
train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

# ---- Utility: adapt ResNet-18 first conv to MNIST (1 channel, 28x28) ----
def build_resnet18(num_classes=10):
    net = models.resnet18(weights=None)
    # 替换 conv1 为 1 通道输入，去掉最大池化以适配小尺寸
    net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

# ---- ConvNeXt-Tiny (torchvision 支持) ----
def build_convnext_tiny(num_classes=10):
    net = models.convnext_tiny(weights=None)
    # 第一层 features[0][0] 是 Conv2d(3, 96, kernel_size=4, stride=4)
    # 将其替换为 1 通道输入（保持其他结构）
    # 根据 torchvision 版本，features 的结构可能有差异；这里按常见结构改写
    first_block = net.features[0]
    if isinstance(first_block[0], nn.Conv2d):
        in_channels_new = 1
        out_channels = first_block[0].out_channels
        kernel_size = first_block[0].kernel_size
        stride = first_block[0].stride
        padding = first_block[0].padding
        first_block[0] = nn.Conv2d(in_channels_new, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
    else:
        # 保险起见：若结构不同，直接查找第一个 Conv2d 并替换
        for i, m in enumerate(net.features):
            for j, n in enumerate(m):
                if isinstance(n, nn.Conv2d):
                    out_ch = n.out_channels
                    k = n.kernel_size
                    s = n.stride
                    p = n.padding
                    net.features[i][j] = nn.Conv2d(1, out_ch, kernel_size=k, stride=s, padding=p)
                    raise StopIteration
    # 分类头调整
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    return net

# ---- Simple ViT-Tiny implementation (轻量，适配 MNIST) ----
class ViTTiny(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, emb_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 num_classes=10, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        # patch embedding via conv (in_chans 可变)
        self.patch_embed = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)
        # cls token + pos emb
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder（batch_first=True）
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                                                   dim_feedforward=int(emb_dim * mlp_ratio), activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # x: B x 1 x H x W
        B = x.shape[0]
        x = self.patch_embed(x)             # B x emb_dim x H' x W'
        x = x.flatten(2).transpose(1, 2)    # B x num_patches x emb_dim
        cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x emb_dim
        x = torch.cat((cls_tokens, x), dim=1)          # B x (1+P) x emb_dim
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)             # B x (1+P) x emb_dim
        x = self.norm(x)
        cls = x[:, 0]                       # B x emb_dim
        out = self.head(cls)
        return out

def build_vit_tiny(num_classes=10):
    # img_size=28, patch_size=4 -> 7x7 patches
    return ViTTiny(img_size=28, patch_size=4, in_chans=1, emb_dim=192, depth=6, num_heads=3, mlp_ratio=2.0, num_classes=num_classes)

# ---- Training orchestrator for three models ----
def train_all():
    experiments = [
        (build_resnet18, "resnet18_mnist",
         lambda net: optim.SGD(net.parameters(), lr=LR_RESNET, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY),
         LR_RESNET),
        (build_vit_tiny, "vit_tiny_mnist",
         lambda net: optim.AdamW(net.parameters(), lr=LR_VIT, weight_decay=WEIGHT_DECAY),
         LR_VIT),
        (build_convnext_tiny, "convnext_tiny_mnist",
         lambda net: optim.SGD(net.parameters(), lr=LR_CONVNEXT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY),
         LR_CONVNEXT),
    ]

    for builder, name, optim_builder, lr in experiments:
        print(f"\n=== Start training: {name} ===")
        net = builder(num_classes=10)
        net = net.to(DEVICE)

        # criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim_builder(net)

        # scheduler: cosine decay over epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # prepare trainer - 假设 NetTrainerFNN 接口与原来一致
        trainer = NetTrainerFNN(
            train_loader=train_loader,
            test_loader=test_loader,
            net=net,
            loss_fn=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            eval_type=None,
            eval_during_training=True,
            eval_interval=1,
        )

        # 打印参数统计（按您原示例）
        trainer.view_parameters(view_net_struct=False, view_params_count=True, view_params_details=False)

        # model save path
        save_path = os.path.join(MODEL_ROOT, f"{name}_mnist.pth")
        print("Saving model to:", save_path)

        # 训练并保存（若 NetTrainerFNN.train_net 的参数名不同，请按实际修改）
        trainer.train_net(net_save_path=save_path)

        # 如果 trainer 内部没有调用 scheduler.step()，可以在此处手动 step（视实现而定）
        # for epoch in range(EPOCHS):
        #     scheduler.step()
        print(f"=== Finished training: {name} ===\n")


if __name__ == "__main__":
    print("Device:", DEVICE)
    train_all()

    # 运行示例输出
    # [train_net] 最佳模型保存地址net_save_path=../../Models/MNIST\vit_tiny_mnist_mnist.pth
    # Epoch 1/10, Train Loss: 1.360520, Train Acc: 0.688017, Test Acc: 0.706500, Time: 102.06s, GPU: 设备0：U0.04+R0.82/T4.00GB
    # Epoch 2/10, Train Loss: 0.815132, Train Acc: 0.808133, Test Acc: 0.840000, Time: 102.98s, GPU: 设备0：U0.04+R0.82/T4.00GB
    # Epoch 9/10, Train Loss: 0.129196, Train Acc: 0.974733, Test Acc: 0.980200, Time: 101.78s, GPU: 设备0：U0.04+R0.82/T4.00GB
    # Epoch 10/10, Train Loss: 0.114328, Train Acc: 0.976633, Test Acc: 0.981200, Time: 97.76s, GPU: 设备0：U0.04+R0.82/T4.00GB
