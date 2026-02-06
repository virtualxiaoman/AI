# 使用timm调用的方法见：https://gemini.google.com/app/ba9563aa475a4056
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

from Utils.train_net import NetTrainerFNN

# ---- 超参数 ----
DATA_ROOT = "../../Datasets/CIFAR10"
MODEL_ROOT = "../../Models/CIFAR10"
PATH = "../../Datasets/JAFFE/jaffe"
os.makedirs(MODEL_ROOT, exist_ok=True)
BATCH_SIZE = 256
NUM_WORKERS = 5
EPOCHS = 100
LR_RESNET = 0.1
LR_VIT = 0.001
LR_CONVNEXT = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Transforms
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.5, inplace=True)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ---- Datasets ----
train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ---- Utility: adapt ResNet-18 first conv to CIFAR size ----
def build_resnet18(num_classes=10):
    net = models.resnet18(weights=None)
    # replace conv1 and remove maxpool to adapt to 32x32
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


# ---- ConvNeXt-Tiny (torchvision supports convnext_tiny) ----
def build_convnext_tiny(num_classes=10):
    net = models.convnext_tiny(weights=None)
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    return net


# ---- Simple ViT-Tiny implementation (lightweight, for CIFAR-10) ----
# 这是一个小型、资源友好的 ViT 实现（可直接训练）。参数设置为 tiny级别。
class ViTTiny(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, emb_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 num_classes=10, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        # patch embedding via conv
        self.patch_embed = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)
        # cls token + pos emb
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                                                   dim_feedforward=int(emb_dim * mlp_ratio), activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

        # init
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
        # x: B x 3 x H x W
        B = x.shape[0]
        x = self.patch_embed(x)  # B x emb_dim x H' x W'
        x = x.flatten(2).transpose(1, 2)  # B x num_patches x emb_dim
        cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x emb_dim
        x = torch.cat((cls_tokens, x), dim=1)  # B x (1+P) x emb_dim
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)  # B x (1+P) x emb_dim
        x = self.norm(x)
        cls = x[:, 0]  # B x emb_dim
        out = self.head(cls)
        return out


def build_vit_tiny(num_classes=10):
    return ViTTiny(img_size=32, patch_size=4, emb_dim=192, depth=6, num_heads=3, mlp_ratio=2.0, num_classes=num_classes)


# ---- Training orchestrator for three models ----
def train_all():
    # 相当于def create_optimizer_for_resnet(net):  return optim.SGD(net.parameters()...)
    experiments = [
        # (build_resnet18, "resnet18",
        #  lambda net: optim.SGD(net.parameters(), lr=LR_RESNET, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)),
        (build_vit_tiny, "vit_tiny",
         lambda net: optim.AdamW(net.parameters(), lr=LR_VIT, weight_decay=WEIGHT_DECAY)),
        (build_convnext_tiny, "convnext_tiny",
         lambda net: optim.AdamW(net.parameters(), lr=LR_CONVNEXT, weight_decay=WEIGHT_DECAY)),
    ]

    for builder, name, optim_builder in experiments:
        net = builder(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim_builder(net)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        save_path = os.path.join(MODEL_ROOT, f"{name}_cifar10.pth")
        # print("Saving model to:", save_path)

        trainer = NetTrainerFNN(
            train_loader=train_loader,
            test_loader=test_loader,
            net=net,
            loss_fn=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            eval_type='acc',
            eval_during_training=True,
            eval_interval=1,
            use_amp=True,
            net_name=name,
            scheduler=scheduler
        )
        # trainer.view_parameters(view_net_struct=False, view_params_count=True, view_params_details=False)
        trainer.train_net(net_save_path=save_path)


if __name__ == "__main__":
    train_all()
    # Epoch 100/100, Train Loss: 0.055291, Train Acc: 0.983680, Test Acc: 0.954300, Time: 21.08s, LR: 0.000000, GPU: 设备0：U0.15+R1.26/T15.92GB
    # >>> [train_net] (*^w^*) Congratulations！resnet18训练结束，总共花费时间: 2125.475818872452秒
    # [train_net] 最佳结果 epoch = 96, acc = 0.9547

    # Epoch 100/100, Train Loss: 0.315281, Train Acc: 0.917580, Test Acc: 0.850700, Time: 20.94s, LR: 0.000000, GPU: 设备0：U0.04+R1.22/T15.92GB
    # >>> [train_net] (*^w^*) Congratulations！vit_tiny训练结束，总共花费时间: 2114.9014189243317秒
    # [train_net] 最佳结果 epoch = 95, acc = 0.8508

    # Epoch 1/2, Train Loss: 1.589917, Train Acc: 0.510220, Test Acc: 0.514600, Time: 159.45s, GPU: 设备0：U0.14+R0.99/T4.00GB
    # Epoch 2/2, Train Loss: 1.054038, Train Acc: 0.667840, Test Acc: 0.673200, Time: 159.60s, GPU: 设备0：U0.14+R0.99/T4.00GB
    # Epoch 9/10, Train Loss: 0.405180, Train Acc: 0.829700, Test Acc: 0.805200, Time: 164.70s, GPU: 设备0：U0.14+R0.99/T4.00GB
    # Epoch 10/10, Train Loss: 0.382158, Train Acc: 0.845620, Test Acc: 0.818200, Time: 165.48s, GPU: 设备0：U0.14+R0.99/T4.00GB
    # Epoch 1/2, Train Loss: 1.747074, Train Acc: 0.410840, Test Acc: 0.423100, Time: 134.04s, GPU: 设备0：U0.04+R1.01/T4.00GB
    # Epoch 2/2, Train Loss: 1.444125, Train Acc: 0.526800, Test Acc: 0.530900, Time: 134.74s, GPU: 设备0：U0.04+R1.01/T4.00GB
    # Epoch 9/10, Train Loss: 0.978681, Train Acc: 0.688900, Test Acc: 0.692000, Time: 136.64s, GPU: 设备0：U0.04+R1.01/T4.00GB
    # Epoch 10/10, Train Loss: 0.945410, Train Acc: 0.703100, Test Acc: 0.694100, Time: 137.56s, GPU: 设备0：U0.04+R1.01/T4.00GB
    # Epoch 1/2, Train Loss: 2.178949, Train Acc: 0.356100, Test Acc: 0.389900, Time: 193.18s, GPU: 设备0：U0.33+R1.04/T4.00GB
    # Epoch 2/2, Train Loss: 1.791617, Train Acc: 0.371720, Test Acc: 0.406900, Time: 194.21s, GPU: 设备0：U0.33+R1.04/T4.00GB
    # Epoch 9/10, Train Loss: 2.251940, Train Acc: 0.167980, Test Acc: 0.171900, Time: 201.28s, GPU: 设备0：U0.33+R1.04/T4.00GB
    # Epoch 10/10, Train Loss: 2.248870, Train Acc: 0.170300, Test Acc: 0.173200, Time: 200.80s, GPU: 设备0：U0.33+R1.04/T4.00GB
