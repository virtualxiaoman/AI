"""
10类图像分类任务
使用 ResNet-18 进行的最小规模的 CIFAR-10 监督训练。
python train_cifar_resnet18_minimal.py --epochs 10 --batch_size 32 --save_dir ./exp_minimal
"""
import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet18(num_classes=None, cifar_adjust=True):
    """
    Build a ResNet-18.
    If num_classes is None -> returns backbone (model.fc = Identity), attr model.feature_dim set.
    If num_classes provided -> returns classifier with model.fc = Linear(num_features, num_classes).
    For CIFAR, adjust conv1 and maxpool for small images.
    """
    model = models.resnet18(weights=None)
    if cifar_adjust:
        # adapt to CIFAR resolution (32x32)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    feat_dim = model.fc.in_features
    if num_classes is None:
        model.fc = nn.Identity()
        model.feature_dim = feat_dim
    else:
        model.fc = nn.Linear(feat_dim, num_classes)
        model.feature_dim = feat_dim
    return model


def get_transforms():
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    val_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return train_t, val_t


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc="Train", leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loop.set_postfix(loss=running_loss / total, acc=correct / total)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc="Val", leave=False)
    for x, y in loop:
        x = x.to(device);
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loop.set_postfix(loss=running_loss / total, acc=correct / total)
    return running_loss / total, correct / total


@torch.no_grad()
def extract_embeddings(model, loader, device, save_path_emb, save_path_lbl):
    """
    Set model.fc = Identity before calling this, or build backbone with num_classes=None.
    This function will run the model and save embeddings and labels as numpy files.
    """
    model.eval()
    embs = []
    labs = []
    for x, y in tqdm(loader, desc="Embeddings", leave=False):
        x = x.to(device)
        z = model(x)  # shape: (B, feat_dim) if fc is Identity (we set that)
        if z.ndim == 4:
            # if still returns (B, C, 1, 1), flatten
            z = z.view(z.size(0), -1)
        embs.append(z.cpu().numpy())
        labs.append(y.numpy())
    embs = np.concatenate(embs, axis=0)
    labs = np.concatenate(labs, axis=0)
    np.save(save_path_emb, embs)
    np.save(save_path_lbl, labs)
    print(f"Saved embeddings: {save_path_emb}, labels: {save_path_lbl}")
    return embs, labs


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_t, val_t = get_transforms()
    train_ds = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=train_t, download=True)
    val_ds = torchvision.datasets.CIFAR10(root=args.data_root, train=False, transform=val_t, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # model with classifier
    model = build_resnet18(num_classes=10, cifar_adjust=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train loss: {tr_loss:.4f}, acc: {tr_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": float(val_acc)
            }
            ckpt_path = os.path.join(args.save_dir, "best_supervised.pth")
            torch.save(ckpt, ckpt_path)
            print("Saved best model to", ckpt_path)

    print("Training finished. Best val acc:", best_acc)

    # --- Extract embeddings with backbone (freeze & remove classifier) ---
    print("Extracting embeddings from best checkpoint...")
    # build backbone (no classifier)
    backbone = build_resnet18(num_classes=None, cifar_adjust=True)
    # load weights from best checkpoint into a full model first
    ckpt = torch.load(os.path.join(args.save_dir, "best_supervised.pth"), map_location="cpu")
    # load into a model with classifier temporarily to match keys
    full_model = build_resnet18(num_classes=10, cifar_adjust=True)
    full_model.load_state_dict(ckpt["model_state"])
    # transfer weights to backbone (copy all except fc)
    bs = backbone.state_dict()
    fs = full_model.state_dict()
    for k in bs.keys():
        if k in fs and bs[k].shape == fs[k].shape:
            bs[k] = fs[k]
    backbone.load_state_dict(bs)
    backbone = backbone.to(device)
    # now run backbone to get embeddings
    emb_path = os.path.join(args.save_dir, "embeddings.npy")
    lbl_path = os.path.join(args.save_dir, "labels.npy")
    extract_embeddings(backbone, val_loader, device, emb_path, lbl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../Datasets/CIFAR10")
    parser.add_argument("--save_dir", type=str, default="../../Models/CIFAR10/resnet18_minimal")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)  # 4GB 显卡推荐 16-32
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
