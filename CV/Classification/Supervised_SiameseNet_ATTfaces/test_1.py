#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ResNet-18 (pretrained) + ArcFace on ATTfaces dataset.
Dataset layout (given):
../../../Datasets/ATTfaces/raw/
    s1/
      img1.jpg
      ...
    s2/
      ...
    ...
    s40/

Usage example:
    python train_arcface_resnet18.py --data-root "../../../Datasets/ATTfaces/raw" --epochs 50 --batch-size 64
"""

import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from torch.amp import GradScaler, autocast

from Utils.loss_fn import ArcFaceLoss

# ---------------------------
# Helpers: train/val split per class (ensure each class has a few images in train/val)
# ---------------------------
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

# ---------------------------
# Training / Validation loops
# ---------------------------
def train_one_epoch(model, arcface, train_loader, optimizer, device, scaler, epoch, print_freq=20):
    model.train()
    arcface.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            embeddings = model(imgs)  # [B, D]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            loss, logits = arcface(embeddings, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        if (i + 1) % print_freq == 0:
            print(f"Epoch {epoch} Iter {i+1}/{len(train_loader)} loss={loss.item():.4f} batch_acc={(preds==labels).float().mean().item():.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, arcface, val_loader, device):
    model.eval()
    arcface.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(imgs)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = arcface.get_logits(embeddings)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 + ArcFace on ATTfaces")
    parser.add_argument("--data-root", type=str, default="../../../Datasets/ATTfaces/raw", help="root folder for ImageFolder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--s", type=float, default=30.0)
    parser.add_argument("--m", type=float, default=0.5)
    parser.add_argument("--easy-margin", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root)
    assert data_root.exists(), f"data root not found: {data_root}"

    # transforms
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

    # dataset (ImageFolder)
    full_dataset = datasets.ImageFolder(str(data_root))
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images of {num_classes} classes in {data_root}")

    train_idx, val_idx = per_class_split(full_dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = Subset(datasets.ImageFolder(str(data_root), transform=train_transform), train_idx)
    val_dataset = Subset(datasets.ImageFolder(str(data_root), transform=val_transform), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)

    # model: pretrained resnet18, replace fc with embedding layer
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Linear(in_feat, args.embedding_dim)  # embedding layer (random init)
    # optionally: backbone = torch.nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(), nn.Linear(in_feat, args.embedding_dim))
    backbone = backbone.to(device)

    # arcface module (contains class weights)
    arcface = ArcFaceLoss(num_classes=num_classes, embedding_dim=args.embedding_dim,
                          s=args.s, m=args.m, easy_margin=False).to(device)

    # optimizer: optimize both backbone params and arcface weight
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(arcface.parameters()),
                                lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # scheduler (cosine or step); small dataset so simple StepLR is fine
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(backbone, arcface, train_loader, optimizer, device, scaler, epoch)
        val_loss, val_acc = validate(backbone, arcface, val_loader, device)

        print(f"Epoch {epoch:03d} Train loss: {train_loss:.4f} Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} Val acc: {val_acc:.4f}")
        scheduler.step()

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "backbone_state": backbone.state_dict(),
            "arcface_state": arcface.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch_{epoch:03d}.pth")
        torch.save(ckpt, ckpt_path)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"New best val acc: {best_val_acc:.4f} saved.")

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
    # Epoch 048 Train loss: 0.0009 Train acc: 1.0000 | Val loss: 0.1080 Val acc: 0.9750
    # Epoch 049 Train loss: 0.0005 Train acc: 1.0000 | Val loss: 0.1054 Val acc: 0.9750
    # Epoch 050 Train loss: 0.0007 Train acc: 1.0000 | Val loss: 0.1026 Val acc: 0.9750
    # Training finished. Best val acc: 0.975