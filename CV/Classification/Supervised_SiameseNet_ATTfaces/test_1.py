# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Train ResNet-18 (pretrained) + ArcFace on ATTfaces dataset.
# Dataset layout (given):
# ../../../Datasets/ATTfaces/raw/
#     s1/
#       img1.jpg
#       ...
#     s2/
#       ...
#     ...
#     s40/
#
# Usage example:
#     python train_arcface_resnet18.py --data-root "../../../Datasets/ATTfaces/raw" --epochs 50 --batch-size 64
# """
#
# import os
# import math
# import random
# import argparse
# from pathlib import Path
# from typing import Tuple
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from torchvision import transforms, datasets, models
# from torch.amp import GradScaler, autocast
#
# from Utils.loss_fn import ArcFaceLoss
# from Utils.train_net import NetTrainerArcFace
#
# # ---------------------------
# # Helpers: train/val split per class (ensure each class has a few images in train/val)
# # ---------------------------
# def per_class_split(dataset: datasets.ImageFolder, val_ratio: float = 0.2, seed: int = 42) -> Tuple[list, list]:
#     """
#     Return (train_indices, val_indices) splitting indices of the ImageFolder dataset such that
#     for each class the split ratio is approximately val_ratio.
#     """
#     random.seed(seed)
#     targets = dataset.targets  # list of class indices per sample
#     class_to_indices = {}
#     for idx, cls in enumerate(targets):
#         class_to_indices.setdefault(cls, []).append(idx)
#
#     train_idx, val_idx = [], []
#     for cls, idxs in class_to_indices.items():
#         n = len(idxs)
#         n_val = max(1, int(round(n * val_ratio)))  # ensure at least 1 sample in val if possible
#         shuffled = idxs.copy()
#         random.shuffle(shuffled)
#         val_subset = shuffled[:n_val]
#         train_subset = shuffled[n_val:]
#         # if class is tiny and train becomes empty, move one from val to train
#         if len(train_subset) == 0 and len(val_subset) > 1:
#             train_subset.append(val_subset.pop())
#         train_idx.extend(train_subset)
#         val_idx.extend(val_subset)
#     return train_idx, val_idx
#
#
# # ---------------------------
# # Training / Validation loops
# # ---------------------------
# # def train_one_epoch(model, arcface, train_loader, optimizer, device, scaler, epoch, print_freq=20):
# #     model.train()
# #     arcface.train()
# #     total_loss = 0.0
# #     total_correct = 0
# #     total_samples = 0
# #
# #     for i, (imgs, labels) in enumerate(train_loader):
# #         imgs = imgs.to(device, non_blocking=True)
# #         labels = labels.to(device, non_blocking=True)
# #
# #         optimizer.zero_grad()
# #
# #         with autocast(device_type=device.type):
# #             embeddings = model(imgs)  # [B, D]
# #             embeddings = F.normalize(embeddings, p=2, dim=1)
# #             loss, logits = arcface(embeddings, labels)
# #         scaler.scale(loss).backward()
# #         scaler.step(optimizer)
# #         scaler.update()
# #
# #         batch_size = labels.size(0)
# #         total_loss += loss.item() * batch_size
# #         preds = logits.argmax(dim=1)
# #         total_correct += (preds == labels).sum().item()
# #         total_samples += batch_size
# #
# #         if (i + 1) % print_freq == 0:
# #             print(f"Epoch {epoch} Iter {i + 1}/{len(train_loader)} loss={loss.item():.4f} "
# #                   f"batch_acc={(preds == labels).float().mean().item():.4f}")
# #
# #     avg_loss = total_loss / total_samples
# #     avg_acc = total_correct / total_samples
# #     return avg_loss, avg_acc
# #
# #
# # @torch.no_grad()
# # def validate(model, arcface, val_loader, device):
# #     model.eval()
# #     arcface.eval()
# #     total_loss = 0.0
# #     total_correct = 0
# #     total_samples = 0
# #
# #     for imgs, labels in val_loader:
# #         imgs = imgs.to(device, non_blocking=True)
# #         labels = labels.to(device, non_blocking=True)
# #
# #         embeddings = model(imgs)
# #         embeddings = F.normalize(embeddings, p=2, dim=1)
# #         logits = arcface.get_logits(embeddings)
# #         loss = F.cross_entropy(logits, labels)
# #
# #         batch_size = labels.size(0)
# #         total_loss += loss.item() * batch_size
# #         preds = logits.argmax(dim=1)
# #         total_correct += (preds == labels).sum().item()
# #         total_samples += batch_size
# #
# #     avg_loss = total_loss / total_samples
# #     avg_acc = total_correct / total_samples
# #     return avg_loss, avg_acc
#
#
# class ResNet18Embedding(nn.Module):
#     """
#     ResNet-18 backbone for embedding extraction.
#     Output: (B, embedding_dim)
#     """
#
#     def __init__(self, embedding_dim: int = 512):
#         super().__init__()
#         self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         in_feat = self.backbone.fc.in_features
#         self.backbone.fc = nn.Linear(in_feat, embedding_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 3, H, W)
#         return: (B, embedding_dim)
#         """
#         x = self.backbone(x)  # [B, embedding_dim]
#         x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings (important for ArcFace)
#         return x
#
#
# # ---------------------------
# # Main
# # ---------------------------
# def main():
#     data_root = "../../../Datasets/ATTfaces/raw"
#     epochs = 50
#     batch_size = 64
#     lr = 0.01
#     weight_decay = 1e-4
#     embedding_dim = 512
#     s = 30.0
#     m = 0.5
#     easy_margin = False
#     val_ratio = 0.2
#     num_workers = 4
#     seed = 42
#     save_dir = "./checkpoints"
#
#     # reproducibility
#     torch.manual_seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#     # data_root = Path(data_root)
#     # assert data_root.exists(), f"data root not found: {data_root}"
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#
#     full_dataset = datasets.ImageFolder(str(data_root))
#     num_classes = len(full_dataset.classes)
#     print(f"Found {len(full_dataset)} images of {num_classes} classes in {data_root}")
#
#     train_idx, val_idx = per_class_split(full_dataset, val_ratio=val_ratio, seed=seed)
#     train_dataset = Subset(datasets.ImageFolder(str(data_root), transform=train_transform), train_idx)
#     val_dataset = Subset(datasets.ImageFolder(str(data_root), transform=val_transform), val_idx)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#     # # model: pretrained resnet18, replace fc with embedding layer
#     # backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#     # in_feat = backbone.fc.in_features
#     # backbone.fc = nn.Linear(in_feat, embedding_dim)  # embedding layer (random init)
#     # # optionally: backbone = torch.nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(), nn.Linear(in_feat, embedding_dim))
#     # backbone = backbone.to(device)
#     backbone = ResNet18Embedding(embedding_dim=embedding_dim).to(device)
#     arcface = ArcFaceLoss(num_classes=num_classes, embedding_dim=embedding_dim,
#                           s=s, m=m, easy_margin=easy_margin).to(device)
#     optimizer = torch.optim.SGD(list(backbone.parameters()) + list(arcface.parameters()),
#                                 lr=lr, momentum=0.9, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#     # scaler = GradScaler()
#
#     trainer = NetTrainerArcFace(
#         net=backbone,
#         train_loader=train_loader,
#         test_loader=val_loader,
#         loss_fn=arcface,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         epochs=epochs,
#         device=device,
#         eval_type='acc',
#         eval_interval=1
#     )
#     trainer.train_net(net_save_path=os.path.join(save_dir, "final.pth"))
#     exit(1)
#
#     best_val_acc = 0.0
#
#     for epoch in range(1, epochs + 1):
#         train_loss, train_acc = train_one_epoch(backbone, arcface, train_loader, optimizer, device, scaler, epoch)
#         val_loss, val_acc = validate(backbone, arcface, val_loader, device)
#
#         print(f"Epoch {epoch:03d} Train loss: {train_loss:.4f} Train acc: {train_acc:.4f} |"
#               f" Val loss: {val_loss:.4f} Val acc: {val_acc:.4f}")
#         scheduler.step()
#
#         # save checkpoint
#         ckpt = {
#             "epoch": epoch,
#             "backbone_state": backbone.state_dict(),
#             "arcface_state": arcface.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "scheduler_state": scheduler.state_dict(),
#             "args": {
#                 "data_root": str(data_root),
#                 "epochs": epochs,
#                 "batch_size": batch_size,
#                 "lr": lr,
#                 "weight_decay": weight_decay,
#                 "embedding_dim": embedding_dim,
#                 "s": s,
#                 "m": m,
#                 "easy_margin": easy_margin,
#                 "val_ratio": val_ratio,
#                 "num_workers": num_workers,
#                 "seed": seed,
#                 "save_dir": save_dir,
#                 "device": str(device),
#             },
#             "train_loss": train_loss,
#             "train_acc": train_acc,
#             "val_loss": val_loss,
#             "val_acc": val_acc,
#         }
#         ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch:03d}.pth")
#         torch.save(ckpt, ckpt_path)
#
#         # save best
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(ckpt, os.path.join(save_dir, "best.pth"))
#             print(f"New best val acc: {best_val_acc:.4f} saved.")
#
#     print("Training finished. Best val acc:", best_val_acc)
#
#
# if __name__ == "__main__":
#     main()
#     # Epoch 048 Train loss: 0.0009 Train acc: 1.0000 | Val loss: 0.1080 Val acc: 0.9750
#     # Epoch 049 Train loss: 0.0005 Train acc: 1.0000 | Val loss: 0.1054 Val acc: 0.9750
#     # Epoch 050 Train loss: 0.0007 Train acc: 1.0000 | Val loss: 0.1026 Val acc: 0.9750
#     # Training finished. Best val acc: 0.975
