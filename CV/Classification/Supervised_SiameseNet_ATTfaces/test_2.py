# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, random_split
# from torchvision import models, transforms
# from PIL import Image
# import os
# import math
# import numpy as np
#
#
# from Utils.loss_fn import ArcFaceLoss
#
# # ==========================================
# # 2. 数据集定义
# # ==========================================
# class ATTFaceDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []
#
#         if not os.path.exists(root_dir):
#             raise RuntimeError(f"Dataset path not found: {root_dir}")
#
#         # 遍历 s1 ~ s40
#         for class_dir in os.listdir(root_dir):
#             dir_path = os.path.join(root_dir, class_dir)
#             if os.path.isdir(dir_path) and class_dir.startswith('s'):
#                 try:
#                     label = int(class_dir[1:]) - 1
#                 except ValueError:
#                     continue
#
#                 for img_name in os.listdir(dir_path):
#                     if img_name.lower().endswith(('.pgm', '.jpg', '.png')):
#                         self.image_paths.append(os.path.join(dir_path, img_name))
#                         self.labels.append(label)
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         # 确保转为 RGB
#         image = Image.open(img_path).convert('RGB')
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, label
#
#
# # ==========================================
# # 3. 模型定义
# # ==========================================
# class ResNet18Embedding(nn.Module):
#     def __init__(self, embedding_dim=512, pretrained=True):
#         super().__init__()
#         self.model = models.resnet18(pretrained=pretrained)
#         fc_in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(fc_in_features, embedding_dim)
#         self.bn = nn.BatchNorm1d(embedding_dim)
#
#     def forward(self, x):
#         features = self.model(x)
#         features = self.bn(features)
#         return features
#
#
# # ==========================================
# # 4. 评估函数 (Evaluation Function)
# # ==========================================
# def evaluate(backbone, head, dataloader, device, criterion):
#     """
#     计算在给定数据集上的 Loss 和 Accuracy
#     """
#     backbone.eval()
#     head.eval()
#
#     total_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # 1. 提取特征并归一化
#             features = backbone(inputs)
#             features = F.normalize(features, p=2, dim=1)
#
#             # 2. 获取 Logits
#             # 注意：测试时我们通常使用 get_logits (不带margin penalty)，
#             # 这样衡量的是特征与类中心的真实余弦距离。
#             # 如果你想监控 "Training Loss (with margin)" 在测试集上的表现，也可以改用 forward。
#             # 这里我们使用 get_logits 来计算标准的分类准确率。
#             logits = head.get_logits(features)
#
#             # 3. 计算 Loss
#             loss = criterion(logits, labels)
#             total_loss += loss.item() * inputs.size(0)  # 累加 total loss
#
#             # 4. 计算 Accuracy
#             _, predicted = logits.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     avg_loss = total_loss / total
#     acc = 100. * correct / total
#     return avg_loss, acc
#
#
# # ==========================================
# # 5. 主程序
# # ==========================================
# def main():
#     # 配置
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     DATA_PATH = "../../../Datasets/ATTfaces/raw"
#     BATCH_SIZE = 32
#     EMBEDDING_DIM = 512
#     NUM_CLASSES = 40
#     LR = 0.001
#     EPOCHS = 20
#
#     print(f"Using device: {DEVICE}")
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#
#     # 1. 加载完整数据集
#     try:
#         full_dataset = ATTFaceDataset(root_dir=DATA_PATH, transform=transform)
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return
#
#     # 2. 划分训练集与测试集 (80% 训练, 20% 测试)
#     total_size = len(full_dataset)
#     train_size = int(0.8 * total_size)
#     test_size = total_size - train_size
#
#     # 固定随机种子以保证结果可复现
#     generator = torch.Generator().manual_seed(42)
#     train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
#
#     print(f"Total images: {total_size} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")
#
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#
#     # 3. 初始化模型
#     backbone = ResNet18Embedding(embedding_dim=EMBEDDING_DIM).to(DEVICE)
#     metric_fc = ArcFaceLoss(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM).to(DEVICE)
#
#     # 4. 优化器 & Loss
#     optimizer = optim.Adam([
#         {'params': backbone.parameters()},
#         {'params': metric_fc.parameters()}
#     ], lr=LR)
#
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
#
#     # 这里的 criterion 用于 evaluate 中计算普通交叉熵
#     criterion_eval = nn.CrossEntropyLoss()
#
#     # ================= Training Loop =================
#     for epoch in range(EPOCHS):
#         backbone.train()
#         metric_fc.train()
#
#         running_loss = 0.0
#
#         # --- 训练阶段 ---
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#
#             optimizer.zero_grad()
#
#             embeddings = backbone(inputs)
#             embeddings_norm = F.normalize(embeddings, p=2, dim=1)
#
#             # 使用 ArcFace 的 forward (带 Margin 惩罚) 进行训练
#             loss, _ = metric_fc(embeddings_norm, labels)
#
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         scheduler.step()
#
#         # --- 评估阶段 (每个Epoch结束都跑一次) ---
#         # 计算训练集指标 (Monitor convergence)
#         train_loss, train_acc = evaluate(backbone, metric_fc, train_loader, DEVICE, criterion_eval)
#         # 计算测试集指标 (Monitor generalization)
#         test_loss, test_acc = evaluate(backbone, metric_fc, test_loader, DEVICE, criterion_eval)
#
#         # 打印结果
#         print(f"Epoch [{epoch + 1}/{EPOCHS}]")
#         print(f"    Train Loss (Arc): {running_loss / len(train_loader):.4f}")  # 训练时的带Margin Loss
#         print(f"    Eval  Train Set : Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
#         print(f"    Eval  Test  Set : Loss {test_loss:.4f}  | Acc {test_acc:.2f}%")
#         print("-" * 60)
#
#     print("Training Finished.")
#     torch.save(backbone.state_dict(), "resnet18_arcface_backbone.pth")
#
#
# if __name__ == '__main__':
#     main()
#     # Epoch [19/20]
#     #     Train Loss (Arc): 0.0118
#     #     Eval  Train Set : Loss 0.0000 | Acc 100.00%
#     #     Eval  Test  Set : Loss 0.2532  | Acc 96.25%
#     # ------------------------------------------------------------
#     # Epoch [20/20]
#     #     Train Loss (Arc): 0.0019
#     #     Eval  Train Set : Loss 0.0000 | Acc 100.00%
#     #     Eval  Test  Set : Loss 0.2601  | Acc 96.25%
#     # ------------------------------------------------------------
#     # Training Finished.